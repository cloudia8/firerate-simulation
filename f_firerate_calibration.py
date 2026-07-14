"""
Firerate Calibration Module
============================
Три метрики для калибровки параметров Firerate:

  1. Spread Containment  — насколько Firerate сдерживает рост spread
                           при стрессе относительно контрфактического
                           мира без Firerate (effective_rate = 1.0, cap = inf)

  2. User Loss Fairness  — потери обычных пользователей (не паникёров)
                           в стрессовые дни: насколько ставка бьёт по тем
                           кто держит Gratis недолго и конвертирует не в панике

  3. Threshold Stability — корреляция GratisTotalBalance → threshold
                           по всем фракциям; frac* — точка перегиба где
                           рост сети начинает делать протокол устойчивее

Использование:
    from firerate_calibration import FirerateCalibration
    cal = FirerateCalibration(sim.results)
    cal.run_all()

    # или по отдельности:
    sc  = cal.spread_containment()
    ulf = cal.user_loss_fairness()
    ts  = cal.threshold_stability()
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Optional

from _firerate_sim import (
    ProtocolParams, MarketImpactParams, AttackParams,
    update_spread_endogenous, classify_stress,
)
from vectorized_threshold import compute_threshold_map_for_day


# ──────────────────────────────────────────────────────────────
# РЕЗУЛЬТИРУЮЩИЕ СТРУКТУРЫ
# ──────────────────────────────────────────────────────────────

@dataclass
class SpreadContainmentResult:
    """Метрика 1: сдерживание spread."""
    stress_days: int                    # дней со стрессом в симуляции
    spread_peak_with:    float          # пиковый spread С Firerate
    spread_peak_without: float          # пиковый spread БЕЗ Firerate (контрфакт)
    containment_ratio:   float          # spread_peak_with / spread_peak_without
    mean_spread_with:    float          # средний spread в стрессовые дни С
    mean_spread_without: float          # средний spread в стрессовые дни БЕЗ
    mean_containment:    float          # mean_with / mean_without
    # по уровням стресса
    by_level: dict = field(default_factory=dict)


@dataclass
class UserLossFairnessResult:
    """Метрика 2: справедливость потерь."""
    # обычные пользователи — конвертируют в нормальные дни или держат < N дней
    normal_user_mean_loss_pct:  float
    normal_user_p95_loss_pct:   float
    # паникёры — конвертируют именно в стрессовые дни
    panic_user_mean_loss_pct:   float
    panic_user_p95_loss_pct:    float
    # отношение: насколько паникёр платит больше обычного
    panic_to_normal_ratio:      float
    # дни без потерь (ставка = 1.0, cap не задействован)
    pct_days_no_loss:           float
    # дни с потерями > threshold
    pct_days_loss_above_10pct:  float
    pct_days_loss_above_30pct:  float


@dataclass
class ThresholdStabilityResult:
    """Метрика 3: стабильность threshold."""
    fractions: list[float]
    correlations_gratis: list[float]     # corr(GratisTotalBalance, threshold_frac)
    correlations_spread: list[float]     # corr(SpreadEnd, threshold_frac)
    mean_thresholds:     list[float]
    std_thresholds:      list[float]
    frac_star:           Optional[float]  # точка перегиба (corr > 0)
    frac_star_interp:    Optional[float]  # интерполированная frac*
    safe_pct:            list[float]      # % дней где threshold = inf


# ──────────────────────────────────────────────────────────────
# КОНТРФАКТИЧЕСКАЯ СИМУЛЯЦИЯ (без Firerate)
# ──────────────────────────────────────────────────────────────

def simulate_spread_without_firerate(
    results: pd.DataFrame,
    market: MarketImpactParams,
    b_network: float,
    initial_spread: float = 0.015,
) -> pd.Series:
    """
    Пересчитывает spread день за днём БЕЗ Firerate:
    effective_rate = 1.0 (нет штрафа), cap = inf (нет ограничения объёма).
    sell_volume_usd = GratisConverted × CoenPrice (весь объём выходит без ограничений).

    Это контрфактический мир — что было бы если бы Firerate не существовал.
    GratisConverted берётся из реальной симуляции (решения пользователей те же),
    но Firerate не снижает effective_rate и не обрезает объём.
    """
    spread = initial_spread
    spreads_no_firerate = []

    for _, row in results.iterrows():
        spreads_no_firerate.append(spread)

        # контрфакт: без Firerate конвертируется ПОЛНЫЙ запрошенный объём
        # GratisRequested = то что пользователи хотели конвертировать ДО cap и ставки
        # GratisConverted = то что реально конвертировалось ПОСЛЕ Firerate
        # Используем GratisRequested чтобы не брать уже срезанный объём
        if 'GratisRequested' in row.index:
            gratis_volume = float(row.get('GratisRequested', 0) or 0)
        else:
            # fallback: если колонки нет — реконструируем из effective_rate
            converted  = float(row.get('GratisConverted', 0) or 0)
            eff_rate   = float(row.get('FirerateEffectiveRate', 1.0) or 1.0)
            fill_ratio = converted / float(row.get('GratisTotalBalance', converted) or converted or 1)
            # приближение: если effective_rate < 1 значит был штраф
            gratis_volume = converted / eff_rate if eff_rate > 0 else converted

        coen_price      = float(row.get('CoenPrice', 1) or 1)
        # без Firerate: весь запрошенный Gratis конвертируется 1:1 в coen
        sell_volume_usd = gratis_volume * coen_price

        spread = update_spread_endogenous(
            prev_spread=spread,
            sell_volume_usd=sell_volume_usd,
            B=b_network,
            params=market,
        )

    return pd.Series(spreads_no_firerate, index=results.index)

def calibrate_thresholds(
    results: pd.DataFrame,
    baseline_days: int = 100,
    theta1_multiplier: float = 2.0,
    theta2_multiplier: float = 4.0,
    spread_col: str = 'SpreadEnd',
    verbose: bool = True,
) -> tuple[float, float]:
    """
    Автоматически калибрует theta1 и theta2 из базового spread первых N дней.

    Идея: "нормальный" spread определяется из первых baseline_days дней
    (до того как сеть входит в устойчивый режим). Пороги стресса должны быть
    значительно выше этого базового уровня — иначе Firerate триггерится
    при обычной рыночной активности.

    theta1 = percentile_75(baseline_spread) × theta1_multiplier  (умеренный стресс)
    theta2 = percentile_75(baseline_spread) × theta2_multiplier  (сильный стресс)

    Args:
        results:             sim.results из CredisSimulation
        baseline_days:       сколько первых дней брать как baseline
        theta1_multiplier:   во сколько раз выше baseline → умеренный стресс
        theta2_multiplier:   во сколько раз выше baseline → сильный стресс
        spread_col:          название колонки spread в results
        verbose:             печатать результат

    Returns:
        (theta1, theta2) — откалиброванные пороги
    """
    if spread_col not in results.columns:
        raise ValueError(f"Колонка '{spread_col}' не найдена в results")

    spread = pd.to_numeric(results[spread_col], errors='coerce')
    baseline = spread.iloc[:baseline_days].dropna()

    if len(baseline) == 0:
        raise ValueError(f"Нет данных в первых {baseline_days} днях")

    baseline_p50 = float(baseline.quantile(0.50))
    baseline_p75 = float(baseline.quantile(0.75))
    baseline_p95 = float(baseline.quantile(0.95))

    theta1 = baseline_p75 * theta1_multiplier
    theta2 = baseline_p75 * theta2_multiplier

    if verbose:
        print("=" * 65)
        print("АВТО-КАЛИБРОВКА ПОРОГОВ FIRERATE")
        print("=" * 65)
        print(f"  Baseline период:          первые {baseline_days} дней")
        print(f"  Baseline spread P50:      {baseline_p50:.4f}  ({baseline_p50*100:.2f}%)")
        print(f"  Baseline spread P75:      {baseline_p75:.4f}  ({baseline_p75*100:.2f}%)")
        print(f"  Baseline spread P95:      {baseline_p95:.4f}  ({baseline_p95*100:.2f}%)")
        print()
        print(f"  theta1 = P75 × {theta1_multiplier}  = {theta1:.4f}  ({theta1*100:.2f}%)")
        print(f"  theta2 = P75 × {theta2_multiplier}  = {theta2:.4f}  ({theta2*100:.2f}%)")
        print()

        # показываем сколько дней попало бы в стресс с новыми порогами
        n_stress1 = int((spread >= theta1).sum())
        n_stress2 = int((spread >= theta2).sum())
        n_total   = len(spread.dropna())
        print(f"  С новыми порогами:")
        print(f"    Дней умеренного стресса:  {n_stress1} ({n_stress1/n_total*100:.1f}%)")
        print(f"    Дней сильного стресса:    {n_stress2} ({n_stress2/n_total*100:.1f}%)")
        print(f"    Дней нормы:               {n_total-n_stress1} ({(n_total-n_stress1)/n_total*100:.1f}%)")
        print()
        print(f"  Обновите ProtocolParams:")
        print(f"    proto = ProtocolParams(theta1={theta1:.4f}, theta2={theta2:.4f})")

    return theta1, theta2


# ──────────────────────────────────────────────────────────────
# ГЛАВНЫЙ КЛАСС
# ──────────────────────────────────────────────────────────────

class FirerateCalibration:
    """
    Калибровочный анализ для трёх метрик Firerate.

    Args:
        results:        sim.results из CredisSimulation (DataFrame)
        proto:          ProtocolParams (theta1, theta2, r1, r2, v_cap1, v_cap2)
        market:         MarketImpactParams для расчёта spread
        b_network:      глубина рынка (USD) — используется для контрфакта
        fraction_cols:  маппинг frac → column_name для threshold колонок
        normal_hold_threshold_days: дни держания ниже которого пользователь
                                    считается "обычным" (не паникёром)
    """

    FRACTION_COLUMNS = {
        # 0.001: 'ThresholdF0_1pct',
        0.005: 'ThresholdF0_5pct',
        # 0.010: 'ThresholdF1pct',
        0.050: 'ThresholdF5pct',
        # 0.100: 'ThresholdF10pct',
        0.250: 'ThresholdF25pct',
        0.500: 'ThresholdF50pct',
        1.000: 'ThresholdF100pct',
    }

    def __init__(
        self,
        results: pd.DataFrame,
        proto: ProtocolParams = None,
        market: MarketImpactParams = None,
        b_network: float = 1_000_000.0,
        fraction_cols: dict = None,
        normal_hold_threshold_days: int = 14,
    ):
        self.results = results.copy()
        self.proto   = proto  or ProtocolParams()
        self.market  = market or MarketImpactParams()
        # guard: sim.firerate_b_network — legacy-поле, по умолчанию None.
        # None здесь раньше ронял контрфакт (сравнение None <= 0).
        if b_network is None or (isinstance(b_network, (int, float)) and b_network <= 0):
            print("⚠ b_network не задан (legacy-поле = None) — fallback $1M. "
                  "Актуален только для replay-контрфакта; при наличии "
                  "SpreadEndNoFR в results не используется вовсе.")
            b_network = 1_000_000.0
        self.b_network = b_network
        self.fraction_cols = fraction_cols or self.FRACTION_COLUMNS
        self.normal_hold_threshold_days = normal_hold_threshold_days

        # числовые версии threshold колонок
        for frac, col in self.fraction_cols.items():
            if col in self.results.columns:
                num_col = col + '_num'
                self.results[num_col] = (
                    pd.to_numeric(self.results[col], errors='coerce')
                    .replace([np.inf, -np.inf], np.nan)
                )

        # маска стрессовых дней.
        # При наличии часовых колонок: стресс-день = >= min_stress_hours
        # часов на уровне >=1. Прежнее определение (max-уровень дня > 0)
        # помечало день стрессовым из-за ОДНОГО часа на границе —
        # осцилляция классификатора у порога раздувала стресс-статистику.
        self.min_stress_hours = 4
        if ('FirerateStress1Hours' in self.results.columns
                and 'FirerateStress2Hours' in self.results.columns):
            h = (pd.to_numeric(self.results['FirerateStress1Hours'], errors='coerce').fillna(0)
                 + pd.to_numeric(self.results['FirerateStress2Hours'], errors='coerce').fillna(0))
            self.stress_mask = h >= self.min_stress_hours
        elif 'FirerateStressLevel' in self.results.columns:
            self.stress_mask = self.results['FirerateStressLevel'] > 0
        elif 'SpreadEnd' in self.results.columns:
            self.stress_mask = self.results['SpreadEnd'] >= self.proto.theta1
        else:
            self.stress_mask = pd.Series(False, index=self.results.index)

    # ──────────────────────────────────────────────────────────
    # МЕТРИКА 1: SPREAD CONTAINMENT
    # ──────────────────────────────────────────────────────────

    def spread_containment(self, verbose: bool = True) -> SpreadContainmentResult:
        """
        Сравнивает spread С Firerate (из results) и БЕЗ Firerate (контрфакт).
        containment_ratio < 1.0 означает что Firerate сдерживает spread.
        """
        spread_with = pd.to_numeric(
            self.results.get('SpreadEnd', pd.Series(dtype=float)), errors='coerce'
        )

        if 'SpreadEndNoFR' in self.results.columns:
            # paired counterfactual: собственная эволюция балансов/spread
            # мира без Firerate, посчитанная внутри симуляции на общих
            # случайных числах. Корректный контрфакт: нет двойного счёта
            # отложенного объёма, B динамический и одинаковый в обоих мирах.
            spread_without = pd.to_numeric(
                self.results['SpreadEndNoFR'], errors='coerce'
            )
            self._counterfactual_mode = 'paired'
        else:
            # legacy fallback: replay GratisRequested с фиксированным B.
            # СИСТЕМАТИЧЕСКИ ЗАВЫШАЕТ спред без Firerate: GratisRequested
            # инфлирован деферралом, который создал сам Firerate.
            spread_without = simulate_spread_without_firerate(
                self.results, self.market, self.b_network,
                initial_spread=float(spread_with.iloc[0]) if len(spread_with) else 0.015,
            )
            self._counterfactual_mode = 'replay (biased, upgrade sim for paired)'

        stress_idx = self.stress_mask
        n_stress   = int(stress_idx.sum())

        sw_peak  = float(spread_with.max())
        swo_peak = float(spread_without.max())
        ratio    = sw_peak / swo_peak if swo_peak > 0 else np.nan

        sw_mean_stress  = float(spread_with[stress_idx].mean())  if n_stress else np.nan
        swo_mean_stress = float(spread_without[stress_idx].mean()) if n_stress else np.nan
        mean_ratio = sw_mean_stress / swo_mean_stress if swo_mean_stress > 0 else np.nan

        # разбивка по уровням стресса
        by_level = {}
        if 'FirerateStressLevel' in self.results.columns:
            for lvl in [1, 2]:
                mask = self.results['FirerateStressLevel'] == lvl
                if mask.any():
                    by_level[lvl] = {
                        'n_days':          int(mask.sum()),
                        'mean_spread_with': float(spread_with[mask].mean()),
                        'mean_spread_wo':   float(spread_without[mask].mean()),
                        'ratio':           float(spread_with[mask].mean() /
                                                 spread_without[mask].mean())
                                           if spread_without[mask].mean() > 0 else np.nan,
                    }

        result = SpreadContainmentResult(
            stress_days=n_stress,
            spread_peak_with=sw_peak,
            spread_peak_without=swo_peak,
            containment_ratio=ratio,
            mean_spread_with=sw_mean_stress,
            mean_spread_without=swo_mean_stress,
            mean_containment=mean_ratio,
            by_level=by_level,
        )

        if verbose:
            self._print_spread_containment(result)
        return result

    def coen_outflow_reduction(self, verbose: bool = True) -> dict:
        """
        Прямое сравнение coen-оттока: paired counterfactual против факта.
        Заменяет прежнюю прокси-оценку (GratisRequested vs GratisConverted),
        которая двойным счётом учитывала деферрал.

        reduction = 1 − Σ CoenFromConversion / Σ CoenFromConversionNoFR
        """
        if 'CoenFromConversionNoFR' not in self.results.columns:
            raise ValueError("CoenFromConversionNoFR не найден — запустите "
                             "симуляцию с hourly mineCoen (firerate_hourly).")
        with_fr = pd.to_numeric(self.results['CoenFromConversion'],     errors='coerce').fillna(0)
        no_fr   = pd.to_numeric(self.results['CoenFromConversionNoFR'], errors='coerce').fillna(0)

        stress = self.stress_mask
        total_red  = 1 - with_fr.sum() / no_fr.sum() if no_fr.sum() > 0 else np.nan
        stress_red = (1 - with_fr[stress].sum() / no_fr[stress].sum()
                      if no_fr[stress].sum() > 0 else np.nan)
        calm_red   = (1 - with_fr[~stress].sum() / no_fr[~stress].sum()
                      if no_fr[~stress].sum() > 0 else np.nan)

        out = {'total': total_red, 'stress_days': stress_red, 'calm_days': calm_red}
        if verbose:
            print("=" * 65)
            print("COEN OUTFLOW REDUCTION (paired counterfactual)")
            print("=" * 65)
            print(f"  Всего:            {total_red*100:.1f}% снижение оттока coen")
            print(f"  Стрессовые дни:   {stress_red*100:.1f}%" if not np.isnan(stress_red)
                  else "  Стрессовые дни:   N/A")
            print(f"  Спокойные дни:    {calm_red*100:.1f}%" if not np.isnan(calm_red)
                  else "  Спокойные дни:    N/A")
            print()
        return out

    def conversion_delay(self, verbose: bool = True, service_roll: int = 7) -> dict:
        """
        МЕТРИКА 4: ЗАДЕРЖКА КОНВЕРТАЦИИ (цена троттлинга)

        Очередь = ConversionQueueGratis (точный учёт: хотел хотя бы раз,
        не исполнен). Время ожидания по закону Литтла:
            W_t = очередь_t / сервис_t,
        где сервис = rolling-среднее GratisConverted за service_roll дней.
        W — ожидаемое время полного исполнения маргинальной заявки в днях.
        """
        if 'ConversionQueueGratis' not in self.results.columns:
            raise ValueError("ConversionQueueGratis не найден — нужен прогон "
                             "в DEX-режиме с queue-трекингом.")
        q = pd.to_numeric(self.results['ConversionQueueGratis'], errors='coerce')
        srv = pd.to_numeric(self.results['GratisConverted'], errors='coerce')                 .rolling(service_roll, min_periods=1).mean()
        wait = (q / srv.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

        stress = self.stress_mask
        out = {
            'wait_median':        float(wait.median()),
            'wait_p95':           float(wait.quantile(0.95)),
            'wait_max':           float(wait.max()),
            'wait_median_calm':   float(wait[~stress].median()),
            'wait_median_stress': float(wait[stress].median()) if stress.any() else np.nan,
            'queue_share_end':    float((q / pd.to_numeric(
                self.results['GratisTotalBalance'], errors='coerce')).iloc[-1]),
            'wait_series':        wait,
        }
        if verbose:
            print("=" * 65)
            print("МЕТРИКА 4: ЗАДЕРЖКА КОНВЕРТАЦИИ (закон Литтла)")
            print("=" * 65)
            print(f"  Медианное ожидание:            {out['wait_median']:.1f} дней")
            print(f"  P95 ожидание:                  {out['wait_p95']:.1f} дней")
            print(f"  Максимум:                      {out['wait_max']:.1f} дней")
            print(f"  Медиана в спокойные дни:       {out['wait_median_calm']:.1f}")
            print(f"  Медиана в стрессовые дни:      {out['wait_median_stress']:.1f}")
            print(f"  Очередь на конец, доля Gratis: {out['queue_share_end']*100:.1f}%")
            if out['wait_median'] > 7:
                print("  ⚠ Хроническая очередь: троттлинг работает как капитальный")
                print("    контроль — смотреть TVL-фронтир / x-cap'ы, а не пороги.")
            print()
        return out

    def _print_spread_containment(self, r: SpreadContainmentResult) -> None:
        print("=" * 65)
        print("МЕТРИКА 1: SPREAD CONTAINMENT")
        print(f"  (контрфакт: {getattr(self, '_counterfactual_mode', '?')})")
        print("=" * 65)
        print(f"  Стрессовых дней:               {r.stress_days}")
        print()
        print(f"  Пиковый spread С Firerate:     {r.spread_peak_with:.4f}  ({r.spread_peak_with*100:.1f}%)")
        print(f"  Пиковый spread БЕЗ Firerate:   {r.spread_peak_without:.4f}  ({r.spread_peak_without*100:.1f}%)")
        ratio_pct = (1 - r.containment_ratio) * 100 if not np.isnan(r.containment_ratio) else np.nan
        print(f"  Containment ratio (пик):       {r.containment_ratio:.4f}  "
              f"(Firerate снизил пик на {ratio_pct:.1f}%)")
        print()
        print(f"  Средний spread в стрессовые дни:")
        print(f"    С Firerate:                  {r.mean_spread_with:.4f}")
        print(f"    БЕЗ Firerate:                {r.mean_spread_without:.4f}")
        mean_ratio_pct = (1 - r.mean_containment) * 100 if not np.isnan(r.mean_containment) else np.nan
        print(f"    Mean containment ratio:      {r.mean_containment:.4f}  "
              f"(снижение {mean_ratio_pct:.1f}%)")
        if r.by_level:
            print()
            print(f"  По уровням стресса:")
            for lvl, d in sorted(r.by_level.items()):
                label = "умеренный" if lvl == 1 else "сильный"
                print(f"    Уровень {lvl} ({label}):  {d['n_days']} дней  "
                      f"spread {d['mean_spread_with']:.4f} vs {d['mean_spread_wo']:.4f}  "
                      f"ratio={d['ratio']:.3f}")
        print()

    # ──────────────────────────────────────────────────────────
    # МЕТРИКА 2: USER LOSS FAIRNESS
    # ──────────────────────────────────────────────────────────

    def user_loss_fairness(self, verbose: bool = True) -> UserLossFairnessResult:
        """
        Считает потери пользователей по типам:
          - обычные: конвертируют в нормальные дни (stress_level = 0)
          - паникёры: конвертируют именно в стрессовые дни (stress_level > 0)

        Потеря = 1 - FirerateEffectiveRate (за каждый день конвертации).
        """
        if 'FirerateEffectiveRate' not in self.results.columns:
            raise ValueError("FirerateEffectiveRate не найден в results. "
                             "Запустите симуляцию с обновлённым CredisSimulation.")

        rate = pd.to_numeric(self.results['FirerateEffectiveRate'], errors='coerce')
        loss = (1.0 - rate).clip(lower=0)  # потеря на конвертацию в этот день

        normal_mask = ~self.stress_mask
        panic_mask  = self.stress_mask

        normal_loss = loss[normal_mask].dropna()
        panic_loss  = loss[panic_mask].dropna()

        normal_mean  = float(normal_loss.mean())  if len(normal_loss) else 0.0
        normal_p95   = float(normal_loss.quantile(0.95)) if len(normal_loss) else 0.0
        panic_mean   = float(panic_loss.mean())   if len(panic_loss)  else 0.0
        panic_p95    = float(panic_loss.quantile(0.95))  if len(panic_loss)  else 0.0
        ratio        = panic_mean / normal_mean if normal_mean > 0 else np.nan

        pct_no_loss  = float((loss == 0).mean()) * 100
        pct_loss_10  = float((loss > 0.10).mean()) * 100
        pct_loss_30  = float((loss > 0.30).mean()) * 100

        result = UserLossFairnessResult(
            normal_user_mean_loss_pct=normal_mean * 100,
            normal_user_p95_loss_pct=normal_p95 * 100,
            panic_user_mean_loss_pct=panic_mean * 100,
            panic_user_p95_loss_pct=panic_p95 * 100,
            panic_to_normal_ratio=ratio,
            pct_days_no_loss=pct_no_loss,
            pct_days_loss_above_10pct=pct_loss_10,
            pct_days_loss_above_30pct=pct_loss_30,
        )

        if verbose:
            self._print_user_loss_fairness(result)
        return result

    def _print_user_loss_fairness(self, r: UserLossFairnessResult) -> None:
        print("=" * 65)
        print("МЕТРИКА 2: USER LOSS FAIRNESS")
        print("=" * 65)
        print(f"  Обычные пользователи (нормальные дни):")
        print(f"    Средние потери:              {r.normal_user_mean_loss_pct:.2f}%")
        print(f"    P95 потери:                  {r.normal_user_p95_loss_pct:.2f}%")
        print()
        print(f"  Паникёры (стрессовые дни):")
        print(f"    Средние потери:              {r.panic_user_mean_loss_pct:.2f}%")
        print(f"    P95 потери:                  {r.panic_user_p95_loss_pct:.2f}%")
        print()
        if not np.isnan(r.panic_to_normal_ratio):
            print(f"  Коэф. асимметрии (паник/норм): {r.panic_to_normal_ratio:.1f}x")
        print()
        print(f"  Дней без потерь:               {r.pct_days_no_loss:.1f}%")
        print(f"  Дней с потерями > 10%:         {r.pct_days_loss_above_10pct:.1f}%")
        print(f"  Дней с потерями > 30%:         {r.pct_days_loss_above_30pct:.1f}%")
        print()
        # интерпретация
        if np.isinf(r.panic_to_normal_ratio):
            print("  ✓ Идеальная асимметрия: в норме потерь нет вовсе — весь")
            print("    haircut ложится только на стрессовые конвертации")
        elif r.panic_to_normal_ratio > 3:
            print("  ✓ Хорошая асимметрия: паникёры платят значительно больше обычных")
        elif r.panic_to_normal_ratio > 1.5:
            print("  ~ Умеренная асимметрия: механизм работает, но дифференциация слабая")
        else:
            print("  ⚠ Слабая асимметрия: Firerate почти одинаково бьёт по всем пользователям")
        if r.normal_user_mean_loss_pct > 5:
            print("  ⚠ Обычные пользователи теряют > 5% в среднем — возможна retention-проблема")
        print()

    # ──────────────────────────────────────────────────────────
    # МЕТРИКА 3: THRESHOLD STABILITY
    # ──────────────────────────────────────────────────────────

    def threshold_stability(self, verbose: bool = True) -> ThresholdStabilityResult:
        """
        Считает корреляцию GratisTotalBalance → threshold для каждой фракции.
        Находит frac* — точку перегиба где корреляция меняет знак.
        """
        available = {
            frac: col + '_num'
            for frac, col in self.fraction_cols.items()
            if (col + '_num') in self.results.columns
        }
        if not available:
            raise ValueError("Не найдены threshold колонки (_num). "
                             "Проверьте что fraction_cols соответствуют results.")

        gratis = pd.to_numeric(
            self.results.get('GratisTotalBalance', pd.Series(dtype=float)),
            errors='coerce'
        )
        spread = pd.to_numeric(
            self.results.get('SpreadEnd', pd.Series(dtype=float)),
            errors='coerce'
        )

        fractions    = sorted(available.keys())
        corr_gratis  = []
        corr_spread  = []
        means        = []
        stds         = []
        safe_pcts    = []

        for frac in fractions:
            col = available[frac]
            thr = self.results[col]
            orig_col = self.fraction_cols[frac]

            # inf → nan для корреляции
            thr_num = pd.to_numeric(
                self.results.get(orig_col, pd.Series(dtype=float)), errors='coerce'
            ).replace([np.inf, -np.inf], np.nan)

            corr_gratis.append(float(thr_num.corr(gratis)))
            corr_spread.append(float(thr_num.corr(spread)))
            means.append(float(thr_num.mean()))
            stds.append(float(thr_num.std()))

            # % safe = inf в оригинальной колонке
            orig = self.results.get(orig_col)
            if orig is not None:
                n_inf = (pd.to_numeric(orig, errors='coerce') == np.inf).sum()
                safe_pcts.append(float(n_inf / len(self.results) * 100))
            else:
                safe_pcts.append(0.0)

        # найти frac*
        frac_star       = None
        frac_star_interp = None
        for i in range(len(fractions) - 1):
            v0, v1 = corr_gratis[i], corr_gratis[i + 1]
            if not (np.isnan(v0) or np.isnan(v1)):
                if v0 < 0 and v1 >= 0:
                    frac_star = fractions[i + 1]
                    # линейная интерполяция
                    f0, f1 = fractions[i], fractions[i + 1]
                    frac_star_interp = f0 + (f1 - f0) * (-v0) / (v1 - v0)
                    break

        result = ThresholdStabilityResult(
            fractions=fractions,
            correlations_gratis=corr_gratis,
            correlations_spread=corr_spread,
            mean_thresholds=means,
            std_thresholds=stds,
            frac_star=frac_star,
            frac_star_interp=frac_star_interp,
            safe_pct=safe_pcts,
        )

        if verbose:
            self._print_threshold_stability(result)
        return result

    def _print_threshold_stability(self, r: ThresholdStabilityResult) -> None:
        print("=" * 75)
        print("МЕТРИКА 3: THRESHOLD STABILITY")
        print("=" * 75)
        print(f"  {'frac':>8}  {'%safe':>7}  {'mean_thr':>9}  {'std_thr':>8}  "
              f"{'corr_Gratis':>12}  {'corr_Spread':>12}")
        print(f"  {'-'*8}  {'-'*7}  {'-'*9}  {'-'*8}  {'-'*12}  {'-'*12}")

        for i, frac in enumerate(r.fractions):
            cg = r.correlations_gratis[i]
            cs = r.correlations_spread[i]
            arrow = " ←" if (not np.isnan(cg) and cg > 0) else ""
            print(f"  {frac:>8.3f}  {r.safe_pct[i]:>6.1f}%  "
                  f"{r.mean_thresholds[i]:>9.3f}  {r.std_thresholds[i]:>8.3f}  "
                  f"{cg:>+12.4f}{arrow}  {cs:>+12.4f}")

        print()
        if r.frac_star_interp is not None:
            print(f"  frac* ≈ {r.frac_star_interp:.4f}  "
                  f"(интерполяция между {r.fractions[r.fractions.index(r.frac_star)-1]:.3f} "
                  f"и {r.frac_star:.3f})")
            print()
            print(f"  При frac < {r.frac_star_interp:.3f}: рост сети → уязвимость растёт")
            print(f"  При frac > {r.frac_star_interp:.3f}: рост сети → устойчивость растёт")
            print()
            print(f"  Практическое требование к ликвидности рынка:")
            print(f"    B_network ≥ GratisTotalBalance × {r.frac_star_interp:.4f}")
        elif all(cg < 0 for cg in r.correlations_gratis if not np.isnan(cg)):
            last_frac = r.fractions[-1]
            last_corr = r.correlations_gratis[-1]
            first_corr = r.correlations_gratis[0]
            print(f"  ⚠ Корреляция отрицательна на всей сетке.")
            print(f"    frac* > {last_frac:.3f} — нужно расширить b_fractions вверх.")
            if not np.isnan(last_corr) and not np.isnan(first_corr) and last_corr > first_corr:
                slope = (last_corr - first_corr) / (last_frac - r.fractions[0])
                if slope > 0:
                    extrap = last_frac + (-last_corr) / slope
                    print(f"    Экстраполяция: frac* ≈ {extrap:.4f} "
                          f"({extrap*100:.1f}% покрытие Gratis)")
        elif all(cg > 0 for cg in r.correlations_gratis if not np.isnan(cg)):
            print(f"  ✓ Корреляция положительна на всей сетке — "
                  f"протокол устойчив при любом покрытии из сетки.")
        print()

    # ──────────────────────────────────────────────────────────
    # СВОДНЫЙ ОТЧЁТ
    # ──────────────────────────────────────────────────────────

    def calibrate_thresholds(
        self,
        baseline_days: int = 100,
        theta1_multiplier: float = 2.0,
        theta2_multiplier: float = 4.0,
        verbose: bool = True,
    ) -> tuple[float, float]:
        """Метод-обёртка над standalone calibrate_thresholds."""
        return calibrate_thresholds(
            self.results,
            baseline_days=baseline_days,
            theta1_multiplier=theta1_multiplier,
            theta2_multiplier=theta2_multiplier,
            verbose=verbose,
        )

    def run_all(self, verbose: bool = True) -> dict:
        """Запускает все три метрики и возвращает словарь результатов."""
        if 'DHatEnd' in self.results.columns:   # DEX-режим
            if verbose:
                print("=" * 65)
                print("АВТО-КАЛИБРОВКА THETA: пропущена (DEX-режим)")
                print("  SpreadEnd в DEX-режиме — displacement пула, в норме он")
                print("  прижат к ~0 арбитражем (no-arb band), baseline вырожден.")
                print("  Пороги DEX-протокола: delta1/delta2 (depth) и disp1/disp2")
                print("  (displacement) в DexProtocolParams; их калибровка — через")
                print("  lp_resilience и attack report, не через P75 baseline.")
                print()
            theta1 = theta2 = None
        else:
            theta1, theta2 = self.calibrate_thresholds(verbose=verbose)
        sc  = self.spread_containment(verbose=verbose)
        ulf = self.user_loss_fairness(verbose=verbose)
        ts  = self.threshold_stability(verbose=verbose)

        outflow = None
        if 'CoenFromConversionNoFR' in self.results.columns:
            outflow = self.coen_outflow_reduction(verbose=verbose)

        delay = None
        if 'ConversionQueueGratis' in self.results.columns:
            delay = self.conversion_delay(verbose=verbose)

        if verbose:
            self._print_summary(sc, ulf, ts)

        return {
            'theta1_calibrated':   theta1,
            'theta2_calibrated':   theta2,
            'spread_containment':  sc,
            'user_loss_fairness':  ulf,
            'threshold_stability': ts,
            'coen_outflow_reduction': outflow,
            'conversion_delay': delay,
        }

    def _print_summary(
        self,
        sc: SpreadContainmentResult,
        ulf: UserLossFairnessResult,
        ts: ThresholdStabilityResult,
    ) -> None:
        print("=" * 65)
        print("СВОДКА ПО КАЛИБРОВКЕ FIRERATE")
        print("=" * 65)

        # spread containment
        ratio_pct = (1 - sc.mean_containment) * 100 if not np.isnan(sc.mean_containment) else np.nan
        flag_sc = "✓" if not np.isnan(ratio_pct) and ratio_pct > 20 else "⚠"
        print(f"  {flag_sc} Spread containment:    {ratio_pct:.1f}% снижение spread в стрессовые дни")

        # user loss fairness
        pn = ulf.panic_to_normal_ratio
        flag_ulf = ("✓" if (np.isinf(pn) or (not np.isnan(pn) and pn > 3))
                    else "~" if (not np.isnan(pn) and pn > 1.5) else "⚠")
        ratio_str = ("∞ (норм.потери=0 — идеально)" if np.isinf(pn)
                     else f"{pn:.1f}x" if not np.isnan(pn) else "N/A")
        print(f"  {flag_ulf} User loss fairness:    "
              f"паникёры платят {ratio_str} больше обычных  "
              f"(норма: {ulf.normal_user_mean_loss_pct:.1f}%  паника: {ulf.panic_user_mean_loss_pct:.1f}%)")

        # threshold stability
        if ts.frac_star_interp is not None:
            flag_ts = "✓" if ts.frac_star_interp < 0.10 else "~"
            print(f"  {flag_ts} Threshold stability:   frac* ≈ {ts.frac_star_interp:.4f}  "
                  f"(B_min = {ts.frac_star_interp*100:.1f}% от Gratis)")
        else:
            print(f"  ⚠ Threshold stability:   frac* не найдена в текущей сетке")

        print()
        print("  Параметры для пересмотра если метрики не достигнуты:")
        print("    spread_containment < 20%  → снизить r1/r2 или v_cap1/v_cap2")
        print("    panic_to_normal < 1.5x    → увеличить разницу r1 vs r0")
        print("    normal_loss > 5%           → поднять r1 ближе к 1.0")
        print("    frac* > 0.10               → ужесточить v_cap или расширить b_network")


# ──────────────────────────────────────────────────────────────
# ДЕМО
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    n = 300

    gratis_bal = np.cumsum(np.random.lognormal(10, 0.3, n))
    spread     = np.clip(0.01 + np.random.normal(0, 0.02, n), 0.005, 0.3)
    stress_lvl = (spread > 0.03).astype(int) + (spread > 0.08).astype(int)
    eff_rate   = np.where(stress_lvl == 0, 1.0,
                 np.where(stress_lvl == 1, 0.75, 0.40))
    converted  = gratis_bal * 0.05 + np.random.normal(0, 100, n)
    coen_conv  = converted * eff_rate

    def fake_thr(frac):
        base = 0.05 + frac * 1.5
        effect = 0.3 * frac * (gratis_bal / gratis_bal.max())
        return np.clip(base + effect + np.random.normal(0, 0.05, n), 0.05, 1.2)

    df = pd.DataFrame({
        'CoenPrice':           np.ones(n) * 1.0,
        'GratisTotalBalance':  gratis_bal,
        'SpreadEnd':           spread,
        'GratisConverted':     np.abs(converted),
        'CoenFromConversion':  np.abs(coen_conv),
        'FirerateStressLevel': stress_lvl,
        'FirerateEffectiveRate': eff_rate,
        'ThresholdF0_1pct':   fake_thr(0.001),
        'ThresholdF0_5pct':   fake_thr(0.005),
        'ThresholdF1pct':     fake_thr(0.010),
        'ThresholdF5pct':     fake_thr(0.050),
        'ThresholdF10pct':    fake_thr(0.100),
        'ThresholdF25pct':    fake_thr(0.250),
    })

    cal = FirerateCalibration(df, b_network=1_000_000)
    cal.run_all()