"""
DEX-native market model для Firerate
=====================================

Заменяет CEX-модель «spread + линейный impact» на AMM-микроструктуру,
по результатам [Research] DEX-Native Depth и [Research] Panic Sales:

1. ПУЛ (constant-product с виртуальным резервом, калиброванным на
   измеримую depth-1%). Смещение цены от продажи V считается ТОЧНО
   по бондинговой кривой — сатурация естественная, артефакт
   MAX_SPREAD=5.0 исчезает. Constant-L ≈ CP для ходов ≤1%
   (медиана пересечённых тиков = 0 в исследовании), для больших
   ходов CP даёт корректную выпуклость.

2. ЛИКВИДНОСТЬ LP как переменная состояния с тремя динамиками:
   - структурная кривая depth-1%/TVL первых 180 дней (эмпирическая
     таблица из исследования: 1.85% → ~0.1%);
   - стрессовый вывод (LP пугаются резких ходов/смещения);
   - медленное возвращение (half-life ~60ч — эмпирия 49–90ч
     восстановления из Panic Sales);
   - POL-floor: protocol-owned liquidity, неустранимая атакующим.

3. КЛАССИФИКАТОР СТРЕССА на DEX-сигналах вместо спреда:
   - d̂ = TWAL(depth) / rolling-baseline (первичный);
   - displacement пула от fair price (со-первичный);
   - depth-only триггер БЕЗ подтверждения потоком ограничен уровнем 1
     (анти liquidity-pull манипуляция);
   - гистерезис: эскалация за 1 окно, деэскалация после N подряд
     спокойных окон с более жёсткими exit-порогами.

4. ЭНДОГЕННЫЕ CAP'Ы: v_cap_i = объём, смещающий цену пула не более
   чем на x_i за окно, — обращение slippage-функции от ТЕКУЩЕГО
   состояния ликвидности. Абсолютных констант нет; условие
   стресс-аттрактора B* закрыто конструктивно: троттлированный поток
   не может создавать смещение быстрее, чем его закрывает арбитраж.

5. COST-OF-ATTACK, выводимый из кривой пула (не placeholder k·B):
   - displacement-атака: слиппедж входа/выхода + fees + стоимость
     удержания против арбитражного потока;
   - liquidity-pull: требуемая доля LP и её feasibility при POL-floor;
   - wash-объём: fee-стоимость инфляции q.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Optional


# ──────────────────────────────────────────────────────────────
# CP-МАТЕМАТИКА (виртуальный резерв ↔ depth-1%)
# ──────────────────────────────────────────────────────────────

_D1_FACTOR = 1.0 / math.sqrt(1.0 - 0.01) - 1.0   # ≈ 0.005038: V(1%) = R * factor


def reserve_from_depth(d1_usd: float) -> float:
    """Виртуальный одностронний USD-резерв R из измеримой depth-1%."""
    return d1_usd / _D1_FACTOR


def displacement_from_sell(v_usd: float, R: float) -> float:
    """Смещение цены вниз от продажи нотионала V в CP-пул: 1 − 1/(1+V/R)²."""
    if R <= 0:
        return 1.0
    return 1.0 - 1.0 / (1.0 + v_usd / R) ** 2


def volume_to_move(x: float, R: float) -> float:
    """Объём (USD), смещающий цену CP-пула на долю x вниз. Обращение slippage."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return float("inf")
    return R * (1.0 / math.sqrt(1.0 - x) - 1.0)


# ──────────────────────────────────────────────────────────────
# ПАРАМЕТРЫ
# ──────────────────────────────────────────────────────────────

# Эмпирическая кривая depth-1% как доля TVL (DEX-Native Depth §XI.4)
DEPTH_CURVE_DAYS = np.array([1, 3, 7, 14, 21, 30, 45, 60, 90, 120, 150, 180], dtype=float)
DEPTH_CURVE_FRAC = np.array([0.0185, 0.0167, 0.0141, 0.0113, 0.0071, 0.0053,
                             0.0036, 0.0049, 0.0008, 0.0013, 0.0015, 0.0011])


def depth_frac_of_tvl(age_days: float) -> float:
    """Структурная depth-1%/TVL по возрасту пула (лог-интерполяция таблицы)."""
    a = max(1.0, min(age_days, 180.0))
    return float(np.interp(np.log(a), np.log(DEPTH_CURVE_DAYS), DEPTH_CURVE_FRAC))


@dataclass
class DexPoolParams:
    fee_tier: float = 0.003          # 0.3% — типовой tier для нового токена
    arb_strength: float = 0.5        # доля гэпа pool↔fair, закрываемая арбитражем за час
    lp_move_trigger: float = 0.02    # |ход цены| за окно, пугающий LP
    lp_disp_trigger: float = 0.03    # displacement, пугающий LP
    lp_out_rate: float = 0.15        # доля health, теряемая за стрессовое окно
    lp_return_halflife_h: float = 60.0  # эмпирия: MM/LP возвращаются 49–90ч
    pol_floor_frac: float = 0.15     # POL: доля baseline-depth (ЭКЗОГЕННЫЙ режим TVL)
    baseline_window_h: int = 30 * 24 # rolling-baseline depth (30 дней часовых окон)

    # ── ЭНДОГЕННЫЙ TVL: доходностное равновесие частных LP ──────
    # Включается, когда DexPoolState.pol_usd > 0. Тогда:
    #   TVL_total(t) = POL + h × TVL_priv(t)
    #   APR частных LP = fee × объём_через_пул(годовой) / TVL_total
    #   равновесие: частный капитал заходит, пока APR ≥ required →
    #   TVL_total* = fee × V_daily × 365 / lp_required_apr
    #   TVL_priv*  = max(0, TVL_total* − POL)
    # TVL_priv адаптируется к цели с полураспадом tvl_adjust_halflife_h
    # (латентность аллокации LP-капитала); испуг h бьёт только по
    # частной части — POL иммунен по построению.
    lp_required_apr: float = 0.35        # 🔴 требуемая доходность LP пула
                                         # молодого токена (IL + риск паник)
    tvl_adjust_halflife_h: float = 168.0 # 🔴 неделя: скорость прихода/ухода
                                         # LP-капитала к доходностной цели
    ext_volume_usd_daily: float = 0.0    # 🔴 внешний (спекулятивный) объём
                                         # торгов, USD/день, СВЕРХ конвертации
                                         # и арбитража. Абсолютный, не доля TVL:
                                         # доля даёт расходимость равновесия
                                         # (выручка ∝ TVL → цель ∝ TVL).
    lp_capital_cap_usd: float = float('inf')  # адресуемый частный LP-капитал:
                                         # потолок TVL_priv (модельная граница;
                                         # при apr_req <= APR_sat_organic
                                         # линейное равновесие не ограничено)


@dataclass
class DexProtocolParams:
    """7 governance-параметров DEX-native Firerate. Все безразмерные."""
    delta1: float = 0.60   # d̂ ниже → уровень ≥1 (при подтверждении потоком/смещением)
    delta2: float = 0.35   # d̂ ниже → уровень 2 (при подтверждении)
    disp1: float = 0.02    # displacement ≥ → уровень 1
    disp2: float = 0.05    # displacement ≥ → уровень 2
    x0: float = 0.0075     # НОВОЕ: flow governor нормального режима.
                           # Эмпирика DEX (depth-1% ≈ 0.1% TVL к дню 90) означает,
                           # что органический поток конвертации превосходит
                           # поглощение пула и БЕЗ стресса; без постоянного
                           # ограничителя LP выбивается самим нормальным потоком.
                           # x0 — максимум хода цены/окно в норме (без штрафа ставки).
                           # ВАЖНО: x0 обязан быть СТРОГО НИЖЕ lp_move_trigger
                           # (0.02), иначе разрешённый в норме поток сам пугает LP
                           # и протокол осциллирует в стресс собственным governor'ом.
    x1: float = 0.005      # cap: поток может двигать цену ≤0.5%/окно при уровне 1
    x2: float = 0.002      # cap: ≤0.2%/окно при уровне 2
    r1: float = 0.75       # ставка конвертации при уровне 1
    r2: float = 0.40       # ставка при уровне 2

    # гистерезис (производные, не governance)
    exit_windows: int = 12          # подряд спокойных окон для деэскалации на 1 уровень
    exit_margin_depth: float = 1.15 # exit требует d̂ > delta_i * margin
    exit_margin_disp: float = 0.7   # exit требует disp < disp_i * margin

    # ── режим displacement-сигнала ──────────────────────────────
    # 'excess' (дефолт): стресс = ПРЕВЫШЕНИЕ displacement над рабочей
    #   точкой собственного управляемого потока d*(level) = fee + x_cap/κ.
    #   Мотивация: при хроническом дефиците поглощения (эмпирика depth)
    #   поток на cap'е сам создаёт предсказуемое смещение; абсолютные
    #   пороги ставят рабочую точку вплотную к триггеру (0.018 vs 0.02)
    #   и дают перманентную осцилляцию 0↔1 (~24% ложных стресс-дней).
    #   Excess-режим нетит собственный поток: органика → excess≈0,
    #   триггерят только внешние шоки. disp1/disp2 = пороги ПРЕВЫШЕНИЯ.
    # 'absolute': legacy-поведение (disp1/disp2 — абсолютные пороги).
    disp_mode: str = 'excess'
    kappa_ref: float = 0.5   # референсная эластичность арбитража для d*.
                             # В проде оценивается по наблюдаемому темпу
                             # схлопывания гэпа; в модели = arb_strength.

    def rate(self, level: int) -> float:
        return {0: 1.0, 1: self.r1, 2: self.r2}[level]

    def x_cap(self, level: int) -> float:
        return {0: self.x0, 1: self.x1, 2: self.x2}[level]


# ──────────────────────────────────────────────────────────────
# СОСТОЯНИЕ ПУЛА
# ──────────────────────────────────────────────────────────────

@dataclass
class DexPoolState:
    """
    Пул COEN/USD. p_fair задаётся извне (внешний ценовой путь симуляции),
    p_pool — эндогенный. Ликвидность = POL-floor + h × структурная кривая.
    """
    params: DexPoolParams
    p_fair: float = 1.0
    p_pool: float = 1.0
    h: float = 1.0                    # LP health ∈ (0, 1]
    age_days: float = 1.0
    tvl_usd: float = 100_000.0
    # ── эндогенный TVL (активен при pol_usd > 0) ──────────────
    pol_usd: float = 0.0              # POL: политика казначейства, USD
    tvl_priv_base: float = 0.0        # частный LP-капитал (без испуга h)
    vol_ewma_daily: float = 0.0       # EWMA дневного объёма через пул, USD
    _vol_today: float = 0.0           # накопитель объёма текущего дня
    _baseline: deque = field(default_factory=deque)
    last_window_move: float = 0.0
    last_window_flow: float = 0.0

    @property
    def endogenous_tvl(self) -> bool:
        return self.pol_usd > 0

    def tvl_total(self) -> float:
        """Эффективный TVL: эндогенный (POL + h×частный) или экзогенный."""
        if self.endogenous_tvl:
            return self.pol_usd + self.h * self.tvl_priv_base
        return self.tvl_usd

    def update_private_tvl(self, hours_in_day: float = 24.0) -> dict:
        """
        Конец дня, доходностное равновесие частных LP:

          APR_realized = fee × vol_ewma_daily × 365 / TVL_total
          TVL_total*   = fee × vol_ewma_daily × 365 / lp_required_apr
          TVL_priv*    = max(0, TVL_total* − POL)

        tvl_priv_base адаптируется к цели с полураспадом
        tvl_adjust_halflife_h (латентность аллокации капитала).
        Объём дня сглаживается EWMA (α≈0.1: LP смотрят на ~2 недели
        выручки, а не на один день). Вызывать раз в день из дневного
        цикла; в экзогенном режиме — no-op.
        """
        vol = self._vol_today
        self._vol_today = 0.0
        if not self.endogenous_tvl:
            return {"vol_day": vol, "apr_realized": np.nan,
                    "tvl_priv": 0.0, "tvl_total": self.tvl_usd}
        p = self.params
        alpha = 0.1
        self.vol_ewma_daily = (vol if self.vol_ewma_daily == 0
                               else (1 - alpha) * self.vol_ewma_daily + alpha * vol)
        fee_rev_annual = p.fee_tier * self.vol_ewma_daily * 365.0
        tvl_now = self.tvl_total()
        apr_realized = fee_rev_annual / tvl_now if tvl_now > 0 else 0.0
        tvl_target_total = fee_rev_annual / p.lp_required_apr \
            if p.lp_required_apr > 0 else 0.0
        tvl_priv_target = min(max(0.0, tvl_target_total - self.pol_usd),
                              p.lp_capital_cap_usd)
        w = 1.0 - 0.5 ** (hours_in_day / p.tvl_adjust_halflife_h)
        self.tvl_priv_base += (tvl_priv_target - self.tvl_priv_base) * w
        return {"vol_day": vol, "apr_realized": apr_realized,
                "tvl_priv": self.tvl_priv_base, "tvl_total": self.tvl_total()}

    def pol_share(self) -> float:
        """Доля POL в эффективном TVL (анти-манипуляционная величина)."""
        t = self.pol_usd + self.h * self.tvl_priv_base
        return self.pol_usd / t if t > 0 else 1.0

    # ── ликвидность ──────────────────────────────────────────
    def depth_structural(self) -> float:
        if self.endogenous_tvl:
            # структурная норма: POL + частная база БЕЗ испуга
            return depth_frac_of_tvl(self.age_days) * (self.pol_usd + self.tvl_priv_base)
        return depth_frac_of_tvl(self.age_days) * self.tvl_usd

    def depth_1pct(self) -> float:
        if self.endogenous_tvl:
            # испуг h бьёт только по частной части; POL иммунен
            return depth_frac_of_tvl(self.age_days) * (self.pol_usd + self.h * self.tvl_priv_base)
        base = self.depth_structural()
        return self.params.pol_floor_frac * base + self.h * base

    def depth_baseline(self) -> float:
        """Rolling median НОРМИРОВАННОЙ (u=d/структурная) глубины."""
        if not self._baseline:
            return 1.0
        return float(np.median(self._baseline))

    def d_hat(self) -> float:
        """d̂ = u_now / median(u): чистый LP-drawdown, без структурного дрейфа."""
        b = self.depth_baseline()
        struct = self.depth_structural()
        u_now = self.depth_1pct() / struct if struct > 0 else 1.0
        return u_now / b if b > 0 else 1.0

    # ── смещение ──────────────────────────────────────────────
    @property
    def displacement(self) -> float:
        """Положительное = пул НИЖЕ fair (давление продаж)."""
        return max(0.0, 1.0 - self.p_pool / self.p_fair)

    # ── один batch-window ─────────────────────────────────────
    def apply_window(self, sell_volume_usd: float, window_len_h: float = 1.0) -> dict:
        """
        Порядок внутри окна: продажи двигают пул вниз по кривой →
        LP реагируют на реализованный ход/смещение → арбитраж закрывает
        часть гэпа к fair → LP health медленно восстанавливается.
        Возвращает метрики окна.
        """
        p = self.params
        R = reserve_from_depth(self.depth_1pct())
        p_before = self.p_pool

        # 1. поток продаж по бондинговой кривой
        if sell_volume_usd > 0:
            disp = displacement_from_sell(sell_volume_usd, R)
            self.p_pool *= (1.0 - disp)
        move = abs(self.p_pool / p_before - 1.0)

        # 2. реакция LP на стресс окна
        stressed = (move >= p.lp_move_trigger) or (self.displacement >= p.lp_disp_trigger)
        if stressed:
            self.h *= (1.0 - p.lp_out_rate) ** window_len_h

        # 3. арбитраж тянет пул к fair (частичное закрытие гэпа).
        #    Нотионал арб-свопа — из CP-кривой: |Δ√ratio| × R — идёт в
        #    fee-выручку пула (эндогенный TVL).
        kappa = 1.0 - (1.0 - p.arb_strength) ** window_len_h
        p_pre_arb = self.p_pool
        self.p_pool = self.p_pool * (self.p_fair / self.p_pool) ** kappa
        vol_arb = R * abs(math.sqrt(self.p_pool / p_pre_arb) - 1.0)

        # накопитель дневного объёма через пул (продажи + арбитраж +
        # внешний спекулятивный) — для доходностного равновесия LP
        vol_ext = p.ext_volume_usd_daily / 24.0 * window_len_h
        self._vol_today += sell_volume_usd + vol_arb + vol_ext

        # 4. медленное возвращение LP (49–90ч)
        w_in = 1.0 - 0.5 ** (window_len_h / p.lp_return_halflife_h)
        self.h = min(1.0, self.h + (1.0 - self.h) * w_in)

        # 5. baseline лог — НОРМИРОВАННАЯ глубина u = d/структурная
        # (возрастная нормализация: сжатие depth/TVL по кривой возраста
        # не должно читаться как drawdown)
        struct = self.depth_structural()
        self._baseline.append(self.depth_1pct() / struct if struct > 0 else 1.0)
        while len(self._baseline) > p.baseline_window_h:
            self._baseline.popleft()

        self.last_window_move = move
        self.last_window_flow = sell_volume_usd
        return {"move": move, "displacement": self.displacement,
                "depth": self.depth_1pct(), "d_hat": self.d_hat()}

    def copy(self) -> "DexPoolState":
        c = DexPoolState(params=self.params, p_fair=self.p_fair, p_pool=self.p_pool,
                         h=self.h, age_days=self.age_days, tvl_usd=self.tvl_usd,
                         pol_usd=self.pol_usd, tvl_priv_base=self.tvl_priv_base,
                         vol_ewma_daily=self.vol_ewma_daily)
        c._baseline = deque(self._baseline, maxlen=None)
        return c


# ──────────────────────────────────────────────────────────────
# КЛАССИФИКАТОР С ГИСТЕРЕЗИСОМ
# ──────────────────────────────────────────────────────────────

class DexStressClassifier:
    """
    Уровень стресса из (d̂, displacement, flow-подтверждение).

    Правила:
      raw-уровень = max(depth-уровень, displacement-уровень), где
      depth-only сигнал БЕЗ подтверждения (нет ни потока, ни смещения)
      ограничен уровнем 1 — вывод ликвидности сам по себе дёшев и
      обратим, полная эскалация требует исполненного давления продаж.

      Эскалация мгновенная (spread/depth TTP < 1ч по эмпирике).
      Деэскалация: только после exit_windows подряд окон, удовлетворяющих
      БОЛЕЕ ЖЁСТКИМ exit-порогам, и только на один уровень за раз —
      длинный хвост из эмпирики восстановления (49–90ч).
    """

    def __init__(self, proto: DexProtocolParams, fee: float = 0.01):
        self.proto = proto
        self.fee = fee
        self.level = 0
        self._calm_streak = 0

    def _d_op(self, level: int) -> float:
        """Рабочая точка displacement собственного потока на cap'е уровня."""
        x = self.proto.x_cap(level)
        if x == float("inf"):
            return self.fee
        k = self.proto.kappa_ref
        return self.fee + x * (1.0 - k) / k

    def _disp_signal(self, disp: float) -> float:
        """Сигнальное смещение: excess над рабочей точкой или абсолют."""
        if self.proto.disp_mode == 'excess':
            return max(0.0, disp - self._d_op(self.level))
        return disp

    def _raw_level(self, d_hat: float, disp: float, flow_confirm: bool) -> int:
        pr = self.proto
        ds = self._disp_signal(disp)
        lvl_disp = 2 if ds >= pr.disp2 else (1 if ds >= pr.disp1 else 0)
        lvl_depth = 2 if d_hat <= pr.delta2 else (1 if d_hat <= pr.delta1 else 0)
        if lvl_depth == 2 and not (flow_confirm or lvl_disp >= 1):
            lvl_depth = 1   # анти liquidity-pull: depth-only → максимум уровень 1
        return max(lvl_disp, lvl_depth)

    def _calm_for_exit(self, d_hat: float, disp: float) -> bool:
        pr = self.proto
        tgt = self.level
        delta = pr.delta1 if tgt == 1 else pr.delta2
        dsp = pr.disp1 if tgt == 1 else pr.disp2
        ds = self._disp_signal(disp)
        return (d_hat > delta * pr.exit_margin_depth) and (ds < dsp * pr.exit_margin_disp)

    def update(self, d_hat: float, disp: float, flow_confirm: bool) -> int:
        raw = self._raw_level(d_hat, disp, flow_confirm)
        if raw > self.level:                      # эскалация мгновенно
            self.level = raw
            self._calm_streak = 0
        elif self.level > 0:
            if raw < self.level and self._calm_for_exit(d_hat, disp):
                self._calm_streak += 1
                if self._calm_streak >= self.proto.exit_windows:
                    self.level -= 1               # деэскалация на 1 уровень
                    self._calm_streak = 0
            else:
                self._calm_streak = 0
        return self.level

    def v_cap_usd(self, pool: DexPoolState) -> float:
        """Эндогенный cap: объём, двигающий цену не более чем на x_level."""
        x = self.proto.x_cap(self.level)
        if x == float("inf"):
            return float("inf")
        return volume_to_move(x, reserve_from_depth(pool.depth_1pct()))


# ──────────────────────────────────────────────────────────────
# COST-OF-ATTACK — из кривой пула, не placeholder
# ──────────────────────────────────────────────────────────────

def attack_cost_displacement(
    pool: DexPoolState,
    proto: DexProtocolParams,
    target_level: int = 2,
    hold_windows: int = 12,
    overshoot: float = 1.2,
) -> dict:
    """
    Displacement-атака: продать в пул достаточно, чтобы сместить цену
    за порог disp_i, и удерживать против арбитража hold_windows окон
    (например, до истечения гистерезиса + запас).

    Компоненты (все выводятся из CP-кривой; предположение — арбитраж
    восстанавливает к fair, т.е. атакующий выкупает/докидывает по ценам
    между displaced и fair):

      C_entry  ≈ V0·α/2 + fee·V0   — слиппедж входа (средняя цена
                 исполнения √(1−α) от fair) плюс комиссия;
      C_hold   ≈ Σ окон [V_arb·(α/2 + fee)] — каждое окно арбитраж
                 закрывает κ гэпа объёмом V_arb; чтобы удержать α,
                 атакующий перепродаёт тот же объём с тем же дисконтом;
      C_exit   ≈ V0·α/2 + fee·V0   — выкуп обратно по восстанавливающейся
                 цене (симметричная оценка сверху).

    Возвращает разбивку и полную стоимость. Заменяет старую модель
    C_capital + C_fill + C_signal: fake-order-вектора на AMM нет,
    капитал реально исполняется в кривую.
    """
    p = pool.params
    alpha = (proto.disp1 if target_level == 1 else proto.disp2) * overshoot
    R = reserve_from_depth(pool.depth_1pct())

    v0 = volume_to_move(alpha, R)
    c_entry = v0 * alpha / 2.0 + p.fee_tier * v0

    kappa = p.arb_strength
    # объём арбитража, закрывающего κ гэпа: движение цены от (1−α) к (1−α(1−κ))
    ratio = (1.0 - alpha * (1.0 - kappa)) / (1.0 - alpha)
    v_arb = R * (math.sqrt(ratio) - 1.0)
    c_hold_per_window = v_arb * (alpha / 2.0 + p.fee_tier)
    c_hold = c_hold_per_window * hold_windows

    c_exit = v0 * alpha / 2.0 + p.fee_tier * v0

    total = c_entry + c_hold + c_exit
    return {
        "target_level": target_level, "alpha": alpha,
        "depth_1pct": pool.depth_1pct(), "virtual_reserve": R,
        "v0_usd": v0, "v_arb_per_window": v_arb,
        "c_entry": c_entry, "c_hold_per_window": c_hold_per_window,
        "c_hold": c_hold, "c_exit": c_exit, "c_total": total,
        "hold_windows": hold_windows,
        "rate_imposed": proto.rate(target_level),
    }


def attack_cost_liquidity_pull(
    pool: DexPoolState,
    proto: DexProtocolParams,
    attacker_lp_share: float = 0.5,
    daily_pool_volume_usd: Optional[float] = None,
) -> dict:
    """
    Liquidity-pull: атакующий-LP выводит ликвидность, чтобы обрушить d̂.

    Прямой издержки ≈ 0 (вывод бесплатен и обратим) — стоимость это
    упущенные fee-доходы за окно удержания. Поэтому защита структурная:
      1. POL-floor: часть depth атакующему недоступна;
      2. depth-only сигнал без flow-подтверждения ограничен уровнем 1;
      3. TWAL/rolling baseline: мгновенный вывод не двигает baseline.

    Функция отвечает: достижим ли уровень 2 выводом ликвидности вообще,
    какая доля LP для этого нужна, и сколько стоит удержание.
    """
    p = pool.params
    base = pool.depth_baseline()
    struct = pool.depth_structural()
    pol = p.pol_floor_frac * struct
    lp_liquid = pool.h * struct                      # выводимая часть
    attacker_liq = attacker_lp_share * lp_liquid

    # d̂ после вывода атакующим всей своей доли:
    d_after = (pol + lp_liquid - attacker_liq) / base if base > 0 else 1.0

    # какая доля LP нужна чтобы пробить delta2:
    need = pol + lp_liquid - proto.delta2 * base     # ликвидность, которую надо убрать
    share_req_l2 = need / lp_liquid if lp_liquid > 0 else float("inf")
    feasible_l2_depth_only = False                   # ограничено уровнем 1 конструктивно

    fee_income_daily = (p.fee_tier * daily_pool_volume_usd * attacker_lp_share
                        if daily_pool_volume_usd else None)

    return {
        "attacker_lp_share": attacker_lp_share,
        "d_hat_after_pull": d_after,
        "level_reachable_depth_only": 1 if d_after <= proto.delta1 else 0,
        "share_required_for_delta2": max(0.0, min(share_req_l2, 1.0))
                                     if share_req_l2 <= 1.0 else float("inf"),
        "level2_reachable_without_flow": feasible_l2_depth_only,
        "pol_floor_usd": pol,
        "cost_forgone_fees_per_day": fee_income_daily,
        "note": ("Depth-only триггер ограничен уровнем 1; уровень 2 требует "
                 "исполненного sell-flow или displacement — атакующему придётся "
                 "добавить displacement-атаку (см. attack_cost_displacement)."),
    }


def attack_cost_wash_volume(pool: DexPoolState, inflate_ratio: float,
                            baseline_hourly_volume_usd: float) -> dict:
    """Инфляция сигнала q wash-свопами: стоимость = fee × добавленный объём."""
    added = baseline_hourly_volume_usd * max(0.0, inflate_ratio - 1.0)
    return {"added_volume_per_h": added,
            "cost_per_window": added * pool.params.fee_tier,
            "note": "Wash-объём на AMM платит полный fee_tier — самоценообразующая защита."}


def print_attack_report(pool: DexPoolState, proto: DexProtocolParams,
                        daily_volume_usd: float = None) -> None:
    print("COST-OF-ATTACK (DEX-native, из кривой пула)")
    print("─" * 72)
    print(f"  depth-1%: ${pool.depth_1pct():,.0f}   d̂={pool.d_hat():.2f}   "
          f"TVL=${pool.tvl_usd:,.0f}   age={pool.age_days:.0f}d   fee={pool.params.fee_tier:.2%}")
    for lvl in (1, 2):
        r = attack_cost_displacement(pool, proto, target_level=lvl,
                                     hold_windows=proto.exit_windows)
        print(f"\n  Displacement-атака → уровень {lvl} (α={r['alpha']:.1%}, "
              f"удержание {r['hold_windows']}ч):")
        print(f"    V0 (вход):            ${r['v0_usd']:>12,.0f}")
        print(f"    C_entry:              ${r['c_entry']:>12,.0f}")
        print(f"    C_hold (всего):       ${r['c_hold']:>12,.0f}  "
              f"(${r['c_hold_per_window']:,.0f}/окно, V_arb=${r['v_arb_per_window']:,.0f})")
        print(f"    C_exit:               ${r['c_exit']:>12,.0f}")
        print(f"    ── C_total:           ${r['c_total']:>12,.0f}")
    lp = attack_cost_liquidity_pull(pool, proto, attacker_lp_share=0.5,
                                    daily_pool_volume_usd=daily_volume_usd)
    print(f"\n  Liquidity-pull (доля атакующего 50% LP):")
    print(f"    d̂ после вывода:        {lp['d_hat_after_pull']:.2f}  "
          f"→ достижимый уровень depth-only: {lp['level_reachable_depth_only']}")
    print(f"    POL-floor:             ${lp['pol_floor_usd']:,.0f}")
    if lp['cost_forgone_fees_per_day'] is not None:
        print(f"    Упущенные fees/день:   ${lp['cost_forgone_fees_per_day']:,.0f}")
    print(f"    {lp['note']}")