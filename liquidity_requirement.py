"""
Требование к ликвидности: три раздельных вопроса вместо одной таблицы
=======================================================================

Прежний run_pol_apr смешивал три разных вопроса в одну решётку (POL×APR
наугад). Разводим их по природе:

  1. TVL_required — ФИЗИКА. Минимальный размер пула, при котором сеть
     устойчива (нет спирали / нет хронического severe / приемлемая
     очередь), НЕЗАВИСИМО от того, кто эти деньги дал. Вопрос к рынку/
     дизайну протокола, не к казначейству. Считается lp_shock_frontier-
     подобным бинарным поиском по TVL при worst-case LP-шоку.

  2. POL_min — GOVERNANCE. Минимальная НЕУВОДИМАЯ доля TVL_required
     (протокольная ликвидность). Не "часть TVL, которую даёт
     казначейство по факту" — а верхняя граница, требуемая анти-
     манипуляционным инвариантом (POL ≥ delta2) и устойчивостью к
     полному бегству частного капитала (worst-case: TVL становится
     POL_min и должен сам держать нулевой severe). Это политика:
     казначейство МОЖЕТ дать больше POL_min (вплоть до 100% TVL) —
     наш расчёт даёт минимум, не рекомендуемое значение.

  3. implied_apr_for_gap — РЫНОК. Если казначейство даёт ровно POL_min,
     остаток gap = TVL_required − POL_min должен закрыть частный
     капитал. Вопрос — на какую доходность он должен рассчитывать,
     чтобы захотеть закрыть именно этот gap (не "что получится при
     произвольном apr_req", а обращение доходностного равновесия под
     ЗАДАННЫЙ целевой TVL_priv = gap).

Использование (на реальной симуляции):
    from dex_credis_simulation import CredisSimulation
    from liquidity_requirement import liquidity_requirement_report

    sim = CredisSimulation(...)
    sim.run(verbose=False)                       # для снятия состояния/эмиссии
    report = liquidity_requirement_report(sim)
"""
from __future__ import annotations
import numpy as np
from dataclasses import replace
from typing import Optional

from dex_market import DexPoolState, DexPoolParams, DexProtocolParams
from firerate_hourly_dex import probe_endogenous_dex
from pol_apr_planner import saturated_apr, absorb_frac_daily


# ──────────────────────────────────────────────────────────────
# 0. TVL_no_throttle — при каком TVL поток вообще не задевает cap
# ──────────────────────────────────────────────────────────────

def tvl_for_zero_throttle(e_daily_usd: float, x0: float, age_days: float) -> float:
    """
    Минимальный TVL, при котором ЕЖЕДНЕВНЫЙ органический поток e_daily_usd
    целиком проходит на уровне 0 (x0-cap), т.е. троттлинг отсутствует
    ВООБЩЕ, не только «нет спирали»/«нет severe» — более жёсткое условие,
    чем min_tvl_for_stability.

    Точное обращение поглощения насыщенного пула (см. absorb_frac_daily,
    §6.4 formulas reference): absorb_frac_daily(x0,age) = доля TVL/день,
    которую cap способен пропустить. Приравнивая e_daily = absorb_frac×TVL:

        TVL_no_throttle = e_daily_usd / absorb_frac_daily(x0, age_days)

    ⚠ Не статичная величина: absorb_frac_daily убывает с возрастом
    (depth_frac сжимается ~×20 за 90 дней), поэтому при неизменной
    эмиссии TVL_no_throttle растёт ~×20 за тот же период — это движущаяся
    цель, планировать нужно траекторию, не число.
    """
    af = absorb_frac_daily(x0, age_days)
    if af <= 0:
        return float('inf')
    return e_daily_usd / af


# ──────────────────────────────────────────────────────────────
# 1. TVL_required — физика устойчивости
# ──────────────────────────────────────────────────────────────

def _survives(tvl_usd: float, gratis_balance, joined_mask, proto,
             pool_params: DexPoolParams, age_days: float,
             sens_grid=(1, 2, 4, 8), probe_hours=96, seed=42,
             worst_case_lp_shock: float = 1.0,
             firerate_on: bool = True,
             n_seeds: int = 1,
             criterion: str = "clean") -> bool:
    """
    Выживает ли пул данного размера при worst-case LP-шоком (вся частная
    ликвидность уходит) под сеткой панической чувствительности.

    criterion='clean' — требуем НОЛЬ severe-часов (жёсткий, устойчивость);
    criterion='no_spiral' — требуем только отсутствие спирали (мягкий).
    """
    pool = DexPoolState(params=pool_params, tvl_usd=tvl_usd, age_days=age_days)
    for _ in range(48):
        pool.apply_window(0.0)   # прогреть baseline
    # мульти-seed консервативное голосование: «выживает» = выживает при
    # ВСЕХ seed'ах. Одиночный probe даёт шум оценки порога ×2-3 между
    # расчётами; 2-3 seed'а стабилизируют месячный трекер.
    for k in range(max(1, n_seeds)):
        sk = None if seed is None else seed + 101 * k
        for sens in sens_grid:
            r = probe_endogenous_dex(
                gratis_balance, joined_mask, pool, clf_level=0, proto=proto,
                hours=probe_hours, panic_sens=sens,
                initial_lp_shock=worst_case_lp_shock,
                firerate_on=firerate_on, seed=sk,
            )
            if r.spiral:
                return False
            if criterion == "clean" and r.hours_severe > 0:
                return False
    return True


def min_tvl_for_stability(
    gratis_balance: np.ndarray,
    joined_mask: np.ndarray,
    proto: DexProtocolParams,
    pool_params: DexPoolParams,
    age_days: float,
    tvl_lo: float = 10_000.0,
    tvl_hi: float = 100_000_000.0,
    tol: float = 0.05,
    criterion: str = "no_spiral",
    firerate_on: bool = False,
    abs_floor: float = 1_000.0,
    abs_ceil: float = 1e9,
    **kwargs,
) -> dict:
    """
    Бинарный поиск минимального TVL, устойчивого к worst-case LP-шоку
    (уходит 100% частной ликвидности).

    ВОПРОС ЗАДАЁТСЯ МИРУ БЕЗ FIRERATE (firerate_on=False): «какая глубина
    нужна рынку, чтобы паника не спирализовалась сама по себе». С
    включённым Firerate вопрос вырожден — спираль недостижима при ЛЮБОМ
    TVL (валюта провала — троттлинг, не коллапс), и поиск деградирует
    в возврат нижней границы (источник храповика warm-start → 0).

    Критерий — "не спирализуется", а не "ноль severe": мгновенный
    100%-шок создаёт легитимный переходный severe (см. lp_resilience).

    Если граница выживает/проваливается — она РАСШИРЯЕТСЯ (до
    abs_floor/abs_ceil), а не возвращается как ответ: возврат границы
    поиска как «ответа» и был механикой храповика.
    """
    lo, hi = max(tvl_lo, abs_floor), min(tvl_hi, abs_ceil)

    # расширяем hi, пока не найдём выживающий (или упор в abs_ceil)
    while not _survives(hi, gratis_balance, joined_mask, proto, pool_params,
                        age_days, criterion=criterion,
                        firerate_on=firerate_on, **kwargs):
        if hi >= abs_ceil:
            return {"tvl_required_no_fr": None, "lo": lo, "hi": hi, "iterations": [],
                   "note": f"даже TVL=${hi:,.0f} не хватает (abs_ceil)"}
        lo, hi = hi, min(hi * 10.0, abs_ceil)

    # расширяем lo вниз, пока не найдём проваливающийся (или упор в abs_floor)
    while _survives(lo, gratis_balance, joined_mask, proto, pool_params,
                    age_days, criterion=criterion,
                    firerate_on=firerate_on, **kwargs):
        if lo <= abs_floor:
            return {"tvl_required_no_fr": lo, "lo": lo, "hi": hi, "iterations": [],
                   "at_floor": True,
                   "note": f"выживает даже на abs_floor=${lo:,.0f} — "
                           f"требование ниже разрешающей способности модели"}
        hi, lo = lo, max(lo / 10.0, abs_floor)

    iterations = []
    while (hi - lo) / hi > tol:
        mid = (lo * hi) ** 0.5   # лог-шаг: TVL-пространство логарифмическое
        ok = _survives(mid, gratis_balance, joined_mask, proto, pool_params,
                       age_days, criterion=criterion, **kwargs)
        iterations.append((mid, ok))
        if ok:
            hi = mid
        else:
            lo = mid
    return {"tvl_required_no_fr": hi, "lo": lo, "hi": hi, "iterations": iterations}


# ──────────────────────────────────────────────────────────────
# 2. POL_min — governance-минимум неуводимой доли
# ──────────────────────────────────────────────────────────────

def pol_min_requirement(
    gratis_balance: np.ndarray,
    joined_mask: np.ndarray,
    proto: DexProtocolParams,
    pool_params: DexPoolParams,
    tvl_required: float,
    age_days: float,
    pol_grid: tuple = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50),
    **kwargs,
) -> dict:
    """
    Минимальная НЕУВОДИМАЯ доля от tvl_required.

    A. Устойчивость: если весь частный капитал уходит, TVL эффективно
       падает до POL — какая минимальная доля POL от tvl_required САМА
       по себе (без единого доллара частных LP) держит ноль severe.
    B. Анти-манипуляция (замкнутая форма): POL_frac >= delta2, иначе
       liquidity-pull одним депозитом-выводом пробивает depth-сигнал
       уровня 2 без единого свопа.

    POL_min = max(A, B) × tvl_required. Это ВЕРХНЯЯ граница требования,
    не рекомендация — казначейство свободно дать больше.
    """
    results = {}
    for frac in sorted(pol_grid):
        ok = _survives(frac * tvl_required, gratis_balance, joined_mask,
                       proto, pool_params, age_days,
                       worst_case_lp_shock=0.0,  # POL — это и есть весь TVL здесь
                       criterion="clean", **kwargs)
        results[frac] = ok

    ok_fracs = [f for f, ok in results.items() if ok]
    pol_frac_A = min(ok_fracs) if ok_fracs else None
    pol_frac_B = proto.delta2

    pol_frac_final = (max(pol_frac_A, pol_frac_B) if pol_frac_A is not None
                      else None)
    return {
        "grid": results,
        "pol_frac_stability": pol_frac_A,
        "pol_frac_antimanip": pol_frac_B,
        "pol_frac_min": pol_frac_final,
        "pol_min_usd": (pol_frac_final * tvl_required
                        if pol_frac_final is not None else None),
        "binding": (None if pol_frac_final is None else
                   "устойчивость" if pol_frac_A is not None and pol_frac_A > pol_frac_B
                   else "анти-манипуляция"),
    }


# ──────────────────────────────────────────────────────────────
# 3. implied_apr_for_gap — рыночный вопрос про остаток
# ──────────────────────────────────────────────────────────────

def implied_apr_for_gap(
    gap_usd: float,
    fee_tier: float,
    vol_ewma_daily_usd: float,
) -> dict:
    """
    Обращение доходностного равновесия (dex_market.update_private_tvl)
    под ЗАДАННЫЙ целевой частный TVL = gap, а не наоборот:

        TVL_priv* = fee × vol_daily × 365 / apr_req
        ⟺ apr_req = fee × vol_daily × 365 / TVL_priv*

    То есть: "при каком apr_req частный капитал САМ выберет вложить
    ровно gap_usd" — прямая, не итеративная формула (в отличие от
    run_pol_apr, где apr_req — вход, а TVL_priv — равновесный выход).

    vol_ewma_daily_usd нужно взять из реального прогона (эмиссия,
    цена, органический поток) — если недоступен, используйте
    saturated_apr() как верхнюю оценку (насыщенный поток = cap).
    """
    if gap_usd <= 0:
        return {"gap_usd": gap_usd, "implied_apr": 0.0,
                "note": "gap <= 0 — POL_min уже покрывает TVL_required, "
                        "частный капитал не нужен"}
    fee_revenue_annual = fee_tier * vol_ewma_daily_usd * 365.0
    apr = fee_revenue_annual / gap_usd
    return {"gap_usd": gap_usd, "fee_revenue_annual": fee_revenue_annual,
           "implied_apr": apr}


# ──────────────────────────────────────────────────────────────
# ОТЧЁТ ЦЕЛИКОМ
# ──────────────────────────────────────────────────────────────

def liquidity_requirement_report(
    gratis_balance: np.ndarray,
    joined_mask: np.ndarray,
    proto: DexProtocolParams,
    pool_params: DexPoolParams,
    age_days: float,
    vol_ewma_daily_usd: float = None,
    fee_tier: float = None,
    verbose: bool = True,
    **search_kwargs,
) -> dict:
    """
    Собирает все три ответа в один вызов. vol_ewma_daily_usd — реальный
    дневной объём органического потока (эмиссия × sell_through + арбитраж);
    если не передан, используется APR_sat-эквивалент (насыщенный поток на
    cap'е x0 из proto) как консервативная верхняя оценка объёма относительно
    gap: vol_daily ≈ (APR_sat/365/fee) × gap — то есть предполагаем, что
    ВЕСЬ насыщенный поток (в его текущей относительной интенсивности к TVL)
    достаточен для фондирования именно gap, не полного TVL_required.
    """
    fee_tier = fee_tier if fee_tier is not None else pool_params.fee_tier
    vol_was_given = vol_ewma_daily_usd is not None

    tvl_kwargs = {k: v for k, v in search_kwargs.items()
                  if k not in ('criterion', 'pol_grid')}
    tvl_res = min_tvl_for_stability(gratis_balance, joined_mask, proto,
                                    pool_params, age_days,
                                    criterion=search_kwargs.get('criterion', 'no_spiral'),
                                    **tvl_kwargs)
    tvl_req = tvl_res["tvl_required_no_fr"]

    if tvl_req is None:
        if verbose:
            print("=" * 70)
            print("TVL_required НЕ НАЙДЕН:", tvl_res["note"])
            print("=" * 70)
        return {"tvl_required_no_fr": None, "note": tvl_res["note"]}

    # pol_min_requirement передаёт kwargs напрямую в _survives (probe),
    # а не в поиск — search-only параметры бинарного поиска TVL там чужие
    probe_kwargs = {k: v for k, v in search_kwargs.items()
                    if k not in ('tol', 'tvl_lo', 'tvl_hi', 'criterion')}
    pol_res = pol_min_requirement(gratis_balance, joined_mask, proto,
                                  pool_params, tvl_req, age_days,
                                  **probe_kwargs)
    pol_min_usd = pol_res["pol_min_usd"] or tvl_req  # если не найден — весь TVL
    gap_usd = max(0.0, tvl_req - pol_min_usd)

    if vol_ewma_daily_usd is None:
        # консервативная оценка ТОЛЬКО как ручка по умолчанию: доля от TVL,
        # подразумеваемая APR_sat при насыщении, применённая к gap.
        # ⚠ Это НЕ прогноз реального объёма — передавайте настоящий
        # vol_ewma_daily_usd с прогона симуляции, где он есть.
        vol_over_tvl = saturated_apr(fee_tier, proto.x0, age_days) / (fee_tier * 2.0 * 365.0)
        vol_ewma_daily_usd = vol_over_tvl * gap_usd

    if gap_usd < max(1_000.0, 0.01 * tvl_req):
        # вырожденный gap (в т.ч. tvl_req на floor поиска) — не считаем
        # implied_apr от микроскопического знаменателя
        apr_res = {"gap_usd": gap_usd, "implied_apr": float('nan'),
                   "note": "gap вырожден"}
    else:
        apr_res = implied_apr_for_gap(gap_usd, fee_tier, vol_ewma_daily_usd)

    out = {
        "tvl_required_no_fr": tvl_req,
        # UX-порог: TVL, при котором Firerate ВООБЩЕ не троттлит (замкнутая
        # форма tvl_for_zero_throttle). Считается от РЕАЛЬНОГО объёма, если
        # он был передан явно; если объём выведен эвристикой-фолбэком
        # (circular: зависит от gap, который сам зависит от tvl_no_throttle
        # через pol_min), оставляем None с пометкой — не показываем число,
        # посчитанное из собственного следствия.
        "tvl_no_throttle": (tvl_for_zero_throttle(vol_ewma_daily_usd, proto.x0, age_days)
                            if vol_was_given else None),
        "pol_min_usd": pol_min_usd,
        "pol_min_frac": pol_res["pol_frac_min"],
        "pol_binding": pol_res["binding"],
        "gap_usd": gap_usd,
        "implied_apr_for_gap": apr_res["implied_apr"],
        "vol_assumed_daily": vol_ewma_daily_usd,
    }
    if verbose:
        print("=" * 70)
        print("ТРЕБОВАНИЕ К ЛИКВИДНОСТИ (три раздельных вопроса)")
        print("=" * 70)
        print(f"  1. TVL_required_no_fr (физика краха):  ${tvl_req:,.0f}")
        print(f"     — минимум глубины, чтобы паника НЕ СПИРАЛИЗОВАЛАСЬ САМА")
        print(f"       (мир без Firerate; с Firerate это почти всегда")
        print(f"       тривиально мало — см. tvl_no_throttle для реального")
        print(f"       порога 'устойчивости' в присутствии Firerate)")
        if out['tvl_no_throttle'] is not None:
            print(f"\n     UX-порог (tvl_no_throttle): ${out['tvl_no_throttle']:,.0f}")
            print(f"     — TVL, при котором Firerate ВООБЩЕ не режет поток "
                  f"(0% троттлинга).")
            print(f"     Firerate не снижает ЭТОТ порог — только защищает от "
                  f"краха ниже него.")
        print()
        print(f"  2. POL_min (governance):       ${pol_min_usd:,.0f} "
              f"({out['pol_min_frac']:.0%} от TVL_required)")
        print(f"     — binding: {out['pol_binding']}")
        print(f"     — это ВЕРХНЯЯ граница обязательного минимума, не")
        print(f"       рекомендация; казначейство может дать и больше")
        print()
        print(f"  3. Остаток на частный капитал: ${gap_usd:,.0f}")
        print(f"     Требуемая доходность для привлечения именно этой суммы:")
        print(f"     implied_apr ≈ {out['implied_apr_for_gap']:.1%}")
        print(f"     (при допущении объёма ${vol_ewma_daily_usd:,.0f}/день)")
        if out['implied_apr_for_gap'] > 0.30:
            print(f"     ⚠ Нереалистично высокая требуемая доходность —")
            print(f"       gap либо надо закрывать большим POL, либо принять")
            print(f"       хронический троттлинг (объём/эмиссия недостаточны)")
        print()
    return out


def liquidity_requirement_for_day(
    gratis_balance: np.ndarray,
    joined_mask: np.ndarray,
    proto: DexProtocolParams,
    pool_params: DexPoolParams,
    age_days: float,
    day: int,
    vol_ewma_daily_usd: float,
    prev_tvl_required: float = None,
    compute_every_n_days: int = 30,
    warm_start_factor: float = 3.0,
    sens_grid: tuple = (4,),
    probe_hours: int = 48,
    tol: float = 0.15,
    pol_grid: tuple = None,
    max_probe_users: int = 50_000,
    n_seeds: int = 2,
    seed: int = None,
) -> Optional[dict]:
    """
    Дешёвая обёртка для ежедневного вызова из симуляции — тот же паттерн
    троттлинга/carry-forward, что у panic_threshold_map_for_day.

    Экономия относительно полного liquidity_requirement_report:
      - warm-start: диапазон поиска TVL = [prev/factor, prev×factor]
        вместо [10k, 100M] — на медленно дрейфующей величине это режет
        число итераций бинарного поиска с ~13 до ~3-4;
      - sens_grid по умолчанию — одна точка (4×), а не полная (1,2,4,8);
      - pol_grid сужен вокруг delta2, а не полная сетка 9 точек;
      - сабсэмпл холдеров (как в panic_threshold_map_for_day).

    Возвращает None в дни, когда расчёт пропускается (carry-forward
    делает вызывающая сторона), иначе dict от liquidity_requirement_report.
    """
    if day % compute_every_n_days != 0 and day != 1:
        return None

    bal = np.where(joined_mask, gratis_balance, 0.0).astype(np.float64)
    holders = np.flatnonzero(bal > 0)
    if holders.size == 0:
        return {"tvl_required_no_fr": None, "note": "нет активного Gratis"}
    if holders.size > max_probe_users:
        rng_sub = np.random.default_rng(seed)
        pick = rng_sub.choice(holders, size=max_probe_users, replace=False)
        probe_bal = bal[pick] * (bal.sum() / bal[pick].sum())
    else:
        probe_bal = bal[holders]
    probe_mask = np.ones(probe_bal.size, dtype=bool)

    if prev_tvl_required is not None and prev_tvl_required > 0:
        # warm-start сужает диапазон, но границы РАСШИРЯЕМЫЕ (см.
        # min_tvl_for_stability) — храповик «ответ=граница→новая граница»
        # исключён конструкцией; floor дополнительно страхует
        tvl_lo = max(prev_tvl_required / warm_start_factor, 1_000.0)
        tvl_hi = prev_tvl_required * warm_start_factor
    else:
        tvl_lo, tvl_hi = 10_000.0, 100_000_000.0   # холодный старт (день 1)

    if pol_grid is None:
        d2 = proto.delta2
        pol_grid = tuple(sorted({0.05, d2, d2 + 0.05, d2 + 0.15, d2 + 0.30}))

    return liquidity_requirement_report(
        probe_bal, probe_mask, proto, pool_params, age_days,
        vol_ewma_daily_usd=vol_ewma_daily_usd, verbose=False,
        sens_grid=sens_grid, probe_hours=probe_hours, tol=tol,
        tvl_lo=tvl_lo, tvl_hi=tvl_hi, pol_grid=pol_grid, seed=seed,
        n_seeds=n_seeds,
    )


if __name__ == "__main__":
    # демо на синтетическом состоянии
    rng = np.random.default_rng(0)
    n = 100_000
    bal = rng.lognormal(3, 0.8, n)
    mask = rng.random(n) < 0.5
    proto = DexProtocolParams()
    pp = DexPoolParams(fee_tier=0.01)

    report = liquidity_requirement_report(
        bal, mask, proto, pp, age_days=90,
        vol_ewma_daily_usd=50_000,   # пример: реальная эмиссия × sell_through
    )