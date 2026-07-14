"""
Hourly mineCoen на DEX-микроструктуре + эндогенный стабилити-probe
===================================================================

Тот же paired-counterfactual дизайн, что в firerate_hourly.py (общие
случайные числа, два мира), но рынок — DexPoolState вместо линейного
spread-impact:

  FR-мир:   классификатор с гистерезисом → эндогенный v_cap
            (объём ≤ x_i% хода цены за окно) + ставка r_i;
  NoFR-мир: та же физика пула, но без cap/rate/cancellation.

Паника (p_convert_stress_sens) реагирует на уровень стресса СВОЕГО
мира — в NoFR-мире классификатор тоже считается (рыночный стресс
существует без Firerate), но ничего не ограничивает.

Плюс стабилити-инструменты:
  probe_endogenous_dex()      — прогон вперёд от состояния дня:
                                самораскручивается ли паника;
  find_min_panic_threshold()  — минимальная паническая чувствительность
                                p_convert_stress_sens, при которой сеть
                                уходит в устойчивый severe (метрика
                                хрупкости, сменившая impact-sweep).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from dex_market import (
    DexPoolState, DexPoolParams, DexProtocolParams, DexStressClassifier,
)
# самодостаточность DEX-стека: две утилиты инлайнены, зависимость от
# CEX-модулей (f_firerate_hourly → _firerate_sim) устранена
DRAIN_EPS = 1e-6   # float32-дружелюбный порог «баланс исчерпан»


def daily_to_hourly_prob(p_day: np.ndarray, hours: int) -> np.ndarray:
    """p_hour: P(конвертировать хотя бы раз за hours окон) = p_day."""
    return 1.0 - np.power(1.0 - np.clip(p_day, 0.0, 1.0), 1.0 / hours)


# ──────────────────────────────────────────────────────────────
# РЕЗУЛЬТАТ ДНЯ
# ──────────────────────────────────────────────────────────────

@dataclass
class DexDayResult:
    # FR-мир
    gratis_requested: float = 0.0
    gratis_converted: float = 0.0
    coen_from_conversion: float = 0.0
    effective_rate_realized: float = 1.0
    stress_hours: dict = field(default_factory=lambda: {0: 0, 1: 0, 2: 0})
    stress_level_max: int = 0
    displacement_end: float = 0.0
    depth_end: float = 0.0
    d_hat_end: float = 1.0
    lp_health_end: float = 1.0
    queue_gratis_end: float = 0.0   # Gratis в очереди: хотят, но не исполнены
    queue_users_end: int = 0
    tvl_info: dict = field(default_factory=dict)   # эндогенный TVL: vol/apr/tvl
    tvl_info_nofr: dict = field(default_factory=dict)  # то же для мира без Firerate
    # NoFR-мир
    gratis_converted_nofr: float = 0.0
    coen_from_conversion_nofr: float = 0.0
    stress_hours_nofr: dict = field(default_factory=lambda: {0: 0, 1: 0, 2: 0})
    displacement_end_nofr: float = 0.0
    depth_end_nofr: float = 0.0
    lp_health_end_nofr: float = 1.0


def _p_hour_by_level(balance, balance_day, day, trend, hours,
                     p_base, p_price, p_hold, p_stress):
    has = balance > 0
    hold_days = np.where(has, day - balance_day, 0).clip(min=0)
    base_arr = p_base * (1.0 + p_price * np.clip(trend, 0.0, 1.0)
                         + p_hold * np.tanh(hold_days / 90.0))
    return {lvl: daily_to_hourly_prob(
                np.clip(base_arr * (1.0 + p_stress * lvl), 0.0, 1.0), hours)
            for lvl in (0, 1, 2)}


# ──────────────────────────────────────────────────────────────
# ОДИН ДЕНЬ, ДВА МИРА
# ──────────────────────────────────────────────────────────────

def run_mine_coen_day_dex(
    day: int,
    gratis_balance: np.ndarray,
    gratis_balance_day: np.ndarray,
    gratis_balance_nofr: np.ndarray,
    gratis_balance_day_nofr: np.ndarray,
    pool_fr: DexPoolState,
    pool_nofr: DexPoolState,
    clf_fr: DexStressClassifier,
    clf_nofr: DexStressClassifier,
    trend: float,
    coen_price: float,          # внешний fair price дня
    tvl_usd: float,             # TVL пула сегодня (растёт с сетью)
    proto: DexProtocolParams,
    rng: np.random.Generator,
    *,
    hours: int = 24,
    p_convert_base: float = 0.05,
    p_convert_price_sens: float = 2.0,
    p_convert_hold_sens: float = 0.3,
    p_convert_stress_sens: float = 2.0,
    p_cancel_on_penalty: float = 0.25,
    sell_through_rate: float = 0.6,
    seeking: "Optional[np.ndarray]" = None,   # bool(N), персистентен между
        # днями: True = пользователь хотя бы раз хотел конвертировать и ещё
        # не исполнен полностью. Обновляется in place. Даёт точную очередь
        # конвертации для метрики задержки (закон Литтла: W = очередь/сервис).
) -> DexDayResult:
    """Прогон дня как `hours` batch-окон на общих случайных числах."""
    window_len = 24.0 / hours

    # экзогенные обновления дня — одинаковы в обоих мирах.
    # fair price меняется НЕПРЕРЫВНО внутри дня: геометрическая интерполяция
    # от вчерашнего p_fair к сегодняшнему coen_price по окнам. Разовый
    # дневной степ создавал искусственные 1-оконные ходы цены >= 2%
    # (ложный LP-испуг) и displacement-спайки (ложный триггер уровня).
    fair_step_fr = ((coen_price / pool_fr.p_fair) ** (1.0 / hours)
                    if pool_fr.p_fair > 0 else 1.0)
    fair_step_no = ((coen_price / pool_nofr.p_fair) ** (1.0 / hours)
                    if pool_nofr.p_fair > 0 else 1.0)
    for pool in (pool_fr, pool_nofr):
        if not pool.endogenous_tvl:
            pool.tvl_usd = tvl_usd     # экзогенный режим (legacy)
        pool.age_days = day

    cand = np.flatnonzero((gratis_balance > 0) | (gratis_balance_nofr > 0))
    res = DexDayResult()
    if cand.size == 0:
        for _ in range(hours):
            pool_fr.p_fair *= fair_step_fr
            pool_nofr.p_fair *= fair_step_no
            for pool, clf, sh in ((pool_fr, clf_fr, res.stress_hours),
                                  (pool_nofr, clf_nofr, res.stress_hours_nofr)):
                m = pool.apply_window(0.0, window_len)
                sh[clf.update(m["d_hat"], m["displacement"], False)] += 1
        res.stress_level_max = max(l for l, h in res.stress_hours.items() if h > 0)
        _fill_end_state(res, pool_fr, pool_nofr)
        return res

    m = cand.size
    bal = gratis_balance[cand].astype(np.float64)
    bal_no = gratis_balance_nofr[cand].astype(np.float64)

    ph_fr = _p_hour_by_level(bal, gratis_balance_day[cand], day, trend, hours,
                             p_convert_base, p_convert_price_sens,
                             p_convert_hold_sens, p_convert_stress_sens)
    ph_no = _p_hour_by_level(bal_no, gratis_balance_day_nofr[cand], day, trend, hours,
                             p_convert_base, p_convert_price_sens,
                             p_convert_hold_sens, p_convert_stress_sens)

    req_fr = burned_fr = coen_fr = 0.0
    req_no = 0.0

    for _w in range(hours):
        pool_fr.p_fair *= fair_step_fr
        pool_nofr.p_fair *= fair_step_no
        u_convert = rng.random(m)
        u_cancel = rng.random(m)
        fraction = rng.uniform(0.1, 1.0, size=m)

        # ══════ FR-мир ══════
        lvl = clf_fr.level
        res.stress_hours[lvl] += 1
        wants = (bal > 0) & (u_convert < ph_fr[lvl])
        sell_usd = 0.0
        if wants.any():
            if seeking is not None:
                seeking[cand[wants]] = True
            requested = np.where(wants, bal * fraction, 0.0)
            req_fr += float(requested.sum())
            if lvl > 0 and p_cancel_on_penalty > 0:
                requested = np.where(wants & (u_cancel < p_cancel_on_penalty),
                                     0.0, requested)
            total = float(requested.sum())
            if total > 0:
                # эндогенный cap: поток в пул = burned × rate × sell_through × price
                # ограничиваем burned так, чтобы поток ≤ cap_usd
                cap_usd = clf_fr.v_cap_usd(pool_fr) 
                # cap_usd = clf_fr.v_cap_usd(pool_fr) if lvl > 0 else float('inf')
                # cap_usd = float('inf')

                denom = proto.rate(lvl) * sell_through_rate * coen_price
                cap_coen = cap_usd / denom if cap_usd < np.inf else np.inf
                fill = min(1.0, cap_coen / total) if cap_coen < np.inf else 1.0
                burned = requested * fill
                bal -= burned
                bal[bal <= DRAIN_EPS] = 0.0
                if seeking is not None and fill >= 0.999:
                    # заявки окна исполнены полностью → выход из очереди
                    seeking[cand[wants]] = False
                b_sum = float(burned.sum())
                c_sum = b_sum * proto.rate(lvl)
                burned_fr += b_sum
                coen_fr += c_sum
                sell_usd = c_sum * sell_through_rate * coen_price
        w_m = pool_fr.apply_window(sell_usd, window_len)
        clf_fr.update(w_m["d_hat"], w_m["displacement"],
                      flow_confirm=sell_usd > 0.05 * w_m["depth"])

        # ══════ NoFR-мир ══════
        lvl_no = clf_nofr.level
        res.stress_hours_nofr[lvl_no] += 1
        wants_no = (bal_no > 0) & (u_convert < ph_no[lvl_no])
        sell_usd_no = 0.0
        if wants_no.any():
            requested_no = np.where(wants_no, bal_no * fraction, 0.0)
            t_no = float(requested_no.sum())
            req_no += t_no
            bal_no -= requested_no
            bal_no[bal_no <= DRAIN_EPS] = 0.0
            res.coen_from_conversion_nofr += t_no
            sell_usd_no = t_no * sell_through_rate * coen_price
        w_n = pool_nofr.apply_window(sell_usd_no, window_len)
        clf_nofr.update(w_n["d_hat"], w_n["displacement"],
                        flow_confirm=sell_usd_no > 0.05 * w_n["depth"])

    gratis_balance[cand] = bal.astype(gratis_balance.dtype, copy=False)
    gratis_balance_nofr[cand] = bal_no.astype(gratis_balance_nofr.dtype, copy=False)

    # доходностное равновесие частных LP (no-op в экзогенном режиме).
    # NoFR-мир ведёт СВОЁ равновесие: его объём/выручка/TVL отвечают на
    # вопрос «самоподдерживается ли пул БЕЗ Firerate».
    res.tvl_info = pool_fr.update_private_tvl(24.0)
    res.tvl_info_nofr = pool_nofr.update_private_tvl(24.0)

    if seeking is not None:
        seeking &= gratis_balance > DRAIN_EPS       # исполненные — из очереди
        res.queue_gratis_end = float(gratis_balance[seeking].sum())
        res.queue_users_end = int(seeking.sum())

    res.gratis_requested = req_fr
    res.gratis_converted = burned_fr
    res.coen_from_conversion = coen_fr
    res.effective_rate_realized = coen_fr / burned_fr if burned_fr > 0 else 1.0
    res.gratis_converted_nofr = res.coen_from_conversion_nofr
    res.stress_level_max = max(l for l, h in res.stress_hours.items() if h > 0)
    _fill_end_state(res, pool_fr, pool_nofr)
    return res


def _fill_end_state(res: DexDayResult, pool_fr, pool_nofr):
    res.displacement_end = pool_fr.displacement
    res.depth_end = pool_fr.depth_1pct()
    res.d_hat_end = pool_fr.d_hat()
    res.lp_health_end = pool_fr.h
    res.displacement_end_nofr = pool_nofr.displacement
    res.depth_end_nofr = pool_nofr.depth_1pct()
    res.lp_health_end_nofr = pool_nofr.h


# ──────────────────────────────────────────────────────────────
# СТАБИЛЬНОСТЬ ПОД ЭНДОГЕННЫМИ ПРОЦЕССАМИ
# ──────────────────────────────────────────────────────────────

@dataclass
class ProbeResult:
    hours_severe: int
    hours_stress: int
    max_level: int
    min_d_hat: float
    max_displacement: float
    final_lp_health: float
    spiral: bool     # severe устойчив до конца горизонта (не погашен)


def probe_endogenous_dex(
    gratis_balance: np.ndarray,
    joined_mask: np.ndarray,
    pool: DexPoolState,
    clf_level: int,
    proto: DexProtocolParams,
    hours: int,
    panic_sens: float,
    firerate_on: bool = True,
    p_convert_base_daily: float = 0.05,
    p_cancel_on_penalty: float = 0.25,
    sell_through_rate: float = 0.6,
    initial_displacement: float = 0.0,
    initial_lp_shock: float = 0.0,
    seed: Optional[int] = None,
) -> ProbeResult:
    """
    Resilience probe: от состояния конца дня прогоняем hours окон вперёд
    и смотрим, раскручивается ли эндогенная паника (flow → displacement →
    LP-вывод → depth ↓ → стресс → паника ↑) или Firerate её гасит.

    Опциональные шоки на старте: initial_displacement (мгновенное смещение
    пула вниз) и initial_lp_shock (доля LP health, снятая сразу) —
    моделируют внешний триггер bank run / LP exodus.
    """
    rng = np.random.default_rng(seed)
    bal = np.where(joined_mask, gratis_balance, 0.0).astype(np.float64)
    idx = np.flatnonzero(bal > 0)
    bal = bal[idx]
    n = bal.size

    pool = pool.copy()
    clf = DexStressClassifier(proto, fee=pool.params.fee_tier)
    clf.level = clf_level
    if initial_displacement > 0:
        pool.p_pool = pool.p_fair * (1.0 - initial_displacement)
    if initial_lp_shock > 0:
        pool.h *= (1.0 - initial_lp_shock)

    p_hour = {lvl: 1.0 - (1.0 - min(1.0, p_convert_base_daily
                                    * (1.0 + panic_sens * lvl))) ** (1.0 / 24.0)
              for lvl in (0, 1, 2)}

    hours_sev = hours_str = 0
    max_lvl = clf.level
    min_dhat = pool.d_hat()
    max_disp = pool.displacement
    last_levels = []

    for _h in range(hours):
        lvl = clf.level
        hours_sev += (lvl == 2)
        hours_str += (lvl > 0)
        max_lvl = max(max_lvl, lvl)

        sell_usd = 0.0
        if n > 0 and bal.max() > 0:
            wants = (bal > 0) & (rng.random(n) < p_hour[lvl])
            if wants.any():
                requested = np.where(wants, bal * rng.uniform(0.1, 1.0, n), 0.0)
                if firerate_on and lvl > 0 and p_cancel_on_penalty > 0:
                    requested = np.where(wants & (rng.random(n) < p_cancel_on_penalty),
                                         0.0, requested)
                total = float(requested.sum())
                if total > 0:
                    if firerate_on:
                        cap_usd = clf.v_cap_usd(pool)
                        # cap_usd = clf.v_cap_usd(pool) if lvl > 0 else float('inf')
                        # cap_usd = float('inf')

                        rate = proto.rate(lvl)
                        denom = rate * sell_through_rate * pool.p_fair
                        cap_coen = cap_usd / denom if cap_usd < np.inf else np.inf
                        fill = min(1.0, cap_coen / total) if cap_coen < np.inf else 1.0
                    else:
                        fill, rate = 1.0, 1.0
                    burned = requested * fill
                    bal -= burned
                    bal[bal <= DRAIN_EPS] = 0.0
                    sell_usd = float(burned.sum()) * rate * sell_through_rate * pool.p_fair

        w = pool.apply_window(sell_usd, 1.0)
        clf.update(w["d_hat"], w["displacement"],
                   flow_confirm=sell_usd > 0.05 * w["depth"])
        min_dhat = min(min_dhat, w["d_hat"])
        max_disp = max(max_disp, w["displacement"])
        last_levels.append(clf.level)

    tail = last_levels[-max(1, hours // 8):]
    spiral = all(l == 2 for l in tail)
    return ProbeResult(hours_severe=hours_sev, hours_stress=hours_str,
                       max_level=max_lvl, min_d_hat=min_dhat,
                       max_displacement=max_disp, final_lp_health=pool.h,
                       spiral=spiral)


def panic_threshold_map_for_day(
    gratis_balance: np.ndarray,
    joined_mask: np.ndarray,
    pool: "DexPoolState",
    proto: DexProtocolParams,
    b_fractions: list[float],
    sens_grid: tuple = (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64),
    probe_hours: int = 72,
    initial_displacement: float = 0.06,
    seed: Optional[int] = None,
    day: Optional[int] = None,
    compute_every_n_days: int = 1,
    max_probe_users: int = 200_000,
    firerate_on: bool = True,
) -> Optional[dict]:
    """
    DEX-преемник compute_threshold_map_for_day: тот же интерфейс
    (сетка фракций, троттлинг раз в N дней, carry-forward снаружи),
    но метрика — ПАНИЧЕСКИЙ порог вместо impact-порога.

    Семантика фракции СОХРАНЕНА: f = TVL probe-пула как доля суммарного
    активного Gratis (раньше: B = f × active_gratis). Возвращает
    {f: min p_convert_stress_sens для устойчивого severe | inf}, плюс
    ключ 'total_active_gratis'. Корреляционный анализ frac*
    (credis_correlation_analysis) работает поверх без изменений.

    Сабсэмплинг: при числе холдеров > max_probe_users probe гоняется на
    случайной подвыборке с масштабированием балансов (агрегатный поток
    сохранён) — иначе 5 фракций × 9 точек сетки × 72ч на 3M-массивах
    неподъёмны по времени.

    Возвращает None в дни, когда расчёт пропускается (day % N != 0) —
    вызывающая сторона делает carry-forward, как раньше.
    """
    if day is not None and compute_every_n_days > 1:
        if (day % compute_every_n_days) != 0 and day != 1:
            return None

    bal = np.where(joined_mask, gratis_balance, 0.0).astype(np.float64)
    total_active = float(bal.sum())
    if total_active <= 0:
        out = {f: float("inf") for f in b_fractions}
        out['total_active_gratis'] = 0.0
        return out

    holders = np.flatnonzero(bal > 0)
    if holders.size > max_probe_users:
        rng_sub = np.random.default_rng(seed)
        pick = rng_sub.choice(holders, size=max_probe_users, replace=False)
        sub_bal = bal[pick]
        sub_bal *= total_active / float(sub_bal.sum())   # сохранить агрегат
        probe_bal = sub_bal
        probe_mask = np.ones(probe_bal.size, dtype=bool)
    else:
        probe_bal = bal[holders]
        probe_mask = np.ones(probe_bal.size, dtype=bool)

    out = {}
    for i, f in enumerate(b_fractions):
        probe_pool = pool.copy()
        probe_pool.tvl_usd = max(10_000.0, f * total_active)
        out[f] = find_min_panic_threshold(
            probe_bal, probe_mask, probe_pool, proto,
            sens_grid=sens_grid,
            probe_hours=probe_hours,
            initial_displacement=initial_displacement,
            firerate_on=firerate_on,
            seed=None if seed is None else seed + i,
        )
    out['total_active_gratis'] = total_active
    return out


def find_min_panic_threshold(
    gratis_balance: np.ndarray,
    joined_mask: np.ndarray,
    pool: DexPoolState,
    proto: DexProtocolParams,
    sens_grid: list[float] = (0.5, 1, 2, 4, 8, 16, 32, 64),
    probe_hours: int = 72,
    severe_hours_trigger: int = 6,
    trigger: str = "spiral",     # "spiral" — устойчивый severe в хвосте probe;
                                 # "hours"  — суммарные severe-часы ≥ trigger
    firerate_on: bool = True,
    seed: Optional[int] = 42,
    **probe_kwargs,
) -> float:
    """
    Минимальная паническая чувствительность p_convert_stress_sens,
    при которой сеть самораскручивается. По умолчанию критерий —
    'spiral': severe УСТОЙЧИВ в конце горизонта (паника не погашена),
    а не просто набрал часы (одиночный шок сам по себе даёт severe-часы,
    вопрос стабильности — вернулась ли система).
    inf = устойчива на всей сетке.

    Один и тот же seed на все точки сетки — честное сравнение
    (та же «удача», отличается только поведенческий параметр).
    """
    for sens in sorted(sens_grid):
        r = probe_endogenous_dex(
            gratis_balance, joined_mask, pool, clf_level=0, proto=proto,
            hours=probe_hours, panic_sens=sens, firerate_on=firerate_on,
            seed=seed, **probe_kwargs,
        )
        hit = r.spiral if trigger == "spiral" else (r.hours_severe >= severe_hours_trigger)
        if hit:
            return sens
    return float("inf")