"""
Векторизованный поиск порога самораскрутки для больших баз пользователей
==========================================================================

Контекст: внешняя симуляция генерирует базу пользователей и их Gratis
балансы на ежедневной основе (массивы numpy, до миллионов строк).
По окончании каждого дня нужно:

  1. Взять joined_customer_base (булева маска активных/присоединившихся
     пользователей), gratis_balance (их баланс) и spread_end
     (spread на конец дня)
  2. Прогнать эндогенную симуляцию вперёд (resilience probe) и найти
     минимальный impact_coefficient, при котором сеть самораскрутится
     до сильного стресса — для каждого B из сетки

Почему векторизация обязательна на этом масштабе:
  Оригинальный run_simulation_endogenous хранит каждого пользователя как
  Python-объект User и обновляет балансы в Python-цикле. При 3,000,000
  пользователей и многократных прогонах (по сетке impact × по сетке B ×
  по каждому дню) это даёт миллиарды операций на чистом Python — слишком
  медленно. Векторизованная версия ниже работает с numpy-массивами на
  протяжении всего цикла часов: решение о майнинге, объём тикета,
  cancellation, pro-rata ставка — всё применяется ко всем пользователям
  одновременно через булевы маски и арифметику массивов.

Семантика протокола идентична run_simulation_endogenous из firerate_sim.py —
изменился только способ вычисления, не правила.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional

from firerate_sim import (
    ProtocolParams, AttackParams, MarketImpactParams,
    classify_stress, get_base_rate, get_v_cap,
    update_spread_endogenous,
)


# ──────────────────────────────────────────────────────────────
# ВЕКТОРИЗОВАННЫЙ ЭНДОГЕННЫЙ ПРОГОН
# ──────────────────────────────────────────────────────────────

@dataclass
class VectorizedSimResult:
    """Итог векторизованного прогона — облегчённый, без per-user истории."""
    spreads: np.ndarray            # spread на начало каждого часа, shape (hours,)
    stress_levels: np.ndarray      # уровень стресса каждого часа, shape (hours,)
    total_mined: np.ndarray        # суммарно намайнено coen каждый час
    final_gratis: np.ndarray       # финальные балансы Gratis (только активные)
    max_stress_level_reached: int  # максимум за весь прогон
    hours_severe: int              # сколько часов был сильный стресс


def run_simulation_endogenous_vectorized(
    gratis_balance: np.ndarray,
    joined_mask: np.ndarray,
    spread_start: float,
    hours: int,
    proto: ProtocolParams,
    attack: AttackParams,
    market: MarketImpactParams,
    coen_price_usd: float = 1.0,
    p_mine_normal: float = 0.10,
    p_mine_stress1: float = 0.30,
    p_mine_stress2: float = 0.60,
    rng: Optional[np.random.Generator] = None,
) -> VectorizedSimResult:
    """
    Векторизованный симулятор mineCoen с Firerate на конвертации.
    balance = накопленный Gratis; каждый час пользователи конвертируют часть.
    Firerate применяет effective_rate и v_cap. sell pressure = coen_received.

    Args:
        gratis_balance: массив балансов Gratis, shape (N,), dtype float32/64.
                         Неактивные/не присоединившиеся пользователи могут
                         иметь произвольный баланс — они исключаются маской.
        joined_mask:    булев массив shape (N,) — True для пользователей,
                         реально участвующих в сети (joined_customer_base).
        spread_start:   spread на начало прогона (обычно spread_end
                         предыдущего дня из внешней симуляции).
        hours:          длина прогона вперёд (resilience probe horizon).
        rng:            numpy Generator для воспроизводимости; если None —
                         создаётся новый с энтропией ОС (не воспроизводимо).

    Returns:
        VectorizedSimResult с агрегатами по часам и финальными балансами.
    """
    if rng is None:
        rng = np.random.default_rng()

    # рабочая копия балансов — только активные пользователи участвуют,
    # у неактивных принудительно ставим 0 чтобы они не майнили
    balance = np.where(joined_mask, gratis_balance, 0.0).astype(np.float64)
    n = balance.shape[0]

    spread = spread_start
    spreads_log       = np.empty(hours, dtype=np.float64)
    stress_levels_log = np.empty(hours, dtype=np.int8)
    total_mined_log   = np.empty(hours, dtype=np.float64)

    p_mine_by_level = {0: p_mine_normal, 1: p_mine_stress1, 2: p_mine_stress2}

    for h in range(hours):
        # Firerate классифицирует стресс по текущему spread
        stress_level  = classify_stress(spread, proto)
        firerate_rate = get_base_rate(stress_level, proto)
        firerate_cap  = get_v_cap(stress_level, proto)
        p_convert     = p_mine_by_level[stress_level]  # вероятность конвертации при данном стрессе

        spreads_log[h]       = spread
        stress_levels_log[h] = stress_level

        has_balance = balance > 0
        if not has_balance.any():
            total_mined_log[h:] = 0.0
            spreads_log[h:]       = spread
            stress_levels_log[h:] = stress_level
            balance_final = balance
            break

        # шаг 1: кто хочет конвертировать Gratis → coen в этот час
        wants_convert = has_balance & (rng.random(n) < p_convert)

        if not wants_convert.any():
            total_mined_log[h] = 0.0
            balance_final = balance
            spread = update_spread_endogenous(spread, 0.0, attack.B, market)
            continue

        # шаг 2: запрошенный объём конвертации (случайная доля баланса)
        fraction  = rng.uniform(0.1, 1.0, size=n)
        requested = np.where(wants_convert, balance * fraction, 0.0)
        total_requested = requested.sum()

        # шаг 3: Firerate — pro-rata если суммарный запрос превышает cap
        if firerate_cap < float("inf") and total_requested > firerate_cap:
            fill_ratio = firerate_cap / total_requested
        else:
            fill_ratio = 1.0

        # шаг 4: settlement
        # gratis_burned = сколько Gratis сожжено (с учётом cap)
        # coen_received = gratis_burned × effective_rate (< 1 при стрессе)
        gratis_burned = requested * fill_ratio
        coen_received = gratis_burned * firerate_rate
        balance       = balance - gratis_burned
        total_coen    = coen_received.sum()

        total_mined_log[h] = total_coen
        balance_final = balance

        # шаг 5: обратная связь
        # sell pressure = coen вышедший на рынок (не Gratis, а именно coen)
        sell_volume_usd = total_coen * coen_price_usd
        spread = update_spread_endogenous(spread, sell_volume_usd, attack.B, market)

    else:
        balance_final = balance

    hours_severe = int((stress_levels_log == 2).sum())
    max_level    = int(stress_levels_log.max()) if hours > 0 else 0

    return VectorizedSimResult(
        spreads=spreads_log,
        stress_levels=stress_levels_log,
        total_mined=total_mined_log,
        final_gratis=balance_final,
        max_stress_level_reached=max_level,
        hours_severe=hours_severe,
    )


# ──────────────────────────────────────────────────────────────
# ПОИСК ПОРОГА: ВЕКТОРИЗОВАННАЯ ВЕРСИЯ
# ──────────────────────────────────────────────────────────────

def find_min_impact_threshold_vectorized(
    gratis_balance: np.ndarray,
    joined_mask: np.ndarray,
    spread_end: float,
    B: float,
    proto: ProtocolParams,
    impact_grid: list[float],
    probe_hours: int = 48,
    sell_through_rate: float = 0.6,
    spread_decay: float = 0.3,
    spread_floor: float = 0.01,
    p_mine_normal: float = 0.10,
    p_mine_stress1: float = 0.30,
    p_mine_stress2: float = 0.60,
    p_cancel_on_penalty: float = 0.25,
    seed: Optional[int] = None,
) -> float:
    """
    Находит минимальный impact_coefficient, при котором сеть (с заданным
    состоянием gratis_balance/joined_mask на конец дня и заданным spread_end
    как стартовой точкой) самораскрутится до сильного стресса в течение
    probe_hours часов вперёд.

    Возвращает float('inf') если ни одно значение из impact_grid
    не вызывает self-trigger, или если активных пользователей с балансом
    не осталось (физически нечем создавать sell pressure).

    Это прямой векторизованный аналог find_min_impact_threshold из
    daily_threshold_evolution.py, рассчитанный на N в миллионы пользователей.
    """
    active_balance_total = float(np.where(joined_mask, gratis_balance, 0.0).sum())
    if active_balance_total <= 0:
        return float("inf")

    for impact in sorted(impact_grid):
        market = MarketImpactParams(
            sell_through_rate=sell_through_rate,
            impact_coefficient=impact,
            spread_decay=spread_decay,
            spread_floor=spread_floor,
        )
        attack = AttackParams(B=B)
        rng = np.random.default_rng(seed)  # один и тот же seed для каждого impact -> честное сравнение

        result = run_simulation_endogenous_vectorized(
            gratis_balance=gratis_balance,
            joined_mask=joined_mask,
            spread_start=spread_end,
            hours=probe_hours,
            proto=proto,
            attack=attack,
            market=market,
            p_mine_normal=p_mine_normal,
            p_mine_stress1=p_mine_stress1,
            p_mine_stress2=p_mine_stress2,
            rng=rng,
        )
        if result.hours_severe > 0:
            return impact

    return float("inf")



def build_dynamic_b_grid(
    gratis_balance: np.ndarray,
    joined_mask: np.ndarray,
    fractions: list[float] = None,
) -> tuple[list[float], float]:
    """
    Строит B-сетку как доли от суммарного активного Gratis баланса.

    Почему это правильнее фиксированной сетки:
      При 3M пользователях суммарный Gratis может быть $1B-$100B.
      Фиксированная сетка [$50K..$10M] бессмысленна — любой impact_coefficient
      даст self-trigger потому что объём в тысячи раз превышает B.
      Динамическая сетка масштабируется вместе с сетью и показывает:
      "при какой доле рыночной ликвидности относительно накопленного Gratis
       система устойчива?" — это интерпретируемая метрика на всём горизонте.

    Args:
        fractions: доли от total_active_gratis. По умолчанию:
                   [0.001, 0.005, 0.01, 0.05, 0.10, 0.25]
                   т.е. от "рынок покрывает 0.1% Gratis" до "25%".

    Returns:
        (b_grid, total_active_gratis) — список абсолютных B и базовый объём.
    """
    if fractions is None:
        fractions = [0.001, 0.005, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0]

    active_gratis = float(np.where(joined_mask, gratis_balance, 0.0).sum())
    if active_gratis <= 0:
        # нет активного Gratis — возвращаем placeholder
        return [1.0] * len(fractions), 0.0

    b_grid = [active_gratis * f for f in fractions]
    return b_grid, active_gratis


def compute_threshold_map_for_day(
    gratis_balance: np.ndarray,
    joined_mask: np.ndarray,
    spread_end: float,
    proto: ProtocolParams,
    impact_grid: list[float],
    b_grid: list[float] = None,
    b_fractions: list[float] = None,
    probe_hours: int = 48,
    seed: Optional[int] = None,
    day: Optional[int] = None,
    compute_every_n_days: int = 1,
    **kwargs,
) -> "dict | None":
    """
    Считает threshold map для каждого B из сетки.

    B-сетка задаётся одним из двух способов (взаимоисключающих):

      b_grid:      абсолютные значения B в USD — фиксированная сетка.
                   Подходит для малых демо или если B известен заранее.

      b_fractions: доли от суммарного активного Gratis — динамическая сетка.
                   Пересчитывается каждый вызов от текущего состояния балансов.
                   Рекомендуется для production-симуляции с 3M+ пользователей.
                   Пример: [0.001, 0.01, 0.05, 0.10, 0.25]
                   Ключи возвращаемого dict — это доли (float), не абс. USD.

    Если заданы оба — b_fractions имеет приоритет.
    Если не задан ни один — используется b_grid.

    Returns:
        dict или None (если день пропущен по compute_every_n_days).
        Ключи dict:
          - при b_fractions: float-доля (0.01 = "рынок покрывает 1% Gratis")
          - при b_grid:      абсолютный B в USD
        Значения: min impact_coefficient для self-trigger, или float('inf').
    """
    if day is not None and compute_every_n_days > 1:
        if day % compute_every_n_days != 0:
            return None

    # определяем сетку B
    if b_fractions is not None:
        abs_b_grid, total_gratis = build_dynamic_b_grid(
            gratis_balance, joined_mask, b_fractions
        )
        keys = b_fractions  # ключи dict — доли, не абс. числа
    elif b_grid is not None:
        abs_b_grid  = b_grid
        keys        = b_grid
        total_gratis = float(np.where(joined_mask, gratis_balance, 0.0).sum())
    else:
        raise ValueError("Нужно задать b_grid или b_fractions")

    thresholds = {"total_active_gratis": total_gratis}
    for bi, (b, key) in enumerate(zip(abs_b_grid, keys)):
        b_seed = None if seed is None else seed + bi
        thresholds[key] = find_min_impact_threshold_vectorized(
            gratis_balance=gratis_balance,
            joined_mask=joined_mask,
            spread_end=spread_end,
            B=b,
            proto=proto,
            impact_grid=impact_grid,
            probe_hours=probe_hours,
            seed=b_seed,
            **kwargs,
        )
    return thresholds


# ──────────────────────────────────────────────────────────────
# ДЕМО / ПРОВЕРКА НА МАСШТАБЕ
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    proto = ProtocolParams()

    n = 3_000_000
    rng = np.random.default_rng(0)
    gratis_balance = rng.lognormal(mean=8, sigma=1.5, size=n).astype(np.float32)
    joined_mask    = rng.random(n) < 0.4   # ~40% базы реально присоединились
    spread_end     = 0.018                  # пример: конец дня закрылся в лёгком стрессе

    print(f"N пользователей в базе: {n:,}")
    print(f"Присоединившихся: {joined_mask.sum():,}")
    print(f"Суммарный активный Gratis: {np.where(joined_mask, gratis_balance, 0).sum():,.0f}")
    print(f"Spread на конец дня: {spread_end}")
    print()

    b_fractions = [0.001, 0.005, 0.01, 0.05, 0.10, 0.25]
    impact_grid = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2]

    total_gratis = float(np.where(joined_mask, gratis_balance, 0.0).sum())
    print(f"Суммарный активный Gratis: ${total_gratis:,.0f}")
    print(f"B-сетка (доли от Gratis):  {b_fractions}")
    print(f"B-сетка (абс. USD):        {[f'${total_gratis*f:,.0f}' for f in b_fractions]}")
    print()
    print(f"Считаем threshold map по {len(b_fractions)} значениям B "
          f"(до {len(impact_grid)} прогонов на каждое, с early-exit)...")
    t0 = time.time()
    thresholds = compute_threshold_map_for_day(
        gratis_balance=gratis_balance,
        joined_mask=joined_mask,
        spread_end=spread_end,
        proto=proto,
        b_fractions=b_fractions,
        impact_grid=impact_grid,
        probe_hours=48,
        seed=42,
    )
    t1 = time.time()
    print(f"Готово за {t1-t0:.2f}с\n")

    total_g = thresholds.pop("total_active_gratis", total_gratis)
    print(f"  {'B (доля Gratis)':>16}  {'B (USD)':>14}  {'min impact':>12}")
    print(f"  {'-'*16}  {'-'*14}  {'-'*12}")
    for frac, t in thresholds.items():
        b_usd = total_g * frac
        t_str = f"{t:.2f}" if not np.isinf(t) else "safe"
        print(f"  {frac:>16.3f}  ${b_usd:>13,.0f}  {t_str:>12}")