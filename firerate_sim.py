"""
Firerate Simulator — упрощённая трёхуровневая модель
=====================================================

Протокол:
  Нормальный режим    spread < θ1           → ставка 1.0, без cap
  Умеренный стресс    θ1 ≤ spread < θ2      → ставка r1, cap V1
  Сильный стресс      spread ≥ θ2           → ставка r2, cap V2

Стоимость атаки в упрощённой модели:
  Нет ramp, нет sigmoid → атакующему не нужно удерживать манипуляцию
  несколько окон подряд. Один час держишь spread выше порога → Firerate
  активирован → пользователи получают штраф.

  C_attack = C_capital + C_fill + C_signal (за одно окно)

  Критически: в упрощённой модели N всегда = 1 для одного уровня стресса.
  Атака дешевле, чем в оригинальном документе — это честный вывод
  из упрощения архитектуры. Защита теперь строится на fill-risk и
  стоимости поддержания spread, а не на ramp-мультипликаторе.
"""

from __future__ import annotations
import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy


# ──────────────────────────────────────────────────────────────
# GOVERNANCE ПАРАМЕТРЫ (6 штук — упрощённая модель)
# ──────────────────────────────────────────────────────────────

@dataclass
class ProtocolParams:
    """
    6 governance параметров упрощённой модели.
    Все остальные — константы или производные.
    """
    theta1: float = 0.03    # порог умеренного стресса (spread)
    theta2: float = 0.08    # порог сильного стресса (spread)
    r1: float     = 0.75    # ставка при умеренном стрессе
    r2: float     = 0.40    # ставка при сильном стрессе
    v_cap1: float = 50_000  # объёмный cap при умеренном стрессе (coen/час)
    v_cap2: float = 20_000  # объёмный cap при сильном стрессе (coen/час)

    # производные (не governance)
    @property
    def r0(self) -> float:
        return 1.0

    @property
    def v_cap0(self) -> float:
        return float("inf")


TAU = 1.0  # длина batch window (часов) — константа протокола


# ──────────────────────────────────────────────────────────────
# СТОИМОСТЬ АТАКИ — АДАПТИРОВАНО ПОД УПРОЩЁННУЮ МОДЕЛЬ
# ──────────────────────────────────────────────────────────────

@dataclass
class AttackParams:
    """
    Рыночные параметры для расчёта стоимости атаки.

    B         — глубина order book (USD, одна сторона)
    alpha     — во сколько раз нужно раздуть spread для достижения порога
    r_opp     — opportunity cost капитала (в час, ~10% годовых = 0.0001/ч)
    p_fill    — вероятность fill ордеров за одно окно
    delta     — slippage при fill (доля объёма)
    c_signal  — стоимость поддержания spread за одно окно (USD)

    Примечание: r_down и m_min из оригинального документа здесь не нужны —
    в упрощённой модели нет ramp. Атака действует за одно окно.
    """
    B: float        = 100_000
    r_opp: float    = 0.0001    # ~10% годовых / 8760ч
    p_fill: float   = 0.01
    delta: float    = 0.03
    k_signal: float = 0.002     # C_signal = k_signal * B


def attack_cost_per_window(params: AttackParams, alpha: float = 1.0) -> dict:
    """
    C_attack = C_capital + C_fill + C_signal

    В упрощённой модели это ПОЛНАЯ стоимость атаки за один эпизод (N=1).

    C_capital = α · B · τ · r_opp   (opportunity cost замороженного капитала)
    C_fill    = P_fill · α · B · δ  (риск исполнения арбитражёрами)
    C_signal  = k_signal · B        (поддержание spread выше порога)

    C_signal теперь линейно зависит от B — глубже рынок, дороже манипуляция.
    При k_signal=0.002: B=$100K → $200, B=$10M → $20,000.
    """
    c_capital = alpha * params.B * TAU * params.r_opp
    c_fill    = params.p_fill * alpha * params.B * params.delta
    c_signal  = params.k_signal * params.B
    return {
        "c_capital":      c_capital,
        "c_fill":         c_fill,
        "c_signal":       c_signal,
        "c_per_window":   c_capital + c_fill + c_signal,
        "windows_needed": 1,
    }


def attack_cost_to_level(
    target_level: int,
    params: AttackParams,
    proto: ProtocolParams,
) -> dict:
    """
    Стоимость атаки для активации конкретного уровня стресса.

    target_level: 1 = умеренный (spread ≥ θ1), 2 = сильный (spread ≥ θ2)

    alpha рассчитывается автоматически: сколько нужно раздуть baseline spread
    чтобы перейти нужный порог. Baseline spread принимаем = θ1 * 0.5 (спокойный рынок).
    """
    baseline = proto.theta1 * 0.5
    threshold = proto.theta1 if target_level == 1 else proto.theta2
    target_spread = threshold * 1.2  # 20% выше порога для надёжной активации
    alpha_needed  = target_spread / baseline if baseline > 0 else 1.0

    result = attack_cost_per_window(params, alpha=alpha_needed)
    result["target_level"]  = target_level
    result["target_spread"] = target_spread
    result["alpha_needed"]  = alpha_needed
    result["rate_imposed"]  = proto.r1 if target_level == 1 else proto.r2
    result["cap_imposed"]   = proto.v_cap1 if target_level == 1 else proto.v_cap2
    return result


def attack_cost_sensitivity(
    b_values: list[float],
    params: AttackParams,
    proto: ProtocolParams,
    target_level: int = 2,
) -> list[dict]:
    """
    Кривая C_total(B) — sensitivity по глубине рынка.
    """
    results = []
    for b in b_values:
        p = deepcopy(params)
        p.B = b
        row = attack_cost_to_level(target_level, p, proto)
        row["B"] = b
        results.append(row)
    return results


def print_attack_breakdown(
    params: AttackParams,
    proto: ProtocolParams,
) -> None:
    """Детальный расчёт стоимости атаки под упрощённую модель."""
    print("РАСЧЁТ СТОИМОСТИ АТАКИ — упрощённая трёхуровневая модель")
    print("─" * 60)
    print(f"  B (глубина order book):  ${params.B:>12,.0f}")
    print(f"  r_opp:                   {params.r_opp:>12.6f}/ч  (~10% годовых)")
    print(f"  P_fill:                  {params.p_fill:>12.4f}")
    print(f"  δ (slippage):            {params.delta:>12.3f}")
    print(f"  k_signal (C_signal/B):   {params.k_signal:>12.4f}  (C_signal = k·B)")
    print()

    for level in [1, 2]:
        r = attack_cost_to_level(level, params, proto)
        label = "Умеренный стресс" if level == 1 else "Сильный стресс"
        threshold = proto.theta1 if level == 1 else proto.theta2
        print(f"  Атака на уровень {level} ({label}):")
        print(f"    Порог spread:          {threshold:.1%}  →  нужно {r['target_spread']:.1%}")
        print(f"    α (инфляция spread):   {r['alpha_needed']:.1f}x")
        print(f"    C_capital = α·B·τ·r:  ${r['c_capital']:>10,.2f}")
        print(f"    C_fill    = P·αB·δ:   ${r['c_fill']:>10,.2f}")
        print(f"    C_signal = k·B:        ${r['c_signal']:>10,.2f}  (k={params.k_signal})")
        print(f"    ── C_total (1 окно):   ${r['c_per_window']:>10,.2f}")
        print(f"    Ставка для жертв:      {r['rate_imposed']:.0%}")
        print(f"    Cap эмиссии:           {r['cap_imposed']:,.0f} coen/час")
        print()

    print("  Ключевое отличие от оригинального документа:")
    print("  Нет ramp → N=1 всегда. Атака эффективна за одно окно.")
    print("  Защита строится только на fill-risk и c_signal,")
    print("  а не на мультипликаторе N×C.")
    print("─" * 60)


# ──────────────────────────────────────────────────────────────
# СТРУКТУРЫ ДАННЫХ
# ──────────────────────────────────────────────────────────────

@dataclass
class User:
    user_id: str
    gratis_balance: float
    gratis_balance_initial: float
    coen_received: float   = 0.0
    gratis_burned: float   = 0.0
    hours_participated: int = 0
    hours_cancelled: int    = 0
    history: list[dict]    = field(default_factory=list)

    @property
    def nominal_loss(self) -> float:
        return self.gratis_burned - self.coen_received

    @property
    def loss_pct(self) -> float:
        if self.gratis_burned == 0:
            return 0.0
        return self.nominal_loss / self.gratis_burned * 100


@dataclass
class HourResult:
    hour: int
    spread: float
    stress_level: int       # 0 / 1 / 2
    base_rate: float
    effective_rate: float   # после pro-rata cap
    v_cap: float
    total_requested: float
    total_mined: float
    participants: int
    cancelled: int
    c_window: float = 0.0   # оценочная стоимость атаки за окно
    sell_volume_usd: float = 0.0   # объём coen проданных на рынке в этот час (эндогенный режим)
    spread_endogenous: bool = False  # True если spread посчитан эндогенно, а не задан извне


@dataclass
class AttackSummary:
    stress1_hours: int         = 0
    stress2_hours: int         = 0
    c_per_window_lvl1: float   = 0.0
    c_per_window_lvl2: float   = 0.0
    c_actual_observed: float   = 0.0  # за фактические стресс-окна
    breakeven_note: str        = ""


@dataclass
class SimulationResult:
    users: list[User]
    hour_results: list[HourResult]
    attack_summary: AttackSummary
    proto: ProtocolParams

    total_coen_mined: float    = 0.0
    total_gratis_burned: float = 0.0
    total_nominal_loss: float  = 0.0
    hours_by_level: dict       = field(default_factory=lambda: {0:0, 1:0, 2:0})

    def summary(self) -> str:
        lines = []
        lines.append("=" * 65)
        lines.append("ИТОГИ СИМУЛЯЦИИ")
        lines.append("=" * 65)
        lines.append(f"Часов симуляции:            {len(self.hour_results)}")
        lines.append(f"  Нормальных:               {self.hours_by_level[0]}")
        lines.append(f"  Умеренный стресс:         {self.hours_by_level[1]}")
        lines.append(f"  Сильный стресс:           {self.hours_by_level[2]}")
        lines.append("")
        lines.append(f"Всего coen намайнено:       {self.total_coen_mined:,.0f}")
        lines.append(f"Всего Gratis сожжено:       {self.total_gratis_burned:,.0f}")
        lines.append(f"Номинальные потери:         {self.total_nominal_loss:,.0f} coen")
        if self.total_gratis_burned > 0:
            pct = self.total_nominal_loss / self.total_gratis_burned * 100
            lines.append(f"Потери %:                   {pct:.1f}%")

        lines.append("")
        lines.append("─" * 65)
        lines.append("СТОИМОСТЬ АТАКИ (упрощённая модель, N=1)")
        lines.append("─" * 65)
        a = self.attack_summary
        lines.append(f"  Окон умеренного стресса:  {a.stress1_hours}")
        lines.append(f"  Окон сильного стресса:    {a.stress2_hours}")
        lines.append(f"  Стоимость окна lvl1:      ${a.c_per_window_lvl1:,.0f}")
        lines.append(f"  Стоимость окна lvl2:      ${a.c_per_window_lvl2:,.0f}")
        lines.append(f"  Суммарно за стресс-окна:  ${a.c_actual_observed:,.0f}")
        if a.breakeven_note:
            lines.append(f"  {a.breakeven_note}")

        lines.append("")
        lines.append("─" * 65)
        lines.append("ТОП-5 ПОСТРАДАВШИХ")
        lines.append("─" * 65)
        lines.append(f"  {'ID':<15} {'Gratis сожжено':>14} {'coen получено':>14} {'Потери %':>9}")
        lines.append(f"  {'-'*15} {'-'*14} {'-'*14} {'-'*9}")
        for u in sorted(self.users, key=lambda u: u.nominal_loss, reverse=True)[:5]:
            lines.append(
                f"  {u.user_id:<15} {u.gratis_burned:>14,.0f} "
                f"{u.coen_received:>14,.0f} {u.loss_pct:>8.1f}%"
            )
        lines.append("=" * 65)
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ──────────────────────────────────────────────────────────────

def classify_stress(spread: float, proto: ProtocolParams) -> int:
    if spread >= proto.theta2:
        return 2
    elif spread >= proto.theta1:
        return 1
    return 0


def get_base_rate(stress_level: int, proto: ProtocolParams) -> float:
    return {0: proto.r0, 1: proto.r1, 2: proto.r2}[stress_level]


def get_v_cap(stress_level: int, proto: ProtocolParams) -> float:
    return {0: proto.v_cap0, 1: proto.v_cap1, 2: proto.v_cap2}[stress_level]


def pro_rata_rate(base_rate: float, v_cap: float, total_requested: float) -> float:
    if total_requested == 0 or v_cap == float("inf"):
        return base_rate
    if total_requested <= v_cap:
        return base_rate
    return base_rate * (v_cap / total_requested)


# ──────────────────────────────────────────────────────────────
# ОБРАТНАЯ СВЯЗЬ: MINING → SELL PRESSURE → SPREAD
# ──────────────────────────────────────────────────────────────

@dataclass
class MarketImpactParams:
    """
    Параметры эндогенной обратной связи рынка.

    sell_through_rate — какая доля намайненного coen продаётся
                         на открытом рынке в течение того же часа
                         (а не держится / переводится off-exchange).
    impact_coefficient — коэффициент price impact: насколько растёт
                          spread на единицу sell volume относительно
                          глубины рынка B. Простая линейная модель
                          square-root impact упрощена до линейной для
                          прозрачности; можно заменить на sqrt-модель
                          позже без изменения интерфейса.
    spread_decay        — скорость возврата spread к базовому уровню
                          в часы без давления продаж (mean reversion).
    spread_floor        — минимальный "здоровый" spread рынка.
    """
    sell_through_rate: float  = 0.6     # 60% намайненного продаётся сразу
    impact_coefficient: float = 0.8     # spread_impact = coeff * (sell_vol / B)
    spread_decay: float       = 0.3     # 30% возврата к floor за час
    spread_floor: float       = 0.01    # базовый "здоровый" spread


def price_impact(
    sell_volume_usd: float,
    B: float,
    params: MarketImpactParams,
) -> float:
    """
    Сколько добавляется к spread из-за давления продаж за это окно.

    Линейная impact-модель: impact = k * (sell_volume / B)
    Если продают объём равный половине глубины рынка при k=0.8 —
    spread вырастет на 0.40 (40 п.п.), что уже далеко в зоне сильного стресса.
    """
    if B <= 0:
        return 0.0
    return params.impact_coefficient * (sell_volume_usd / B)


MAX_SPREAD = 5.0
def update_spread_endogenous(
    prev_spread: float,
    sell_volume_usd: float,
    B: float,
    params: MarketImpactParams,
) -> float:
    """
    Новый spread = decay к floor + price impact от продаж этого часа.

    spread_t = floor + (prev_spread - floor) * (1 - decay) + impact(sell_vol)

    Это даёт: без давления продаж spread плавно возвращается к floor.
    При давлении продаж — растёт пропорционально объёму относительно B.
    """
    decayed = params.spread_floor + (prev_spread - params.spread_floor) * (1 - params.spread_decay)
    impact  = price_impact(sell_volume_usd, B, params)
    return min(max(params.spread_floor, decayed + impact), MAX_SPREAD)


# ──────────────────────────────────────────────────────────────
# ОСНОВНОЙ СИМУЛЯТОР
# ──────────────────────────────────────────────────────────────

def run_simulation(
    bid_ask_spreads: list[float],
    users: list[dict],
    proto: Optional[ProtocolParams]  = None,
    attack: Optional[AttackParams]   = None,
    p_mine_normal: float             = 0.10,
    p_mine_stress1: float            = 0.30,
    p_mine_stress2: float            = 0.60,
    p_cancel_on_penalty: float       = 0.25,
    random_seed: Optional[int]       = 42,
) -> SimulationResult:
    """
    Почасовой симулятор Firerate (упрощённая трёхуровневая модель).
    """
    if proto  is None: proto  = ProtocolParams()
    if attack is None: attack = AttackParams()

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    sim_users = [
        User(
            user_id=u["user_id"],
            gratis_balance=u["gratis_balance"],
            gratis_balance_initial=u["gratis_balance"],
        )
        for u in users
    ]

    # заранее считаем стоимость атаки по уровням
    atk1 = attack_cost_to_level(1, attack, proto)
    atk2 = attack_cost_to_level(2, attack, proto)
    atk_by_level = {0: 0.0, 1: atk1["c_per_window"], 2: atk2["c_per_window"]}

    p_mine_by_level = {0: p_mine_normal, 1: p_mine_stress1, 2: p_mine_stress2}
    hour_results: list[HourResult] = []

    for hour, spread in enumerate(bid_ask_spreads):
        stress_level   = classify_stress(spread, proto)
        base_rate      = get_base_rate(stress_level, proto)
        v_cap          = get_v_cap(stress_level, proto)
        p_mine         = p_mine_by_level[stress_level]

        # шаг 1: кто майнит
        tickets: list[tuple[User, float]] = []
        for user in sim_users:
            if user.gratis_balance <= 0:
                continue
            if random.random() < p_mine:
                fraction = random.uniform(0.1, 1.0)
                tickets.append((user, user.gratis_balance * fraction))

        # шаг 2: cancellation window
        cancelled_count = 0
        if stress_level > 0 and p_cancel_on_penalty > 0:
            surviving = []
            for (user, amount) in tickets:
                if random.random() < p_cancel_on_penalty:
                    user.hours_cancelled += 1
                    cancelled_count += 1
                else:
                    surviving.append((user, amount))
            tickets = surviving

        # шаг 3: батч и pro-rata
        total_requested = sum(a for _, a in tickets)
        effective_rate  = pro_rata_rate(base_rate, v_cap, total_requested)

        # шаг 4: settlement
        total_mined = 0.0
        for user, amount in tickets:
            coen_out = amount * effective_rate
            user.gratis_balance   -= amount
            user.coen_received    += coen_out
            user.gratis_burned    += amount
            user.hours_participated += 1
            total_mined += coen_out
            user.history.append({
                "hour":             hour,
                "stress_level":     stress_level,
                "spread":           spread,
                "gratis_submitted": amount,
                "effective_rate":   effective_rate,
                "coen_received":    coen_out,
                "loss":             amount - coen_out,
            })

        hour_results.append(HourResult(
            hour=hour,
            spread=spread,
            stress_level=stress_level,
            base_rate=base_rate,
            effective_rate=effective_rate,
            v_cap=v_cap,
            total_requested=total_requested,
            total_mined=total_mined,
            participants=len(tickets),
            cancelled=cancelled_count,
            c_window=atk_by_level[stress_level],
        ))

    # ── агрегация
    result = SimulationResult(
        users=sim_users,
        hour_results=hour_results,
        attack_summary=AttackSummary(),
        proto=proto,
    )
    result.total_coen_mined    = sum(u.coen_received  for u in sim_users)
    result.total_gratis_burned = sum(u.gratis_burned   for u in sim_users)
    result.total_nominal_loss  = sum(u.nominal_loss    for u in sim_users)
    for h in hour_results:
        result.hours_by_level[h.stress_level] += 1

    c_actual = sum(h.c_window for h in hour_results)
    note = ""
    if atk2["c_per_window"] < 5_000:
        note = f"⚠  Атака дешёвая (${atk2['c_per_window']:,.0f}/окно). Рынок тонкий."
    elif atk2["c_per_window"] > 100_000:
        note = f"✓  Атака дорогая (${atk2['c_per_window']:,.0f}/окно). Хорошая защита."
    else:
        note = f"~  Атака умеренная (${atk2['c_per_window']:,.0f}/окно). Требует калибровки."

    result.attack_summary = AttackSummary(
        stress1_hours=result.hours_by_level[1],
        stress2_hours=result.hours_by_level[2],
        c_per_window_lvl1=atk1["c_per_window"],
        c_per_window_lvl2=atk2["c_per_window"],
        c_actual_observed=c_actual,
        breakeven_note=note,
    )
    return result


def run_simulation_endogenous(
    initial_spread: float,
    users: list[dict],
    hours: int,
    proto: Optional[ProtocolParams]       = None,
    attack: Optional[AttackParams]        = None,
    market: Optional[MarketImpactParams]  = None,
    coen_price_usd: float                 = 1.0,
    p_mine_normal: float                  = 0.10,
    p_mine_stress1: float                 = 0.30,
    p_mine_stress2: float                 = 0.60,
    p_cancel_on_penalty: float            = 0.25,
    random_seed: Optional[int]            = 42,
) -> SimulationResult:
    """
    Эндогенная версия симулятора: spread больше не подаётся извне как
    готовый временной ряд — он формируется КАЖДЫЙ ЧАС из реального
    давления продаж, создаваемого самим майнингом.

    Цикл обратной связи:
      spread(t) -> классификация стресса -> майнинг батч ->
      coen продан на рынке -> price impact -> spread(t+1)

    Это модель ОРГАНИЧЕСКОГО bank run: без внешнего атакующего,
    просто из-за того что много пользователей одновременно решают
    майнить и продавать. Cost-of-attack здесь неприменим — нет
    атакующего, которому можно приписать стоимость; вместо этого
    интересна сама динамика — раскручивается ли паника сама себя
    или гасится механизмом Firerate.
    """
    if proto  is None: proto  = ProtocolParams()
    if attack is None: attack = AttackParams()
    if market is None: market = MarketImpactParams()

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    sim_users = [
        User(
            user_id=u["user_id"],
            gratis_balance=u["gratis_balance"],
            gratis_balance_initial=u["gratis_balance"],
        )
        for u in users
    ]

    atk1 = attack_cost_to_level(1, attack, proto)
    atk2 = attack_cost_to_level(2, attack, proto)
    atk_by_level = {0: 0.0, 1: atk1["c_per_window"], 2: atk2["c_per_window"]}

    p_mine_by_level = {0: p_mine_normal, 1: p_mine_stress1, 2: p_mine_stress2}
    hour_results: list[HourResult] = []

    spread = initial_spread

    for hour in range(hours):
        stress_level = classify_stress(spread, proto)
        base_rate    = get_base_rate(stress_level, proto)
        v_cap        = get_v_cap(stress_level, proto)
        p_mine       = p_mine_by_level[stress_level]

        tickets: list[tuple[User, float]] = []
        for user in sim_users:
            if user.gratis_balance <= 0:
                continue
            if random.random() < p_mine:
                fraction = random.uniform(0.1, 1.0)
                tickets.append((user, user.gratis_balance * fraction))

        cancelled_count = 0
        if stress_level > 0 and p_cancel_on_penalty > 0:
            surviving = []
            for (user, amount) in tickets:
                if random.random() < p_cancel_on_penalty:
                    user.hours_cancelled += 1
                    cancelled_count += 1
                else:
                    surviving.append((user, amount))
            tickets = surviving

        total_requested = sum(a for _, a in tickets)
        effective_rate  = pro_rata_rate(base_rate, v_cap, total_requested)

        total_mined = 0.0
        for user, amount in tickets:
            coen_out = amount * effective_rate
            user.gratis_balance   -= amount
            user.coen_received    += coen_out
            user.gratis_burned    += amount
            user.hours_participated += 1
            total_mined += coen_out
            user.history.append({
                "hour":             hour,
                "stress_level":     stress_level,
                "spread":           spread,
                "gratis_submitted": amount,
                "effective_rate":   effective_rate,
                "coen_received":    coen_out,
                "loss":             amount - coen_out,
            })

        sell_volume_coen = total_mined * market.sell_through_rate
        sell_volume_usd  = sell_volume_coen * coen_price_usd

        hour_results.append(HourResult(
            hour=hour,
            spread=spread,
            stress_level=stress_level,
            base_rate=base_rate,
            effective_rate=effective_rate,
            v_cap=v_cap,
            total_requested=total_requested,
            total_mined=total_mined,
            participants=len(tickets),
            cancelled=cancelled_count,
            c_window=atk_by_level[stress_level],
            sell_volume_usd=sell_volume_usd,
            spread_endogenous=True,
        ))

        spread = update_spread_endogenous(spread, sell_volume_usd, attack.B, market)

    result = SimulationResult(
        users=sim_users,
        hour_results=hour_results,
        attack_summary=AttackSummary(),
        proto=proto,
    )
    result.total_coen_mined    = sum(u.coen_received  for u in sim_users)
    result.total_gratis_burned = sum(u.gratis_burned   for u in sim_users)
    result.total_nominal_loss  = sum(u.nominal_loss    for u in sim_users)
    for h in hour_results:
        result.hours_by_level[h.stress_level] += 1

    result.attack_summary = AttackSummary(
        stress1_hours=result.hours_by_level[1],
        stress2_hours=result.hours_by_level[2],
        c_per_window_lvl1=atk1["c_per_window"],
        c_per_window_lvl2=atk2["c_per_window"],
        c_actual_observed=0.0,
        breakeven_note=("Эндогенный режим: стресс возникает из органического "
                        "давления продаж, а не из внешней атаки. Cost-of-attack "
                        "не определён — нет атакующего, которому приписать стоимость."),
    )
    return result


# ──────────────────────────────────────────────────────────────
# ГЕНЕРАТОРЫ ВРЕМЕННЫХ РЯДОВ
# ──────────────────────────────────────────────────────────────

def generate_spread_calm(hours: int, base_spread: float = 0.01) -> list[float]:
    return [max(0.001, base_spread + np.random.normal(0, 0.003)) for _ in range(hours)]


def generate_spread_with_stress_spike(
    hours: int, spike_start: int, spike_duration: int,
    spike_level: float = 0.10, base_spread: float = 0.01,
) -> list[float]:
    spreads = generate_spread_calm(hours, base_spread)
    for i in range(spike_start, min(spike_start + spike_duration, hours)):
        spreads[i] = spike_level + np.random.normal(0, 0.005)
    return spreads


def generate_spread_sustained_attack(
    hours: int, attack_start: int, attack_duration: int = 7,
    attack_spread: float = 0.09, base_spread: float = 0.01,
) -> list[float]:
    spreads = generate_spread_calm(hours, base_spread)
    for i in range(attack_start, min(attack_start + attack_duration, hours)):
        spreads[i] = attack_spread + np.random.normal(0, 0.003)
    return spreads


def generate_spread_bank_run(hours: int) -> list[float]:
    spreads = []
    for h in range(hours):
        if h < 6:
            s = 0.01 + np.random.normal(0, 0.002)
        elif h < 10:
            s = 0.01 + (h - 6) * 0.015 + np.random.normal(0, 0.003)
        elif h < 14:
            s = 0.12 + np.random.normal(0, 0.01)
        elif h < 20:
            s = 0.12 - (h - 14) * 0.015 + np.random.normal(0, 0.005)
        else:
            s = 0.01 + np.random.normal(0, 0.002)
        spreads.append(max(0.001, s))
    return spreads


# ──────────────────────────────────────────────────────────────
# ДЕМО
# ──────────────────────────────────────────────────────────────

def make_demo_users(n: int = 100, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [
        {"user_id": f"user_{i:03d}",
         "gratis_balance": round(float(rng.lognormal(mean=8, sigma=1.5)), 2)}
        for i in range(n)
    ]


if __name__ == "__main__":
    np.random.seed(0)
    proto  = ProtocolParams()
    attack = AttackParams(B=100_000)

    print("=" * 65)
    print("FIRERATE SIMULATOR — упрощённая трёхуровневая модель")
    print("=" * 65)
    print()

    # расчёт стоимости атаки
    print_attack_breakdown(attack, proto)
    print()

    # sensitivity по глубине рынка
    print("SENSITIVITY: стоимость атаки lvl2 vs глубина рынка  (k_signal=0.002)")
    print("─" * 75)
    print(f"  {'B (USD)':>12}  {'C_capital':>10}  {'C_fill':>10}  {'C_signal':>10}  {'C/окно':>10}  Защита")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*9}")
    for b in [50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]:
        r = attack_cost_to_level(2, AttackParams(B=b), proto)
        flag = ("⚠ тонкий" if r["c_per_window"] < 5_000
                else ("✓ хорошо" if r["c_per_window"] > 100_000 else "~ умер."))
        print(f"  ${b:>11,.0f}  ${r['c_capital']:>9,.0f}  ${r['c_fill']:>9,.0f}  "
              f"${r['c_signal']:>9,.0f}  ${r['c_per_window']:>9,.0f}  {flag}")
    print()

    # влияние k_signal
    print("ВЛИЯНИЕ k_signal на стоимость атаки (B=$1M, lvl2)")
    print("─" * 65)
    print(f"  {'k_signal':>10}  {'C_signal':>10}  {'C/окно':>10}  Интерпретация")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*25}")
    for k, note in [(0.001, "тонкий рынок, низкие fees"),
                    (0.002, "базовый placeholder ◄"),
                    (0.005, "ликвидный рынок"),
                    (0.010, "высокий fill-risk")]:
        r = attack_cost_to_level(2, AttackParams(B=1_000_000, k_signal=k), proto)
        print(f"  {k:>10.3f}  ${r['c_signal']:>9,.0f}  ${r['c_per_window']:>9,.0f}  {note}")
    print()

    # симуляция
    users = make_demo_users(n=100)
    print(f"Пользователей: {len(users)}, суммарный Gratis: {sum(u['gratis_balance'] for u in users):,.0f}")
    print()

    scenarios = {
        "Спокойный рынок (24ч)": generate_spread_calm(24),
        "Острый стресс-spike (ч.8-11)": generate_spread_with_stress_spike(
            24, spike_start=8, spike_duration=3, spike_level=0.12),
        "Sustained атака 7ч (с ч.6)": generate_spread_sustained_attack(
            24, attack_start=6, attack_duration=7),
        "Bank run сценарий (24ч)": generate_spread_bank_run(24),
    }

    for name, spreads in scenarios.items():
        print(f"\n{'─'*65}")
        print(f"СЦЕНАРИЙ: {name}")
        print(f"{'─'*65}")
        result = run_simulation(
            bid_ask_spreads=spreads,
            users=deepcopy(users),
            proto=proto,
            attack=attack,
            p_mine_normal=0.08,
            p_mine_stress1=0.30,
            p_mine_stress2=0.65,
            p_cancel_on_penalty=0.20,
            random_seed=42,
        )
        print(result.summary())
        labels = {0: ".", 1: "~", 2: "!"}
        timeline = "".join(labels[h.stress_level] for h in result.hour_results)
        print(f"Timeline: [{timeline}]  (. норма  ~ умерен  ! сильный)")

    # ── ЭНДОГЕННЫЙ РЕЖИМ: spread формируется самим майнингом
    print(f"\n{'='*65}")
    print("ЭНДОГЕННЫЙ РЕЖИМ: bank run без внешней атаки")
    print(f"{'='*65}")
    print("Spread больше не задан заранее — он формируется из давления")
    print("продаж, которое создаёт сам майнинг. Проверяем: раскручивается")
    print("ли паника сама себя, или Firerate её гасит.")
    print()

    market = MarketImpactParams(
        sell_through_rate=0.6,
        impact_coefficient=0.8,
        spread_decay=0.3,
        spread_floor=0.01,
    )

    # сценарий A: спокойный старт, нет начального шока
    print(f"{'─'*65}")
    print("Сценарий A: спокойный старт (spread=0.01), без внешнего триггера")
    print(f"{'─'*65}")
    result_a = run_simulation_endogenous(
        initial_spread=0.01,
        users=deepcopy(users),
        hours=24,
        proto=proto,
        attack=attack,
        market=market,
        p_mine_normal=0.08,
        p_mine_stress1=0.30,
        p_mine_stress2=0.65,
        p_cancel_on_penalty=0.20,
        random_seed=42,
    )
    print(result_a.summary())
    labels = {0: ".", 1: "~", 2: "!"}
    timeline_a = "".join(labels[h.stress_level] for h in result_a.hour_results)
    spread_path_a = " ".join(f"{h.spread:.3f}" for h in result_a.hour_results[:12])
    print(f"Timeline: [{timeline_a}]")
    print(f"Spread путь (первые 12ч): {spread_path_a} ...")

    # сценарий B: небольшой начальный шок — проверяем самораскрутку
    print(f"\n{'─'*65}")
    print("Сценарий B: начальный шок (spread=0.05), проверка самораскрутки")
    print(f"{'─'*65}")
    result_b = run_simulation_endogenous(
        initial_spread=0.05,
        users=deepcopy(users),
        hours=24,
        proto=proto,
        attack=attack,
        market=market,
        p_mine_normal=0.08,
        p_mine_stress1=0.30,
        p_mine_stress2=0.65,
        p_cancel_on_penalty=0.20,
        random_seed=42,
    )
    print(result_b.summary())
    timeline_b = "".join(labels[h.stress_level] for h in result_b.hour_results)
    spread_path_b = " ".join(f"{h.spread:.3f}" for h in result_b.hour_results[:12])
    print(f"Timeline: [{timeline_b}]")
    print(f"Spread путь (первые 12ч): {spread_path_b} ...")

    print()
    print("Интерпретация: если spread в сценарии B растёт сам по себе")
    print("после начального шока (несмотря на сниженную ставку и cap),")
    print("значит price impact от sell-through перевешивает defensive")
    print("эффект Firerate — паника самораскручивается. Если spread")
    print("затухает к floor — Firerate гасит панику успешно.")
