"""
Планировщик POL / APR: какая протокольная ликвидность и какая требуемая
доходность частных LP дают самоподдерживающийся пул
========================================================================

Модель (вариант 2): TVL эндогенный.
  TVL(t) = POL + h × TVL_priv(t)
  TVL_priv* = max(0, fee × V_daily × 365 / APR_req − POL)
  V_daily = поток конвертации (side FR-мира) + арбитраж + внешний объём

ЗАМКНУТАЯ ФОРМА — потолок доходности насыщенного пула.
При хроническом дефиците поглощения поток = cap(x0), тогда объём через
пул НЕ зависит от TVL в долях:
  absorb_daily/TVL = 24 × (d1_frac(age)/0.005038) × ((1−x0)^-½ − 1)
  APR_sat = fee × vol_mult × absorb_daily/TVL × 365
где vol_mult ≈ 2 (каждый доллар продаж порождает ~доллар арбитража).
Это ВЕРХНЯЯ граница органической доходности пула: если APR_sat ниже
требуемой доходности LP — частный капитал не придёт ни при каком TVL,
и пул навсегда POL-only. Единственные рычаги: fee tier, x0, внешний
(спекулятивный) объём.

Использование:
  python pol_apr_planner.py                # замкнутые формы + фронтир
  from pol_apr_planner import saturated_apr, pol_apr_frontier
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd
from dataclasses import replace

from dex_market import (DexPoolState, DexPoolParams, DexProtocolParams,
                        DexStressClassifier, depth_frac_of_tvl)
from firerate_hourly_dex import run_mine_coen_day_dex

D1_FACTOR = 1.0 / (1.0 / np.sqrt(0.99) - 1.0)   # ≈ 198.5


# ──────────────────────────────────────────────────────────────
# ЗАМКНУТЫЕ ФОРМЫ
# ──────────────────────────────────────────────────────────────

def absorb_frac_daily(x0: float, age_days: float) -> float:
    """Поглощение насыщенного пула, доля TVL в день (24 окна на cap x0)."""
    R_frac = depth_frac_of_tvl(age_days) * D1_FACTOR       # R/TVL
    per_window = R_frac * (1.0 / np.sqrt(1.0 - x0) - 1.0)
    return 24.0 * per_window


def saturated_apr(fee: float, x0: float, age_days: float,
                  vol_mult: float = 2.0) -> float:
    """Потолок ОРГАНИЧЕСКОЙ доходности пула при насыщенной очереди.
    apr_req <= APR_sat → линейное равновесие TVL не ограничено (модельная
    граница; на практике упирается в адресуемый LP-капитал)."""
    return fee * vol_mult * absorb_frac_daily(x0, age_days) * 365.0


def required_pol_bootstrap(target_tvl: float, fee: float, x0: float,
                           age_days: float, apr_req: float,
                           vol_mult: float = 2.0) -> float:
    """
    Минимальный POL-сид: частный капитал приходит только на выручку,
    выручка требует объёма, объём — глубины. Если APR_sat >= apr_req,
    равновесие самоподдерживается при ЛЮБОМ TVL (объём ∝ TVL) — POL
    нужен только как анти-манипуляционная доля (>= delta2 от总). Если
    APR_sat < apr_req — частный TVL* = 0 и POL = весь target_tvl.
    """
    if saturated_apr(fee, x0, age_days, vol_mult) >= apr_req:
        return 0.35 * target_tvl        # анти-манипуляционный минимум POL-доли
    return target_tvl


def print_closed_forms():
    print("=" * 78)
    print("ЗАМКНУТЫЕ ФОРМЫ: потолок доходности насыщенного пула (APR_sat)")
    print("=" * 78)
    print(f"{'возраст':>8} {'d1/TVL':>8} {'поглощ./день':>13} | "
          f"{'APR@fee 0.3%':>13} {'APR@fee 1%':>11}")
    for age in [14, 30, 60, 90, 180]:
        af = absorb_frac_daily(0.0075, age)
        a03 = saturated_apr(0.003, 0.0075, age)
        a10 = saturated_apr(0.010, 0.0075, age)
        print(f"{age:>7}д {depth_frac_of_tvl(age):>8.4f} {af:>12.1%} | "
              f"{a03:>13.1%} {a10:>11.1%}")
    print()
    print("Чтение: если требуемая LP-доходность выше APR_sat своего столбца —")
    print("частный капитал не приходит НИ ПРИ КАКОМ TVL (объём ∝ TVL, доходность")
    print("от TVL не зависит). Тогда TVL = POL, и вопрос POL — вопрос フронтира F.")
    print("Рычаги подъёма APR_sat: fee tier (×3.3), x0 (линейно), внешний объём.")
    print()


# ──────────────────────────────────────────────────────────────
# СИМУЛЯЦИОННЫЙ ФРОНТИР
# ──────────────────────────────────────────────────────────────

N, DAYS, SEED = 50_000, 300, 42
TARGET_USD = 20.0


def run_pol_apr(pol_usd: float, apr_req: float, fee: float = 0.003,
                ext_usd: float = 0.0, stress_sens: float = 2.0,
                days: int = DAYS) -> dict:
    """Органический прогон с эндогенным TVL. Возвращает сводку равновесия."""
    rng = np.random.default_rng(SEED)
    bal = np.zeros(N); bald = np.zeros(N, np.int32)
    baln = np.zeros(N); baldn = np.zeros(N, np.int32)
    joined = np.zeros(N, bool); seeking = np.zeros(N, bool)
    pp = replace(DexPoolParams(), fee_tier=fee, lp_required_apr=apr_req,
                 ext_volume_usd_daily=ext_usd)
    proto = DexProtocolParams()
    pool_fr = DexPoolState(params=pp, pol_usd=pol_usd)
    pool_no = DexPoolState(params=pp, pol_usd=pol_usd)
    clf_fr = DexStressClassifier(proto, fee=fee)
    clf_no = DexStressClassifier(proto, fee=fee)
    price = np.cumprod(1 + rng.normal(0.001, 0.015, days + 1))
    rows = []
    for day in range(1, days + 1):
        n_act = int(N * min(1, day / 120))
        joined[:n_act] = True
        idx = np.flatnonzero(joined)
        miners = idx[rng.random(idx.size) < 0.05]
        amt = rng.lognormal(3, 0.8, miners.size)
        bal[miners] += amt; bald[miners] = day
        baln[miners] += amt; baldn[miners] = day
        r = run_mine_coen_day_dex(
            day, bal, bald, baln, baldn, pool_fr, pool_no, clf_fr, clf_no,
            trend=0.1, coen_price=float(price[day]), tvl_usd=0.0,  # эндогенный
            proto=proto, rng=rng, p_convert_stress_sens=stress_sens,
            seeking=seeking)
        info = r.tvl_info
        rows.append(dict(
            H1=r.stress_hours.get(1, 0), H2=r.stress_hours.get(2, 0),
            Coen=r.coen_from_conversion, CoenNo=r.coen_from_conversion_nofr,
            Queue=r.queue_gratis_end, Total=float(bal.sum()),
            Conv=r.gratis_converted,
            TVL=info.get("tvl_total", np.nan),
            TVLpriv=info.get("tvl_priv", np.nan),
            APR=info.get("apr_realized", np.nan),
            Vol=info.get("vol_day", np.nan)))
    df = pd.DataFrame(rows)
    tail = df.tail(60)
    srv = df.Conv.rolling(7, min_periods=1).mean()
    wait = (df.Queue / srv.replace(0, np.nan)).iloc[-1]
    return {
        "pol": pol_usd, "apr_req": apr_req, "fee": fee,
        "tvl_eq": float(tail.TVL.mean()),
        "tvl_priv_eq": float(tail.TVLpriv.mean()),
        "apr_realized": float(tail.APR.mean()),
        "priv_share": float(tail.TVLpriv.mean() / tail.TVL.mean())
        if tail.TVL.mean() > 0 else 0.0,
        "stress_days_pct": float(((df.H1 + df.H2) >= 4).mean() * 100),
        "throttle_pct": float((1 - df.Coen.sum() / df.CoenNo.sum()) * 100)
        if df.CoenNo.sum() > 0 else np.nan,
        "queue_share": float(df.Queue.iloc[-1] / df.Total.iloc[-1]),
        "wait_days_end": float(wait) if np.isfinite(wait) else np.inf,
        "df": df,
    }


def pol_apr_frontier(pol_grid=(2_000_000, 20_000_000),
                     apr_grid=(0.05, 0.10),
                     fee_grid=(0.003, 0.01),
                     ext_frac: float = 0.0) -> list:
    print("=" * 100)
    print(f"ФРОНТИР POL × APR (эндогенный TVL, внешний объём {ext_frac:.0%} TVL/день)")
    print("=" * 100)
    print(f"{'fee':>5} {'POL':>11} {'APR_req':>8} | {'TVL_eq':>11} {'приват':>8} "
          f"{'APR_real':>9} | {'стресс':>7} {'throttle':>9} {'очередь':>8}")
    print("-" * 100)
    out = []
    for fee in fee_grid:
        for pol in pol_grid:
            for apr in apr_grid:
                r = run_pol_apr(pol, apr, fee=fee)
                out.append(r)
                print(f"{fee:>5.1%} ${pol:>10,.0f} {apr:>8.0%} | "
                      f"${r['tvl_eq']:>10,.0f} {r['priv_share']:>8.1%} "
                      f"{r['apr_realized']:>9.1%} | {r['stress_days_pct']:>6.1f}% "
                      f"{r['throttle_pct']:>8.1f}% {r['queue_share']:>8.1%}")
        print("-" * 100)
    return out


if __name__ == "__main__":
    t0 = time.time()
    print_closed_forms()
    pol_apr_frontier()
    print(f"\nВсего: {time.time()-t0:.1f}s")
