# -*- coding: utf-8 -*-
"""
Прогон Credis-симуляции под два вопроса дизайна
================================================

В1: БЕЗ Firerate — какая протокольная ликвидность (POL) и требуемая
    доходность частных LP (APR_req) дают самоподдерживающийся пул?
В2: С Firerate — то же самое, плюс потери пользователей при стрессе
    и среднее время ожидания конвертации.

Оба мира считаются В ОДНОМ прогоне (paired counterfactual на общих
случайных числах): FR-мир — колонки PoolTVL/…, NoFR-мир — PoolTVLNoFR/….

Запуск:  python run_pol_apr_questions.py
"""
import numpy as np
import pandas as pd

from dex_credis_simulation import CredisSimulation
from f_firerate_calibration import FirerateCalibration
from pol_apr_planner import saturated_apr

# ── кандидаты (POL, требуемый APR частных LP) и fee tier ────────────
CANDIDATES = [
    (100_000,   0.10),
    (500_000,   0.10),
    (500_000,   0.25),
    (2_000_000, 0.25),
]
FEE_TIER = 0.01           # 1% — по выводу планировщика: 0.3% не зажигает петлю
EXT_VOLUME_USD_DAILY = 0  # внешний спекулятивный объём; поставьте оценку, если есть

SIM_KWARGS = dict(
    # ← сюда ваши обычные аргументы конструктора CredisSimulation
    # coen_price_path=..., s_curve_path=..., ...
    coen_price_path='/Users/aminakaltayeva/Desktop/crypto_joyslab/misc/coen_price_simulation_v6.xlsx',
    s_curve_path='/Users/aminakaltayeva/Desktop/crypto_joyslab/misc/s_curve_price_simulation_v2.xlsx'
)


def viability_verdict(s: dict) -> tuple[str, str]:
    """(вердикт NoFR-мира, вердикт FR-мира) по сводке run_pol_apr."""
    # Без Firerate «самоподдержка» требует ЖИВОГО пула, а не только APR:
    # объём умирающего пула (спираль) создаёт фиктивную выручку.
    nofr_alive = (s['disp_max_nofr'] <= 0.20) and (s['lp_min_nofr'] >= 0.30)
    nofr_ignites = s['priv_share_nofr'] > 0.5
    v1 = ("САМОПОДДЕРЖИВАЕТСЯ" if (nofr_alive and nofr_ignites) else
          "POL-ONLY (петля не зажглась)" if nofr_alive else
          "НЕЖИЗНЕСПОСОБЕН (коллапс пула)")
    fr_ignites = s['priv_share'] > 0.5
    v2 = ("САМОПОДДЕРЖИВАЕТСЯ" if fr_ignites else "POL-ONLY (петля не зажглась)")
    return v1, v2


def run_all():
    print(f"APR_sat (потолок органики, зрелый пул, fee={FEE_TIER:.1%}): "
          f"{saturated_apr(FEE_TIER, 0.0075, 180):.1%}"
          f"  — apr_req выше него без внешнего объёма не зажигается\n")

    rows = []
    for pol, apr in CANDIDATES:
        print("#" * 78)
        print(f"# POL = ${pol:,}   APR_req = {apr:.0%}   fee = {FEE_TIER:.1%}")
        print("#" * 78)

        sim = CredisSimulation(**SIM_KWARGS)        # свежий инстанс на точку
        s = sim.run_pol_apr(
            pol_usd=pol, apr_req=apr, fee_tier=FEE_TIER,
            ext_volume_usd_daily=EXT_VOLUME_USD_DAILY, verbose=True,
        )
        v1, v2 = viability_verdict(s)

        # ── потери при стрессе и ожидание — из калибровочных метрик
        cal = FirerateCalibration(sim.results)
        ulf = cal.user_loss_fairness(verbose=False)
        dly = cal.conversion_delay(verbose=False)

        print("  ── В1 (БЕЗ Firerate):", v1)
        print(f"     TVL_eq ${s['tvl_eq_nofr']:,.0f}, частный "
              f"{s['priv_share_nofr']:.0%}, disp max {s['disp_max_nofr']:.0%}, "
              f"LP min {s['lp_min_nofr']:.2f}")
        print("  ── В2 (С Firerate):  ", v2)
        print(f"     TVL_eq ${s['tvl_eq']:,.0f}, частный {s['priv_share']:.0%}, "
              f"APR_real {s['apr_realized']:.1%}")
        print(f"     Потери в стресс-дни: средн {ulf.panic_user_mean_loss_pct:.1f}%, "
              f"P95 {ulf.panic_user_p95_loss_pct:.1f}%; "
              f"в норме {ulf.normal_user_mean_loss_pct:.1f}%")
        print(f"     Ожидание конвертации: медиана "
              f"{dly['wait_median']:.1f} дн (норма {dly['wait_median_calm']:.1f}, "
              f"стресс {dly['wait_median_stress']:.1f}), "
              f"P95 {dly['wait_p95']:.1f} дн")
        print()

        rows.append(dict(
            pol=pol, apr_req=apr,
            verdict_no_fr=v1, verdict_fr=v2,
            tvl_fr=s['tvl_eq'], priv_fr=s['priv_share'],
            tvl_nofr=s['tvl_eq_nofr'], disp_max_nofr=s['disp_max_nofr'],
            loss_stress_mean=ulf.panic_user_mean_loss_pct,
            loss_stress_p95=ulf.panic_user_p95_loss_pct,
            wait_median_d=dly['wait_median'],
            wait_p95_d=dly['wait_p95'],
            throttle_pct=s['throttle_pct'],
        ))

    df = pd.DataFrame(rows)
    print("=" * 78)
    print("СВОДНАЯ ТАБЛИЦА")
    print("=" * 78)
    with pd.option_context('display.width', 160, 'display.max_columns', 20):
        print(df.to_string(index=False))
    df.to_csv('pol_apr_questions_results.csv', index=False)
    return df


if __name__ == "__main__":
    run_all()
