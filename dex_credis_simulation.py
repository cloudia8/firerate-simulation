from viyasa_1911 import optimization_algorithm
from lysis_fractions_algo_jan2026 import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from utils import *
import time

pd.options.display.float_format = '{:,.5f}'.format

from Assumptions import Assumptions
from SimulationParameters import *

from dataclasses import replace as dc_replace
from dex_market import DexPoolState, DexPoolParams, DexProtocolParams, DexStressClassifier
from firerate_hourly_dex import run_mine_coen_day_dex, panic_threshold_map_for_day
from liquidity_requirement import liquidity_requirement_for_day, tvl_for_zero_throttle


# ---------------------------------------------------------------------------
# Axis formatters
# ---------------------------------------------------------------------------

def billions_formatter(x, pos):
    return f'{x / 10**9}'

def millions_formatter(x, pos):
    return f'{x / 10**6}'

def thousands_formatter(x, pos):
    return f'{x / 10**3}'


# ---------------------------------------------------------------------------
# CredisSimulation
# ---------------------------------------------------------------------------

class CredisSimulation:
    """
    Usage
    -----
    sim = CredisSimulation(
        coen_price_path='path/to/coen_price.xlsx',
        s_curve_path='path/to/s_curve.xlsx',
    )
    sim.run()
    df = sim.results
    """

    OUTBE_TRANSACTIONS = {
        'top_up', 'notify TX', 'request credis', 'reserve gratis',
        'reserve stablecoin', 'lock ticket', 'pledge gratis',
        'order to vault send stablecoin to cca',
    }
    OTHER_NETWORKS_TRANSACTIONS = {
        'lock stablecoin', 'order to vault send stablecoin to cca',
        'escrow stablecoin', 'create intent for stablecoin to customer card account',
    }

    _EOD_COLUMNS = [
        'Day', 'GreenDay', 'ReadyToSellDay', 'CoenPrice',
        'ConsumerBaseCells', 'ConsumerBase', 'AllocationLimit',
        'ConsumerCanSell', 'ConsumerReasonableSell', 'ConsumerDecidedSell',
        'NodsBalance', 'NodsQualifiedBalance', 'GratisMined', 'StrikePricePaid',
        'PromisDemand', 'PromisAllocation', 'PromisProceeds', 'PromisReserve',
        'IntexIssuedBalance', 'IntexSettledInflow', 'IntexSettledPaidInflow',
        'Rewards', 'Touch',
        'SellProbMean', 'JoinProbMean',
        'trend', 'n_active_consumers', 'tam_f', 'factual_deficit', 'avg_wait', 'user_gain',
        # ── Firerate additions
        'GratisRequested', 'GratisConverted', 'CoenFromConversion', 
        'GratisTotalBalance',
        'FirerateStressLevel', 'FirerateEffectiveRate',
        'FirerateStress1Hours', 'FirerateStress2Hours',
        'SpreadEnd',
        # ── контрфактический мир (paired counterfactual, no Firerate)
        'SpreadEndNoFR', 'GratisConvertedNoFR', 'CoenFromConversionNoFR',
        # ── DEX-режим (NaN в CEX-режиме). В DEX-режиме SpreadEnd/SpreadEndNoFR
        #    несут displacement пула (стресс-мера DEX) — Metric 1 калибровки
        #    работает поверх без изменений.
        'DepthEnd', 'DHatEnd', 'LPHealth', 'LPHealthNoFR',
        'ConversionQueueGratis',
        # ── эндогенный TVL (NaN в legacy/CEX-режимах)
        'PoolTVL', 'PoolTVLPrivate', 'PoolAPRRealized', 'PoolVolumeDay',
        'PoolTVLNoFR', 'PoolTVLPrivateNoFR', 'PoolAPRRealizedNoFR',
        # 'TVLRequiredNoFR', 'PolMinUSD', 'PolMinFrac', 'GapUSD', 'ImpliedAPRGap',
        # # 'ThresholdF0_1pct', 
        # 'ThresholdF0_5pct', 
        # # 'ThresholdF1pct',
        # 'ThresholdF5pct', 
        # # 'ThresholdF10pct', 
        # 'ThresholdF25pct',
        # 'ThresholdF50pct', 
        # 'ThresholdF100pct',
        'TVLToday'
    ]

    def __init__(
        self,
        coen_price_path: str,
        s_curve_path: str,
        *,
        seed: int = 42,
        initial_capacity: int = 3_000_000,
        target_consumer_base: int = 10_000_000,
        simulation_length: int = 2000,
        target_gratis_balance_usd: float = 20.0,
        total_intex_demand_fiat: float = 3_000_000.0,
        clearing_price_share: float = 0.08,
        allocation_limit_others_share: float = 0.12,
        deificit_start: float = 0.32,
        L: int = 5,
        # ── Firerate config
        firerate_compute_every_n_days: int = 7,
        firerate_probe_hours: int = 48,
        firerate_b_fractions: list = None,
        firerate_b_mcap_fraction: float = 0.01,  # B = active_users × target_gratis_usd × coen_price × fraction
        # ── Gratis conversion model
        p_convert_base: float = 0.05,    # базовая вероятность конвертации в день
        p_convert_price_sens: float = 2.0,  # чувствительность к росту цены
        p_convert_hold_sens: float = 0.3,   # чувствительность к длительности держания
        p_convert_stress_sens: float = 2.0, # канал паники: множитель (1 + sens·stress_level).
                                            # 0.0 → прежнее поведение (без паники)
        p_cancel_on_penalty: float = 0.25,  # доля тикетов, отзываемых при виде штрафа (только FR-мир)
        firerate_hours_per_day: int = 24,   # часовых batch-окон в дне (единицы протокола = час)
        firerate_tvl_mcap_fraction: float = 0.05,  # TVL пула = active × target_usd × price × fraction
        firerate_panic_grid: tuple = (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64),
        firerate_probe_max_users: int = 200_000,   # сабсэмпл probe на больших базах
        firerate_pol_usd: float = 0.0,
        firerate_liqreq_every_n_days: int = 30,   # раз в N дней; медленно
                                                  # дрейфующая величина
        firerate_liqreq_sens_grid: tuple = (4,),  # одна представительная
                                                  # точка паники, не полная сетка
        firerate_liqreq_probe_hours: int = 48,
        firerate_liqreq_tol: float = 0.15,        # грубая точность (±15%)
        firerate_liqreq_warm_start_factor: float = 3.0,
        firerate_liqreq_max_probe_users: int = 50_000,   # >0 → ЭНДОГЕННЫЙ TVL: POL (USD, политика
                                         # казначейства) + частный LP-капитал из
                                         # доходностного равновесия (см. dex_market
                                         # update_private_tvl). 0 → legacy-режим
                                         # TVL = mcap × firerate_tvl_mcap_fraction.
    ):
        self.assumptions = Assumptions()
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        self.initial_capacity              = initial_capacity
        self.target_consumer_base          = target_consumer_base
        self.simulation_length             = simulation_length
        self.target_gratis_balance_usd     = target_gratis_balance_usd
        self.total_intex_demand_fiat       = total_intex_demand_fiat
        self.clearing_price_share          = clearing_price_share
        self.allocation_limit_others_share = allocation_limit_others_share
        self.deificit_start                = deificit_start
        self.L                             = L
        self.N_GROUPS                      = 2 ** L

        self.a0, self.a1, self.a2, self.a3, self.a4 = 1, 3.0, 1, 1, 2

        # ── Firerate config

        # ── Firerate config (DEX)
        self.firerate_b_mcap_fraction      = firerate_b_mcap_fraction
        self.firerate_compute_every_n_days = firerate_compute_every_n_days
        self.firerate_probe_hours          = firerate_probe_hours
        self.firerate_b_fractions = firerate_b_fractions or [
            0.005,  0.05,  0.25, 0.5, 1
        ]

        # ── Gratis conversion model
        self.p_convert_base        = p_convert_base
        self.p_convert_price_sens  = p_convert_price_sens
        self.p_convert_hold_sens   = p_convert_hold_sens
        self.p_convert_stress_sens = p_convert_stress_sens
        self.p_cancel_on_penalty   = p_cancel_on_penalty
        self.firerate_hours_per_day = firerate_hours_per_day

        # ── DEX-режим (market_model='dex')
        self.market_model                = 'dex'   # CEX-путь удалён
        self.firerate_tvl_mcap_fraction  = firerate_tvl_mcap_fraction
        self.firerate_panic_grid         = firerate_panic_grid
        self.firerate_probe_max_users    = firerate_probe_max_users
        self.dex_proto        = DexProtocolParams()
        self.dex_pool_params  = DexPoolParams()
        self.firerate_pol_usd = firerate_pol_usd
        self.firerate_liqreq_every_n_days       = firerate_liqreq_every_n_days
        self.firerate_liqreq_sens_grid          = firerate_liqreq_sens_grid
        self.firerate_liqreq_probe_hours        = firerate_liqreq_probe_hours
        self.firerate_liqreq_tol                = firerate_liqreq_tol
        self.firerate_liqreq_warm_start_factor  = firerate_liqreq_warm_start_factor
        self.firerate_liqreq_max_probe_users    = firerate_liqreq_max_probe_users
        self._last_liquidity_requirement: dict | None = None
        self._last_TVL = 10_000

        df_coen = pd.read_excel(coen_price_path)
        df_coen.rename(columns={'Unnamed: 0': 'day'}, inplace=True)
        self.coen_price_arr = df_coen['CoenPriceBase'].to_numpy()

        df_sc = pd.read_excel(s_curve_path)
        self.s_curve_arr = df_sc['0.64Price'].values

        self._build_consumer_model()
        self._init_state()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_consumer_model(self):
        N = self.initial_capacity
        rng = self.rng
        A = self.assumptions

        seg_ids  = [0, 1, 2]
        seg_dist = [0.2, 0.6, 0.2]
        segs = rng.choice(seg_ids, size=N, p=seg_dist)

        def _map(vals):
            return np.array(vals)[segs]

        self.sim_enter_base      = _map([-3, -5, -6])
        self.sim_enter_adoption  = _map([4, 3, 2])
        self.sim_enter_deficit   = _map([0.5, 2, 3])
        self.sim_enter_trend     = _map([0, 0.5, 1])
        self.tam_k               = 1 / 180

        x_idx   = np.arange(N)
        p_gauss = np.exp(-((x_idx - N / 2) ** 2) / (2 * (N / 8) ** 2))
        P_gauss = p_gauss / p_gauss.sum()
        additional = np.round((self.target_consumer_base - N) * P_gauss)
        self.final_consumer_base_count = np.ones(N, dtype=int) + additional

        self.trian_a, self.trian_b, self.trian_c = A.consumption['T']

    def _init_state(self):
        N = self.initial_capacity
        G = self.N_GROUPS

        self.dt_nods = np.dtype([
            ("Group", np.int32), ("SymbolicLoad", np.float64),
            ("TargetPrice", np.float64), ("Day", np.int32),
        ])
        self.dt_nods_q = np.dtype([
            ("Group", np.int32), ("SymbolicLoad", np.float64),
            ("StrikePrice", np.float64), ("Day", np.int32),
        ])
        self.dt_intex_issued = np.dtype([
            ("SymbolicLoad", np.float64), ("CallThreshold", np.float64),
            ("StrikePrice", np.float64), ("Day", np.int32),
        ])
        self.dt_intex_called = np.dtype([
            ("SymbolicLoad", np.float64), ("CallThreshold", np.float64),
            ("StrikePrice", np.float64), ("Day", np.int32), ("CallDay", np.int32),
        ])

        self.nods_issued          = np.array([], dtype=self.dt_nods)
        self.nods_qualified       = np.array([], dtype=self.dt_nods_q)
        self.nods_issued_detailed = [[]]

        self.intex_issued         = np.array([], dtype=self.dt_intex_issued)
        self.intex_called_history = np.array([], dtype=self.dt_intex_called)

        self.gratis_balance           = np.zeros(N, dtype=np.float32)
        self.gratis_balance_day       = np.zeros(N, dtype=np.int32)   # день последнего начисления Gratis
        self.promis_proceeds          = np.zeros(G, dtype=np.float32)
        self.joined_customer_base     = np.zeros(N, dtype=np.bool_)
        self.joined_customer_base_day = np.zeros(N, dtype=np.int32)

        self.total_unallocated_promis = 0.0
        self.factual_deficit          = self.deificit_start
        self.block_number             = 16
        self.day_sell_prob            = 0.0

        # ── Firerate state
        # spread-колонки в DEX несут displacement; стартуем с нуля
        self.spread                    = 0.0
        self._last_firerate_thresholds = {f: float('inf') for f in self.firerate_b_fractions}

        # ── Контрфактический (no-Firerate) мир: собственная эволюция
        #    балансов и spread. Приток Gratis (mineGratis) идентичен
        #    основному миру; расходится только конвертация.
        self.gratis_balance_nofr     = np.zeros(N, dtype=np.float32)
        self.gratis_balance_day_nofr = np.zeros(N, dtype=np.int32)
        self.spread_nofr             = self.spread

        # ── DEX-состояние: пул и классификатор на каждый мир
        # DEX-состояние: пул и классификатор на каждый мир
        if True:
            self.pool_fr   = DexPoolState(params=self.dex_pool_params,
                                          pol_usd=self.firerate_pol_usd)
            self.pool_nofr = DexPoolState(params=self.dex_pool_params,
                                          pol_usd=self.firerate_pol_usd)
            self.clf_fr    = DexStressClassifier(self.dex_proto, fee=self.dex_pool_params.fee_tier)
            self.clf_nofr  = DexStressClassifier(self.dex_proto, fee=self.dex_pool_params.fee_tier)
            # очередь конвертации: хотел хотя бы раз, ещё не исполнен
            self.conversion_seeking = np.zeros(N, dtype=bool)

        self.results = pd.DataFrame(columns=self._EOD_COLUMNS)

    # ------------------------------------------------------------------
    # Pure / static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tam_curve(tam, day, simulation_length, tam_k):
        return round(1 / (1 + np.exp(-tam_k * (day - simulation_length / 2))) * tam + 1)

    @staticmethod
    def _consumer_enter_prob(base, adoption_sens, deficit_level, trend_sens,
                             n_active, tam_f, factual_deficit, deificit_start, trend):
        x1 = np.log(n_active + 1) / np.log(tam_f)
        x2 = factual_deficit / deificit_start
        x3 = np.clip(trend / 0.2, -1, 1)
        return 1 / (1 + np.exp(-(base + adoption_sens * x1 + deficit_level * x2 + trend_sens * x3)))

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_pol_apr(self, pol_usd: float, apr_req: float,
                    fee_tier: float = None,
                    ext_volume_usd_daily: float = None,
                    lp_capital_cap_usd: float = 50e6,
                    verbose: bool = True) -> dict:
        """
        Доходностное равновесие POL/APR на РЕАЛЬНОЙ симуляции — интерфейс
        pol_apr_planner.run_pol_apr, но с настоящей эмиссией, базой
        пользователей и ценовым путём Credis (а не синтетическим harness).

            sim = CredisSimulation(...)
            summary = sim.run_pol_apr(pol_usd=500_000, apr_req=0.25,
                                      fee_tier=0.01)

        Конфигурирует dex_pool_params, включает эндогенный TVL,
        пере-инициализирует состояние и запускает полный sim.run().
        Возвращает сводку равновесия; полные ряды — в sim.results
        (PoolTVL / PoolTVLPrivate / PoolAPRRealized / PoolVolumeDay).
        """
        upd = dict(lp_required_apr=apr_req, lp_capital_cap_usd=lp_capital_cap_usd)
        if fee_tier is not None:
            upd['fee_tier'] = fee_tier
        if ext_volume_usd_daily is not None:
            upd['ext_volume_usd_daily'] = ext_volume_usd_daily
        self.dex_pool_params  = dc_replace(self.dex_pool_params, **upd)
        self.firerate_pol_usd = float(pol_usd)
        self._init_state()               # чистое состояние под новую конфигурацию
        self.run(verbose=verbose)
        return self._pol_apr_summary(verbose=verbose)

    def _pol_apr_summary(self, tail_days: int = 60, verbose: bool = True) -> dict:
        """Сводка доходностного равновесия по хвосту self.results."""
        df = self.results
        num = lambda c: pd.to_numeric(df[c], errors='coerce')
        tail = df.tail(tail_days)
        tnum = lambda c: pd.to_numeric(tail[c], errors='coerce')
        conv = num('GratisConverted')
        srv = conv.rolling(7, min_periods=1).mean()
        queue = num('ConversionQueueGratis')
        wait = (queue / srv.replace(0, np.nan)).iloc[-1]
        h = num('FirerateStress1Hours').fillna(0) + num('FirerateStress2Hours').fillna(0)
        coen, coen_no = num('CoenFromConversion'), num('CoenFromConversionNoFR')
        out = {
            'pol_usd':        self.firerate_pol_usd,
            'apr_req':        self.dex_pool_params.lp_required_apr,
            'fee_tier':       self.dex_pool_params.fee_tier,
            'tvl_eq':         float(tnum('PoolTVL').mean()),
            'tvl_priv_eq':    float(tnum('PoolTVLPrivate').mean()),
            'priv_share':     float(tnum('PoolTVLPrivate').mean()
                                    / tnum('PoolTVL').mean())
                              if tnum('PoolTVL').mean() > 0 else 0.0,
            'apr_realized':   float(tnum('PoolAPRRealized').mean()),
            'stress_days_pct': float((h >= 4).mean() * 100),
            'throttle_pct':   float((1 - coen.sum() / coen_no.sum()) * 100)
                              if coen_no.sum() > 0 else np.nan,
            'queue_share':    float(queue.iloc[-1]
                                    / num('GratisTotalBalance').iloc[-1]),
            'wait_days_end':  float(wait) if np.isfinite(wait) else np.inf,
            # ── мир БЕЗ Firerate: самоподдержка пула и его состояние
            'tvl_eq_nofr':      float(tnum('PoolTVLNoFR').mean()),
            'tvl_priv_eq_nofr': float(tnum('PoolTVLPrivateNoFR').mean()),
            'apr_realized_nofr': float(tnum('PoolAPRRealizedNoFR').mean()),
            'priv_share_nofr':  float(tnum('PoolTVLPrivateNoFR').mean()
                                      / tnum('PoolTVLNoFR').mean())
                                if tnum('PoolTVLNoFR').mean() > 0 else 0.0,
            'disp_mean_nofr':   float(tnum('SpreadEndNoFR').mean()),
            'disp_max_nofr':    float(num('SpreadEndNoFR').max()),
            'lp_min_nofr':      float(num('LPHealthNoFR').min()),
        }
        if verbose:
            print("=" * 66)
            print("POL/APR РАВНОВЕСИЕ (реальная симуляция, хвост "
                  f"{tail_days} дней)")
            print("=" * 66)
            print(f"  POL: ${out['pol_usd']:,.0f}   APR_req: {out['apr_req']:.0%}"
                  f"   fee: {out['fee_tier']:.1%}")
            print(f"  Равновесный TVL:      ${out['tvl_eq']:,.0f}  "
                  f"(частный {out['priv_share']:.1%})")
            print(f"  Реализованный APR:    {out['apr_realized']:.1%}")
            print(f"  Стресс-дней:          {out['stress_days_pct']:.1f}%")
            print(f"  Троттлинг оттока:     {out['throttle_pct']:.1f}%")
            print(f"  Очередь (доля/ожид.): {out['queue_share']:.1%} / "
                  f"{out['wait_days_end']:.0f} дн")
            print("  ── Мир БЕЗ Firerate (paired counterfactual):")
            print(f"  Равновесный TVL:      ${out['tvl_eq_nofr']:,.0f}  "
                  f"(частный {out['priv_share_nofr']:.1%}, "
                  f"APR {out['apr_realized_nofr']:.1%})")
            print(f"  Displacement ср/макс: {out['disp_mean_nofr']:.1%} / "
                  f"{out['disp_max_nofr']:.1%}   LP min: {out['lp_min_nofr']:.2f}")
            if out['disp_max_nofr'] > 0.20 or out['lp_min_nofr'] < 0.3:
                print("  ⚠ Без Firerate пул нежизнеспособен: displacement/LP в")
                print("    коллапсе; частный TVL и APR NoFR-мира не читать как")
                print("    равновесие — объём там создан смертельной спиралью.")
            if out['tvl_priv_eq'] >= 0.98 * self.dex_pool_params.lp_capital_cap_usd:
                print("  ⚠ Упор в потолок адресуемого LP-капитала — равновесие")
                print("    не ограничено моделью (apr_req <= APR_sat): см. planner")
            print()
        return out

    def run(self, verbose: bool = True) -> 'CredisSimulation':
        RED_DAY_REDUCTION_COEF = 8

        rng                       = self.rng
        N                         = self.initial_capacity
        G                         = self.N_GROUPS
        coen_price_arr            = self.coen_price_arr
        s_curve_arr               = self.s_curve_arr
        final_consumer_base_count = self.final_consumer_base_count
        A                         = self.assumptions
        trian_a, trian_b, trian_c = self.trian_a, self.trian_b, self.trian_c

        nods_issued          = self.nods_issued
        nods_qualified       = self.nods_qualified
        nods_issued_detailed = self.nods_issued_detailed
        intex_issued         = self.intex_issued
        intex_called_history = self.intex_called_history
        gratis_balance       = self.gratis_balance
        gratis_balance_day   = self.gratis_balance_day
        promis_proceeds      = self.promis_proceeds
        joined_customer_base = self.joined_customer_base

        for day in range(1, self.simulation_length + 1):
            t0 = time.perf_counter()

            coen_price    = coen_price_arr[day]
            s_curve_price = s_curve_arr[day]

            block_numbers = np.arange(
                self.block_number + 1,
                self.block_number + params['blocks_per_day'] + 1,
            )
            block_rewards = reward_exponential_vectorized(
                block_numbers, params['k_soft'], params['minimal_coens_per_block']
            )
            total_allocation_limit = float(np.sum(block_rewards))
            day_metadosis_limit    = total_allocation_limit * (1 - self.allocation_limit_others_share)
            self.block_number     += params['blocks_per_day']

            promis_proceeds_today    = np.zeros(G)
            day_promis_allocation    = 0.0
            stat_user_gain           = -1.0
            day_intex_settled_inflow = 0.0
            day_intex_paid_inflow    = 0.0
            total_intex_demand       = 0.0

            trend_short = coen_price_arr[day]  / coen_price_arr[max(0, day - 7)]
            trend_long  = coen_price_arr[day]  / coen_price_arr[max(0, day - 30)]
            trend = float(np.clip((trend_short - trend_long) / trend_long, -1, 1))

            # --- 1. Add new users ---
            tam_f = self._tam_curve(N, day, self.simulation_length, self.tam_k)
            n_active_consumers = int(np.sum(joined_customer_base[:tam_f]))
            not_joined_mask    = ~joined_customer_base[:tam_f]
            idx_non_active     = np.nonzero(not_joined_mask)[0]

            u = rng.random(idx_non_active.size)
            day_enter_probability = self._consumer_enter_prob(
                self.sim_enter_base[:tam_f][not_joined_mask],
                self.sim_enter_adoption[:tam_f][not_joined_mask],
                self.sim_enter_deficit[:tam_f][not_joined_mask],
                self.sim_enter_trend[:tam_f][not_joined_mask],
                n_active_consumers, tam_f,
                self.factual_deficit, self.deificit_start, trend,
            )
            joined_customer_base[idx_non_active[u < day_enter_probability]] = True

            # --- 2. Active users ---
            active_idx = np.nonzero(joined_customer_base)[0]

            # --- 3. Day type + spending ---
            day_flag  = int(coen_price > coen_price_arr[day - 1])
            x_divisor = 1 if day_flag else RED_DAY_REDUCTION_COEF

            u = rng.random(active_idx.size)
            spend_idx     = active_idx[u < rng.uniform(0.7, 0.9)]
            spend_amounts = rng.triangular(trian_a, trian_b, trian_c, spend_idx.size).astype(np.float32)
            spend_amounts = np.multiply(spend_amounts, final_consumer_base_count[spend_idx])
            spend_amounts = calculate_symbolic_load(spend_amounts, s_curve_price, A.symbolic_rate)
            fi_groups     = rng.integers(0, G, size=spend_idx.size)

            # --- 4. Lysis ---
            trib              = np.bincount(fi_groups, weights=spend_amounts, minlength=G)
            total_trib_amount = float(trib.sum())
            y   = trib / total_trib_amount
            wts = final_consumer_base_count[spend_idx].astype(float)
            p   = np.bincount(fi_groups, weights=wts, minlength=G) / wts.sum()

            x     = min(self.deificit_start / x_divisor,
                        (day_metadosis_limit / x_divisor) / total_trib_amount)
            x_max = 2 * x
            fractions           = lysis_fractions(y, p, self.L, spend_idx.size, f=x, fmax=x_max)
            self.factual_deficit = x

            gratis_load      = trib * fractions
            floor_price_rate = 1.08 * coen_price

            nods_issued_today = [
                (g, gl, floor_price_rate, day)
                for g, gl in enumerate(gratis_load)
                if gl > 0
            ]
            nods_issued_detailed.append([np.where(fi_groups == fi)[0] for fi in range(G)])

            if nods_issued_today:
                nods_issued = np.concatenate(
                    (nods_issued, np.array(nods_issued_today, dtype=self.dt_nods))
                )

            # --- 4.1 Promis ---
            gratis_total     = gratis_load.sum()
            day_promis_limit = day_metadosis_limit - gratis_total

            if day_flag:
                total_intex_demand    = rng.random() * self.total_intex_demand_fiat / coen_price
                day_promis_demand     = min(total_trib_amount, total_intex_demand)
                day_promis_allocation = min(day_promis_demand,
                                            day_promis_limit + self.total_unallocated_promis)
                promis = day_promis_limit - min(day_promis_allocation, day_promis_limit)

                if gratis_total > 0:
                    promis_proceeds_today = (
                        gratis_load / gratis_total * self.clearing_price_share * day_promis_allocation
                    )
                promis_proceeds[:G] += promis_proceeds_today
                self.total_unallocated_promis += promis - promis_proceeds_today.sum()

                intex_strike_price  = coen_price * 1.16
                intex_call_treshold = intex_strike_price * 1.64
                intex_issued = np.concatenate((intex_issued, np.array(
                    [(day_promis_allocation, intex_call_treshold, intex_strike_price, day)],
                    dtype=self.dt_intex_issued,
                )))
            else:
                day_promis_allocation          = 0.0
                self.total_unallocated_promis += day_promis_limit

            # --- 5. Nod Qualification ---
            qualified_mask = (
                (nods_issued['TargetPrice'] <= coen_price) &
                (nods_issued['Day'] <= day - 21)
            )
            if qualified_mask.any():
                nods_q_today = np.array(nods_issued[qualified_mask], dtype=self.dt_nods_q)
                nods_q_today['StrikePrice'] = coen_price_arr[day - 1]
                nods_qualified = np.concatenate((nods_qualified, nods_q_today))
                nods_issued    = nods_issued[~qualified_mask]

            # --- 5.1 Intex Settlement ---
            settlement_mask = intex_issued["Day"] + 30 <= day
            if settlement_mask.any():
                prices_30    = coen_price_arr[day - 30:day]
                eligible     = intex_issued[settlement_mask]
                above_counts = (prices_30[:, None] >= eligible["CallThreshold"][None, :]).sum(axis=0)
                pay_mask     = above_counts >= 20

                if pay_mask.any():
                    to_pay                   = eligible[pay_mask]
                    day_intex_settled_inflow = float(to_pay["SymbolicLoad"].sum())
                    day_intex_paid_inflow    = float((to_pay["SymbolicLoad"] * to_pay["StrikePrice"]).sum())
                    intex_issued             = np.delete(intex_issued, np.nonzero(settlement_mask)[0][pay_mask])
                    intex_called_history     = np.concatenate((
                        intex_called_history,
                        np.array([(r[0], r[1], r[2], r[3], day) for r in to_pay], dtype=self.dt_intex_called),
                    ))

            # --- 6. Mine gratis ---
            today_gratis_mined      = 0.0
            strike_price_to_reserve = 0.0
            can_mine        = 0
            reasobable_mine = 0
            decided_mine    = 0
            stat_avg_wait   = 0.0
            day_ready_to_sell_flag = 0

            ready_mask = nods_qualified["StrikePrice"] < coen_price
            if ready_mask.any():
                nq = nods_qualified[ready_mask]

                nod_id_lists = [nods_issued_detailed[d][g] for d, g in zip(nq["Day"], nq["Group"])]
                counts       = np.array([len(x) for x in nod_id_lists], dtype=np.float64)

                valid_mask = counts > 0
                if valid_mask.any():
                    nq           = nq[valid_mask]
                    counts       = counts[valid_mask]
                    nod_id_lists = [nod_id_lists[i] for i in np.nonzero(valid_mask)[0]]

                    gain_per_nod = nq["SymbolicLoad"] * (coen_price - nq["StrikePrice"])
                    nod_sizes    = np.array([len(x) for x in nod_id_lists], dtype=np.int64)
                    all_user_ids = np.concatenate(nod_id_lists).astype(np.intp)

                    user_gain    = np.bincount(all_user_ids,
                                               weights=np.repeat(gain_per_nod / counts, nod_sizes),
                                               minlength=N)
                    user_load    = np.bincount(all_user_ids,
                                               weights=np.repeat(nq["SymbolicLoad"] / counts, nod_sizes),
                                               minlength=N)
                    user_reserve = np.bincount(all_user_ids,
                                               weights=np.repeat(nq["SymbolicLoad"] * nq["StrikePrice"] / counts, nod_sizes),
                                               minlength=N)
                    user_day_sum = np.bincount(all_user_ids,
                                               weights=np.repeat(nq["Day"].astype(np.float64), nod_sizes),
                                               minlength=N)
                    user_count   = np.bincount(all_user_ids, minlength=N).astype(np.int32)

                    eligible_users = user_gain >= self.target_gratis_balance_usd
                    if eligible_users.any():
                        day_ready_to_sell_flag = 1

                        avg_wait = (day - user_day_sum[eligible_users] / user_count[eligible_users]) / 90
                        x_mine = (self.a0
                                  + self.a2 * (user_gain[eligible_users] / self.target_gratis_balance_usd)
                                  - self.a3 * avg_wait
                                  - self.a4 * trend)
                        prob        = self._sigmoid(x_mine)
                        mine_mask   = rng.random(prob.size) < prob
                        mined_users = np.nonzero(eligible_users)[0][mine_mask]

                        # mineGratis: Gratis зачисляется на баланс пользователя.
                        # Конвертация в coen (mineCoen) происходит отдельно — ниже,
                        # в шаге конвертации. gratis_balance накапливается до тех пор
                        # пока пользователь не решит конвертировать.
                        gratis_balance[mined_users] += user_load[mined_users]
                        gratis_balance_day[mined_users] = day   # день последнего начисления
                        # контрфакт: тот же приток Gratis (mineGratis не зависит от Firerate)
                        self.gratis_balance_nofr[mined_users] += user_load[mined_users]
                        self.gratis_balance_day_nofr[mined_users] = day
                        today_gratis_mined      = float(user_load[mined_users].sum())
                        strike_price_to_reserve = float(user_reserve[mined_users].sum())

                        mined_flag = np.zeros(N, dtype=bool)
                        mined_flag[mined_users] = True

                        remove_per_nod = np.array([
                            mined_flag[ids_i].sum() * (sl / c)
                            for ids_i, sl, c in zip(nod_id_lists, nq["SymbolicLoad"], counts)
                        ])
                        nods_qualified["SymbolicLoad"][np.nonzero(ready_mask)[0][valid_mask]] -= remove_per_nod

                        can_mine        = int((user_count > 0).sum())
                        reasobable_mine = prob.size
                        decided_mine    = mined_users.size
                        self.day_sell_prob = float(prob.mean())
                        stat_avg_wait   = float(avg_wait.mean() * 90)
                        stat_user_gain  = float(user_gain[eligible_users].mean())

                        pairs = np.unique(np.stack([nq["Day"], nq["Group"]], axis=1), axis=0)
                        for d, g in pairs:
                            arr = nods_issued_detailed[d][g]
                            if arr.size:
                                nods_issued_detailed[d][g] = arr[~np.isin(arr, mined_users)]

            nods_issued    = nods_issued[nods_issued['SymbolicLoad'] > 1e-4]
            nods_qualified = nods_qualified[nods_qualified['SymbolicLoad'] > 1e-4]

            if day % 100 == 0:
                active_days = set(np.unique(
                    np.concatenate((nods_qualified['Day'], nods_issued['Day']))
                ))
                for d in range(day):
                    if d not in active_days:
                        nods_issued_detailed[d] = []

            # ── mineCoen: конвертация накопленного Gratis в coen
            # ЧАСОВОЙ САБ-ЦИКЛ (24 batch-окна/день). Firerate применяется
            # на выходе (конвертация), в единицах спеки: v_cap = coen/час,
            # spread_decay = mean reversion за час. Параллельно на тех же
            # случайных числах эволюционирует контрфактический мир без
            # Firerate (rate=1, cap=inf) — собственные балансы и spread.
            # См. firerate_hourly.run_mine_coen_day.

            n_active_today   = int(final_consumer_base_count.sum())

            # ── DEX-путь: AMM-пул, эндогенные cap'ы, LP-динамика,
            #    paired counterfactual на общих случайных числах.
            tvl_today = max(
                # 10_000.0,
                self._last_TVL,
                # n_active_today * self.target_gratis_balance_usd
                # * coen_price * self.firerate_tvl_mcap_fraction,
                self.gratis_balance.sum() * coen_price * self.firerate_tvl_mcap_fraction
            )

            self._last_TVL = tvl_today

            day_conv = run_mine_coen_day_dex(
                day,
                gratis_balance, gratis_balance_day,
                self.gratis_balance_nofr, self.gratis_balance_day_nofr,
                self.pool_fr, self.pool_nofr, self.clf_fr, self.clf_nofr,
                trend=trend, coen_price=coen_price, tvl_usd=tvl_today,
                proto=self.dex_proto, rng=rng,
                hours=self.firerate_hours_per_day,
                p_convert_base=self.p_convert_base,
                p_convert_price_sens=self.p_convert_price_sens,
                p_convert_hold_sens=self.p_convert_hold_sens,
                p_convert_stress_sens=self.p_convert_stress_sens,
                p_cancel_on_penalty=self.p_cancel_on_penalty,
                seeking=self.conversion_seeking,
            )
            # SpreadEnd-колонки в DEX-режиме несут displacement пула
            self.spread      = day_conv.displacement_end
            self.spread_nofr = day_conv.displacement_end_nofr
            depth_end   = day_conv.depth_end
            dhat_end    = day_conv.d_hat_end
            lp_end      = day_conv.lp_health_end
            lp_end_nofr = day_conv.lp_health_end_nofr

            gratis_requested_today = day_conv.gratis_requested
            gratis_converted_today = day_conv.gratis_converted
            coen_from_conversion   = day_conv.coen_from_conversion
            # уровень стресса дня = максимальный часовой уровень;
            # часы по уровням логируются отдельными колонками
            firerate_stress_level  = day_conv.stress_level_max
            # реализованная ставка за день: coen / сожжённый Gratis
            firerate_base_rate     = day_conv.effective_rate_realized

            # ── Firerate: threshold map (раз в N дней, остальные — carry-forward)
            # DEX-режим: панический порог (min p_convert_stress_sens для
            # устойчивого severe) на probe-пулах с TVL = f × active_gratis —
            # семантика фракций сохранена, frac*-анализ работает поверх.
            # firerate_thresholds = panic_threshold_map_for_day(
            #     gratis_balance=gratis_balance,
            #     joined_mask=joined_customer_base,
            #     pool=self.pool_fr,
            #     proto=self.dex_proto,
            #     b_fractions=self.firerate_b_fractions,
            #     sens_grid=self.firerate_panic_grid,
            #     probe_hours=self.firerate_probe_hours,
            #     seed=day,
            #     day=day,
            #     compute_every_n_days=self.firerate_compute_every_n_days,
            #     max_probe_users=self.firerate_probe_max_users,
            # )
            # if firerate_thresholds is not None:
            #     self._last_firerate_thresholds = firerate_thresholds
            # else:
            #     firerate_thresholds = self._last_firerate_thresholds

            # ── Требование к ликвидности (раз в N дней, carry-forward).
            # Дешёвая обёртка: warm-start от вчерашнего ответа, урезанная
            # сетка паники, сабсэмпл — см. liquidity_requirement_for_day.
            # prev_tvl_req = (self._last_liquidity_requirement or {}).get('tvl_required_no_fr')
            # liqreq = liquidity_requirement_for_day(
            #     gratis_balance=gratis_balance,
            #     joined_mask=joined_customer_base,
            #     proto=self.dex_proto,
            #     pool_params=self.dex_pool_params,
            #     age_days=day,
            #     day=day,
            #     vol_ewma_daily_usd=day_conv.tvl_info.get('vol_day', 0.0),
            #     prev_tvl_required=prev_tvl_req,
            #     compute_every_n_days=self.firerate_liqreq_every_n_days,
            #     warm_start_factor=self.firerate_liqreq_warm_start_factor,
            #     sens_grid=self.firerate_liqreq_sens_grid,
            #     probe_hours=self.firerate_liqreq_probe_hours,
            #     tol=self.firerate_liqreq_tol,
            #     max_probe_users=self.firerate_liqreq_max_probe_users,
            #     seed=day,
            # )
            # if liqreq is not None:
            #     self._last_liquidity_requirement = liqreq
            # liqreq = self._last_liquidity_requirement or {}

            self.results.loc[day] = (
                day, day_flag, day_ready_to_sell_flag,
                coen_price,
                joined_customer_base.sum(),
                np.sum(final_consumer_base_count[joined_customer_base]),
                total_allocation_limit,
                can_mine, reasobable_mine, decided_mine,
                nods_issued['SymbolicLoad'].sum(),
                nods_qualified['SymbolicLoad'].sum(),
                today_gratis_mined, strike_price_to_reserve,
                total_intex_demand, day_promis_allocation,
                promis_proceeds_today.sum(), self.total_unallocated_promis,
                intex_issued['SymbolicLoad'].sum(),
                day_intex_settled_inflow, day_intex_paid_inflow,
                total_allocation_limit * self.allocation_limit_others_share,
                0,
                self.day_sell_prob, day_enter_probability.mean(),
                trend, n_active_consumers, tam_f, self.factual_deficit,
                stat_avg_wait, stat_user_gain,
                # ── Firerate columns
                gratis_requested_today,
                gratis_converted_today,
                coen_from_conversion,
                # firerate_thresholds.get('total_active_gratis', float(gratis_balance.sum())),
                self.gratis_balance.sum(),
                firerate_stress_level,
                firerate_base_rate,
                day_conv.stress_hours.get(1, 0),
                day_conv.stress_hours.get(2, 0),
                self.spread,
                # ── контрфактический мир
                self.spread_nofr,
                day_conv.gratis_converted_nofr,
                day_conv.coen_from_conversion_nofr,
                # ── DEX-колонки (NaN в CEX-режиме)
                depth_end, dhat_end, lp_end, lp_end_nofr,
                getattr(day_conv, 'queue_gratis_end', np.nan),
                getattr(day_conv, 'tvl_info', {}).get('tvl_total', np.nan),
                getattr(day_conv, 'tvl_info', {}).get('tvl_priv', np.nan),
                getattr(day_conv, 'tvl_info', {}).get('apr_realized', np.nan),
                getattr(day_conv, 'tvl_info', {}).get('vol_day', np.nan),
                getattr(day_conv, 'tvl_info_nofr', {}).get('tvl_total', np.nan),
                getattr(day_conv, 'tvl_info_nofr', {}).get('tvl_priv', np.nan),
                getattr(day_conv, 'tvl_info_nofr', {}).get('apr_realized', np.nan),
                # liqreq.get('tvl_required_no_fr', np.nan),
                # liqreq.get('pol_min_usd', np.nan),
                # liqreq.get('pol_min_frac', np.nan),
                # liqreq.get('gap_usd', np.nan),
                # liqreq.get('implied_apr_for_gap', np.nan),
                # *[firerate_thresholds[f] for f in self.firerate_b_fractions],
                tvl_today
            )

            if verbose:
                print(f'Day {day:4d} | {time.perf_counter() - t0:.3f}s')

        # write-back
        self.nods_issued          = nods_issued
        self.nods_qualified       = nods_qualified
        self.nods_issued_detailed = nods_issued_detailed
        self.intex_issued         = intex_issued
        self.intex_called_history = intex_called_history
        self.joined_customer_base = joined_customer_base
        self.gratis_balance       = gratis_balance
        self.gratis_balance_day   = gratis_balance_day
        self.promis_proceeds      = promis_proceeds

        return self