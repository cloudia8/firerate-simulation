import pandas as pd
import numpy as np
# import numexpr as ne
from random import randrange, shuffle
import math
import random
from collections import Counter, defaultdict
import datetime
from dateutil.relativedelta import relativedelta
# import polars as pl
from lysis_binary_4dec2025 import *


def divide_by_two_integer(x):
    integer_part, remainder = divmod(x, 2)
    return integer_part+remainder, integer_part

def create_equal_groups_distribution(dist: list, n: int):
    temp_dist = [math.floor(n*d) for d in dist]
    diff = n - sum(temp_dist)
    group = randrange(len(dist))
    temp_dist[group] += diff
    return temp_dist

def create_equal_groups_count(cohort_granularity: int, cohort: int):
    temp_list = [cohort // cohort_granularity + (1 if x < cohort % cohort_granularity else 0) 
                 for x in range (cohort_granularity)]
    temp_list = [x for x in temp_list if x!=0]
    return temp_list

def create_cash_in_schedule(agent_type, current_day, assumptions):
    # 1. Frequency
    freq = round(365 / assumptions.cash_in_frequency[agent_type] * assumptions.years)
    # 2. Create schedule
    prev = [0 for _ in range(0, current_day)]
    to_be = [0 for _ in range(0, assumptions.model_length - freq - current_day+1)] + [1 for _ in range(0, freq)]
    shuffle(to_be)
    schedule = prev + to_be
    return schedule

def generate_consumption_records_amount(agent_type, assumptions):
    """
    used
    """
    a,b,c = assumptions.consumption[agent_type]
    x = np.random.triangular(a,b,c)
    return x

def calculate_consumption_record(amount, gem_price):
    """
    not used
    """
    # The gems amount in consumption record
    return amount/(gem_price)

def calculate_symbolic_load(amount, gem_price, symbolic_rate):
    """
    used
    """
    # The gems amount in consumption record
    return (amount*symbolic_rate)/(gem_price*(1+symbolic_rate))

def create_consumption_record(day, cohort, amount, gem_price, symbolic_rate):
    """
    used
    """
    # target_price = gem_price * (1+pgt)
    gems_to_be_given = calculate_symbolic_load(amount, gem_price, symbolic_rate)
    return (day, cohort, gems_to_be_given)

def exp_consumer_base_change(day, assumptions, 
                             N_start=2500, N_max=2*10**6, k = 0.004,
                             phase_0_years = 1,
                             phase_1_years = 3):
    """
    used
    """

    start_date = assumptions.start_date
    phase_0_end_date = datetime.date(  
        start_date.year + phase_0_years,
        start_date.month,
        start_date.day)
    
    phase_1_end_date = datetime.date(  
        phase_0_end_date.year + phase_1_years,
        phase_0_end_date.month,
        phase_0_end_date.day)

    phase_0_length = (phase_0_end_date - start_date).days # in days
    phase_1_length = (phase_1_end_date - phase_0_end_date).days # in days
    
    if day <= phase_0_length:
        N = random.choices([1, 5, 10], 
                           weights=[0.1, 0.7, 0.2])[0]

    elif day < (phase_1_length+phase_0_length):
        # Time or steps (x-axis)
        x = np.linspace(0, phase_1_length, phase_1_length)  # Adjust the range and steps as needed

        # Exponential growth function
        N = N_max * (1 - np.exp(-k * x))
        # Shift the curve to start at N_start
        N = N_start + (N - N_start * (1 - np.exp(-k * x[0])))

        N = round(N[day-phase_0_length] - N[day-phase_0_length-1])

    else:
        N = random.choices([0, 10**2, 10**3], 
                           weights=[0.1, 0.7, 0.2])[0]
        
    return N

def customer_base_schedule_for_cra_reward_discussion(customer_base_schedule, day):
    try:
        N = customer_base_schedule[day]
    except:
        N = random.choices([0, 10**2, 10**3], 
                           weights=[0.1, 0.7, 0.2])[0]
    return N

def raffle_algorithm(day:int, 
                     gem_price: float, 
                     submitted_cu: list, 
                     minting_pool: float,
                     cohort_counts:dict,
                     pools_ids: list,
                     pools_deficit_distribution: list,
                     pools_pgt: list):
    """
    used
    """

    # 1. Shuffle CU
    shuffle(submitted_cu)

    # 2. Select winners
    lysis_limit = minting_pool / len(pools_ids)
    min_value = min([x[2] for x in submitted_cu])

    winners = []
    lysis_winners_ids = []

    for pool in range(1,24):
        temp_lysis_limit = lysis_limit

        for x in submitted_cu:
            if x[2] <= temp_lysis_limit:
                winners.append(x)
                temp_lysis_limit -= x[2]
                submitted_cu.remove(x)
                lysis_winners_ids.append(pool)

            elif temp_lysis_limit <= min_value:
                break

            elif x[2]/cohort_counts[x[1]] <= temp_lysis_limit:
                temp_per_consumer_symbolic_load = x[2]/cohort_counts[x[1]]
                temp_consumers_to_win = temp_lysis_limit // temp_per_consumer_symbolic_load

                temp_lysis_limit -= temp_consumers_to_win*temp_per_consumer_symbolic_load

                winners.append((x[0], x[1], temp_consumers_to_win*temp_per_consumer_symbolic_load))

                submitted_cu.append((x[0], x[1], x[2]-temp_consumers_to_win*temp_per_consumer_symbolic_load))
                submitted_cu.remove(x)

                lysis_winners_ids.append(pool)


    # 3. Calculate the distribution of lysis
    lysis_winners_pgt = []

    for i in lysis_winners_ids:
        lysis_winners_pgt.append(pools_pgt[i-1])

    # Add (1) day of raffle and (2) target price to each selected CU 
    selected_cr = [(a[0],a[1],a[2],b,day,gem_price*(1+c)) for a,b,c in zip(winners, lysis_winners_ids, lysis_winners_pgt)]
        
    # return losers, selected_cr
    return selected_cr, submitted_cu

def minting_winners_statistics(winners_cr: list, day: int):
    """
    not used

    Update: (1) cohort_gems, (2) raffle_winners_count, (3) raffle_winners_gems, (4) pool_cycle_length
    """
    if winners_cr:
        temp_df = pd.DataFrame(winners_cr, columns=['null_date', 'cohort', 'gems', 'pool', 'start_date','target_price'])
        temp_df['current_day'] = day
        temp_df['wait_time'] = temp_df['current_day'] - temp_df['start_date']

        # cohort_gems
        temp_cohort_gems = temp_df.groupby('cohort')['gems'].sum().to_dict()

        # raffle_winners_count, raffle_winners_gems, pool_cycle_length
        temp_df = temp_df.groupby('pool').agg({'gems':'sum', 'cohort':'count', 'wait_time':'sum'}).reset_index()
        temp_pools = temp_df['pool'].to_list()
        temp_raffle_winners_count = temp_df['cohort'].to_list()
        temp_raffle_winners_gems = temp_df['gems'].to_list()
        temp_pool_cycle_length = temp_df['wait_time'].to_list()

    else:
        temp_cohort_gems,temp_pools,temp_raffle_winners_count,temp_raffle_winners_gems,temp_pool_cycle_length = [],[],[],[],[]

    return temp_cohort_gems, temp_pools, temp_raffle_winners_count, temp_raffle_winners_gems, temp_pool_cycle_length

def exp_minting_pool(day, assumptions, N_start=0, N_max=10**12, k = 0.001,
                     phase_1_years = 8):
    """
    not used
    """
    start_date = assumptions.start_date
    end_date = datetime.date(  
        start_date.year + phase_1_years,
        start_date.month,
        start_date.day)

    model_length = (end_date - start_date).days # in days

    if day < model_length:
        # Time or steps (x-axis)
        x = np.linspace(0, model_length, model_length)  # Adjust the range and steps as needed

        # Exponential growth function
        N = N_max * (1 - np.exp(-k * x))
        # Shift the curve to start at N_start
        N = N_start + (N - N_start * (1 - np.exp(-k * x[0])))
        N = float(N[day] - N[day-1])
        
    else:
        N = 10**3
        
    return N

def _raffle_winners_statistics(winners_cr: list, day: int):
    """
    not used

    Update: (1) cohort_gems, (2) raffle_winners_count, (3) raffle_winners_gems, (4) pool_cycle_length
    """
    if winners_cr:
        temp_df = pd.DataFrame(winners_cr, columns=['start_date', 'cohort', 'target_price', 'gems', 'pool'])
        temp_df['current_day'] = day
        temp_df['wait_time'] = temp_df['current_day'] - temp_df['start_date']

        # cohort_gems
        temp_cohort_gems = temp_df.groupby('cohort')['gems'].sum().to_dict()

        # raffle_winners_count, raffle_winners_gems, pool_cycle_length
        temp_df = temp_df.groupby('pool').agg({'gems':'sum', 'cohort':'count', 'wait_time':'sum'}).reset_index()
        temp_pools = temp_df['pool'].to_list()
        temp_raffle_winners_count = temp_df['cohort'].to_list()
        temp_raffle_winners_gems = temp_df['gems'].to_list()
        temp_pool_cycle_length = temp_df['wait_time'].to_list()

    else:
        temp_cohort_gems,temp_pools,temp_raffle_winners_count,temp_raffle_winners_gems,temp_pool_cycle_length = [],[],[],[],[]

    return temp_cohort_gems, temp_pools, temp_raffle_winners_count, temp_raffle_winners_gems, temp_pool_cycle_length

def _raffle_algorithm(day:int, gem_price: float, submitted_cr: list, minting_pool: float, pools_base_prob:list):
    """
    not used
    """
    # 1. Split submitted CR to out- and in- raffle lists
    out_of_raffle_cr = [x for x in submitted_cr if x[2]>=gem_price]
    in_raffle_cr = [x for x in submitted_cr if x[2]<=gem_price]

    # CHECK 1: in_of_raffle_cr is not empty
    if in_raffle_cr:

        # CHECK 2: Total gem value of in-raffle CR <> Minting pool size
        check_1_total_gems_cr = sum([x[3] for x in in_raffle_cr])

        if check_1_total_gems_cr <= minting_pool:
            losers = []
            winners = in_raffle_cr

        else:
            winners = []
            losers = []

            # 2. Allocate minting pool between Pools 
            pools_nominated = [x[4] for x in in_raffle_cr]
            pools_nominated_no_duplicates = list(set(pools_nominated))
            pools_nominated_base_prob = [pools_base_prob[i-1] for i in pools_nominated_no_duplicates]
            pools_nominated_prob = [x/sum(pools_nominated_base_prob) for x in pools_nominated_base_prob]
            pools_nominated_minting_pool = [minting_pool*x for x in pools_nominated_prob]
            
            for p,m in zip(pools_nominated_no_duplicates, pools_nominated_minting_pool):
                pool_in_raffle_cr = [x for x in in_raffle_cr if x[4]==p and x[3]<=m]
                pool_out_raffle_cr = [x for x in in_raffle_cr if x[4]==p and x[3]>m]

                # CHECK 3: pool_in_raffle_cr is empty
                if pool_in_raffle_cr:
                    min_value = min([x[3] for x in pool_in_raffle_cr])
                    minting_pool_left = m
                    random.shuffle(pool_in_raffle_cr)

                    while pool_in_raffle_cr and minting_pool_left >= min_value:
                        consumption_record_e = pool_in_raffle_cr.pop(0)
                        if consumption_record_e[3] <= minting_pool_left:
                            minting_pool_left -= consumption_record_e[3]
                            winners.append(consumption_record_e)
                        else:
                            losers.append(consumption_record_e)
                
                losers.extend(pool_out_raffle_cr)
                    
    else:
        losers = []
        winners = []
        
    # Combine output
    updated_submitted_cr = losers + out_of_raffle_cr
    cohort_gems, pools, raffle_winners_count, raffle_winners_gems, pool_cycle_length = raffle_winners_statistics(winners, day)
        
    return updated_submitted_cr, cohort_gems, pools, raffle_winners_count, raffle_winners_gems, pool_cycle_length

def get_cumulative_tokens(block_number, initial_rate, decay, years=8, block_time_seconds=5, inflation_rate=0.02):
    """
    Рассчитывает общее накопленное количество токенов до указанного блока.
    
    Формула: 
    - В первые 8 лет: total_tokens = initial_rate * (1 - exp(-decay * block_number)) / decay
    - После 8 лет: добавляем токены с учетом годовой инфляции, используя формулу сложных процентов
    
    Параметры:
    -----------
    block_number : int
        Номер блока
    initial_rate : float
        Начальное количество токенов за блок
    decay : float
        Коэффициент экспоненциального снижения
    years : int, optional
        Количество лет в основном периоде эмиссии (по умолчанию: 8)
    block_time_seconds : float, optional
        Время между блоками в секундах (по умолчанию: 5)
    inflation_rate : float, optional
        Годовая инфляция после основного периода эмиссии (по умолчанию: 0.02 или 2%)
    
    Возвращает:
    -----------
    float: Общее количество токенов, выпущенных до данного блока
    """
    seconds_per_year = 365.25 * 24 * 60 * 60
    blocks_per_year = seconds_per_year / block_time_seconds
    total_blocks_in_period = int(years * blocks_per_year)
    
    if block_number <= total_blocks_in_period:
        # В основной период (первые 8 лет) используем интеграл экспоненциальной функции
        return initial_rate * (1 - np.exp(-decay * block_number)) / decay
    else:
        # После основного периода используем сложные проценты с учетом дробной части года
        
        # Рассчитываем общее количество токенов на конец основного периода
        total_at_end_of_period = initial_rate * (1 - np.exp(-decay * total_blocks_in_period)) / decay
        
        # Рассчитываем, сколько лет прошло после основного периода
        years_after_period = (block_number - total_blocks_in_period) / blocks_per_year
        
        # Используем формулу сложных процентов: P * (1 + r)^t
        # Где P - начальная сумма, r - годовая ставка, t - время в годах
        return total_at_end_of_period * (1 + inflation_rate) ** years_after_period
    
def get_tokens_for_day(day, initial_rate, decay, years=8, block_time_seconds=5, inflation_rate=0.02):
    seconds_per_day = 24 * 60 * 60
    blocks_per_day = seconds_per_day / block_time_seconds

    block_number_start = (day-1) * blocks_per_day
    block_number_end = day * blocks_per_day

    tokens_for_today = (get_cumulative_tokens(block_number_end, initial_rate, decay, years=years, block_time_seconds=block_time_seconds, inflation_rate=inflation_rate)
                        -
                        get_cumulative_tokens(block_number_start, initial_rate, decay, years=years, block_time_seconds=block_time_seconds, inflation_rate=inflation_rate))
    
    return tokens_for_today

def price_delta_bad(values, day):
    try:
        x = values[day]
    except:
        try:
            x = values[day - 2695]
        except:
            x = random.choices([-0.0025, -0.001, -0.00001,   0.00001,  0.001,  0.003, 0.005], 
                     weights=[0.13,   0.3,      0.14,         0.06,   0.2,    0.1, 0.07])[0] 
    return x

def price_delta_bad_for_cra_discussion(values, day, values_norm):
    try:
        x = values[day]
    except:
        try:
            x = values_norm[day - 1825]
        except:
            x = random.choices([-0.0025, -0.001, -0.00001,   0.00001,  0.001,  0.003, 0.005], 
                     weights=[0.13,   0.3,      0.14,         0.06,   0.2,    0.1, 0.07])[0] 
    return x

def reward_exponential(block_number, total_supply, initial_tokens_per_block, blocks_per_year, k_soft, inflation_rate_per_year):
    block_reward = initial_tokens_per_block * np.exp(k_soft * block_number)
    if total_supply == 0:
        current_inflation = (block_reward * blocks_per_year) / initial_tokens_per_block
    else:
        current_inflation = (block_reward * blocks_per_year) / total_supply
        
    if current_inflation < 0.02:
        return (inflation_rate_per_year * total_supply) / blocks_per_year
    else:
        return  block_reward
    
def reward_exponential_vectorized(
    block_numbers: np.ndarray,
    k_soft: float,
    minimal_coens_per_block: float
) -> np.ndarray:
    """
    Vectorized version: Computes block rewards for an array of block_numbers.
    """
    # Compute exponential block rewards for all blocks
    block_rewards = k_soft * block_numbers + 2**13

    # If inflation is too low, override reward
    adjusted_reward = np.array([minimal_coens_per_block]*len(block_numbers))

    # Use np.where to apply the condition element-wise
    final_rewards = np.maximum(adjusted_reward, block_rewards)

    return final_rewards

def raffle_algorithm_nods_control(day:int, 
                     gem_price: float, 
                     submitted_cu: list, 
                     nods: list,
                     minting_pool: float,
                     cohort_counts:dict,
                     pools_ids: list,
                     pools_deficit_distribution: list,
                     pools_pgt: list):
    """
    used
    """

    # 0. Calculate
    # output_distr, output_pgts = lysis_control(gem_price, nods, minting_pool, day)
    output_distr, output_pgts, output_pgt_step_sizes = lysis_control_1(gem_price, nods, minting_pool)

    # 1. Shuffle CU
    shuffle(submitted_cu)

    # 2. Select winners
    lysis_limit = minting_pool / len(pools_ids)

    winners = []
    lysis_winners_ids = []

    for pool in range(1,len(pools_ids)+1):
        temp_lysis_limit = lysis_limit

        while min([x[2]/cohort_counts[x[1]] for x in submitted_cu]) <= temp_lysis_limit:
            
            for x in submitted_cu:
                if x[2] <= temp_lysis_limit:
                    winners.extend([(x[0], x[1], x[2]/cohort_counts[x[1]])] * cohort_counts[x[1]])
                    temp_lysis_limit -= x[2]
                    submitted_cu.remove(x)
                    lysis_winners_ids.extend([pool]*cohort_counts[x[1]])

                elif x[2]/cohort_counts[x[1]] <= temp_lysis_limit:
                    temp_per_consumer_symbolic_load = x[2]/cohort_counts[x[1]]
                    temp_consumers_to_win = int(temp_lysis_limit // temp_per_consumer_symbolic_load)

                    temp_lysis_limit -= temp_consumers_to_win*temp_per_consumer_symbolic_load
                    cohort_counts[x[1]] -= temp_consumers_to_win

                    winners.extend([(x[0], x[1], temp_per_consumer_symbolic_load)] * temp_consumers_to_win)

                    submitted_cu.append((x[0], x[1], x[2]-temp_consumers_to_win*temp_per_consumer_symbolic_load))
                    submitted_cu.remove(x)

                    lysis_winners_ids.extend([pool]*temp_consumers_to_win)

    # 3. Calculate the distribution of lysis
    winners_symbolic_load = [x[2] for x in winners]
    output_distr_symbolic_load = [x*minting_pool for x in output_distr]
    lysis_pgt_count = []

    i = 0
    direction = 1
    temp_target_pgt_symbolic_load = output_distr_symbolic_load.pop(0)

    for x in winners_symbolic_load:
        if x <= temp_target_pgt_symbolic_load:
            i += 1
            temp_target_pgt_symbolic_load -= x
        
        elif x>temp_target_pgt_symbolic_load:
            # Need to choose this pgt or next
            try: 
                temp_target_pgt_symbolic_load = output_distr_symbolic_load.pop(0)
                if direction == 1:
                    direction = 2
                    # winner stays at this pgt
                    i += 1
                    lysis_pgt_count.append(i)
                    i = 0
                else:
                    direction = 1
                    lysis_pgt_count.append(i)

                    i = 1
                    temp_target_pgt_symbolic_load -= x
            except:
                left_winners = len(winners) - sum(lysis_pgt_count)
                lysis_pgt_count.append(left_winners)

        else:
            print('Unexpected turn')

    if len(output_distr) > len(lysis_pgt_count):
        lysis_pgt_count.append(i)

    lysis_winners_pgt = []
    lysis_winners_pgt_step = []
    lysis_winners_id = []


    for a, b, c in zip(lysis_pgt_count, output_pgts, output_pgt_step_sizes):
        if a!=0:
            lysis_winners_pgt.extend(a*[b])
            lysis_winners_pgt_step.extend(a*[c/a])
            lysis_winners_id.extend(range(a))

    # Add (1) day of raffle and (2) target price to each selected CU 

    temp_df = pd.DataFrame(winners, columns=['tribute_day', 'cohort_id', 'symbolic_load'])
    temp_df['pool'] = lysis_winners_ids
    temp_df['nod_day'] = day
    temp_df['target_price'] = lysis_winners_pgt
    temp_df['pgt_step'] = lysis_winners_pgt_step
    temp_df['pgt_coef'] = lysis_winners_id

    rounding = 4
    if gem_price / 10**(-3) > 100:
        rounding = 3
    elif gem_price / 10**(-3) > 1000:
        rounding = 2
    elif gem_price / 10**(-3) > 10000:
        rounding = 1

    temp_df['target_price'] = (1+temp_df['target_price']+temp_df['pgt_step']*temp_df['pgt_coef']) * gem_price
    temp_df['target_price'] = round(temp_df['target_price'], rounding)
    temp_df['symbolic_load'] = round(temp_df['symbolic_load'], 4)
    temp_df = temp_df.groupby(['tribute_day', 'cohort_id', 'pool', 'nod_day','target_price'])['symbolic_load'].sum().reset_index()
    selected_cr = list(temp_df[['tribute_day', 'cohort_id', 'symbolic_load', 
                           'pool', 'nod_day', 'target_price']].itertuples(index=False, name=None))
    
    
    # return losers, selected_cr
    return selected_cr#, submitted_cu, output_distr, output_pgts

def lysis_vinyasa_sim(day:int,
                      current_coen_price: float,
                      deficit:float,
                      nods: list,
                      cohort_count):
    """
    used
    """

    n_tributes_to_generate = sum([cohort_count[x] for x in cohort_count.keys()])

    minting_pool, max_share_of_single_tribute, winners_symbolic_load = lysis_vinyasa(n_tributes_to_generate, 
                                                                                     current_coen_price, 
                                                                                     deficit)
    output_distr_symbolic_load, floor_price_lb, floor_price_distance = nods_distribution(current_coen_price, 
                                                                                         minting_pool, 
                                                                                         max_share_of_single_tribute, 
                                                                                         nods)
    selected_cr = nods_fomation(day, output_distr_symbolic_load, floor_price_lb, floor_price_distance, winners_symbolic_load)
    
    # return losers, selected_cr
    return selected_cr

def lysis_control(current_coen_price: float,
                  nods: list,
                  minting_pool: float,
                  day:int):
    
    # 0. Granularity parameter
    step_size = 0.04
    rounding = 4

    df_nods = pd.DataFrame(nods, columns =['TributeDay', 'CohortID', 'SymbolicLoad', 'Lysis', 'NodDay', 'TargetPrice'])
    upper_bound_nods = df_nods['TargetPrice'].max() / current_coen_price - 1

    # 1. Coen price range 
    lower_bound = 0.08
    upper_bound = 0.96 # min(0.96, upper_bound_nods) 
    # upper_bound = max(upper_bound, 0.96*2)  

    # 2. Bins
    n_bins = int((upper_bound-lower_bound) // step_size)
    bins = [round(current_coen_price * (1+lower_bound+x*step_size), rounding) for x in range(0,n_bins+2)]
    bins = list(set(bins))
    bins.sort()

    # 3. Nods analytics
    
    df_nods['TargetPrice'] = round(df_nods['TargetPrice'], rounding)
    df_nods['bins'] = pd.cut(df_nods['TargetPrice'], bins=bins, include_lowest=True)
    # df_nods.dropna(inplace=True)
    df_nods_summary = pd.DataFrame(df_nods.groupby('bins', observed=False)['SymbolicLoad'].sum()).reset_index()

    if df_nods_summary['SymbolicLoad'].sum() == 0:
        df_nods_summary['ToBeAdded'] = 1 / len(df_nods_summary)

    else:
        if df_nods_summary['SymbolicLoad'].median() == 0:
            symbolic_load_mean = df_nods_summary['SymbolicLoad'].mean()
        else:
            symbolic_load_mean = df_nods_summary['SymbolicLoad'].median()

        # cap = symbolic_load_mean / minting_pool

        df_nods_summary['ToBeAdded'] = symbolic_load_mean - df_nods_summary['SymbolicLoad']
        df_nods_summary.loc[df_nods_summary[df_nods_summary['ToBeAdded']<0].index,'ToBeAdded'] = 0
        df_nods_summary['ToBeAdded'] = df_nods_summary['ToBeAdded'] / minting_pool

        left_minting_pool = 1 - df_nods_summary['ToBeAdded'].sum()

        if left_minting_pool < 0:
            df_nods_summary['ToBeAdded'] = df_nods_summary['ToBeAdded'] / df_nods_summary['ToBeAdded'].sum()
        
        elif left_minting_pool > 0:
            df_nods_summary['ToBeAdded'] = df_nods_summary['ToBeAdded'] + left_minting_pool / len(df_nods_summary)


        # df_nods_summary['ToBeAdded'] = df_nods_summary['ToBeAdded'] / df_nods_summary['ToBeAdded'].sum()


        # df_nods_summary = df_nods_summary[df_nods_summary['ToBeAdded']>0]
    # df_nods_summary['pgt'] = df_nods_summary['bins'].apply(lambda x: x.right/current_coen_price-1)
    df_nods_summary['pgt'] = bins[:-1]#bins[1:]
    df_nods_summary['pgt'] = df_nods_summary['pgt'] /current_coen_price-1
    df_nods_summary = df_nods_summary[df_nods_summary['ToBeAdded']>0]

    # 5. Output

    output_distr = df_nods_summary['ToBeAdded'].to_list()
    output_pgts = df_nods_summary['pgt'].to_list()
    output_pgts = [round(x,rounding) for x in output_pgts]

    return output_distr, output_pgts

def lysis_control_1(current_coen_price: float,
                    nods: list,
                    minting_pool: float):

    # 0. Precision (rounding)
    # The function decides rounding precision based on the scale of the price:
    # Default: 4 decimal places
    # If price > 0.1 → round to 3 decimals
    # If price > 1 → round to 2 decimals
    # This prevents bins from being too granular when prices are large.

    rounding = 4
    if current_coen_price / 10**(-3) > 100:
        rounding = 3
    elif current_coen_price / 10**(-3) > 1000:
        rounding = 2
    elif current_coen_price / 10**(-3) > 10000:
        rounding = 1
    
    df_nods = pd.DataFrame(nods, columns =['TributeDay', 'CohortID', 'SymbolicLoad', 'Lysis', 'NodDay', 'TargetPrice'])
    df_nods['TargetPrice0'] = df_nods['TargetPrice']
    df_nods['TargetPrice'] = round(df_nods['TargetPrice'], rounding)

    # 1. Coen price range 
    lower_bound = 0.08
    upper_bound = 0.16

    # 2. Bins
    n_bins = math.ceil((current_coen_price * (1+upper_bound)-current_coen_price * (1+lower_bound)) / 10**(-rounding))
    bins = [round(current_coen_price * (1+lower_bound), rounding) + 10**(-rounding)*x for x in range(0,n_bins+1)]
    bins = [round(x,rounding) for x in bins]

    # 3. Nods analytics
    df_nods_summary = pd.DataFrame(df_nods.groupby('TargetPrice')['SymbolicLoad'].sum()).reset_index()

    # 3.1 Add missing bins
    bins_to_be_added = [x for x in bins if x not in df_nods_summary['TargetPrice'].to_list()]

    for new_bin in bins_to_be_added:
        df_nods_summary.loc[len(df_nods_summary), :] = (new_bin, 0)

    # 3.2 Remove out-of-bound bins
    df_nods_summary = df_nods_summary[(df_nods_summary['TargetPrice']>=bins[0])&(df_nods_summary['TargetPrice']<=bins[-1])].copy()

    # 4. Distribute minting pool
    if df_nods_summary['SymbolicLoad'].sum() == 0:
        df_nods_summary['ToBeAdded'] = 1 / len(df_nods_summary)

    else:
        if df_nods_summary['SymbolicLoad'].median() == 0:
            symbolic_load_mean = minting_pool/len(df_nods_summary) + df_nods_summary['SymbolicLoad'].mean()
        else:
            symbolic_load_mean = minting_pool/len(df_nods_summary) + df_nods_summary['SymbolicLoad'].median()

        df_nods_summary['ToBeAdded'] = symbolic_load_mean - df_nods_summary['SymbolicLoad']
        df_nods_summary.loc[df_nods_summary[df_nods_summary['ToBeAdded']<0].index,'ToBeAdded'] = 0
        df_nods_summary['ToBeAdded'] = df_nods_summary['ToBeAdded'] / minting_pool

        left_minting_pool = 1 - df_nods_summary['ToBeAdded'].sum()

        if left_minting_pool < 0:
            df_nods_summary['ToBeAdded'] = df_nods_summary['ToBeAdded'] / df_nods_summary['ToBeAdded'].sum()
        
        elif left_minting_pool > 0:
            df_nods_summary['ToBeAdded'] = df_nods_summary['ToBeAdded'] + left_minting_pool / len(df_nods_summary)

    # 5. Compute (1) pgt and (2) pgt step size
    df_nods_summary['min'] = df_nods_summary['TargetPrice'] - 5 * 10**(-rounding-1)
    df_nods_summary['max'] = df_nods_summary['TargetPrice'] + 4.9 * 10**(-rounding-1)

    df_nods_summary['pgt'] = df_nods_summary['min'] / current_coen_price-1
    df_nods_summary['pgt_step_size'] = df_nods_summary['max'] / current_coen_price-1 - df_nods_summary['pgt']

    # 6. Remove zero-value bins
    df_nods_summary = df_nods_summary[df_nods_summary['ToBeAdded']>0]

    output_distr = df_nods_summary['ToBeAdded'].to_list()
    output_pgts = df_nods_summary['pgt'].to_list()
    output_pgt_step_sizes = df_nods_summary['pgt_step_size'].to_list()

    return output_distr, output_pgts, output_pgt_step_sizes


# INTEX
def intex_demand_generation(coen_price, df_coen_price, today):

    # 1. Compute Intex Nominal
    intex_nominal = 10**6
    # - Check if coen price increased / decreased
    сhange_condition_met_days = df_coen_price[(df_coen_price['day']>today-30)&
                                                  (df_coen_price['day']<=today)].copy()
    сhange_condition_met_days['price_change'] = ((сhange_condition_met_days['CoenPriceBase'] // 10**(-3)) // 10) * 10 

    if len(сhange_condition_met_days[сhange_condition_met_days['price_change']>=10]) > 20:
        factor = сhange_condition_met_days[сhange_condition_met_days['price_change']>=10]['price_change'].min()
        intex_nominal = intex_nominal / factor  

    # 2. The number of IBAs
    ibas = random.choice(range(20))

    # 3.1. Bid sizes
    bids  = random.choices([5, 10, 50, 100, 1000], k=ibas)
    
    # 3.2 Intex demand
    intex_demand = sum(bids) * intex_nominal

    # 4. Limit price
    limit_prices  = random.choices([500, 1000, 5000, 10000, 15000], k=ibas)

    return intex_nominal, intex_demand, bids, limit_prices


# LYSIS
def p_distribution(L=12, kind='concentrated', concentration_size=0.9, concentration_groups=100, concentration_humps=[100, 2000, 3000]):
    if kind == 'uniform':
        return [1/2**L] * 2**L
    elif kind == 'concentrated':
        p_concentrated = concentration_size/concentration_groups
        m = 1 - concentration_size
        u = 2**L - concentration_groups
        p_final = [m / u] * u

        base = concentration_groups // len(concentration_humps)
        remainder = concentration_groups % len(concentration_humps)

        for h in concentration_humps:
            for i in range(base):
                p_final.insert(i+h, p_concentrated)
        if remainder>0:
            for r in range(remainder):
                p_final.insert(i+h+r, p_concentrated)
        return p_final
    else:
        print('Please set kind as either uniform or concentrated')