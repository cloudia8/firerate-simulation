from Assumptions import Assumptions
A = Assumptions()

initial_tokens_per_block = 2**14  # 65536 токенов за блок
seconds_per_block = 1
blocks_per_day = (24 * 60 * 60) // seconds_per_block
blocks_per_year = blocks_per_day * 365
years_before_inflation = 8
inflation_rate_per_year = 0.02

blocks_in_8_years = blocks_per_year * years_before_inflation
target_total_tokens = 2**40
total_years_to_simulate = 16
blocks_total = blocks_per_year * total_years_to_simulate
k_soft = -1/2**15
minimal_coens_per_block = 512

params = {    
    'blocks_per_day' : blocks_per_day,
    'blocks_per_year' : blocks_per_year,
    'initial_tokens_per_block' : initial_tokens_per_block,
    'inflation_rate_per_year' : inflation_rate_per_year,
    'k_soft':k_soft,
    'minimal_coens_per_block' : minimal_coens_per_block
    }


