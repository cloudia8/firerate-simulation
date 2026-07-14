import datetime
from dateutil.relativedelta import relativedelta

class Assumptions:
    def __init__(self, assumptions=None) -> None:

        defaults = {
            'years': 8
            }

        if assumptions:
            defaults.update(assumptions)

        self.years = defaults['years']
        self.start_date = datetime.date(2026, 2, 28)
        self.end_date = datetime.date(  
            self.start_date.year + self.years,
            self.start_date.month,
            self.start_date.day)

        self.model_length = (self.end_date - self.start_date).days # in days
        self.cutoff_dates = [
            datetime.date(self.start_date.year, 
                          self.start_date.month, 
                          self.start_date.day)+relativedelta(months=i*3) for i in range(0,self.years*4+1)]

        self.initial_token_price = 1 * 10**(-3)
        self.expected_daily_growth = 1.001

        self.pools_ids = [x for x in range(1,24)]
        self.pools_deficit_distribution = [x/sum(range(1,24)) for x in range(1,24)]
        self.pools_pgt = [0.08+0.04*x for x in range(0,23)]

        # The distribution of agent types in network
        self.agent_types = ['T'] 
        self.agent_types_dict = {'T': 'TestRun'} 
        self.agent_distribution = [1] 

        # Granularity per cohort
        self.cohort_granularity = 1

        # Consumption distribution - approximated using triangular distribution
        self.consumption = {'T': (90,178,410)}

        # Tribute -> Nod -> Gratis
        self.gratis_churn_ratio = 0.05

        # Sprout -> Intent -> Intentis
        self.gratis_to_sprout_ratio = 1

        # Validators
        self.validators = 100
        self.validators_apr = 0.04
        self.staked_coens = 2**23
        self.transaction_fee = 0.0075#0,005–0,01 random фигануть 

        # CRA
        self.CRA = 5
        self.CRA_reward_cap = 0.04
        self.responsiveness_factor = 0.1
        self.fiat_target_reward = 0.2

        self.symbolic_rate = 0.08

        self.initial_rate=18475.316174578344
        self.decay=5e-08

        # Dead wallets 
        self.dead_wallet_share = 0.20

        # Lysis
        self.L = 12
        self.total_fraction = 0.08
        self.max_fraction = 0.16

        self.gold_price_1kg = 138000

                               

                            






