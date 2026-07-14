
from __future__ import annotations
import math, time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
# import polars as pl

@dataclass
class Params:
    Nt: int = 200_000
    L: int = 11
    Fc: int = 3
    zr: float = 1.0/10000.0
    @property
    def Ng(self) -> int: return 2**self.L

def _pair_sum(vec):
    vec = np.asarray(vec, dtype=np.float64)
    assert vec.size % 2 == 0, "Vector length must be even."
    return vec.reshape(-1,2).sum(axis=1)

def build_levels(y, p, L):
    Y = [np.asarray(y, dtype=np.float64)]
    P = [np.asarray(p, dtype=np.float64)]
    for _ in range(1, L):
        Y.append(_pair_sum(Y[-1]))
        P.append(_pair_sum(P[-1]))
    return Y, P

def eta(i, j, L): return (i + 1.0) / (1.0 + 2.0*i + min(j, (2**(L-i)) + 1 - j))

def _sr(a, b, eps=1e-18):
    if b <= eps or a <= 0.0:
        return 0.0
    return math.sqrt(a / b)

def compute_fractions(x, y, p, L, log_etad=True, x_max=0.16):
    Ng = y.size; assert Ng == (1<<L)

    Y, P = build_levels(y, p, L)
    # if x > 17.0/25.0:
        # print("Reducing allocation"); x = 17.0/25.0

    Frc: Dict[int, np.ndarray] = {}
    Frc[L] = np.array([x], dtype=np.float64)

    # Level L-1
    y1 = float(Y[L-1][0]); y2 = float(Y[L-1][1])
    # Formula 2.18
    del0 = eta(L,1,L) * min(_sr(y1, y2) * (x_max - x), _sr(y2, y1) * x) 
    f = np.array([x + _sr(y2,y1)*del0, x - _sr(y1,y2)*del0], dtype=np.float64)
    Frc[L-1] = f.copy()

    # Level L-2 deltas
    YL2 = Y[L-2]
    # Formula 2.21
    del1 = eta(L-1,1,L) * min(_sr(float(YL2[0]),float(YL2[1]))*(x_max - f[0]),
                               _sr(float(YL2[1]),float(YL2[0]))*(f[0] - f[1]))
    del2 = eta(L-1,2,L) * min(_sr(float(YL2[2]),float(YL2[3]))*(f[0]-f[1]-_sr(float(YL2[0]),float(YL2[1]))*del1),
                               _sr(float(YL2[3]),float(YL2[2]))*(f[1]))
    delta = np.array([del1, del2], dtype=np.float64)

    def K(a): return 2**(L-a-1)

    i = L-2
    while i > 0:
        Yi = Y[i]; Ki = K(i)

        # Expand f -> next row
        ff = np.empty(2*Ki, dtype=np.float64)
        for j in range(1, Ki+1):
            j0 = j-1; a = 2*j-2; b = 2*j-1
            ff[2*j0]   = f[j0] + _sr(float(Yi[b]), float(Yi[a])) * delta[j0]
            ff[2*j0+1] = f[j0] - _sr(float(Yi[a]), float(Yi[b])) * delta[j0]
        f = ff.copy()
        Frc[i] = f.copy()

        # Left deltas
        Yim1 = Y[i-1]
        ldelta: List[float] = []
        pdelta = eta(i,1,L) * min(_sr(float(Yim1[0]),float(Yim1[1]))*(x_max - f[0]),
                                  _sr(float(Yim1[1]),float(Yim1[0]))*(f[0] - f[1]))
        ldelta.append(pdelta)
        for j in range(2, Ki):
            a0 = 2*j-4; a1 = 2*j-3; a2 = 2*j-2
            newpdelta = eta(i,j,L) * min(_sr(float(Yim1[2*j-1-1]),float(Yim1[2*j-1]))*(f[j-1-1]-f[j-1]-pdelta*_sr(float(Yim1[2*j-3-1]),float(Yim1[2*j-2-1]))),
                                         _sr(float(Yim1[2*j-1]),float(Yim1[2*j-1-1]))*(f[j-1]-f[j+1-1]))
            pdelta = newpdelta; ldelta.append(pdelta)

        # Right deltas — M1 fix: index against the RIGHT edge of Y[i-1]
        rdelta: List[float] = []
        width = 2**(L-i)  # len(f)
        # Use the far-right pair in Y[i-1]: indices 2*width-2 and 2*width-1 (0-based)
        left_idx_Y  = 2*width - 2   # y(i-1)_{2*width-1} in 1-based
        right_idx_Y = 2*width - 1   # y(i-1)_{2*width}   in 1-based
        # Initial mdelta uses the LAST f element and last difference
        mdelta = eta(i, width, L) * min(
            _sr(float(Yim1[right_idx_Y]), float(Yim1[left_idx_Y])) * f[width-1],
            _sr(float(Yim1[left_idx_Y]), float(Yim1[right_idx_Y])) * (f[width-2] - f[width-1])
        )
        rdelta.append(mdelta)

        j = width - 1
        while j > Ki + 1:
            aL = 2*j - 2; aR = 2*j - 1; aR2 = 2*j; aR3 = 2*j + 1
            newmdelta = eta(i,j,L) * min(
                _sr(float(Yim1[aR]), float(Yim1[aL])) * (f[j-1] - f[j] - mdelta * _sr(float(Yim1[aR3]), float(Yim1[aR2]))),
                _sr(float(Yim1[aL]), float(Yim1[aR])) * (f[j-2] - f[j-1])
            )
            mdelta = newmdelta
            rdelta.insert(0, mdelta)
            j -= 1

        # Middle sowing
        c1 = _sr(float(Yim1[2*Ki - 2]), float(Yim1[2*Ki - 1]))
        c2 = _sr(float(Yim1[2*Ki + 1]), float(Yim1[2*Ki]))
        a1 = c1 * ((f[Ki-2] - f[Ki-1]) - ldelta[-1]*_sr(float(Yim1[2*Ki - 3-1]), float(Yim1[2*Ki - 2-1])))
        a2 = c2 * ((f[Ki]   - f[Ki+1]) - rdelta[0]*_sr(float(Yim1[2*Ki + 3]), float(Yim1[2*Ki + 2])))
        b = f[Ki-1] - f[Ki]
        denom = c1*a1 + c2*a2
        etad = min(1.0, (b/denom) if denom > 0 else 0.0)
        # if log_etad: print(f"{etad:.6f}")
        pdelta_mid = eta(i, Ki,   L) * a1 * etad
        mdelta_mid = eta(i, Ki+1, L) * a2 * etad
        ldelta.append(pdelta_mid); rdelta.insert(0, mdelta_mid)

        delta = np.array(ldelta + rdelta, dtype=np.float64)
        i -= 1

    # Final base row
    Ki0 = 2**(L-1)
    fr = np.empty(2*Ki0, dtype=np.float64)
    for j in range(1, Ki0+1):
        j0 = j-1; a = 2*j-2; b = 2*j-1
        fr[2*j0]   = f[j0] + _sr(float(y[b]), float(y[a])) * delta[j0]
        fr[2*j0+1] = f[j0] - _sr(float(y[a]), float(y[b])) * delta[j0]
    return fr, Frc

def run(y, p, L, x_percent: float = 8.0, x_max: float = 16.0, seed: Optional[int] = 123):
    x = float(x_percent)/100.0
    x_max = float(x_max)/100.0
    fr, Frc_levels = compute_fractions(x, y, p, L, log_etad=True, x_max=x_max)
    return {"y": y, "p": p, "fr": fr, "Frc_levels": Frc_levels}

def main():
    Nt=2000000
    L=5

    ids = np.arange(Nt)
    fi = np.random.choice(range(0,2**L), size=Nt)
    amount = np.random.rand(Nt).astype(np.float32)*410

    df = pl.DataFrame({
        "id": ids,
        "amount": amount,
        "fi_group": fi
    })

    Trib = df.group_by('fi_group').sum().sort(by='fi_group')['amount'].to_numpy()
    p = df.group_by('fi_group').len().sort(by='fi_group')['len'].to_numpy()
    y = Trib / sum(Trib)
    p = p / sum(p)

    out = run(y, p, L, x_percent= 8.0, x_max= 16)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))

    plt.bar(range(1, len(out['fr']) + 1), out['fr'], color="orange")
    # plt.ylim([0,1.1])
    plt.xlabel("Tribute groups")
    plt.ylabel("Fraction of gratis")
    plt.title(f"Fractions, 8% allocation")
    plt.gca().set_facecolor("black")
    plt.show()

def lysis_vinyasa(n_tributes_to_generate, coen_price, x_target = 8.0):
    # new lysis
    L = 2

    if n_tributes_to_generate < 30:
        L = 2
        params = Params(Nt=200_000, L=L, Fc=3, zr=1.0/10000.0)
        fi = ([0,1,2,3] * 8)[:n_tributes_to_generate]
    elif n_tributes_to_generate < 200:
        L = 3
        params = Params(Nt=200_000, L=L, Fc=3, zr=1.0/10000.0)
        fi = np.random.choice(range(0,2**params.L), size=n_tributes_to_generate)
    elif n_tributes_to_generate < 300:
        L = 4
        params = Params(Nt=200_000, L=L, Fc=3, zr=1.0/10000.0)
        fi = np.random.choice(range(0,2**params.L), size=n_tributes_to_generate)
    elif n_tributes_to_generate < 600:
        L=5
        params = Params(Nt=200_000, L=L, Fc=3, zr=1.0/10000.0)
        fi = np.random.choice(range(0,2**params.L), size=n_tributes_to_generate)
    elif n_tributes_to_generate < 1500:
        L=6
        params = Params(Nt=200_000, L=L, Fc=3, zr=1.0/10000.0)
        fi = np.random.choice(range(0,2**params.L), size=n_tributes_to_generate)
    elif n_tributes_to_generate < 2000:
        L=7
        params = Params(Nt=200_000, L=L, Fc=3, zr=1.0/10000.0)
        fi = np.random.choice(range(0,2**params.L), size=n_tributes_to_generate)
    elif n_tributes_to_generate < 10000:
        L=8
        params = Params(Nt=200_000, L=L, Fc=3, zr=1.0/10000.0)
        fi = np.random.choice(range(0,2**params.L), size=n_tributes_to_generate)
    else :
        L=9
        params = Params(Nt=200_000, L=L, Fc=3, zr=1.0/10000.0)
        fi = np.random.choice(range(0,2**params.L), size=n_tributes_to_generate)

    

    ids = np.arange(n_tributes_to_generate)
    amount = np.random.rand(n_tributes_to_generate).astype(np.float32) * (410 *0.08)/(coen_price*(1.08))


    df = pl.DataFrame({
        "id": ids,
        "amount": amount,
        "fi_group": fi
    })

    Trib = df.group_by('fi_group').sum().sort(by='fi_group')['amount'].to_numpy()
    FI = df['fi_group'].unique()

    out = run(Trib, FI, params, x_percent=x_target, seed=123)

    fi_to_fraction_map = {}
    for i in range(2**params.L):
        fi_to_fraction_map[i] = out['fr'][i]

    df = df.with_columns(replaced=pl.col("fi_group").replace_strict(fi_to_fraction_map))
    df = df.with_columns((pl.col('amount') * pl.col('replaced')).alias('f_amount'))
    df = df.sort(by='fi_group')

    # return
    minting_pool = df['f_amount'].sum()
    max_share_of_single_tribute = df['f_amount'].max() / minting_pool
    winners_symbolic_load = df['f_amount'].to_list()

    return minting_pool, max_share_of_single_tribute, winners_symbolic_load

def nods_distribution(current_coen_price, minting_pool, max_share_of_single_tribute, nods):
    # Step size computation
    lower_bound = 0.08
    upper_bound = 0.16
    ideal_step_size = 10**(-4)
    rounding = 4

    lb_price = current_coen_price * (1+lower_bound)
    ub_price = current_coen_price * (1+upper_bound)

    step_size_bound = round((ub_price-lb_price)/(1/max_share_of_single_tribute), rounding)
    step_size = max(step_size_bound, ideal_step_size)

    # Bins
    n_bins = int((ub_price-lb_price) // step_size)
    bins = [round(lb_price+x*step_size, rounding) for x in range(n_bins)]
    bins.append(round(ub_price, rounding))

    # df_nods = pd.DataFrame(nods, columns =['ConsumerID', 'SymbolicLoad', 'TargetPrice'])
    # df_nods['TargetPrice0'] = df_nods['TargetPrice']
    # df_nods['TargetPrice'] = round(df_nods['TargetPrice'], rounding)

    # # 3. Nods analytics
    # df_nods_summary = pd.DataFrame(df_nods.groupby('TargetPrice')['SymbolicLoad'].sum()).reset_index()

    df_nods = pl.DataFrame(nods,schema=["ConsumerID", "SymbolicLoad", "TargetPrice"])
    df_nods = df_nods.with_columns([pl.col("TargetPrice").alias("TargetPrice0")])
    df_nods = df_nods.with_columns([pl.col("TargetPrice").round(rounding).alias("TargetPrice")])
    df_nods_summary = (df_nods.group_by("TargetPrice").agg(pl.col("SymbolicLoad").sum()))
    df_nods_summary = pd.DataFrame(df_nods_summary, columns=['TargetPrice', 'SymbolicLoad'])

    # 3.1 Add missing bins
    bins_to_be_added = [x for x in bins if x not in df_nods_summary['TargetPrice'].to_list()]

    for new_bin in bins_to_be_added:
        df_nods_summary.loc[len(df_nods_summary), :] = (new_bin, 0)

    # 3.2 Remove out-of-bound bins
    df_nods_summary = df_nods_summary[(df_nods_summary['TargetPrice']>=bins[0])&(df_nods_summary['TargetPrice']<=bins[-1])].copy()

    # 4. Distribute minting pool

    thresholds = df_nods_summary[df_nods_summary['SymbolicLoad']>0].sort_values(by=['SymbolicLoad'])['SymbolicLoad'].to_list()
    df_nods_summary['SymbolicLoad_'] = df_nods_summary['SymbolicLoad']

    if df_nods_summary['SymbolicLoad'].sum() == 0:
        df_nods_summary['ToBeAdded'] = minting_pool / len(df_nods_summary)

    else:
        minting_pool_to_be_distr = minting_pool

        while thresholds:
            # 1. Select minimal bin value
            min_bin_value = thresholds.pop(0)

            # 2.
            df_nods_summary['ToBeAdded'] = min_bin_value - df_nods_summary['SymbolicLoad_']
            df_nods_summary.loc[df_nods_summary[df_nods_summary['ToBeAdded']<0].index,'ToBeAdded'] = 0

            left_minting_pool = minting_pool_to_be_distr - df_nods_summary['ToBeAdded'].sum()

            if left_minting_pool < 0:
                # nothing left
                df_nods_summary['ToBeAdded'] = df_nods_summary['ToBeAdded'] / df_nods_summary['ToBeAdded'].sum() * minting_pool_to_be_distr
                minting_pool_to_be_distr -= df_nods_summary['ToBeAdded'].sum()
                df_nods_summary['SymbolicLoad_'] = df_nods_summary['SymbolicLoad_'] + df_nods_summary['ToBeAdded']
                df_nods_summary['ToBeAdded'] = 0

                break
            
            elif left_minting_pool > 0:
                # new iteration to be done
                minting_pool_to_be_distr -= df_nods_summary['ToBeAdded'].sum()
                df_nods_summary['SymbolicLoad_'] = df_nods_summary['SymbolicLoad_'] + df_nods_summary['ToBeAdded']
                df_nods_summary['ToBeAdded'] = 0

        if minting_pool_to_be_distr>0:
            df_nods_summary['SymbolicLoad_'] = df_nods_summary['SymbolicLoad_']+ minting_pool_to_be_distr / len(df_nods_summary)
        df_nods_summary['ToBeAdded'] = (df_nods_summary['SymbolicLoad_'] - df_nods_summary['SymbolicLoad'])
        
    df_nods_summary = df_nods_summary.sort_values(by=['TargetPrice'])

    # return
    output_distr_symbolic_load = df_nods_summary[df_nods_summary['ToBeAdded']>0]['ToBeAdded'].to_list()
    output_prices = df_nods_summary[df_nods_summary['ToBeAdded']>0]['TargetPrice'].to_list()

    return output_distr_symbolic_load, output_prices
            
def nods_fomation(day, output_distr_symbolic_load, output_prices, winners_symbolic_load, winners_cohort_id):

    dt = np.dtype([
    ("ConsumerID", np.int32),
    ("SymbolicLoad", np.float64),
    ("TargetPrice", np.float64)
])
    
    output_distr_symbolic_load_len = len(output_distr_symbolic_load)
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
                left_winners = len(winners_symbolic_load) - sum(lysis_pgt_count)
                lysis_pgt_count.append(left_winners)
                break

        else:
            print('Unexpected turn')

    if output_distr_symbolic_load_len > len(lysis_pgt_count):
        lysis_pgt_count.append(i)

    lysis_winners_pgt = []

    for a, b in zip(lysis_pgt_count, output_prices):
        if a!=0:
            lysis_winners_pgt.extend(a*[b])

    temp_df = pd.DataFrame(winners_symbolic_load, columns=['SymbolicLoad'])
    temp_df['ConsumerID'] = winners_cohort_id
    temp_df['TargetPrice'] = lysis_winners_pgt

    temp_df = temp_df.groupby(['ConsumerID', 'TargetPrice'])['SymbolicLoad'].sum().reset_index()
    selected_cr = list(temp_df[['ConsumerID', 'SymbolicLoad', 'TargetPrice']].itertuples(index=False, name=None))
    
    return np.array(selected_cr, dtype=dt)
