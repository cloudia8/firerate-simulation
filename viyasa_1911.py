# Lysis version as of Nov 20, 2025

import math
import time
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

def Psi(G: List[float], k: int) -> List[float]:
    """Psi[G_, k_] := Table[If[i < k, G[[i]], G[[i + 1]]], {i, 1, Length[G] - 1}]"""
    return [G[i] if i < k-1 else G[i+1] for i in range(len(G)-1)]

def deaggregate_fractions(pn: list, fractions:list, fi_group_count, L: int = 10):
    if pn:
        fi_groups = [x for x in range(fi_group_count)]
        fr_full = []
        pointer_i = 0
        fi_group_i = 0

        for _ in range(len(fi_groups)):
            if pn[pointer_i] == fi_groups[fi_group_i]:
                fr_full.append(fractions[pointer_i])
                pointer_i +=1
                fi_group_i +=1

            elif pn[pointer_i+1] == fi_groups[fi_group_i+1]:
                fr_full.append(fractions[pointer_i])
                pointer_i +=1
                fi_group_i +=1

            else:
                fr_full.append(fractions[pointer_i])
                fi_group_i +=1
    else:
        fr_full = fractions
    return fr_full

def optimization_algorithm(y:list,
                           p:list,
                           x_input: float,
                           xm_input: float,
                           L: int = 10):
    """
    CU: list of tuples (tribute_index_or_value, FI_list)
    x_input and xm_input are percentages already given as decimals (e.g., 0.01 for 1%).
    L: tree depth (default 10)
    """
    t0_total = time.time()
    
    x = x_input
    xm = xm_input
    if x > xm:
        print("Reducing allocation")
        x = 0.99 * xm

    # Ng = 2**L
    Ng = len(y)

    # Optimization algorithm

    # helper t[i] := Sum[y[[j]], {j, 1, i}];
    def t_func(i):
        if i <= 0:
            return 0.0
        return float(np.sum(y[:i]))

    def th(i):
        return 1.0 - t_func(i)

    tsum = sum(t_func(i) for i in range(1, Ng+1))
    t2sum = sum(t_func(i)**2 for i in range(1, Ng+1))
    
    def xi(i):
        return t_func(i)/tsum if tsum != 0 else 0.0

    g0 = tsum/(Ng + 1)
    gp = t2sum/tsum if tsum != 0 else 0.0
    gm = g0*(1 - gp)/(1 - g0) if (1 - g0) != 0 else 0.0
    bet = (Ng + 1)*g0/gp if gp != 0 else float('inf')
    beth = (Ng + 1)*(1 - g0)/(1 - gm) if (1 - gm) != 0 else float('inf')
    xist = 1 - math.sqrt(1 - 1/bet) if bet != float('inf') and bet > 1 else 0.0
    xihst = 1 - math.sqrt(1 - 1/beth) if beth != float('inf') and beth > 1 else 0.0

    def xih(i):
        return (th(i)/(Ng + 1))/(1 - g0) if (1 - g0) != 0 else 0.0

    imin = 1
    dis = (xi(1) - xist)**2
    i = 2
    while i <= Ng and xi(i) < (1/bet if bet!=0 else float('inf')):
        di = (xi(i) - xist)**2
        if di < dis:
            imin = i
            dis = di
        i += 1

    ihmin = Ng
    dish = (xih(Ng) - xihst)**2
    i = Ng - 1
    while i > 0 and xih(i) < (1/beth if beth!=0 else float('inf')):
        dih = (xih(i) - xihst)**2
        if dih < dish:
            ihmin = i
            dish = dih
        i -= 1

    Dist = [(xi(i) - xist)**2 for i in range(0, Ng+1)]
    Disht = [(xih(i) - xihst)**2 for i in range(0, Ng+1)]
    ti1 = time.time()

    # Psi is defined above
    pr = list(p)
    vx = x / xm if xm != 0 else 0.0
    Nn = Ng
    xir = [xi(i) for i in range(1, Nn+1)]
    cmin = 0
    if vx > gp:
        cmin = imin
    elif vx < gm:
        cmin = ihmin
    gamma0 = g0
    gammam = gm
    gammap = gp

    if cmin == 0:
        pn = []
    else:
        if vx > gp:
            # Note: loop may be long; porting logic carefully
            while vx > gammap and Nn > 1:
                xic = xir[cmin-1]
                xich = (1 - (Nn + 1) * xic * gamma0)/(Nn + 1)/(1 - gamma0)
                gamma0n = gamma0*(1 + 1/Nn)*(1 - xic)
                betn = bet*(1 - xic)**2/(1 - bet*xic**2) if (1 - bet*xic**2) != 0 else bet
                bethn = beth*(1 - xich)**2/(1 - beth*xich**2) if (1 - beth*xich**2) != 0 else beth
                gammamn = 1 - (1 - gammam)*(1 - beth*xich**2)/(1 - xich) if (1 - xich) != 0 else gammam
                gammapn = gammap*(1 - bet*xic**2)/(1 - xic) if (1 - xic) != 0 else gammap
                Nn -= 1
                xin = Psi(xir, cmin)
                xin = [v/(1 - xic) for v in xin]
                pn = []
                for i in range(1, Nn+1):
                    if i == cmin:
                        pn.append(pr[cmin-1] + pr[cmin])
                    elif i < cmin:
                        pn.append(pr[i-1])
                    else:
                        pn.append(pr[i])
                xir = xin
                pr = pn
                gammam = gammamn
                gammap = gammapn
                gamma0 = gamma0n
                bet = betn
                beth = bethn
                xist = 1 - math.sqrt(1 - 1/bet) if bet>1 else xist
                cmin = 1
                dis = (xir[0] - xist)**2
                l = 2
                while l <= Nn and xir[l-1] < (1/bet if bet!=0 else float('inf')):
                    di = (xir[l-1] - xist)**2
                    if di < dis:
                        cmin = l
                        dis = di
                    l += 1
        elif vx < gm:
            while vx < gammam and Nn > 1:
                xic = xir[cmin-1]
                xich = (1 - (Nn + 1) * xic * gamma0)/(Nn + 1)/(1 - gamma0)
                gamma0n = gamma0*(1 + 1/Nn)*(1 - xic)
                betn = bet*(1 - xic)**2/(1 - bet*xic**2) if (1 - bet*xic**2) != 0 else bet
                bethn = beth*(1 - xich)**2/(1 - beth*xich**2) if (1 - beth*xich**2) != 0 else beth
                gammamn = 1 - (1 - gammam)*(1 - beth*xich**2)/(1 - xich) if (1 - xich) != 0 else gammam
                gammapn = gammap*(1 - bet*xic**2)/(1 - xic) if (1 - xic) != 0 else gammap
                Nn -= 1
                xin = Psi(xir, cmin)
                xin = [v/(1 - xic) for v in xin]
                pn = []
                for i in range(1, Nn+1):
                    if i == cmin:
                        pn.append(pr[cmin-1] + pr[cmin])
                    elif i < cmin:
                        pn.append(pr[i-1])
                    else:
                        pn.append(pr[i])
                xir = xin
                pr = pn
                gammam = gammamn
                gammap = gammapn
                gamma0 = gamma0n
                bet = betn
                beth = bethn
                xihst = 1 - math.sqrt(1 - 1/beth) if beth>1 else xihst
                xihr = [((1/(Nn + 1)) - gamma0*v)/(1 - gamma0) if (1 - gamma0)!=0 else 0 for v in xir]
                cmin = Nn
                l = Nn - 1
                dish = (xihr[cmin-1] - xihst)**2
                while l > 0 and xihr[l-1] < (1/beth if beth!=0 else float('inf')):
                    dih = (xihr[l-1] - xihst)**2
                    if dih < dish:
                        cmin = l
                        dish = dih
                    l -= 1

    # Plotting fractions / optimization
    fa = (xm/(Nn + 1))*(gammap - vx)/(gammap - gamma0) if (gammap - gamma0)!=0 else 0.0
    fb = xm*(vx - gamma0)/(gammap - gamma0) if (gammap - gamma0)!=0 else 0.0
    fro = [sum([fa + fb * xir[j] for j in range(k-1, Nn)]) for k in range(1, Nn+1)]
    Energy0 = float((xm - fro[0])**2 + sum((fro[i] - fro[i+1])**2 for i in range(0, Nn-1)) + (fro[-1])**2)
    # FinalOptim = [(pr[Nn - i], fro[Nn - i]) for i in range(1, Nn+1)]

    fro = deaggregate_fractions(pn, fro, len(y), L)

    return {
        'y': y,
        'p': p,
        # 'FinalOptim': FinalOptim,
        'EnergyOpt': Energy0,
        'fro': fro,
        'pointer':pn
    }

if __name__ == "__main__":
    import polars as pl
    Nt=200000
    L=10

    ids = np.arange(Nt)
    fi = np.random.choice(range(0,2**L), size=Nt)
    amount = np.random.rand(Nt).astype(np.float32)*410

    df = pl.DataFrame({
        "id": ids,
        "amount": amount,
        "fi_group": fi
    })

    Trib = df.group_by('fi_group').sum().sort(by='fi_group')['amount'].to_numpy()
    p = df['fi_group'].unique()
    y = Trib / sum(Trib)

    results = optimization_algorithm(y, p, x_input=0.08, xm_input=0.16, L=L)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    resulting_balance = sum([a*b for a,b in zip(results['fro'], results['y'])])*100

    plt.bar(range(1, len(results['fro']) + 1), results['fro'], color="orange")
    # plt.ylim([0,1.1])
    # plt.xlabel("FI")
    plt.ylabel("Fraction of gratis")
    plt.title(f"Fractions, final balance = {resulting_balance:.1f}%")
    plt.gca().set_facecolor("black")
    plt.show()



