from typing import Sequence, Tuple
import numpy as np
from lysis_binary_4dec2025 import *

def kappa_calculation (y: Sequence[float], p: Sequence[float], N:int) -> np.ndarray:
    return np.zeros(len(y))

def policy_tau_P(a: float, b: float, c: float, p: Sequence[float], Nt) -> np.ndarray:
    """
    Implements the current policy P(a,b,c) for tau_i (Eq. 3.54).
    Input:
      - a,b,c: policy parameters (a>0, b>0, 0<c<1)
      - p: array-like population fractions p_i for i=1..N (length N)
    Output:
      - tau: length N+1 array of tau_i, i=1..N+1 (we return 0-based index correspondingly)
    
    Notes & assumptions:
      - The paper uses special fallbacks when p_i or p_{i-1} == 0 ("NaNb t"). Here we
        apply a robust fallback:
          * if p[i]==0 or p[i-1]==0 -> use a small positive epsilon (eps_fallback)
        This avoids division by zero while keeping policy behavior (you can change it
        later to your precise scheme).
      - Indexing: p provided is p[0..N-1] representing p_1..p_N in the paper.
      - Returns tau as numpy array tau[0..N] representing τ_1..τ_{N+1}.
    """
    Ng = len(p)

    # Precompute numerator factor (i - 1/2)^a for i=2..N as in Eq 3.54
    # We'll produce tau for i = 1..N+1 (N+1 entries)
    tau = np.zeros(Ng + 1, dtype=float)

    for i in range(1, Ng):
        pi = p[i]
        pim1 = p[i-1]

        if pi!=0 and pim1!=0:
            tau[i] = (i-0.5)**a / min(pi**b, pim1**b)
        else:
            tau[i] = Ng**a * Nt**b

    # Now set tau_1 and tau_{N+1} to satisfy normalization T = 2 * sum_{i=2..N} tau_i
    # per the paper T = 2 sum_{i=2..N} τ_i and τ_1 = c * sum_{i=2..N} τ_i, τ_{N+1} = (1-c) sum_{i=2..N} τ_i
    sum_middle = np.sum(tau)

    tau[0] = c * sum_middle            # tau_1
    tau[-1] = (1.0 - c) * sum_middle  # tau_{N+1}
    
    return tau

def compute_moments(y, tau, kappa):

    # masses -> 0...N+1
    T = np.sum(tau)
    m = tau / T              # m_0 ... m_N (tau_1...tau_{N+1})

    # Y
    Y = np.cumsum(y)             # Υ_i
    Y = np.insert(Y, 0, 0)
    Y[-1] = 1
    
    # X
    if np.sum(kappa) > 0:
        X = np.cumsum(kappa) / np.sum(kappa)
        X = np.insert(X, 0, 0)
    else:
        X = np.zeros_like(Y)

    Ey = np.sum(m * Y)
    Ey2 = np.sum(m * Y**2)
    var_y = Ey2 - Ey**2
    Ex = np.sum(m * X)
    Exy = np.sum(m * X * Y)
    cov_xy = Exy - Ex * Ey

    return m, X, Y, Ex, Ey, var_y, cov_xy

def compute_f1(y, tau, kappa, f, fmax):
    N = len(y)

    m, X, Y, Ex, Ey, var_y, cov_xy = compute_moments(y, tau, kappa)
    
    alpha = (np.sum(kappa)* np.sum(tau) ) / fmax
    beta = (f / fmax - Ey - alpha * cov_xy) / var_y

    f1 = np.zeros(N)
    I_bad = np.zeros(N)
    for i in range(1, N+1):
        f1[i-1] = fmax * np.sum(
            m[i:] * (1 + alpha * (X[i:] - Ex) + beta * (Y[i:] - Ey))
        )
        I_bad[i-1] = (1 + alpha * (X[i] - Ex) + beta * (Y[i] - Ey))


    vals = 1.0 + beta * (Y - Ey)
    I_bad_2026 = np.where(vals < -0)[0]

    return f1, I_bad, I_bad_2026.tolist(), m, beta, Y, Ey

def select_I(f0, fmax, m, beta, Upsilon, y_mean, I_bad):
    """
    Section 3.2 (linear path)
    Computes t_i for i in I_bad and returns the index I to aggregate.

    Parameters
    ----------
    f0 : np.ndarray, shape (N,)
        Initial monotone fractions
    fmax : float
    m : np.ndarray, shape (N,)
        Masses m_i (note: m[0] corresponds to i=0)
    beta : float
    Upsilon : np.ndarray, shape (N+1,)
    y_mean : float
    I_bad : list[int]

    Returns
    -------
    I : int
        Index to aggregate (0-based)
    t_I : float
        Corresponding hitting time
    """
    best_score = -np.inf
    I_star = None
    t_star = None

    for i in I_bad:
        gap = f0[i] - f0[i + 1]
        if gap <= 0:
            continue  # should not happen if f0 is monotone

        pressure = abs(1.0 + beta * (Upsilon[i] - y_mean))
        score = m[i] * pressure / gap

        if score > best_score:
            best_score = score
            I_star = i

            denom = gap - fmax * m[i] * (1.0 + beta * (Upsilon[i] - y_mean))
            t_star = gap / denom

    return I_star, t_star

def interpolate_f(f0, f1, t):
    return (1.0 - t) * f0 + t * f1

def lysis_fractions(y, p, L, Nt, f=0.08, fmax=0.16, a=0.2, b=0.1, c=0.2):
    tau = policy_tau_P(a, b, c, p, Nt)
    kappa = kappa_calculation(y,p,2**L)

    f_final, violations, violations_1, m, beta, Upsilon, y_mean = compute_f1(y, tau, kappa, f, fmax)
    # f_hit = f_final

    if len(violations_1) == 0:
        # print("Quasi-solution is already monotone.")
        f_hit = f_final
    else:
        # Section 3.2
        f0 = run(y, p, L, x_percent= f*100, x_max= fmax*100)['fr']
        I, t_I = select_I(f0, fmax, m, beta, Upsilon, y_mean, violations_1)
        f_hit = interpolate_f(f0, f_final, t_I)
        # print("Aggregate indices:", I, I+1)
    return f_hit

