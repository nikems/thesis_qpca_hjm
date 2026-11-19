"""
================================================================================
 B0 - MANDATORY CLASSICAL BASELINE   (Chapter 4, protocol section 2)
 Quantum pricing of an interest-rate caplet - classical reference pipeline.

 SINGLE SELF-CONTAINED FILE: every function is defined here (no local imports),
 every protocol point PRINTS its result and its verification, every figure and
 table is SAVED in  results/b0/ .

 Run:   python b0_baseline.py            (about 30-60 seconds on a laptop)

 THE IDEA IN FIVE STEPS (each step's output is the next step's input)
   1. DATA -> C       daily changes of the ECB forward rates (2022-23) give the
                      covariance matrix C: how much each maturity moves and how
                      the maturities move together.
   2. C -> PCA        C = U Lambda U^T splits the movement into independent
                      "directions" u_i (level, slope, ...) with size lambda_i.
   3. PCA -> HJM      sigma_i(tau) = sqrt(lambda_i) u_i(tau) are the volatility
                      functions of the HJM model; no-arbitrage fixes the drift.
   4. HJM -> PRICE    two independent routes to the same caplet price:
                      (a) closed form (Black formula, exact)      -> V, V_h
                      (b) Monte Carlo (simulate the curve M times) -> V_M +- CI
   5. CHECKS          MC must agree with the exact price of the same model;
                      the error sources (truncation, time step, sampling) are
                      measured one at a time; 2024 data test stability.
 The qPCA experiments (E2-E4) replace step 2 with a quantum estimate of lambda, u;
 E6 then repeats steps 3-4 with those values and compares with V(2x2, k=2).

 Pipeline (one line per protocol point):
   1.1   contract: ATM caplet, V0 = E^Q[ D(0,T1) * delta*(L-K)^+ ], random discount
   1.2   frozen inputs: ECB AAA forward curve, CALIBRATION 2022-2023 / HELD-OUT 2024
   1.3   configuration (versions, seeds, date) -> results/b0/b0_config.json
   2.1.1 sample covariance of DAILY forward-rate CHANGES, annualised (x252)
   2.1.2 C = U Lambda U^T + checks ||CU-U Lambda||, ||U^T U - I||
   2.1.3 variance explained R_k, covariance reconstruction error vs k
   2.1.4 two separate things:
         (a) HJM volatilities  sigma_i(tau) = sqrt(lambda_i) u_i(tau)  (from the PCA)
         (b) drift mu(tau), COMPUTED FROM the sigma_i by the no-arbitrage condition
             (continuous mu and discrete mu_hat, both verified numerically)
   2.2.1 Euler scheme on a grid of step dt, short rate and discount per path
   2.2.2 Monte Carlo price = average of the discounted payoffs of the M paths:
           m         = path number, m = 1..M (M = 100,000)
           Pi_m      = payoff of path m, paid at T1: delta * max(L_m - K, 0)
           D_m(0,T1) = discount along path m: exp(-sum of the daily rates r), random
           Y_m       = D_m(0,T1) * Pi_m = value TODAY of path m
           V_M       = mean(Y_m) = (1/M) sum_m Y_m = Monte Carlo price (3.8736 bp)
         (code: mc_caplet,  Y = exp(-R[:, n1]) * max(exp(X) - kappa, 0))
   2.2.3 SE, 95% CI (CLT), CI coverage checked on 500 independent replications,
         EXACT references: continuous model (closed form) and discretised model
   2.2.4 the price has three error sources; each one is measured ON ITS OWN,
         changing one setting and keeping the others fixed:
           (a) PCA truncation  = keeping only k of the n factors:
               price for k = 1..n, error V(k) - V(n)       (4x4, k=3: -0.00004 bp)
           (b) discretisation  = time step dt instead of continuous time:
               change dt, M fixed;  error V_h - V ~ dt^2    (dt=1/252: -0.00005 bp)
           (c) sampling        = only M paths instead of infinitely many:
               change M, dt fixed;  half-width 1.96 SE ~ 1/sqrt(M)  (M=100k: 0.035 bp)
         Changing M and dt together would mix (b) and (c): we could not tell which
         error went down.
   2.2.5 risk-neutral consistency P(0,T) = E^Q[exp(-int r)]
   OOS   held-out 2024: stability of covariance, PCA and price

 Matrices 2x2 (3M,6M), 3x3 (3M,6M,9M), 4x4 (3M,6M,9M,1Y).
 Outputs (results/b0/):
   b0_results.csv          main table (k, dt, M, price, CI, time, errors, R_k)
   b0_reference.json       reference numbers reused by E1-E6 (contract, C, PCA, prices)
   b0_config.json          versions, seeds, settings, run date
   b0_curve.csv/.png       initial forward curve f(0,tau) of the valuation date
   b0_pca.png, b0_hjm.png, b0_convergence.png, b0_error_decomposition.png
================================================================================
"""
import os, sys, json, time, platform, datetime, hashlib
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")                         # save figures to file, never open a window
import matplotlib.pyplot as plt

pd.set_option("future.infer_string", False)   # avoid pandas>=3.0 pyarrow-string CSV quirk
np.set_printoptions(suppress=True, precision=6, floatmode="fixed")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "results", "b0")
os.makedirs(OUT, exist_ok=True)

# ============================================================================
#  CONFIGURATION  (protocol 1.3 - everything that determines the numbers)

#  Same file + same seeds -> exactly the same numbers at every run.
#  Time is always measured in YEARS: 1 business day = 1/252, 3 months = 0.25.
# ============================================================================
CAL_START, CAL_END = "2022-01-01", "2023-12-31"   # covariance estimated ONLY here
OOS_START, OOS_END = "2024-01-01", "2024-12-31"   # held out, verification only
DT_HIST = 1 / 252          # observation step of the data: 1 business day, in years
T0, DELTA = 0.25, 0.25     # caplet: reset in 3M, accrual 3M  -> payment T1 = 0.5y
T1 = T0 + DELTA
# NOTATION (two different quantities, both measured in years):
#   DELTA (delta) = accrual period of the CAPLET = T1 - T0, one fixed number (0.25).
#   tau           = TIME TO MATURITY of a forward rate f(t, t + tau): the variable of
#                   the volatility functions sigma_i(tau); "tau" arrays hold the observed
#                   maturities (0.25, 0.5, 0.75, 1.0 for 3M, 6M, 9M, 1Y).
#   They coincide numerically for the 3M rate (0.25), but they are NOT the same thing.

# The production configuration (FULL: 4x4, k=3, dt=1/252) is defined in BLOCK_DEFS
# further below, next to the analysis that uses it.
Z_DIM = 4                         # normals drawn per step (>= max k): common random numbers across k
BATCH = 25_000                    # paths per batch (memory control only, not a parameter of the estimator)

M_MAIN, H_MAIN, SEED_MAIN = 100_000, 1 / 252, 20240101          # main table
M_LIST = [1_000, 4_000, 16_000, 64_000, 256_000]                 # convergence in M
H_MC_LIST = [1 / 12, 1 / 24, 1 / 52, 1 / 252]                    # convergence in dt (MC)
M_DT = 100_000                                                   # paths per dt point (same as the main table)
H_EXACT_LIST = [1/4, 1/8, 1/12, 1/24, 1/52, 1/252, 1/504, 1/1008]  # exact discretised prices
COV_R, COV_M, COV_H = 500, 2_000, 1 / 52                         # CI-coverage experiment
BOND_M, BOND_T = 200_000, [0.25, 0.5, 1.0]                       # risk-neutral bond check

# thesis palette (purple identity, checked for colour-blind separation);
# every series also has its own marker, so identity never relies on colour alone
COL = {"c1": "#5E35B1", "c2": "#C2185B", "c3": "#0004EF", "ref": "#B87EE4", "muted": "#9E9E9E"}
DIM_STYLE = {"2x2": (COL["c1"], "o"), "3x3": (COL["c2"], "s"), "4x4": (COL["c3"], "^")}
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.6,
    "legend.frameon": False, "legend.fontsize": 7.5, "savefig.dpi": 200, "savefig.bbox": "tight"})


def header(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)

def rnum(x, sig=2):
    """Readable number WITHOUT the 'e' notation (for the printed logs).
    |x| < 1e-12            -> '~0 (< 10^-12)'   (zero to machine precision)
    |x| >= 100000          -> thousands separators, e.g. 13,000,000
    otherwise              -> plain decimals with `sig` significant digits, e.g. 0.0000026"""
    x = float(x)
    if x == 0:
        return "0"
    if abs(x) < 1e-12:
        return "~0 (< 10^-12)"
    if abs(x) >= 1e5:
        return f"{x:,.0f}"
    d = max(0, sig - 1 - int(np.floor(np.log10(abs(x)))))
    return f"{x:.{d}f}"


def pct2(x):
    """Covariance / eigenvalue in (%)^2 per year: 5.6522e-05 (decimal^2) -> '0.5652'.
    Its square root is directly a volatility in %: sqrt(0.5652) = 0.752%."""
    return f"{1e4 * float(x):.4f}"



def tag(ax, letter):
    """Writes the panel letter (a), (b), (c) on a figure, so the thesis can refer to it."""
    ax.text(-0.12, 1.04, letter, transform=ax.transAxes, fontweight="bold")


# ============================================================================
#  0. EXACT INTEGRALS OF PIECEWISE-LINEAR FUNCTIONS
#  THEORY: the pricing formulas contain integrals of the forward curve and of
#  the volatilities, e.g.  P(0,T) = exp(-int_0^T f(0,u) du).  But the data give
#  f(0,tau) and sigma_i(tau) ONLY at the observed maturities (3M, 6M, 9M, 1Y, ...).
#  Model choice: between two maturities the function is a straight line
#  (linear interpolation), below the first / beyond the last it is constant
#  (flat extrapolation, the np.interp convention).
#  FORMULA: on a piece [a, b] where g goes linearly from g_a to g_b, the area
#  covered up to a + y is   g_a * y + (g_b - g_a)/(b - a) * y^2 / 2
#  (rectangle + triangle). Summing the pieces gives the integral EXACTLY,
#  so the reference prices carry no numerical-integration error.
#  USED BY: P(0,T) (initial bonds), the drift mu(tau), the variance s^2.
# ============================================================================
def pl_integral(x, knots, vals):
    """Exact area under a curve known only at a few points.
    WHY:     the pricing formulas need integrals of the forward curve and of sigma_i,
             but we know them only at the maturities 3M, 6M, ... (see section 0).
    IN:      x     = upper limit(s) of the integral, in years (number or array)
             knots = maturities where the function is known, e.g. [0.25, 0.5]
             vals  = function values at those maturities, e.g. sigma_1 = [0.00672, 0.01046]
    OUT:     int_0^x g(u) du, with g = straight lines between knots, flat outside.
    USED IN: B0 (initial bonds P(0,T)), drift_continuous, logbond_var_continuous."""
    x = np.atleast_1d(np.asarray(x, float))
    out = vals[0] * np.minimum(x, knots[0])                       # flat part [0, knots[0]]
    for a, b, va, vb in zip(knots[:-1], knots[1:], vals[:-1], vals[1:]):
        y = np.clip(x - a, 0.0, b - a)                            # length covered in [a, b]
        out = out + va * y + 0.5 * (vb - va) / (b - a) * y ** 2
    return out + vals[-1] * np.maximum(x - knots[-1], 0.0)        # flat part beyond last knot


# ============================================================================
#  2.1  PCA AND HJM VOLATILITY  
#   (1) COVARIANCE. HJM describes how the forward curve MOVES, so we measure the
#       daily changes df = f_t - f_{t-1} of each maturity, not the levels.
#       C_ij = Cov(df_i, df_j) / dt_hist   ->  diagonal = variance of maturity i
#       per year; off-diagonal = how maturities i and j move together.
#   (2) PCA. C is symmetric positive semidefinite, so  C = U Lambda U^T  with
#       orthonormal eigenvectors u_i (columns of U) and eigenvalues lambda_i >= 0.
#       u_i = a SHAPE of curve movement (u_1 ~ all rates together = "level",
#       u_2 ~ short vs long opposite = "slope"); lambda_i = its variance per year.
#       Variance explained R_k = (lambda_1+...+lambda_k)/(sum of all lambda):
#       how much of the movement k factors capture.
#   (3) HJM VOLATILITIES. Model:  df(t,t+tau) = mu(tau) dt + sum_i sigma_i(tau) dW_i
#       with independent Brownian motions W_i. Its covariance per year is
#       sum_i sigma_i sigma_i^T; choosing  sigma_i = sqrt(lambda_i) u_i  makes it
#       equal U Lambda U^T = C: the model moves exactly like the data.
#       sqrt(lambda_i) = SIZE (vol per sqrt(year)), u_i = SHAPE across maturities.
#   (4) DRIFT. It is NOT estimated: no-arbitrage under Q forces
#       mu(tau) = sum_i sigma_i(tau) * int_0^tau sigma_i(u) du   (HJM condition).
#       On the simulation grid we use its exact discrete version mu_hat (below).
# ============================================================================
def sample_covariance(levels):
    """2.1.1  Covariance matrix of the daily changes, per year.
    WHY:     C is the ONLY thing the model takes from the historical data: it says how
             much each maturity moves and how the maturities move together.
    IN:      levels = table of forward rates (decimal), one row per day, one column per
             maturity, already cut to a window (e.g. CAL[["IF_3M","IF_6M"]], 512 days).
    DO:      df_t = f_t - f_{t-1}  (daily changes: HJM models changes, levels are not
             stationary); unbiased sample covariance (divisor N-1); divide by
             DT_HIST = 1/252 because Var(df) ~ sigma^2 * (1/252) -> per-year units.
    OUT:     (C, N): C = n x n matrix in decimal^2 per year, N = number of changes (511).
             e.g. 2x2: C = [[5.652e-5, 6.302e-5], [6.302e-5, 1.141e-4]].
    USED IN: PCA loop (section 2.1 on CAL) and the out-of-sample check (on OOS 2024)."""
    d = levels.diff().dropna()
    return d.cov(ddof=1).values / DT_HIST, len(d)


def pca(C):
    """2.1.2  Principal component analysis: C = U Lambda U^T.
    WHY:     splits the curve movement into independent factors (level, slope, ...);
             this is exactly the step that qPCA replaces with a quantum algorithm.
    IN:      C = covariance matrix from sample_covariance (n x n).
    DO:      numpy.eigh (solver for symmetric matrices); sort eigenvalues from largest
             to smallest; fix the sign of each eigenvector (largest entry positive),
             since u and -u are both valid and we want reproducible plots.
    OUT:     (eigvals, eigvecs): eigvals = eigenvalues lambda_1 >= lambda_2 >= ...,
             eigvecs = matrix whose COLUMN i is the eigenvector u_i (the U of C = U Lambda U^T;
             e.g. 2x2: lambda = (1.546e-4, 1.603e-5), u_1 = (0.54, 0.84), u_2 = (0.84, -0.54).
    USED IN: PCA loop (section 2.1) and out-of-sample check."""
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx].copy()
    for i in range(eigvecs.shape[1]):
        eigvecs[:, i] *= np.sign(eigvecs[np.argmax(np.abs(eigvecs[:, i])), i])
    return eigvals, eigvecs


def variance_explained(eigvals):
    """2.1.3  Share of the total variance captured by the first k factors.
    WHY:     tells how many factors are worth keeping.
    IN:      eigvals = eigenvalues from pca (descending).
    OUT:     array R with R[k-1] = (lambda_1+...+lambda_k) / (sum of all lambda).
             e.g. 2x2: R_1 = 0.906 (level alone = 90.6%), R_2 = 1.
    USED IN: PCA loop (print), mk_row (column var_explained), b0_pca.png, OOS check."""
    return np.cumsum(eigvals) / np.sum(eigvals)


def reconstruction_error(C, eigvals, eigvecs):
    """2.1.3  How badly C is rebuilt when only k factors are kept.
    WHY:     second view of truncation (on the matrix, not on the variance).
    IN:      C, eigvals, eigvecs from pca.
    OUT:     array of ||C - U_k Lambda_k U_k^T||_F / ||C||_F for k = 1..n
             (Frobenius norm; exactly 0 at k = n because nothing is left out).
    USED IN: PCA loop (print, section 2.1.3)."""
    return np.array([np.linalg.norm(C - eigvecs[:, :k] @ np.diag(eigvals[:k]) @ eigvecs[:, :k].T)
                     / np.linalg.norm(C) for k in range(1, len(eigvals) + 1)])


def hjm_vol_factors(eigvals, eigvecs, k):
    """2.1.4  From PCA to HJM: the volatility functions sigma_i = sqrt(lambda_i) * u_i.
    WHY:     HJM needs one volatility function per random shock; this choice makes the
             model's covariance equal to the data's C (sum_i sigma_i sigma_i^T = C at k = n).
    IN:      eigvals, eigvecs from pca; k = how many factors to keep (1..n).
    OUT:     array (k, n_maturities): row i = sigma_i at the observed maturities,
             in decimal per sqrt(year).  sqrt(lambda_i) = SIZE, u_i = SHAPE.
             e.g. 2x2: sigma_1 = (0.672%, 1.046%), sigma_2 = (0.337%, -0.216%).
    USED IN: V_cont, V_disc, main table, analyse_block, OOS check.
             In E6 the same function receives lambda, u from qPCA instead."""
    return np.sqrt(np.maximum(eigvals[:k], 0.0))[:, None] * eigvecs[:, :k].T


def sigma_at(sig, tau, x):
    """Volatility at ANY time-to-maturity, not only at 3M, 6M, ...
    WHY:     the simulation needs sigma_i(d*h) for every grid day d (1 day, 2 days, ...).
    IN:      sig = output of hjm_vol_factors (k x n); tau = observed maturities;
             x = times-to-maturity where we want sigma (years, array).
    OUT:     array (k, len(x)): straight line between maturities, flat below 3M and
             beyond the last maturity (np.interp).
    USED IN: drift_discrete (sigma on the grid), logbond_var_discrete."""
    return np.vstack([np.interp(x, tau, s) for s in sig])


def drift_continuous(sig, tau, x):
    """2.1.4  Continuous HJM no-arbitrage drift under Q.
    WHY:     in HJM the drift is not free: it is fixed by the volatilities, otherwise
             the model would allow arbitrage (bonds would not be priced consistently).
    FORMULA: mu(tau) = sum_i sigma_i(tau) * int_0^tau sigma_i(u) du   (integral exact).
    IN:      sig, tau (as above); x = times-to-maturity (array, years).
    OUT:     mu at each x (decimal per year).
    USED IN: only CHECKS in analyse_block (2.1.4 a/b):
             the simulation uses the discrete version drift_discrete."""
    return sum(np.interp(x, tau, s) * pl_integral(x, tau, s) for s in sig)


def drift_discrete(sig, tau, h, n):
    """2.1.4  DISCRETE no-arbitrage drift (Glasserman 2004, Sec. 3.6.2) for the Euler grid.
    With discrete bonds P(t_i,t_j) = exp(-h sum_{l=i}^{j-1} f(t_i,t_l)) and discount
    D(0,t_j) = exp(-h sum_{l<j} r(t_l)), discounted bonds are EXACT martingales on the
    grid iff, for time-to-maturity d*h (d = 1, 2, ...):
        mu_hat(d) = (h/2) sum_i [ S_i(d)^2 - S_i(d-1)^2 ],   S_i(d) = sum_{e=1}^{d} sigma_i(e h).
    mu_hat -> mu(tau) as h -> 0 (checked below).
    WHY:     with the continuous mu the simulated grid model would have a small
             arbitrage error; mu_hat removes it exactly, so MC converges to V_h.
    IN:      sig, tau; h = time step (1/252); n = number of grid days needed (n1 = 126).
    OUT:     (S, mu, cs): S = sigma on the grid, shape (k, n+1), column d = sigma_i(d h);
             mu = mu_hat for d = 0..n (entry 0 unused); cs = running sums S_i(d).
    USED IN: mc_caplet and mc_bonds (inputs of the Euler step), checks in analyse_block."""
    S = sigma_at(sig, tau, np.arange(n + 1) * h)
    cs = np.concatenate([np.zeros((S.shape[0], 1)), np.cumsum(S[:, 1:], axis=1)], axis=1)
    mu = np.concatenate([[0.0], 0.5 * h * (cs[:, 1:] ** 2 - cs[:, :-1] ** 2).sum(axis=0)])
    return S, mu, cs


# ============================================================================
#  REFERENCE PRICES (2.2.3)  -  closed form, no Monte Carlo
#  WHY: to judge whether Monte Carlo is right we need the TRUE price of the same
#  model, computed without simulation.
#  THEORY:
#   - The volatilities sigma_i do not depend on the rates and the shocks are
#     Gaussian, so every simulated rate is a sum of Gaussians -> Gaussian.
#     Hence ln P(T0,T1) (bond read at the reset) is Gaussian, with variance s^2.
#   - Payoff rewritten: delta (L-K)^+ = (1/P - kappa)^+ paid at T1, kappa = 1+delta K;
#     discounted to T0 it equals (1 - kappa P(T0,T1))^+ : a PUT on the zero bond.
#   - A put on a lognormal asset has the Black price (computed under the
#     T0-forward measure, which turns the random discount into P(0,T0)):
#       V0 = P(0,T0) Phi(d1) - kappa P(0,T1) Phi(d2)
#       d1 = [ ln(P(0,T0)/(kappa P(0,T1))) + s^2/2 ] / s ,  d2 = d1 - s
#   - s^2 = Var[ln P(T0,T1)] is the ONLY model-dependent input: it collects how
#     much the 3M-6M forwards can move between today and T0.
#  TWO VERSIONS of s^2, same formula:
#     continuous model  -> s^2 from integrals      -> V   (the model price)
#     model on step h   -> s^2 from sums on the grid -> V_h (what MC simulates)
#  V_h - V = discretisation error, computed exactly (no random noise).
# ============================================================================
def black_caplet(P0T0, P0T1, kappa, s2):
    """The closed-form caplet price (Black formula for a put on the zero bond).
    WHY:     gives the exact price of the model, to check Monte Carlo against.
    IN:      P0T0 = P(0,T0) = 0.99132558, P0T1 = P(0,T1) = 0.98334542 (today's curve),
             kappa = 1 + delta K = 1.0081153, s2 = Var[ln P(T0,T1)] (from the model).
    OUT:     caplet price per unit notional (x 1e4 -> bp).
             s2 continuous 9.52e-7 -> 3.8584 bp (= V).
    USED IN: V_cont (with logbond_var_continuous), V_disc (with logbond_var_discrete),
             OOS check. It is only the "calculator": V and V_h differ only by s2."""
    s = np.sqrt(s2)
    d1 = (np.log(P0T0 / (kappa * P0T1)) + 0.5 * s2) / s
    return P0T0 * norm.cdf(d1) - kappa * P0T1 * norm.cdf(d1 - s)


def logbond_var_continuous(sig, tau, T0, T1, n_gl=8):
    """s^2 = sum_i int_0^T0 [ int_{T0}^{T1} sigma_i(u-s) du ]^2 ds.
    Inner integral exact (pl_integral); the outer integrand is piecewise polynomial
    (degree 4) with breakpoints where T0-s or T1-s hits a maturity knot, so
    Gauss-Legendre on each piece is exact up to rounding.
    WHY:     s^2 is the only model input of black_caplet; this is the CONTINUOUS one.
    MEANING: for each time s before T0, the inner integral is how much the bond
             P(T0,T1) reacts to a shock at s; squaring and summing over s and factors
             gives the total variance of ln P(T0,T1).
    IN:      sig, tau (volatilities); T0, T1; n_gl = Gauss-Legendre points (8; 16 for the check).
    OUT:     s^2 (number). e.g. 2x2, k=2: 9.52e-7.
    USED IN: V_cont (-> V), OOS check, accuracy check GL8 vs GL16."""
    br = {0.0, T0} | {T - t for T in (T0, T1) for t in tau if 0.0 < T - t < T0}
    br = np.array(sorted(br))
    xg, wg = np.polynomial.legendre.leggauss(n_gl)
    tot = 0.0
    for a, b in zip(br[:-1], br[1:]):
        s = 0.5 * (b - a) * xg + 0.5 * (a + b)
        for si in sig:
            inner = pl_integral(T1 - s, tau, si) - pl_integral(T0 - s, tau, si)
            tot += 0.5 * (b - a) * np.sum(wg * inner ** 2)
    return tot


def logbond_var_discrete(sig, tau, T0, T1, h):
    """Same variance for the Euler model on step h:
        s_h^2 = sum_{i=0}^{n0-1} h * sum_k ( h * sum_{l=n0}^{n1-1} sigma_k(t_l - t_i) )^2.
    WHY:     the exact s^2 of the model that Monte Carlo actually simulates (step h):
             the integrals of logbond_var_continuous become sums over grid days.
    IN:      sig, tau; T0, T1; h = step (e.g. 1/252 -> n0 = 63, n1 = 126).
    OUT:     s_h^2 (number); with black_caplet it gives V_h.
    USED IN: V_disc (-> V_h)."""
    n0, n1 = round(T0 / h), round(T1 / h)
    tot = 0.0
    for i in range(n0):
        S = sigma_at(sig, tau, (np.arange(n0, n1) - i) * h)
        tot += h * np.sum((h * S.sum(axis=1)) ** 2)
    return tot


# ============================================================================
#  MONTE CARLO (2.2.1 - 2.2.2)
#  THEORY: the price is an expectation,  V0 = E^Q[ D(0,T1) * Pi ].  Monte Carlo
#  estimates it by simulating M possible futures of the curve (paths) and
#  averaging the discounted payoffs:  V_M = (1/M) sum_m Y_m,  Y_m = D_m(0,T1) Pi_m.
#  Law of large numbers: V_M -> V0;  central limit theorem: error ~ s_Y/sqrt(M).
#
#  WHY DISCRETISE: the HJM equation is in continuous time (dt, dW), but a
#  computer can only make finite steps. So time becomes a grid and:
#     dt -> h ,   dW -> sqrt(h) * Z  with Z ~ N(0,1) ,   integrals -> sums.
#  Grid t_i = i h  (h = 1/252 -> one step = one trading day; T0 = t_63, T1 = t_126).
#  Every integer index below is a grid day: i = "current day" of the simulation,
#  j = maturity day of a forward. In the code, column j of array f = maturity t_j.
#  State: discrete forwards f(t_i, t_j), j >= i  (only maturities not yet passed).
#
#  ONE PATH, step by step (inner level):
#   - start: today's curve as cell averages f(0,t_j) = (1/h) int_{t_j}^{t_{j+1}} f(0,u) du,
#     so the discrete bonds reproduce P(0,t_j) EXACTLY for every h;
#   - each day every future forward moves (time-to-maturity d = j - i >= 1):
#       f(t_{i+1}, t_j) = f(t_i, t_j) + mu_hat(d) h + sum_k sigma_k(d h) sqrt(h) Z_k,i
#     (drift = no-arbitrage push, shocks = random movement shaped by the PCA);
#   - the rate paid today is the forward maturing today:  r(t_i) = f(t_i, t_i);
#   - discount = accumulated rates:  D(0,t_j) = exp(-h sum_{l<j} r(t_l))  (random);
#   - at the reset (day n0) read the bond  P(T0,T1) = exp(-h sum_{l=n0}^{n1-1} f(T0,t_l)),
#     hence L = (1/P - 1)/delta  and the payoff  Pi = delta (L-K)^+ = (1/P - kappa)^+ at T1.
#  MANY PATHS (outer level): the same day-loop moves ALL paths at once (one numpy
#  row per path); BATCH only splits them into groups to limit memory.
#  The discrete drift mu_hat keeps the simulation arbitrage-free ON THE GRID, so
#  MC converges to V_h (exact price of the step-h model), not to V.
# ============================================================================
def _euler_batch(S, mu, f0_cells, h, n_steps, Mb, rng, reset=None):
    """The heart of Monte Carlo: moves the forward curve day by day for Mb paths at once.
    WHY:     produces, for every simulated future, the two things the payoff needs:
             the discount D(0,T1) and the bond P(T0,T1) at the reset.
    IN:      S, mu      = sigma and drift on the grid (from drift_discrete)
             f0_cells   = today's curve on the grid (from f0_cells_on_grid)
             h, n_steps = step and number of days to simulate (126)
             Mb         = number of paths in this batch (25,000)
             rng        = random generator (seeded -> reproducible)
             reset      = (n0, n1) = (63, 126) to read the bond, or None (bond check)
    DO:      f has one ROW per path and one COLUMN per maturity day j. For i = 0..125:
               - if i == 63: X = h * sum of f[:, 63:126]  (= -ln P(T0,T1))
               - R += h * f[:, i]    (today's rate r(t_i) added to the discount)
               - draw k normals Z per path; all future columns move by
                 drift * h + sigma * sqrt(h) * Z  (Euler step).
    OUT:     R (Mb x n_steps+1): R[:, j] = h * sum_{l<j} r(t_l), so D(0,t_j) = exp(-R[:, j]);
             X (Mb,): -ln P(T0,T1) per path (None if reset is None).
    USED IN: mc_caplet (price), mc_bonds (bond check)."""
    k = S.shape[0]
    f = np.tile(f0_cells[:n_steps], (Mb, 1))
    R = np.zeros((Mb, n_steps + 1))
    X = None
    sqh = np.sqrt(h)
    for i in range(n_steps):
        if reset is not None and i == reset[0]:
            X = h * f[:, reset[0]:reset[1]].sum(axis=1)        # bond seen at the reset date
        R[:, i + 1] = R[:, i] + h * f[:, i]                    # r(t_i) = f(t_i, t_i)
        Z = rng.standard_normal((Mb, Z_DIM))[:, :k]            # always Z_DIM draws -> CRN across k
        nd = n_steps - i - 1                                   # live maturities j = i+1 .. n_steps-1
        if nd > 0:
            f[:, i + 1:] += h * mu[1:nd + 1] + sqh * (Z @ S[:, 1:nd + 1])
    return R, X


def f0_cells_on_grid(h, n):
    """Today's forward curve f(0, t_j) on the simulation grid (starting point of every path).
    WHY:     using the AVERAGE of f(0,.) over each day [t_j, t_j+h] (not its value at t_j)
             makes the discrete bonds equal P(0,t_j) exactly, for any h.
    IN:      h = step; n = number of grid days (126).
    OUT:     array of n values: (B0(t_{j+1}) - B0(t_j)) / h, j = 0..n-1.
    USED IN: mc_caplet, mc_bonds (passed to _euler_batch)."""
    t = np.arange(n + 1) * h
    return np.diff(B0(t)) / h


def mc_caplet(sig, tau, h, M, seed):
    """V_M = (1/M) sum_m Y_m,  Y_m = D_m(0,T1) * Pi_m.
    SE = s_Y / sqrt(M) (s_Y with divisor M-1); 95% CI = V_M +- 1.96 SE.
    CI assumption: Y_m i.i.d. with finite variance (true: Pi is bounded by 1/P) -> CLT.
    WHY:     the Monte Carlo price, the second (independent) route to the caplet price.
    IN:      sig, tau = HJM volatilities; h = step (1/252); M = number of paths (100,000);
             seed = starting point of the random numbers (same seed -> same result).
    DO:      prepare grid (n0 = 63, n1 = 126), drift, curve; for each batch call
             _euler_batch; payoff per path Y = exp(-R[:, 126]) * max(exp(X) - kappa, 0)
             (= discount * (1/P - kappa)^+); accumulate sum of Y and of Y^2.
    OUT:     dict price (= V_M), se, lo/hi (95% CI), time (seconds).
             e.g. FULL: 3.8736 +- 0.0351 bp.
    USED IN: main table, analyse_block (convergence, coverage, cost, error budget)."""
    n0, n1 = round(T0 / h), round(T1 / h)
    S, mu, _ = drift_discrete(sig, tau, h, n1)
    f0c = f0_cells_on_grid(h, n1)
    rng = np.random.default_rng(seed)
    s1 = s2 = 0.0
    t0 = time.perf_counter()
    for start in range(0, M, BATCH):
        Mb = min(BATCH, M - start)
        R, X = _euler_batch(S, mu, f0c, h, n1, Mb, rng, reset=(n0, n1))
        Y = np.exp(-R[:, n1]) * np.maximum(np.exp(X) - KAPPA, 0.0)
        s1 += Y.sum(); s2 += np.dot(Y, Y)
    el = time.perf_counter() - t0
    mean = s1 / M
    se = np.sqrt(max(s2 - M * mean ** 2, 0.0) / (M - 1) / M)
    return dict(price=mean, se=se, lo=mean - 1.96 * se, hi=mean + 1.96 * se, time=el)


def mc_bonds(sig, tau, h, T_list, M, seed):
    """2.2.5  Simulated bond prices E^Q[D(0,T)], to compare with today's P(0,T).
    WHY:     if the simulation is arbitrage-free, the average simulated discount must
             give back today's curve. It is a test of drift + discretisation.
    IN:      sig, tau, h; T_list = maturities to test (0.25, 0.5, 1.0); M paths; seed.
    OUT:     (mean, SE) arrays, one entry per T.
    USED IN: analyse_block, bond check (b0_bond_check.csv)."""
    N = round(max(T_list) / h)
    S, mu, _ = drift_discrete(sig, tau, h, N)
    f0c = f0_cells_on_grid(h, N)
    idx = [round(T / h) for T in T_list]
    rng = np.random.default_rng(seed)
    s1 = np.zeros(len(idx)); s2 = np.zeros(len(idx))
    for start in range(0, M, BATCH):
        Mb = min(BATCH, M - start)
        R, _ = _euler_batch(S, mu, f0c, h, N, Mb, rng)
        D = np.exp(-R[:, idx])
        s1 += D.sum(axis=0); s2 += (D ** 2).sum(axis=0)
    mean = s1 / M
    return mean, np.sqrt(np.maximum(s2 - M * mean ** 2, 0.0) / (M - 1) / M)   # 0 if D is deterministic


# ============================================================================
#  1.2  LOAD DATA, PRE-TREATMENT, CALIBRATION / OUT-OF-SAMPLE SPLIT
#  WHAT: ECB AAA euro-area curve, INSTANTANEOUS forward rates f(t, t+tau) at 12
#  fixed maturities tau (3M ... 30Y), one curve per business day.
#  WHY THE SPLIT: the model is estimated on 2022-2023 (CALIBRATION) only;
#  2024 is kept aside (HELD-OUT) to test whether the estimated volatility is
#  stable on data the model has never seen. Rates: percent in the file -> decimal.
#  Days with a missing value are dropped (all maturities must exist on a day).
# ============================================================================
RUN_START = time.time()
ALLCOLS = ["IF_3M", "IF_6M", "IF_9M", "IF_1Y", "IF_2Y", "IF_3Y", "IF_5Y", "IF_7Y",
           "IF_10Y", "IF_15Y", "IF_20Y", "IF_30Y"]
TAU_ALL = np.array([.25, .5, .75, 1, 2, 3, 5, 7, 10, 15, 20, 30])
meta = json.load(open(os.path.join(DATA, "dataset_metadata.json")))
# frozen input: the file on disk must be exactly the one recorded in dataset_metadata.json
_sha = hashlib.sha256(open(os.path.join(DATA, "forward_rates_raw.csv"), "rb").read()).hexdigest()
if _sha != meta["raw_sha256"]:
    sys.exit(f"data/forward_rates_raw.csv does not match dataset_metadata.json (sha256 {_sha[:16]}... "
             f"vs {meta['raw_sha256'][:16]}...): the frozen input has changed.")
raw = pd.read_csv(os.path.join(DATA, "forward_rates_raw.csv"), index_col=0, parse_dates=True)[ALLCOLS]
n_before, n_missing = len(raw), int(raw.isna().sum().sum())
clean = raw.dropna()
dec = clean / 100.0                                   # percent -> decimal
CAL = dec.loc[CAL_START:CAL_END]
OOS = dec.loc[OOS_START:OOS_END]

header("1.2  FROZEN INPUTS AND PRE-TREATMENT")
print(f"  source      : {meta['source']}")
print(f"  series      : {meta['series_key_template']}  ({meta['curve']})")
print(f"  units       : percent in file -> converted to decimal;  frequency: business daily")
print(f"  sha256      : {meta['raw_sha256'][:16]}...  (downloaded {meta['download_utc'][:10]})")
print(f"  rows        : {n_before}   missing values: {n_missing}   dates dropped (listwise): {n_before - len(clean)}")
print(f"  CALIBRATION : {CAL.index.min().date()} -> {CAL.index.max().date()}  ({len(CAL)} days)  <- covariance, PCA, valuation curve")
print(f"  HELD-OUT    : {OOS.index.min().date()} -> {OOS.index.max().date()}  ({len(OOS)} days)  <- only the OOS check at the end")

# ============================================================================
#  1.1  CONTRACT  (valuation date = last calibration day)
#  WHAT: a "3M x 3M" caplet = it STARTS in 3 months and LASTS 3 months
#  (market naming "start x length"; NOT a 3x3 matrix). The contract is ONE and the
#  same for every covariance matrix (2x2, 4x4, ...): only the sigma_i change.
#  Today t = 0 (29/12/2023). At T0 = 0.25 the 3-month
#  rate L for the period T0 -> T1 is fixed; at T1 = 0.5 the holder receives
#  delta * max(L - K, 0) (fixed in advance, paid in arrears, like a floating loan).
#  FORMULAS:
#    P(0,T) = exp(-int_0^T f(0,u) du)          bond price from today's curve
#    L      = (1/P(T0,T1) - 1) / delta          simple rate of the period
#    K      = (P(0,T0)/P(0,T1) - 1) / delta     ATM strike = today's forward rate
#    kappa  = 1 + delta K  ->  payoff = (1/P(T0,T1) - kappa)^+
#  Prices are per unit notional and printed in basis points (1 bp = 0.0001).
# ============================================================================
VAL_DATE = CAL.index[-1]
f0_vals = CAL.iloc[-1].values                          # f(0, tau) at the 12 maturities


def f0(T):
    """Today's forward curve f(0,T) at any T (years); data of 29/12/2023, linear between
    maturities, flat below 3M. Kept for reference/plots; prices use B0 and P0."""
    return np.interp(np.asarray(T, float), TAU_ALL, f0_vals)


def B0(T):
    """Area under today's curve, int_0^T f(0,u) du (exact, via pl_integral).
    USED IN: P0 (bonds) and f0_cells_on_grid (curve on the grid)."""
    return pl_integral(T, TAU_ALL, f0_vals)


def P0(T):
    """Today's zero-coupon bond P(0,T) = exp(-int_0^T f(0,u) du): value today of 1 euro at T.
    IN: T (years). OUT: P(0,T). e.g. P(0,0.25) = 0.99132558, P(0,0.5) = 0.98334542.
    USED IN: strike K, black_caplet inputs, bond check (target of E[D(0,T)])."""
    return np.exp(-B0(T))


P0T0, P0T1 = float(P0(T0)[0]), float(P0(T1)[0])
K = (P0T0 / P0T1 - 1.0) / DELTA                        # ATM: strike = initial simple forward
KAPPA = 1.0 + DELTA * K

header("1.1  CONTRACT")
print("  ATM caplet on the 3M simple rate L(T0;T0,T1), payoff paid at T1:")
print("     Pi = delta * max(L - K, 0),   L = (1/P(T0,T1) - 1)/delta")
print("     V0 = E^Q[ D(0,T1) Pi ],  D(0,T1) = exp(-int_0^T1 r(s) ds)  (RANDOM discount)")
print("     measure Q = risk-neutral, numeraire = bank account, continuous compounding")
print(f"  T0 = {T0} y, delta = {DELTA} y, T1 = {T1} y, notional = 1 (prices in bp of notional)")
print(f"  valuation date (initial curve f(0,.)) = {VAL_DATE.date()}  (last calibration day)")
print(f"  f(0,tau) at 3M/6M/9M/1Y = {np.round(f0_vals[:4] * 100, 4)} %  (flat below 3M)")
print(f"  P(0,T0) = {P0T0:.8f}   P(0,T1) = {P0T1:.8f}   K = F(0;T0,T1) = {K * 100:.5f} %")

# ---- today's curve f(0,tau): the starting point of every simulated path ------------------------
# WHY SHOW IT: it is the second input taken from the data (the first is C). It fixes P(0,T0),
# P(0,T1), the ATM strike K and the first column of every Monte Carlo path (f0_cells_on_grid).
# The caplet only uses the part up to T1 = 0.5 (3M and 6M points); the rest is shown for context.
print(f"\n  INITIAL CURVE f(0,tau) on {VAL_DATE.date()}  (12 maturities; -> results/b0/b0_curve.csv, b0_curve.png)")
print("     maturity   tau (y)   f(0,tau) (%)   P(0,tau)")
for c_, t_, v_ in zip(ALLCOLS, TAU_ALL, f0_vals):
    print(f"     {c_.replace('IF_', ''):>6s}    {t_:6.2f}     {100 * v_:8.4f}     {float(P0(t_)[0]):.6f}")
pd.DataFrame(dict(maturity=[c.replace("IF_", "") for c in ALLCOLS], tau=TAU_ALL, f0_pct=100 * f0_vals,
                  P0=[float(P0(t)[0]) for t in TAU_ALL])).to_csv(os.path.join(OUT, "b0_curve.csv"), index=False)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
xx = np.linspace(0, 30, 600)
a1.plot(xx, 100 * f0(xx), color=COL["c1"], label="linear interpolation, flat below 3M")
a1.plot(TAU_ALL, 100 * f0_vals, "o", color=COL["c1"], label="ECB data (12 maturities)")
a1.set_xlabel("maturity $\\tau$ (years)"); a1.set_ylabel("$f(0,\\tau)$ (%)")
a1.set_title(f"Initial forward curve, {VAL_DATE.date()}"); a1.legend(); tag(a1, "a")
xz = np.linspace(0, 1, 200)
a2.plot(xz, 100 * f0(xz), color=COL["c1"])
a2.plot(TAU_ALL[:4], 100 * f0_vals[:4], "o", color=COL["c1"])
a2.axvspan(T0, T1, color=COL["c2"], alpha=0.12, label=f"caplet period $T_0$-$T_1$, K = {100 * K:.3f}%")
a2.axhline(100 * K, color=COL["c2"], ls="--", lw=1)
a2.set_xlabel("maturity $\\tau$ (years)"); a2.set_ylabel("$f(0,\\tau)$ (%)")
a2.set_title("Zoom on the first year (the part the caplet uses)"); a2.legend(); tag(a2, "b")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "b0_curve.png")); plt.close(fig)

# ============================================================================
#  2.1  PCA + HJM  for each matrix dimension (calibration window only)
#  WHAT: apply steps (1)-(3) of the 2.1 theory to three sub-matrices:
#    2x2 (3M, 6M)   3x3 (3M, 6M, 9M)   4x4 (3M, 6M, 9M, 1Y).
#  2x2 is the size used by the qPCA experiments (one system qubit).
#  CHECKS printed for each matrix (all must be ~1e-16, i.e. rounding only):
#    ||C U - U Lambda||       the u_i really are eigenvectors
#    ||U^T U - I||            the u_i are orthonormal
#    ||sum sigma_i sigma_i^T - C||   the HJM vols reproduce C exactly (k = n)
# ============================================================================
BLOCKS = {"2x2": (["IF_3M", "IF_6M"], np.array([.25, .5])),
          "3x3": (["IF_3M", "IF_6M", "IF_9M"], np.array([.25, .5, .75])),
          "4x4": (["IF_3M", "IF_6M", "IF_9M", "IF_1Y"], np.array([.25, .5, .75, 1.0]))}
PCA = {}
header("2.1  PCA AND HJM VOLATILITY  (calibration 2022-2023)")
for name, (cols, tau) in BLOCKS.items():
    C, nchg = sample_covariance(CAL[cols])                                   # 2.1.1
    eigvals, eigvecs = pca(C)                                                           # 2.1.2
    R = variance_explained(eigvals); rec = reconstruction_error(C, eigvals, eigvecs)         # 2.1.3
    res_eig = np.linalg.norm(C @ eigvecs - eigvecs @ np.diag(eigvals))
    res_orth = np.linalg.norm(eigvecs.T @ eigvecs - np.eye(len(eigvals)))
    sig_full = hjm_vol_factors(eigvals, eigvecs, len(eigvals))
    res_vol = np.linalg.norm(sig_full.T @ sig_full - C)                      # sum sigma sigma^T = C
    PCA[name] = dict(cols=cols, tau=tau, C=C, eigvals=eigvals, eigvecs=eigvecs, R=R, rec=rec)
    print(f"\n[{name}]  maturities {cols}   ({nchg} daily changes)")
    print("  2.1.1 C in (%)^2 per year (= decimal^2 x 10^4), divisor N-1, annualised x252:")
    for row in C:
        print("        " + "  ".join(f"{pct2(x):>8s}" for x in row))
    print("        implied vol sqrt(C_jj) = " + ", ".join(f"{100 * np.sqrt(c):.3f}%" for c in np.diag(C)) + "  (absolute, per year)")
    print("  2.1.2 eigenvalues (desc, (%)^2 per year) = " + ", ".join(pct2(x) for x in eigvals))
    print(f"        ||CU-U Lambda|| = {rnum(res_eig)}   ||U^T U - I|| = {rnum(res_orth)}   -> "
          f"{'OK' if max(res_eig / np.linalg.norm(C), res_orth) < 1e-12 else 'CHECK'}")
    for i in range(len(eigvals)):
        print(f"        u_{i + 1} = {np.round(eigvecs[:, i], 4)}")
    print("  2.1.3 R_k = " + ", ".join(f"R_{k + 1}={r:.4f}" for k, r in enumerate(R)))
    print("        rel. reconstruction error ||C-C_k||/||C|| = " + ", ".join(f"k={k + 1}: {rnum(r, 2)}" for k, r in enumerate(rec)))
    print(f"  2.1.4 sigma_i = sqrt(lambda_i) u_i ;  ||sum_i sigma_i sigma_i^T - C|| = {rnum(res_vol)} (k = n)")

# ============================================================================
#  REFERENCE PRICES: continuous V(k) for every matrix and k; discretised V_h

#  One exact price = a chain of 3 steps (no random numbers):
#     lambda, u (PCA) --hjm_vol_factors--> sigma_i --logbond_var_*--> s^2 --black_caplet--> price
#     e.g. 2x2, k=2:  sigma_1=(0.672%,1.046%), sigma_2=(0.337%,-0.216%) -> s^2=9.52e-7 -> 3.85845 bp
#  The two (V-cont and V_disc)functions differ ONLY in how s^2 is computed:
#     V_cont(name, k)    -> V    : continuous s^2  (exact price of the HJM model)
#     V_disc(name, k, h) -> V_h  : s^2 on step h   (exact price of what Monte Carlo simulates)
#  Then:
#     REF[name][k]  = V_cont for every matrix (2x2, 3x3, 4x4) and every k = 1..n  (9 prices)
#     printed table = those 9 prices + truncation error V(k) - V(n) (factors left out), exact
#     Gauss-Legendre check: s^2 with 8 and 16 integration points -> identical -> s^2 is exact
#     mk_row        = builds ONE row of the results table (used later, after each MC run)
#  KEY NUMBER FOR E6: V(2x2, k=2) = 3.85845 bp, the price obtained when lambda, u
#  come from the classical PCA; E6 recomputes it with lambda, u from qPCA.
# ============================================================================
def V_cont(name, k):
    """V = exact price of the CONTINUOUS model for matrix `name` ("2x2", ...) with k factors.
    Chain: PCA -> hjm_vol_factors -> logbond_var_continuous (s^2) -> black_caplet.
    OUT: price per unit notional. e.g. V_cont("2x2", 2) = 3.8584 bp (the E6 reference).
    USED IN: REF table (all matrices and k)."""
    P = PCA[name]
    return black_caplet(P0T0, P0T1, KAPPA, logbond_var_continuous(hjm_vol_factors(P["eigvals"], P["eigvecs"], k), P["tau"], T0, T1))


def V_disc(name, k, h):
    """V_h = exact price of the model on step h (what Monte Carlo simulates).
    Same chain as V_cont but with logbond_var_discrete. V_h - V = discretisation error.
    USED IN: main table and analyse_block (MC is checked against V_h)."""
    P = PCA[name]
    return black_caplet(P0T0, P0T1, KAPPA, logbond_var_discrete(hjm_vol_factors(P["eigvals"], P["eigvecs"], k), P["tau"], T0, T1, h))


REF = {name: {k: V_cont(name, k) for k in range(1, len(P["eigvals"]) + 1)} for name, P in PCA.items()}
V_FULL_MODEL = REF["4x4"][4]                 # most complete model: 4 maturities, all factors
header("2.2.3  EXACT REFERENCE PRICES  (closed form, bp of notional)")
for name in PCA:
    n = len(PCA[name]["eigvals"])
    print(f"  [{name}] " + "   ".join(f"V(k={k})={REF[name][k] * 1e4:.5f}" for k in REF[name])
          + f"     truncation V(k)-V(k={n}): " + ", ".join(f"{(REF[name][k] - REF[name][n]) * 1e4:+.4f}" for k in REF[name]))
print("  note: full-rank 2x2 and 3x3 give the SAME price: with linear interpolation the caplet only")
print("        sees sigma(tau) for tau <= T1 = 0.5y, i.e. only the 3M/6M block of C (consistency check).")
_sig = hjm_vol_factors(PCA["4x4"]["eigvals"], PCA["4x4"]["eigvecs"], 3)
s2a = logbond_var_continuous(_sig, PCA["4x4"]["tau"], T0, T1, n_gl=8)
s2b = logbond_var_continuous(_sig, PCA["4x4"]["tau"], T0, T1, n_gl=16)
print(f"  reference accuracy: |s^2(GL8) - s^2(GL16)| / s^2 = {rnum(abs(s2a - s2b) / s2a)}")


def mk_row(study, block, name, k, h, M, seed, mc, Vc, Vh, n_full):
    """one row of the B0 table: MC result next to the exact references and the three error sources.
    z_mc = (MC - V_h)/SE: how many standard errors MC is from the exact price of the SAME
    model; |z| < 2 means agreement within sampling noise (95% level)."""
    return dict(block=block, study=study, matrix=name, k=k, dt=h, M=M, seed=seed,
                price_bp=mc["price"] * 1e4, se_bp=mc["se"] * 1e4, ci_lo_bp=mc["lo"] * 1e4,
                ci_hi_bp=mc["hi"] * 1e4, time_s=round(mc["time"], 3), ref_bp=Vc * 1e4,
                ref_disc_bp=Vh * 1e4, abs_err_bp=abs(mc["price"] - Vc) * 1e4,
                rel_err=abs(mc["price"] - Vc) / Vc, mc_err_bp=(mc["price"] - Vh) * 1e4,
                z_mc=(mc["price"] - Vh) / mc["se"], disc_err_bp=(Vh - Vc) * 1e4,
                trunc_err_bp=(Vc - REF[name][n_full]) * 1e4, var_explained=PCA[name]["R"][k - 1])


# ============================================================================
#  2.2  MAIN TABLE: every matrix, every k, M = 100k, dt = 1/252, SAME seed (CRN)
#  CRN = Common Random Numbers: every run uses the SAME random numbers Z
#  (same seed 20240101; always Z_DIM = 4 normals per step, the first k are used)
#  WHAT: run Monte Carlo for each (matrix, k) and put it next to V and V_h.
#  HOW TO READ IT:  MC vs V_h -> is the simulation correct? (|z| < 2)
#                   V_h vs V  -> discretisation error of daily steps (tiny)
#                   V(k) vs V(n) -> what is lost by keeping only k factors
#  The SAME seed for every k (common random numbers) makes the rows comparable:
#  differences between k come from the model, not from different random draws.
# ============================================================================
rows = []
header(f"2.2  MONTE CARLO vs EXACT REFERENCE  (M={M_MAIN}, dt=1/{round(1 / H_MAIN)}, seed={SEED_MAIN} for all k)")
print("  V = exact continuous model, V_h = exact discretised model (same dt), MC must match V_h within its CI")
for name, P in PCA.items():
    n = len(P["eigvals"])
    for k in range(1, n + 1):
        sig = hjm_vol_factors(P["eigvals"], P["eigvecs"], k)
        mc = mc_caplet(sig, P["tau"], H_MAIN, M_MAIN, SEED_MAIN)
        rows.append(mk_row("main", "all", name, k, H_MAIN, M_MAIN, SEED_MAIN, mc, REF[name][k], V_disc(name, k, H_MAIN), n))
        r = rows[-1]
        print(f"  [{name} k={k}] MC={r['price_bp']:.4f}  CI=({r['ci_lo_bp']:.4f},{r['ci_hi_bp']:.4f})  "
              f"V={r['ref_bp']:.4f}  V_h={r['ref_disc_bp']:.4f}  z(MC-V_h)={r['z_mc']:+.2f}  "
              f"rel.err={100 * r['rel_err']:.2f}%  R_k={r['var_explained']:.4f}  t={r['time_s']:.1f}s")

# ============================================================================
#  PRODUCTION BLOCK (FULL): 4x4, k=3, daily steps - all the checks of the baseline
#  "Production" = the ONE official configuration whose numbers are the baseline result
#  (V = 3.85841, V_h = 3.85836, MC = 3.8736 +- 0.0351 bp). The main table (2.2) tries
#  every matrix and k once; here only FULL is examined in depth with all the checks.
#  Why 4x4, k=3: the largest matrix, with k=3 factors (99.98% of the variance) -> the
#  classical method at its best, with a realistic reduction (not all factors).
#  WHAT IS VERIFIED, and why each check matters:
#   2.1.4 drift   continuous mu = d/dtau[1/2 |int sigma|^2] and discrete mu_hat
#                 satisfy no-arbitrage (residual ~ 0); mu_hat -> mu as h -> 0.
#   2.2.5 bonds   E^Q[D(0,T)] = P(0,T): the simulated discount reproduces today's
#                 curve -> the simulation is arbitrage-free (|z| < 3).
#   2.2.4a M      SE ~ 1/sqrt(M) (log-log slope -0.5) and the 95% CI contains V_h
#                 in ~95% of 500 independent runs -> the error bars are honest.
#   2.2.4b dt     V_h - V shrinks like dt^2 (slope ~2): the time step is under control.
#   cost          paths and steps needed to reach a target precision eps.
#   budget        the error sources side by side, each measured on its own.
#  The caplet only depends on the 3M-6M part of C, so V(4x4, k=4) = V(2x2, k=2)
#  = 3.85845 bp (both use ALL factors -> both reproduce the same 3M-6M block of C):
#  the 2x2 model used by qPCA/E6 loses nothing (model reduction error = 0).
# ============================================================================
BLOCK_DEFS = [
    dict(name="FULL", suffix="", matrix="4x4", k=3, k_list=[1, 2, 3, 4], h_main=H_MAIN, h_mc=H_MC_LIST,
         bond_h=[1 / 12, 1 / 252], drift_h=[1 / 12, 1 / 52], cov_h=COV_H, seed0=0,
         purpose="best classical estimate of the model price (4 maturities, 3 factors, daily steps)"),
]
EPS_LIST, R_COST = [0.2, 0.1, 0.05, 0.02], 10


def analyse_block(B):
    """All the checks of the baseline for one configuration (here FULL: 4x4, k=3, dt=1/252).
    WHY:     shows that the price is right and that each error source is under control.
    IN:      B = one dict of BLOCK_DEFS (matrix, k, time steps to test, seeds).
    DO:      in order: drift check (2.1.4), bond check (2.2.5), convergence in M + CI
             coverage (2.2.4a), convergence in dt (2.2.4b), cost at equal precision,
             error budget, figures b0_hjm.png / b0_convergence.png / b0_error_decomposition.png.
    OUT:     dict with the production numbers (V, V_h, MC, errors, coverage, slopes),
             written into b0_reference.json; also adds rows to `rows` (b0_results.csv).
    USED IN: BLOCK_RESULTS (called once per block)."""
    name, k, h_main, sfx, s0 = B["matrix"], B["k"], B["h_main"], B["suffix"], B["seed0"]
    P = PCA[name]; n_full = len(P["eigvals"])
    sig = hjm_vol_factors(P["eigvals"], P["eigvecs"], k)
    Vc = REF[name][k]; Vh_main = V_disc(name, k, h_main)
    lab = f"{name}, k={k}"; tagB = B["name"]
    header(f"BLOCK {tagB}: {lab}, dt_main=1/{round(1 / h_main)}  -  {B['purpose']}")
    print(f"  exact price V({lab}) = {Vc * 1e4:.5f} bp ;  most complete model V(4x4, k=4) = {V_FULL_MODEL * 1e4:.5f} bp ;"
          f"  model reduction error = {(Vc - V_FULL_MODEL) * 1e4:+.5f} bp")
    print(f"  exact discretised price V_h(dt=1/{round(1 / h_main)}) = {Vh_main * 1e4:.5f} bp")
    out = dict(name=tagB, matrix=name, k=k, dt_main=h_main, V_bp=Vc * 1e4, V_h_main_bp=Vh_main * 1e4)

    # ---- 2.1.4 drift condition --------------------------------------------------------------
    print(f"\n  [{tagB}] 2.1.4 DRIFT CONDITION ({lab})")
    print("   continuous (Q, Musiela): mu(tau) = sum_i sigma_i(tau) int_0^tau sigma_i(u) du")
    xs = np.linspace(0.01, 0.99, 50); e = 1e-6
    half_sq = lambda x: 0.5 * sum(pl_integral(x, P["tau"], s) ** 2 for s in sig)
    fd = (half_sq(xs + e) - half_sq(xs - e)) / (2 * e)
    print(f"   (a) mu = d/dtau[ 1/2 |int sigma|^2 ]: max rel. residual = "
          f"{rnum(np.max(np.abs(fd - drift_continuous(sig, P['tau'], xs)) / drift_continuous(sig, P['tau'], xs)))}")
    for h in B["drift_h"]:
        n = round(1.0 / h)
        S, mu, cs = drift_discrete(sig, P["tau"], h, n)
        lhs = h ** 2 * np.cumsum(mu[1:]); rhs = 0.5 * h ** 3 * (cs[:, 1:] ** 2).sum(axis=0)
        gap = np.max(np.abs(drift_continuous(sig, P["tau"], np.arange(1, n + 1) * h) - mu[1:]))
        print(f"   (b) h=1/{round(1 / h):<3d} discrete no-arbitrage residual = {rnum(np.max(np.abs(lhs - rhs)))}"
              f" ; max|mu_hat - mu| = {rnum(gap)}")
    print("   (c) Monte-Carlo check E^Q[D(0,T)] = P(0,T): below")

    # ---- 2.2.5 bonds ----------------------------------------------------------------------------
    print(f"\n  [{tagB}] 2.2.5 INITIAL-CURVE CONSISTENCY P(0,T) = E^Q[exp(-int r)]  ({lab}, M={BOND_M})")
    bond_rows = []
    for h in B["bond_h"]:
        m, se = mc_bonds(sig, P["tau"], h, BOND_T, BOND_M, seed=s0 + 7)
        for T, mm, ss in zip(BOND_T, m, se):
            Pc = float(P0(T)[0])
            if ss < 1e-12 * Pc:          # one step: D(0,T) = exp(-h r(0)) is not random -> exact equality expected
                z = 0.0 if abs(mm - Pc) < 1e-12 else np.inf
                verdict = "EXACT (D deterministic: single step)" if z == 0.0 else "CHECK"
            else:
                z = (mm - Pc) / ss; verdict = "CONSISTENT" if abs(z) < 3 else "CHECK"
            bond_rows.append(dict(block=tagB, h=h, T=T, P_curve=Pc, P_mc=mm, se=ss, z=z))
            print(f"   h=1/{round(1 / h):<3d} T={T:.2f}  P(0,T)={Pc:.8f}  E[D]={mm:.8f}  SE={rnum(ss)}  z={z:+.2f}  -> {verdict}")
    pd.DataFrame(bond_rows).to_csv(os.path.join(OUT, f"b0_bond_check{sfx}.csv"), index=False)

    # ---- 2.2.4a convergence in M + CI coverage -------------------------------------------------
    print(f"\n  [{tagB}] 2.2.4a CONVERGENCE IN M  ({lab}, dt=1/{round(1 / h_main)}) - sampling error, SE ~ 1/sqrt(M)")
    for j, M in enumerate(M_LIST):
        mc = mc_caplet(sig, P["tau"], h_main, M, seed=s0 + 1000 + j)
        rows.append(mk_row("conv_M" if tagB == "FULL" else f"{tagB.lower()}_conv_M", tagB, name, k, h_main, M,
                           s0 + 1000 + j, mc, Vc, Vh_main, n_full))
        print(f"   M={M:7d}  price={mc['price'] * 1e4:.4f} bp  SE={mc['se'] * 1e4:.4f}  "
              f"CI=({mc['lo'] * 1e4:.4f},{mc['hi'] * 1e4:.4f})  MC-V_h={(mc['price'] - Vh_main) * 1e4:+.4f}  t={mc['time']:.2f}s")
    dM = pd.DataFrame([r for r in rows if r["block"] == tagB and r["study"].endswith("conv_M")])
    slope_M = np.polyfit(np.log(dM.M), np.log(dM.se_bp), 1)[0]
    print(f"   fitted log-log slope of SE vs M = {slope_M:.3f}   (theory -0.5)")
    Vh_cov = V_disc(name, k, B["cov_h"]); t0 = time.perf_counter(); cover = []
    for r in range(COV_R):
        mc = mc_caplet(sig, P["tau"], B["cov_h"], COV_M, seed=s0 + 50_000 + r)
        cover.append(mc["lo"] <= Vh_cov <= mc["hi"])
    cov_rate = float(np.mean(cover)); cov_se = np.sqrt(0.95 * 0.05 / COV_R)
    print(f"   CI COVERAGE: {COV_R} independent replications (M={COV_M}, dt=1/{round(1 / B['cov_h'])}): "
          f"{100 * cov_rate:.1f}% of 95% CIs contain V_h   (expected 95.0 +- {196 * cov_se:.1f}%)  "
          f"-> {'CALIBRATED' if abs(cov_rate - 0.95) < 1.96 * cov_se else 'CHECK'}   t={time.perf_counter() - t0:.1f}s")

    # ---- 2.2.4b convergence in dt ---------------------------------------------------------------
    print(f"\n  [{tagB}] 2.2.4b CONVERGENCE IN dt  ({lab}) - discretisation error")
    exact_dt = pd.DataFrame([dict(h=h, V_h_bp=V_disc(name, k, h) * 1e4) for h in H_EXACT_LIST])
    exact_dt["disc_err_bp"] = exact_dt.V_h_bp - Vc * 1e4
    slope_h = np.polyfit(np.log(exact_dt.h), np.log(np.abs(exact_dt.disc_err_bp)), 1)[0]
    for _, r in exact_dt.iterrows():
        print(f"   exact  dt=1/{round(1 / r.h):<5d} V_h={r.V_h_bp:.6f} bp   V_h - V = {r.disc_err_bp:+.6f} bp")
    print(f"   fitted log-log slope |V_h - V| vs dt = {slope_h:.3f}  (order 2: exact curve + exact discrete drift,")
    print("   sigma evaluated at the cell midpoint of u - s; see the B0 explanation)")
    for j, h in enumerate(B["h_mc"]):
        mc = mc_caplet(sig, P["tau"], h, M_DT, seed=s0 + 2000 + j)
        Vh = V_disc(name, k, h)
        rows.append(mk_row("conv_dt" if tagB == "FULL" else f"{tagB.lower()}_conv_dt", tagB, name, k, h, M_DT,
                           s0 + 2000 + j, mc, Vc, Vh, n_full))
        print(f"   MC     dt=1/{round(1 / h):<5d} price={mc['price'] * 1e4:.4f} bp  SE={mc['se'] * 1e4:.4f}  "
              f"V_h={Vh * 1e4:.4f}  z(MC-V_h)={(mc['price'] - Vh) / mc['se']:+.2f}  t={mc['time']:.2f}s")
    exact_dt.to_csv(os.path.join(OUT, f"b0_exact_dt{sfx}.csv"), index=False)
    dD = pd.DataFrame([r for r in rows if r["block"] == tagB and r["study"].endswith("conv_dt")])

    # ---- cost at equal precision ----------------------------------------------------------------
    # WHY: to reach a 95% half-width eps, the CLT gives M = (1.96 sigma_Y / eps)^2 paths;
    # dt is chosen so that the (exact) bias |V_h - V| is negligible, <= eps/10.
    sigma_Y = float(dM.se_bp.iloc[-1] * np.sqrt(dM.M.iloc[-1]))
    print(f"\n  [{tagB}] COST AT EQUAL PRECISION vs the continuous price V  (sigma_Y = {sigma_Y:.3f} bp; R={R_COST})")
    print("   dt = largest step with exact bias |V_h - V| <= eps/10 ; M = ceil((1.96 sigma_Y / eps)^2)")
    cost_rows = []
    for ie, eps in enumerate(EPS_LIST):
        ok = exact_dt[np.abs(exact_dt.disc_err_bp) <= eps / 10]
        h = float(ok.h.max()); target = Vc
        M_e = int(np.ceil((1.96 * sigma_Y / eps) ** 2))
        errs, times, cov = [], [], []
        for r in range(R_COST):
            mc = mc_caplet(sig, P["tau"], h, M_e, seed=s0 + 3000 + 100 * ie + r)
            errs.append((mc["price"] - target) * 1e4); times.append(mc["time"]); cov.append(mc["lo"] <= target <= mc["hi"])
        errs = np.array(errs)
        cost_rows.append(dict(block=tagB, eps_bp=eps, dt=h, M=M_e, reference="V",
                              bias_vs_V_bp=V_disc(name, k, h) * 1e4 - Vc * 1e4,
                              rmse_bp=float(np.sqrt(np.mean(errs ** 2))), mean_abs_err_bp=float(np.mean(np.abs(errs))),
                              coverage=float(np.mean(cov)), mean_time_s=float(np.mean(times)),
                              paths_x_steps=M_e * round(T1 / h)))
        c = cost_rows[-1]
        print(f"   eps={eps:5.2f} bp -> dt=1/{round(1 / h):<4d} M={M_e:8d}  RMSE={c['rmse_bp']:.4f}  "
              f"mean|err|={c['mean_abs_err_bp']:.4f}  coverage={100 * c['coverage']:.0f}%  time={c['mean_time_s']:.3f}s  "
              f"(paths x steps = {c['paths_x_steps']:,})")
    print("   eps = target 95% half-width (expected RMSE ~ eps/1.96); coverage over 10 runs is indicative only.")
    pd.DataFrame(cost_rows).to_csv(os.path.join(OUT, f"b0_cost_equal_precision{sfx}.csv"), index=False)

    # ---- error budget at the block's production configuration ---------------------------------
    mc = mc_caplet(sig, P["tau"], h_main, M_MAIN, s0 + SEED_MAIN)
    header(f"ERROR BUDGET  block {tagB}: {lab}, dt=1/{round(1 / h_main)}, M={M_MAIN}  (distinct sources, bp)")
    print(f"  model reduction  V({lab}) - V(4x4,k=4) = {(Vc - V_FULL_MODEL) * 1e4:+.5f}   (deterministic: fewer maturities/factors)")
    print(f"  PCA truncation   V(k={k}) - V(k={n_full}) [{name}]  = {(Vc - REF[name][n_full]) * 1e4:+.5f}   (deterministic)")
    print(f"  discretisation   V_h - V                  = {(Vh_main - Vc) * 1e4:+.5f}   (deterministic, O(dt^2))")
    print(f"  sampling         1.96 * SE                = {1.96 * mc['se'] * 1e4:.5f}   (random, O(1/sqrt(M)))")
    print(f"  observed         MC - V_h = {(mc['price'] - Vh_main) * 1e4:+.5f} ;  MC - V = {(mc['price'] - Vc) * 1e4:+.5f}")
    out.update(production=dict(dt=h_main, M=M_MAIN, seed=s0 + SEED_MAIN, price_bp=mc["price"] * 1e4,
                               se_bp=mc["se"] * 1e4, V_bp=Vc * 1e4, V_h_bp=Vh_main * 1e4,
                               model_reduction_bp=(Vc - V_FULL_MODEL) * 1e4,
                               truncation_bp=(Vc - REF[name][n_full]) * 1e4, discretisation_bp=(Vh_main - Vc) * 1e4),
               coverage=cov_rate, slope_M=slope_M, slope_dt=slope_h, sigma_Y_bp=sigma_Y,
               exact_dt=exact_dt.to_dict(orient="records"), cost_equal_precision=cost_rows)

    # ---- figures -------------------------------------------------------------------------------
    # b0_hjm.png - the HJM volatility functions (the random part of the step
    #  f += mu*dt + sum_i sigma_i*sqrt(dt)*Z_i):
    #  sigma_i(tau) = sqrt(lambda_i) u_i: how much shock i moves the forward with time to
    #  maturity tau (dots = PCA values at 3M..1Y, lines = linear interpolation, flat below 3M).
    #  Shaded zone tau <= T1 = 0.5: the only part of the curves the caplet uses.
    #  The drift is NOT plotted: its no-arbitrage property is verified numerically
    #  in 2.1.4 (residuals ~ 0) and 2.2.5 (simulated bonds = P(0,T)).
    x = np.linspace(0, 1.0, 401); cols = [(COL["c1"], "o"), (COL["c2"], "s"), (COL["c3"], "^")]
    with plt.rc_context({"font.size": 11, "axes.titlesize": 11, "legend.fontsize": 9.5}):
        fig, a1 = plt.subplots(1, 1, figsize=(8, 5))
        a1.axvspan(0, T1, color=COL["ref"], alpha=0.10, lw=0)
        a1.text(T1 / 2, 0.97, "used by the caplet\n($\\tau \\leq T_1$ = 0.5)", transform=a1.get_xaxis_transform(),
                ha="center", va="top", fontsize=9.5, color=COL["c1"])
        for i in range(k):
            c, mk = cols[i]
            a1.plot(x, 100 * np.interp(x, P["tau"], sig[i]), color=c, lw=0.9,
                    label=f"$\\sigma_{i + 1}(\\tau) = \\sqrt{{\\lambda_{i + 1}}}\\,u_{i + 1}(\\tau)$")
            a1.plot(P["tau"], 100 * sig[i], mk, color=c, ms=4)
            for tx, sx in zip(P["tau"], sig[i]):
                a1.annotate(f"{100 * sx:.2f}", (tx, 100 * sx), textcoords="offset points", xytext=(4, -13 if i == 2 else 5),
                            fontsize=8.5, color=c)
        a1.axhline(0, color=COL["muted"], lw=0.6); a1.set_xlabel("time to maturity $\\tau$ (years)")
        a1.set_ylabel("volatility (% per $\\sqrt{yr}$)"); a1.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        a1.set_title(f"How much each random shock $Z_i$ moves the forward $f(t, t+\\tau)$\n"
                     f"HJM volatilities from the PCA ({lab}); dots = 3M, 6M, 9M, 1Y")
        a1.legend(loc="lower left")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"b0_hjm{sfx}.png")); plt.close(fig)

    # b0_convergence.png - three panels, read left to right:
    #  (a) MC price for growing M (dt fixed): the 95% bars shrink around V_h
    #  (b) the SIZE of those bars: SE = half-bar / 1.96, on log-log axes; a straight
    #      line of slope -1/2 means "4x paths -> error / 2" (SE ~ 1/sqrt(M))
    #  (c) price for growing dt (M fixed): the exact V_h (line) moves away from V
    #      when the step is coarse; MC points follow V_h, not V
    with plt.rc_context({"font.size": 11, "axes.titlesize": 11, "legend.fontsize": 9.5}):
        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 5.2))
        a1.errorbar(dM.M, dM.price_bp, yerr=1.96 * dM.se_bp, fmt="o", ms=4, color=COL["c1"], capsize=2.5, elinewidth=0.8, capthick=0.8,
                    label="MC price, 95% CI (bar = $\\pm 1.96\\,$SE)")
        a1.axhline(Vh_main * 1e4, color=COL["ref"], ls="--", lw=0.8,
                   label=f"$V_h$ = {Vh_main * 1e4:.4f} exact, $\\Delta t$=1/{round(1 / h_main)}")
        a1.set_xscale("log"); a1.set_xlabel("paths $M$"); a1.set_ylabel("caplet price (bp)")
        a1.set_title(f"(a) More paths $\\to$ narrower bars\n({lab}, $\\Delta t$=1/{round(1 / h_main)} fixed)")
        a1.legend(loc="lower right"); tag(a1, "a")
        a2.loglog(dM.M, dM.se_bp, "o-", ms=4, lw=0.9, color=COL["c1"], label=f"SE measured (fitted slope {slope_M:.2f})")
        a2.loglog(dM.M, dM.se_bp.iloc[0] * np.sqrt(dM.M.iloc[0] / dM.M), "--", lw=0.8, color=COL["muted"],
                  label="theory $\\propto M^{-1/2}$ (slope $-0.5$)")
        for Mx, sx in zip(dM.M, dM.se_bp):
            a2.annotate(f"{sx:.3f}", (Mx, sx), textcoords="offset points", xytext=(8, 6), fontsize=9.5)
        a2.text(0.04, 0.06, "$\\times 4$ paths $\\to$ SE $\\div 2$", transform=a2.transAxes, fontsize=10.5)
        a2.set_xlabel("paths $M$"); a2.set_ylabel("standard error SE (bp)")
        a2.set_title("(b) Size of the bars in (a): sampling error\nSE = std(Y)/$\\sqrt{M}$")
        a2.legend(loc="upper right"); tag(a2, "b")
        a3.plot(exact_dt.h, exact_dt.V_h_bp, "-", lw=0.9, color=COL["c3"], label="$V_h$ exact price of the step-$\\Delta t$ model")
        a3.errorbar(dD.dt, dD.price_bp, yerr=1.96 * dD.se_bp, fmt="s", ms=4, color=COL["c2"], capsize=2.5, elinewidth=0.8, capthick=0.8,
                    label=f"MC, M={M_DT // 1000}k, 95% CI")
        a3.axhline(Vc * 1e4, color=COL["ref"], ls="--", lw=0.8, label=f"$V$ = {Vc * 1e4:.4f} continuous model")
        a3.set_xscale("log"); a3.set_xlabel("time step $\\Delta t$ (years)"); a3.set_ylabel("caplet price (bp)")
        hs = [1 / 252, 1 / 52, 1 / 12, 1 / 4]
        a3.set_xticks(hs); a3.set_xticklabels([f"1/{round(1 / h)}" for h in hs]); a3.minorticks_off()
        a3.set_title(f"(c) Coarser step $\\to$ $V_h$ moves away from $V$\n(discretisation error, M={M_DT // 1000}k fixed)")
        a3.legend(loc="lower left"); tag(a3, "c")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"b0_convergence{sfx}.png")); plt.close(fig)

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 3.5))
    for nm in PCA:
        c, mk = DIM_STYLE[nm]; nn = len(PCA[nm]["eigvals"])
        ks = list(range(1, nn)); a1.semilogy(ks, [abs(REF[nm][kk] - REF[nm][nn]) * 1e4 for kk in ks], "-" + mk, color=c, label=nm)
    a1.set_xticks([1, 2, 3]); a1.set_xlabel("retained factors $k$"); a1.set_ylabel("$|V_k - V_n|$ (bp)")
    a1.set_title("PCA truncation (exact prices)"); a1.legend(); tag(a1, "a")
    a2.loglog(dM.M, 1.96 * dM.se_bp, "o-", color=COL["c1"], label="95% CI half-width")
    a2.loglog(dM.M, np.abs(dM.mc_err_bp), "x", color=COL["c2"], ms=7, label="$|\\hat V_M - V_h|$ realised")
    a2.set_xlabel("paths $M$"); a2.set_ylabel("error (bp)"); a2.set_title(f"Sampling ({lab}, $\\Delta t$=1/{round(1 / h_main)})")
    a2.legend(); tag(a2, "b")
    a3.loglog(exact_dt.h, np.abs(exact_dt.disc_err_bp), "^-", color=COL["c3"], label=f"$|V_h - V|$ (slope {slope_h:.2f})")
    a3.loglog(exact_dt.h, np.abs(exact_dt.disc_err_bp.iloc[0]) * (exact_dt.h / exact_dt.h.iloc[0]) ** 2, "--", color=COL["muted"], label="$\\propto \\Delta t^2$")
    a3.axvline(h_main, color=COL["muted"], lw=0.8, ls=":")
    a3.set_xlabel("time step $\\Delta t$ (years)"); a3.set_ylabel("error (bp)"); a3.set_title(f"Discretisation ({lab})"); a3.legend(); tag(a3, "c")
    fig.suptitle(f"B0 error decomposition, block {tagB} - three independent sources", y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"b0_error_decomposition{sfx}.png")); plt.close(fig)
    return out


BLOCK_RESULTS = {B["name"]: analyse_block(B) for B in BLOCK_DEFS}

# ============================================================================
#  OUT-OF-SAMPLE (held-out 2024): is the calibrated volatility stable?
#  WHY: the price uses volatilities estimated on the past (2022-23). If they
#  changed a lot in 2024, the calibration would be fragile.
#  WHAT: recompute C and PCA on 2024 only and compare:
#    ||C_2024 - C_cal|| / ||C_cal||     overall change of the covariance
#    lambda_2024 / lambda_cal           change of each factor's size
#    1 - |<u_cal, u_2024>|^2            change of each factor's shape (0 = same)
#    price with 2024 vol, SAME initial curve -> effect on the caplet only via vol.
# ============================================================================
header("OUT-OF-SAMPLE CHECK  (2024 never used above)")
oos_rows = []
for name, P in PCA.items():
    Co, _ = sample_covariance(OOS[P["cols"]])
    eigvals_2024, eigvecs_2024 = pca(Co)
    dist = 1 - np.abs(np.sum(P["eigvecs"] * eigvecs_2024, axis=0)) ** 2                # sign-insensitive 1-|<u,u_hat>|^2
    n = len(eigvals_2024)
    Vo_price = black_caplet(P0T0, P0T1, KAPPA,
                            logbond_var_continuous(hjm_vol_factors(eigvals_2024, eigvecs_2024, n), P["tau"], T0, T1))
    oos_rows.append(dict(matrix=name, rel_frob_diff=np.linalg.norm(Co - P["C"]) / np.linalg.norm(P["C"]),
                         R1_cal=P["R"][0], R1_oos=variance_explained(eigvals_2024)[0],
                         V_cal_bp=REF[name][n] * 1e4, V_oos_bp=Vo_price * 1e4))
    print(f"  [{name}] ||C_2024-C_cal||/||C_cal|| = {oos_rows[-1]['rel_frob_diff']:.3f}   "
          f"lambda_2024/lambda_cal = {np.round(eigvals_2024 / P['eigvals'], 3)}   1-|<u,u_2024>|^2 = {np.round(dist, 4)}")
    print(f"          R_1 cal={P['R'][0]:.4f} vs 2024={oos_rows[-1]['R1_oos']:.4f}   "
          f"full-rank price with 2024 vol = {Vo_price * 1e4:.4f} bp vs calibrated {REF[name][n] * 1e4:.4f} bp")
print("  (same initial curve: the difference is due ONLY to the volatility estimated on a different period)")

# ============================================================================
#  OUTPUT TABLES AND REFERENCE FILE
#  b0_results.csv     every MC run with its exact references and error columns
#  b0_reference.json  the numbers later experiments READ (so they never recompute
#                     them differently): contract, matrices C with lambda, u,
#                     prices V(k), production result, and the E6 reference.
#  b0_config.json     every setting + software versions -> reproducibility.
# ============================================================================
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "b0_results.csv"), index=False)
pd.DataFrame(oos_rows).to_csv(os.path.join(OUT, "b0_oos.csv"), index=False)
header("B0 MINIMUM-OUTPUT TABLE  (results/b0/b0_results.csv; prices in bp of notional)")
show = res[["block", "study", "matrix", "k", "dt", "M", "price_bp", "ci_lo_bp", "ci_hi_bp", "time_s",
            "ref_bp", "ref_disc_bp", "abs_err_bp", "rel_err", "var_explained"]].copy()
show["dt"] = show.dt.map(lambda x: f"1/{round(1 / x)}")
print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

header("SUMMARY OF THE PRODUCTION BLOCK")
for nm, b in BLOCK_RESULTS.items():
    p = b["production"]
    print(f"  {nm:8s} {b['matrix']} k={b['k']} dt=1/{round(1 / b['dt_main'])}: V={b['V_bp']:.5f}  V_h={p['V_h_bp']:.5f}  "
          f"MC={p['price_bp']:.4f} +- {1.96 * p['se_bp']:.4f}  | model reduction {p['model_reduction_bp']:+.5f}, "
          f"truncation {p['truncation_bp']:+.5f}, discretisation {p['discretisation_bp']:+.5f}"
          f" | coverage {100 * b['coverage']:.1f}%  slope_M {b['slope_M']:.3f}")
print(f"  E6 reference: V(2x2, k=2) with classical PCA = {REF['2x2'][2] * 1e4:.5f} bp")

# reference file for E1..E6
full = BLOCK_RESULTS["FULL"]
reference = dict(
    contract=dict(T0=T0, delta=DELTA, T1=T1, K=K, kappa=KAPPA, P0T0=P0T0, P0T1=P0T1,
                  valuation_date=str(VAL_DATE.date()), f0_tau=TAU_ALL.tolist(), f0_values=f0_vals.tolist()),
    calibration=dict(start=str(CAL.index.min().date()), end=str(CAL.index.max().date()), dt_hist=DT_HIST,
                     convention="Cov(daily changes of decimal forwards, ddof=1) / dt_hist"),
    matrices={name: dict(maturities=P["cols"], tau=P["tau"].tolist(), C=P["C"].tolist(),
                         eigenvalues=P["eigvals"].tolist(), eigenvectors_columns=P["eigvecs"].tolist(),
                         variance_explained=P["R"].tolist(),
                         price_continuous_bp={str(k): v * 1e4 for k, v in REF[name].items()})
              for name, P in PCA.items()},
    production=dict(matrix=full["matrix"], k=full["k"], **full["production"]),
    qpca_reference=dict(
        note="E6: put lambda, u estimated by qPCA (E2/E4) into the same pricing (2x2, k=2, same curve, "
             "same Black formula) and compare with V_bp, the price with the classical PCA",
        matrix="2x2", k=2, V_bp=REF["2x2"][2] * 1e4),
    blocks={nm: {k_: v for k_, v in b.items() if k_ not in ("exact_dt", "cost_equal_precision")} for nm, b in BLOCK_RESULTS.items()})
json.dump(reference, open(os.path.join(OUT, "b0_reference.json"), "w"), indent=2, default=float)

config = dict(run_date_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
              python=platform.python_version(), platform=platform.platform(),
              numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__,
              matplotlib=matplotlib.__version__, data_sha256=meta["raw_sha256"],
              rng="numpy.random.default_rng (PCG64)", z_dim=Z_DIM, batch=BATCH,
              blocks=[{k_: v for k_, v in B.items()} for B in BLOCK_DEFS],
              seeds=dict(main=SEED_MAIN, per_block="seed0 + (SEED_MAIN | 1000+j conv_M | 2000+j conv_dt | 50000+r coverage"
                                                    " | 7 bonds | 3000+100*i+r cost)"),
              M_main=M_MAIN, dt_main=H_MAIN, M_list=M_LIST, M_dt=M_DT, dt_exact_list=H_EXACT_LIST,
              coverage=dict(R=COV_R, M=COV_M), cost=dict(eps_bp=EPS_LIST, R=R_COST),
              scheme="Euler, discrete no-arbitrage drift (Glasserman 3.6.2), left-point sums for r and P",
              interpolation="linear in maturity, flat extrapolation (curve and volatility)",
              total_runtime_s=round(time.time() - RUN_START, 1))
json.dump(config, open(os.path.join(OUT, "b0_config.json"), "w"), indent=2, default=float)

# ============================================================================
#  FIGURE: PCA (all matrices)
# ============================================================================
# (reconstruction error ||C - C_k||/||C|| is printed in 2.1.3 and in the thesis table; not plotted)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
for name, P in PCA.items():
    c, mk = DIM_STYLE[name]
    a1.plot(range(1, len(P["R"]) + 1), P["R"] * 100, "-" + mk, color=c, label=name)
a1.set_xticks([1, 2, 3, 4]); a1.set_xlabel("factors $k$"); a1.set_ylabel("cumulative variance $R_k$ (%)")
a1.set_title("Variance explained"); a1.legend(); tag(a1, "a")
for i, (c, mk, nm) in enumerate([(COL["c1"], "o", "PC1 level"), (COL["c2"], "s", "PC2 slope"), (COL["c3"], "^", "PC3 curvature")]):
    a2.plot(PCA["4x4"]["tau"], PCA["4x4"]["eigvecs"][:, i], "-" + mk, color=c, label=nm)
a2.axhline(0, color=COL["muted"], lw=0.7); a2.set_xlabel("time to maturity $\\tau$ (years)"); a2.set_ylabel("loading $u_i(\\tau)$")
a2.set_title("Eigenvectors, 4x4 (calibration 2022-23)"); a2.legend(); tag(a2, "b")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "b0_pca.png")); plt.close(fig)

print(f"\nsaved in {os.path.relpath(OUT, os.path.dirname(os.path.abspath(__file__)))}: b0_results.csv, b0_reference.json, b0_config.json, b0_oos.csv, b0_pca.png;")
print("  b0_curve.csv, b0_curve.png, b0_exact_dt.csv, b0_bond_check.csv, b0_cost_equal_precision.csv,")
print("  b0_hjm.png, b0_convergence.png, b0_error_decomposition.png")
print(f"total runtime {time.time() - RUN_START:.0f}s")
