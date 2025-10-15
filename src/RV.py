
# Most Mean Reverting Things
# 1. Johansen cointegration test (classical)

# Find stationary combinations of nonstationary series.
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# df: columns are swap maturities
res = coint_johansen(df, det_order=0, k_ar_diff=1)

# eigenvectors (columns) give cointegration vectors (linear combinations)
coint_vectors = res.evec

# The one with the largest eigenvalue is typically most stationary
best_w = coint_vectors[:, 0]
spread = df @ best_w


# 2. Direct mean-reversion optimization (Ornstein–Uhlenbeck fit)

# Optimize weights to minimize the OU process half-life or maximize reversion speed.
# dz_t = θ(μ - z_t)dt + σ dW_t

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS

def ou_speed(series):
    y = series.shift(1).iloc[1:]
    x = (series - series.shift(1)).iloc[1:]
    model = OLS(x, y).fit()
    return -model.params[0]  # θ > 0 means mean-reverting

X = df.values
X = (X - X.mean(0)) / X.std(0)  # normalize

def mean_reversion_objective(w):
    z = X @ w
    return -ou_speed(pd.Series(z))  # negative because we maximize θ

from scipy.optimize import minimize

n = X.shape[1]
w0 = np.ones(n) / n
res = minimize(mean_reversion_objective, w0, method="SLSQP")
best_w = res.x / np.linalg.norm(res.x)

# 3. PCA + Stationarity check

# If you just want some orthogonal linear combos, compute principal components, then check each for mean

from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller

pca = PCA()
pcs = pca.fit_transform(df)
for i, pc in enumerate(pcs.T):
    stat, pval, *_ = adfuller(pc)
    print(f"PC{i+1}: p={pval:.3f}")

# Cost Function

import numpy as np
import pandas as pd
from numpy.linalg import norm
from statsmodels.regression.linear_model import OLS
from scipy.optimize import minimize

def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std(ddof=0)

def _rho_ar1(z: np.ndarray) -> float:
    z = pd.Series(z).dropna()
    y, x = z.iloc[1:].values, z.shift(1).iloc[1:].values
    if len(y) < 10: return 0.999  # guard
    rho = OLS(y, x).fit().params.item()
    return float(np.clip(rho, -0.999, 0.9999))

def _half_life_from_rho(rho: float) -> float:
    # HL = ln(2)/(-ln rho) for 0<rho<1; large penalty otherwise
    return 1e6 if rho >= 0.9999 else np.log(2.0) / (-np.log(max(rho, 1e-6)))

def _l0_smooth(w: np.ndarray, alpha: float = 50.0) -> float:
    # smooth proxy for "number of names" (cardinality)
    return np.sum(1.0 - np.exp(-alpha * np.abs(w)))

def find_mr_portfolio(
    df: pd.DataFrame,
    bid_ask: pd.Series,
   *,
    lambda_l1: float = 0.0,       # cost on |w_i| * spread_i (enter/roll cost)
    lambda_l0: float = 0.0,       # cost on number of legs (sparsity)
    alpha_l0: float = 50.0,       # smoothness of L0 proxy
    neutrality: str = "sum0",     # "sum0", "none"
    scale: str = "var1"           # "var1" => w'Σw = 1, stabilizes problem
):
    """
    df:      T x N levels (e.g., SOFR 1y..30y), index = dates
    bid_ask: length-N Series of bid-ask widths per leg, aligned to df.columns
    """
    X = _standardize(df.dropna()).values
    cols = list(df.columns)
    N = X.shape[1]
    spreads = bid_ask.reindex(cols).fillna(bid_ask.median()).values

    # precompute covariance for scaling
    Sigma = np.cov(X.T)

    def make_spread(w): return X @ w
    def mean_reversion_metric(w):
        z = make_spread(w)
        rho = _rho_ar1(z)
        return _half_life_from_rho(rho)  # lower is better

    def trading_cost(w):
        # static entry/roll cost proportional to |w_i| * spread_i (L1)
        l1 = np.sum(np.abs(w) * spreads)
        # model "number of legs" with smooth L0
        l0 = _l0_smooth(w, alpha=alpha_l0)
        return lambda_l1 * l1 + lambda_l0 * l0

    def objective(w):
        return mean_reversion_metric(w) + trading_cost(w)

    # constraints
    cons = []
    if scale == "var1":
        cons.append({"type": "eq", "fun": lambda w: w @ Sigma @ w - 1.0})
    if neutrality == "sum0":
        cons.append({"type": "eq", "fun": lambda w: np.sum(w)})

    # bounds optional (e.g., allow both long/short)
    bnds = [(-5.0, 5.0)] * N

    # good starting point: PCA-ish or simple curve butterfly
    w0 = np.zeros(N)
    w0[0], w0[-1] = 0.5, -0.5  # steepener/flattening seed
    # project to var=1 if needed
    if scale == "var1":
        s = np.sqrt(w0 @ Sigma @ w0)
        w0 = w0 / (s if s > 1e-12 else 1.0)

    res = minimize(objective, w0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter": 1000})
    w = res.x

    out = pd.Series(w, index=cols, name="weight")
    hl = mean_reversion_metric(w)
    cost = trading_cost(w)
    return out, {"half_life": hl, "cost": cost, "success": res.success, "message": res.message}
