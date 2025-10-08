from typing import List, Tuple, Callable, Union, Optional
import math
import numpy as np
import pandas as pd

Num = float
DateLike = Union[pd.Timestamp, Num]
Interval = Tuple[DateLike, DateLike]

# --- helpers ---------------------------------------------------------------

def _to_years(x: DateLike, base_date: pd.Timestamp | None) -> Num:
    if isinstance(x, (int, float)):
        return float(x)
    if not isinstance(x, pd.Timestamp):
        raise TypeError("Use floats (years) or pandas.Timestamp. If using dates, pass base_date.")
    if base_date is None:
        raise ValueError("base_date is required when using dates.")
    return (x - base_date).days / 365.0

def _normalize_inputs(
    y: Interval, xs: List[Interval], base_date: pd.Timestamp | None
) -> tuple[tuple[Num, Num], list[tuple[Num, Num]]]:
    Sy, Ey = _to_years(y[0], base_date), _to_years(y[1], base_date)
    xy = [( _to_years(S, base_date), _to_years(E, base_date)) for (S, E) in xs]
    return (Sy, Ey), xy

def _beta_linear(T: Num, pillars: List[Num]) -> np.ndarray:
    """Linear-in-maturity weights across sorted pillars."""
    P = np.array(sorted(set(pillars)), dtype=float)
    m = len(P)
    w = np.zeros(m)
    if T <= 0 or m == 0: 
        return w
    if m == 1:
        w[0] = min(max(T / P[0], 0.0), 1.0)
        return w
    if T <= P[0]:
        w[0] = T / P[0]
        return w
    if T >= P[-1]:
        w[-1] = 1.0
        return w
    k = np.searchsorted(P, T, side="right") - 1
    k = max(0, min(k, m - 2))
    lam = (T - P[k]) / (P[k+1] - P[k]) if P[k+1] > P[k] else 0.0
    w[k], w[k+1] = 1 - lam, lam
    return w

def risk_map_bucket(
    y: Interval,
    xs: List[Interval],
    *,
    base_date: pd.Timestamp | None = None,
    restrict_to_min_bucket: bool = True
) -> List[Num]:
    """
    Uniform risk density on y=[Sy,Ey]; allocate only to the smallest forward buckets in xs.
    dy/dx_i = |y ∩ x_i| / |y| for smallest buckets; 0 for larger (to avoid double counting).
    Inputs can be floats (years) or pandas Timestamps (with base_date provided).
    """
    (Sy, Ey), xy = _normalize_inputs(y, xs, base_date)
    Ly = Ey - Sy
    if Ly <= 0:
        return [0.0] * len(xy)

    lengths = [Ei - Si for (Si, Ei) in xy]
    if not any(l > 0 for l in lengths):
        return [0.0] * len(xy)
    min_len = min(l for l in lengths if l > 0)

    out: List[Num] = []
    for (Si, Ei), li in zip(xy, lengths):
        if restrict_to_min_bucket and (li - min_len) > 1e-12:
            out.append(0.0)
            continue
        overlap = max(0.0, min(Ey, Ei) - max(Sy, Si))
        out.append(overlap / Ly)
    return out

def risk_map_par(
    y: Interval,
    xs: List[Interval],
    *,
    base_date: pd.Timestamp | None = None,
    annuity: Callable[[Num], Num] = lambda T: T,   # plug real A(T) if available
    regularization: float | None = None            # ridge lambda if C is ill-conditioned
) -> List[Num]:
    """
    Curve-consistent mapping using par-swap logic.
    Build a pillar space from END dates of xs. Represent each instrument (y, x_i) as a
    normalized annuity-weighted vector in pillar space, then solve C * alpha = v(y).

    v(z) = [ beta(Ez)*A(Ez) - beta(Sz)*A(Sz) ] / (A(Ez) - A(Sz)), with beta(.) linear in maturity.
    Returns alpha_i = dy/dx_i for the provided xs (works for par swaps and forwards alike).
    """
    (Sy, Ey), xy = _normalize_inputs(y, xs, base_date)

    # pillars are unique END dates of xs
    pillars = sorted(set(E for (_, E) in xy))
    m = len(pillars)
    if m == 0:
        return [0.0] * len(xy)

    def v_vec(S: Num, E: Num) -> np.ndarray:
        AE, AS = annuity(max(E, 0.0)), annuity(max(S, 0.0))
        denom = AE - AS
        if denom <= 0:
            return np.zeros(m)
        bE = _beta_linear(E, pillars)
        bS = _beta_linear(S, pillars)
        return (bE * AE - bS * AS) / denom

    Vy = v_vec(Sy, Ey)
    C = np.column_stack([v_vec(Si, Ei) for (Si, Ei) in xy])  # m x n

    # Solve C alpha = Vy (LS or ridge)
    if regularization and regularization > 0:
        # ridge: (C^T C + λI) α = C^T Vy
        CtC = C.T @ C
        n = CtC.shape[0]
        alpha = np.linalg.solve(CtC + regularization * np.eye(n), C.T @ Vy)
    else:
        alpha, *_ = np.linalg.lstsq(C, Vy, rcond=None)

    return [float(a) for a in alpha]


def map_y_onto_x_from_dates(
    y_start_date,
    y_end_date,
    x_start_dates: List,
    x_end_dates: List,
    *,
    method: str = "par",                    # "bucket", "par", or "both"
    base_date: Optional[pd.Timestamp] = None,
    restrict_to_min_bucket: bool = True,
    annuity = lambda T: T,
    regularization: float | None = None,
):
    """
    Build intervals from datetime-like inputs and apply selected mapping(s).

    Returns
    -------
    If method in {"bucket","par"}:
        df, weights
    If method == "both":
        df, {"bucket": bucket_weights, "par": par_weights}
    """
    # coerce to pandas Timestamps
    yS = pd.to_datetime(y_start_date)
    yE = pd.to_datetime(y_end_date)
    xS = [pd.to_datetime(d) for d in x_start_dates]
    xE = [pd.to_datetime(d) for d in x_end_dates]

    if len(xS) != len(xE):
        raise ValueError("x_start_dates and x_end_dates must be the same length.")
    if any(e < s for s, e in zip(xS, xE)):
        raise ValueError("Each x interval must satisfy end >= start.")
    if yE <= yS:
        raise ValueError("y_end_date must be after y_start_date.")

    if method not in {"bucket", "par", "both"}:
        raise ValueError("method must be 'bucket', 'par', or 'both'.")

    # default base_date: earliest start so year-fractions are non-negative
    if base_date is None:
        base_date = min([yS] + xS)

    y = (yS, yE)
    xs: List[Tuple[pd.Timestamp, pd.Timestamp]] = list(zip(xS, xE))
    idx = [f"x{i+1}" for i in range(len(xs))]

    # compute selected weights
    w = None
    if method in {"bucket"}:
        w = risk_map_bucket(
            y, xs, base_date=base_date, restrict_to_min_bucket=restrict_to_min_bucket
        )
    if method in {"par"}:
        w = risk_map_par(
            y, xs, base_date=base_date, annuity=annuity, regularization=regularization
        )

    # assemble DataFrame
    data = {"Start": xS, "End": xE}
    data["dy/dx"] = w
    
    df = pd.DataFrame(data, index=idx)

    return df

def map_y_onto_x_labeled(
    y_group: str,
    y_name: str,
    y_start_date,
    y_end_date,
    x_groups: List[str],
    x_names: List[str],
    x_start_dates: List,
    x_end_dates: List,
    *,
    method: str = "par",
    base_date: Optional[pd.Timestamp] = None,
    restrict_to_min_bucket: bool = True,
    annuity = lambda T: T,
    regularization: float | None = None,
):
    """Adds labels and aggregates by group. Returns (df_instruments, group_weights)."""
    n = len(x_start_dates)
    if not (len(x_end_dates) == len(x_groups) == len(x_names) == n):
        raise ValueError("x_* lists must all be the same length.")

    # run your existing logic (dates -> dy/dx)
    df = map_y_onto_x_from_dates(
        y_start_date, y_end_date,
        x_start_dates, x_end_dates,
        method=method,
        base_date=base_date,
        restrict_to_min_bucket=restrict_to_min_bucket,
        annuity=annuity,
        regularization=regularization,
    )

    # attach labels and reindex by (group, name)
    df.insert(0, "x_group", x_groups)
    df.insert(1, "x_name",  x_names)
    df.insert(2, "y_group", y_group)
    df.insert(3, "y_name",  y_name)
    df = df.set_index(["x_group", "x_name"]).sort_index()  # rows are instruments (group,name)

    # aggregate to group level (how each group changes)
    group_weights = df["dy/dx"].groupby(level=0).sum().rename("dy/dGroup")

    return df, group_weights

y_group = "A"; y_name = "2y"
y_start_date = "2025-01-01"; y_end_date = "2027-01-01"

x_groups = ["A","A","A","B","B","B","C","C","C"]
x_names  = ["1y","2y","3y","1y","2y","3y","1y","2y","3y"]

x_start_dates = [
    "2025-01-01","2025-01-01","2025-01-01",
    "2025-01-01","2025-01-01","2025-01-01",
    "2025-01-01","2025-01-01","2025-01-01",
]
x_end_dates = [
    "2026-01-01","2027-01-01","2028-01-01",  # A 1y/2y/3y
    "2026-01-01","2027-01-01","2028-01-01",  # B 1y/2y/3y
    "2026-01-01","2027-01-01","2028-01-01",  # C 1y/2y/3y
]

df_inst, group_summary = map_y_onto_x_labeled(
    y_group, y_name,
    y_start_date, y_end_date,
    x_groups, x_names,
    x_start_dates, x_end_dates,
    method="bucket"  # or "par" if you pass a real annuity()
)

# peek
print(df_inst)
print(group_summary)
