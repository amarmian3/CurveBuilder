# monte_carlo_8y1y_from_flies.py
from datetime import datetime as dt
import numpy as np
from rateslib import Curve, IRS, Fly, Solver

# --------- basics ---------
VAL_DATE = dt(2025, 10, 15)
SPEC = "eur_irs"  # change if needed ("usd_irs", etc.)

def make_curve():
    # Seed DF nodes; Solver will reshape them
    return Curve(
        nodes={
            VAL_DATE: 1.0,
            dt(2027, 1, 1): 0.97,
            dt(2030, 1, 1): 0.90,
            dt(2035, 1, 1): 0.80,
        },
        id="zc"
    )

# --------- instruments ---------
def irs_spot(tenor_y):
    # Market par swap starting spot; keep 8y/9y for flies only
    return IRS(termination=f"{tenor_y}y", spec=SPEC)

# Outright tenors (exclude 8y & 9y because they come from flies)
outright_tenors = [1,2,3,4,5,6,7,10,12,15,20,30]
outrights = [irs_spot(t) for t in outright_tenors]

# Build the 3 surrounding IRS needed for each fly
irs7  = irs_spot(7)
irs8  = irs_spot(8)
irs9  = irs_spot(9)
irs10 = irs_spot(10)

fly_789  = Fly(irs7, irs8, irs9)     # 7s8s9s fly (in bps)
fly_8910 = Fly(irs8, irs9, irs10)    # 8s9s10s fly (in bps)

instruments = outrights + [fly_789, fly_8910]

# --------- market quotes ---------
# IMPORTANT: use DECIMALS for outright swap rates; use BPS for flies
s_outrights = [  # example dummy quotes (par swap rates, decimals)
    0.0310, 0.0314, 0.0319, 0.0325, 0.0332, 0.0338, 0.0345,
    0.0362, 0.0372, 0.0386, 0.0398, 0.0407
]
s_fly_789_bp  = +0.30   # bps
s_fly_8910_bp = -0.20   # bps

s_targets = s_outrights + [s_fly_789_bp, s_fly_8910_bp]

# --------- solve base curve ---------
zc = make_curve()
curves4 = [zc, zc, zc, zc]  # single-curve setup: [proj, disc, proj, disc]
Solver(curves=curves4, instruments=instruments, s=s_targets)

# Price the 8y1y forward (starts at 8y, matures at 9y)
fwd_8y1y = IRS(effective="8y", termination="9y", spec=SPEC)
par_8y1y_base = fwd_8y1y.rate(curves=curves4)  # in DECIMAL

print(f"Base 8y1y par rate: {par_8y1y_base*100:.4f}%")

# --------- Monte Carlo on flies (±0.1bp uniform jitter) ---------
def mc_8y1y_from_fly_jitter(n=5000, jitter_bp=0.1, seed=42):
    rng = np.random.default_rng(seed)
    base_out = np.array(s_outrights, dtype=float)
    out = np.empty(n, dtype=float)

    for i in range(n):
        f1 = s_fly_789_bp  + rng.uniform(-jitter_bp, jitter_bp)
        f2 = s_fly_8910_bp + rng.uniform(-jitter_bp, jitter_bp)

        zc_i = make_curve()
        curves4_i = [zc_i, zc_i, zc_i, zc_i]

        Solver(
            curves=curves4_i,
            instruments=instruments,
            s=list(base_out) + [f1, f2]
        )

        out[i] = fwd_8y1y.rate(curves=curves4_i)  # decimal
    return out

samples = mc_8y1y_from_fly_jitter(n=5000, jitter_bp=0.1, seed=1)

mean = samples.mean()
p05, p50, p95 = np.percentile(samples, [5, 50, 95])

print(f"MC mean: {mean*100:.4f}%")
print(f"MC 5/50/95 pct: {p05*100:.4f}% / {p50*100:.4f}% / {p95*100:.4f}%")
print(f"MC stdev: {samples.std(ddof=1)*1e4:.4f} bps")  # show stdev in bps
