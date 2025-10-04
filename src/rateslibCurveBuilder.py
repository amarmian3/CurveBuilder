from rateslib import Curve, IRS, Solver
from datetime import datetime as dt

# 1) Define a DF curve (nodes are Duals once solved)
usd = Curve(
    nodes={dt(2025,1,1):1.0, dt(2025,7,1):0.99, dt(2026,1,1):0.975},
    id="sofr"
)

# 2) Choose the “risk curve” instruments (pillars)
pillars = [
    IRS(dt(2025,1,1), "6M", spec="usd_irs", curves="sofr"),
    IRS(dt(2025,1,1), "1Y", spec="usd_irs", curves="sofr"),
    IRS(dt(2025,1,1), "2Y", spec="usd_irs", curves="sofr"),
]

# 3) Calibrate & store the mapping (s are par rates in %)
solver = Solver(curves=[usd],
                instruments=pillars,
                s=[4.00, 4.20, 4.10],
                instrument_labels=["6M","1Y","2Y"])

# 4) Any swap’s “risk curve” w.r.t. those pillars
my_swap = IRS(dt(2025,1,1), "18M", spec="usd_irs", curves="sofr")
risk_df = my_swap.delta(solver=solver)   # DataFrame of dv01-like deltas per pillar
print(risk_df)
