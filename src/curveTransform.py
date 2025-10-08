import pandas as pd
import numpy as np

def build_transform_df(src_group, src_tenor, *triples):
    """
    triples: any number of (t_group, t_tenor, weights), each length == len(src_group)
    Returns:
      W_df: DataFrame (index=target (group,tenor), columns=source (group,tenor))
    """
    n = len(src_group)
    assert len(src_tenor) == n

    # Long table: one row per (source point, triple)
    frames = []
    for (tg, tt, tw) in triples:
        assert len(tg) == len(tt) == len(tw) == n
        frames.append(pd.DataFrame({
            "s_group": src_group,
            "s_tenor": src_tenor,
            "t_group": tg,
            "t_tenor": tt,
            "w": pd.to_numeric(tw, errors="coerce"),
        }))

    T = pd.concat(frames, ignore_index=True)

    # Clean blanks/None/zeros
    T = T.replace({"": pd.NA, " ": pd.NA})
    T = T.dropna(subset=["t_group", "t_tenor", "w"])
    T = T[T["w"] != 0]

    # Keys (preserve target first-appearance order)
    T["s_key"] = list(zip(T["s_group"], T["s_tenor"]))
    T["t_key"] = list(zip(T["t_group"], T["t_tenor"]))
    t_order = pd.Series(pd.unique(T["t_key"]), name="t_key")
    s_order = list(zip(src_group, src_tenor))  # original order

    # Pivot -> dense transform matrix (sum in case of duplicates)
    W_df = T.pivot_table(index="t_key", columns="s_key", values="w", aggfunc="sum", fill_value=0)
    W_df = W_df.reindex(t_order.values)            # preserve target order of first appearance
    W_df = W_df.reindex(columns=s_order)           # preserve source order

    # Pretty MultiIndex labels
    W_df.index = pd.MultiIndex.from_tuples(W_df.index, names=["group", "tenor"])
    W_df.columns = pd.MultiIndex.from_tuples(W_df.columns, names=["group", "tenor"])
    return W_df

# --- Example with your data ---
InstrumentGroup = ["A","A","A","B","B","B","C","C","C"]
Instrument      = ["1y","2y","3y","1y","2y","3y","1y","2y","3y"]

InstrumentTransGroup1  = ["A","A","A","B","B","B","D","D","D"]
InstrumentTrans1       = ["1y","2y","3y","1y","2y","3y","1y","2y","3y"]
InstrumentTransWeight1 = [1,1,1,1,1,1,1,1,1]

InstrumentTransGroup2  = ["B","B","B","D","D","D","","",""]
InstrumentTrans2       = ["1y","2y","3y","1y","2y","3y","","",""]
InstrumentTransWeight2 = [-1,-1,-1,-1,-1,-1,0,0,0]

W_df = build_transform_df(
    InstrumentGroup, Instrument,
    (InstrumentTransGroup1, InstrumentTrans1, InstrumentTransWeight1),
    (InstrumentTransGroup2, InstrumentTrans2, InstrumentTransWeight2),
)

# Apply the transform: y = W @ x
# x: source vector aligned to columns of W_df (A,B,C x 1y/2y/3y)
x = pd.Series(np.arange(len(W_df.columns), dtype=float), index=W_df.columns)
y = W_df.dot(x)  # pandas aligns by column labels

# If you need raw numpy arrays:
W = W_df.to_numpy()
x_np = x.to_numpy()
y_np = W @ x_np


print(W_df)
print(x)
print(y)
