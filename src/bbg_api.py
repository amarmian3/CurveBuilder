# -*- coding: utf-8 -*-
from __future__ import annotations
import threading
from datetime import datetime, date
from typing import Any, Dict, Iterable, Optional, Sequence, Union
import pandas as pd

# --- PyXLL (safe if not installed) ---
try:
    from pyxll import xl_func
except Exception:  # pragma: no cover
    def xl_func(*_a, **_k):
        def _wrap(f): return f
        return _wrap

# --- Bloomberg (blp) ---
from blp import blp  # pip/conda: 'blp' (next iteration of pdblp)

_BQ_LOCK = threading.Lock()
_BQ: Optional[blp.BlpQuery] = None

def _get_bq() -> blp.BlpQuery:
    """Singleton BlpQuery().start()."""
    global _BQ
    with _BQ_LOCK:
        if _BQ is None:
            _BQ = blp.BlpQuery().start()
        return _BQ

def _as_list(x: Union[str, Iterable[str]]) -> list[str]:
    if x is None:
        return []
    return [x] if isinstance(x, str) else list(x)

def _as_yyyymmdd(d: Union[str, date, datetime]) -> str:
    if isinstance(d, (date, datetime)):
        return d.strftime("%Y%m%d")
    s = str(d).strip()
    if "-" in s:
        return datetime.strptime(s, "%Y-%m-%d").strftime("%Y%m%d")
    if len(s) == 8 and s.isdigit():
        return s
    raise ValueError("Use YYYY-MM-DD, YYYYMMDD, date, or datetime")

def _as_iso(dt: Union[str, datetime]) -> str:
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d"):
            try:
                return datetime.strptime(str(dt), fmt).strftime("%Y-%m-%dT%H:%M:%S")
            except ValueError:
                pass
    except Exception:
        pass
    raise ValueError("Datetime must parse like 'YYYY-MM-DD HH:MM[:SS]' or ISO8601")

# -----------------------------
# BDH: Historical data
# -----------------------------
@xl_func("str[] tickers, str[] fields, var start_date, var end_date=None, var overrides=None, dict options=None: dataframe", auto_resize=True)
def pyxll_bbg_bdh(
    tickers: Union[str, Sequence[str]],
    fields: Union[str, Sequence[str]],
    start_date: Union[str, date, datetime],
    end_date: Optional[Union[str, date, datetime]] = None,
    overrides: Optional[Sequence[tuple[str, Any]]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Bloomberg BDH -> DataFrame indexed by date with MultiIndex (ticker, field) columns.

    Python:
    >>> pyxll_bbg_bdh(["AAPL US Equity","MSFT US Equity"], ["PX_LAST","VOLUME"], "2024-01-01", "2024-06-30",
    ...               overrides=[("CURRENCY","USD")], options={"adjustmentSplit": True})

    Excel (PyXLL):
    =pyxll_bbg_bdh({"AAPL US Equity","MSFT US Equity"},{"PX_LAST","VOLUME"},"2024-01-01","2024-06-30",
                   {{"CURRENCY","USD"}},{"adjustmentSplit":TRUE})
    """
    bq = _get_bq()
    tks, flds = _as_list(tickers), _as_list(fields)
    start, end = _as_yyyymmdd(start_date), _as_yyyymmdd(end_date or date.today())
    df = bq.bdh(tks, flds, start, end, overrides=overrides or [], options=options or {})
    # returned columns: ['date','security', *flds] -> pivot to (ticker, field)
    out = df.pivot(index="date", columns="security")[flds]
    out.columns = pd.MultiIndex.from_product([tks, flds]) if isinstance(out.columns, pd.MultiIndex) else out.columns
    return out.sort_index(axis=1)

# -----------------------------
# BDP: Reference datapoint(s)
# -----------------------------
@xl_func("str[] tickers, str[] fields, var overrides=None, dict options=None: dataframe", auto_resize=True)
def pyxll_bbg_bdp(
    tickers: Union[str, Sequence[str]],
    fields: Union[str, Sequence[str]],
    overrides: Optional[Sequence[tuple[str, Any]]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Bloomberg BDP -> DataFrame indexed by security with field columns.

    Python:
    >>> pyxll_bbg_bdp(["AAPL US Equity","MSFT US Equity"], ["PX_LAST","CUR_MKT_CAP"],
    ...               overrides=[("EQY_FUND_CRNCY","USD")])

    Excel:
    =pyxll_bbg_bdp({"AAPL US Equity","MSFT US Equity"},{"PX_LAST","CUR_MKT_CAP"},{ {"EQY_FUND_CRNCY","USD"} })
    """
    bq = _get_bq()
    tks, flds = _as_list(tickers), _as_list(fields)
    df = bq.bdp(tks, flds, overrides=overrides or [], options=options or {})
    df = df.set_index("security").reindex(tks)
    return df[flds]

# -----------------------------
# BDID: Intraday (bars or ticks)
# -----------------------------
@xl_func("str ticker, str|str[] event_type, var start_dt, var end_dt, int interval=0, var overrides=None, dict options=None: dataframe", auto_resize=True)
def pyxll_bbg_bdid(
    ticker: str,
    event_type: Union[str, Sequence[str]],
    start_dt: Union[str, datetime],
    end_dt: Union[str, datetime],
    interval: int = 0,
    overrides: Optional[Sequence[tuple[str, Any]]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Intraday data:
      - interval > 0  -> BDIB (bars)
      - interval == 0 -> BDIT (ticks)

    Python (bars, 5m):
    >>> pyxll_bbg_bdid("AAPL US Equity","TRADE","2024-06-03 14:30","2024-06-03 20:00",5)

    Python (ticks):
    >>> pyxll_bbg_bdid("AAPL US Equity",["TRADE"],"2024-06-03 14:30","2024-06-03 15:30",0,
    ...                options={"includeConditionCodes": True})

    Excel (bars):
    =pyxll_bbg_bdid("AAPL US Equity","TRADE","2024-06-03 14:30","2024-06-03 20:00",5)

    Excel (ticks):
    =pyxll_bbg_bdid("AAPL US Equity",{"TRADE"},"2024-06-03 14:30","2024-06-03 15:30",0,,
                    {"includeConditionCodes":TRUE})
    """
    bq = _get_bq()
    start, end = _as_iso(start_dt), _as_iso(end_dt)
    ovs, opts = overrides or [], options or {}

    if interval and interval > 0:
        df = bq.bdib(ticker, event_type=str(event_type), interval=interval,
                     start_datetime=start, end_datetime=end,
                     overrides=ovs, options=opts)
        return df.set_index("time").sort_index()
    else:
        evts = _as_list(event_type)
        df = bq.bdit(ticker, evts, start, end, overrides=ovs, options=opts)
        return df.set_index("time").sort_index()

# -----------------------------
# BQL: Bloomberg Query Language
# -----------------------------
@xl_func("str query, var overrides=None, dict options=None: dataframe", auto_resize=True)
def pyxll_bbg_bql(
    query: str,
    overrides: Optional[Sequence[tuple[str, Any]]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    BQL passthrough.

    Python:
    >>> pyxll_bbg_bql('get(px_last) for(["AAPL US Equity","MSFT US Equity"])')

    Excel:
    =pyxll_bbg_bql("get(px_last) for([""AAPL US Equity"",""MSFT US Equity""])")
    """
    bq = _get_bq()
    return bq.bql(expression=query, overrides=overrides or [], options=options or {})

# -----------------------------
# Optional: disconnect
# -----------------------------
@xl_func("void")
def pyxll_bbg_disconnect():
    """Stop the shared Bloomberg session."""
    global _BQ
    with _BQ_LOCK:
        if _BQ is not None:
            try:
                _BQ.stop()
            finally:
                _BQ = None
