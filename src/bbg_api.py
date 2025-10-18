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


import pandas as pd
from typing import Sequence, Union, Optional

def get_intraday(
    con,                                   # pdblp.BCon (already start()'ed)
    tickers: Union[str, Sequence[str]],
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: int = 0,                     # 0 -> tick (bdit), >0 -> bars (bdib)
    event_type: str = "TRADE",
    tz: Optional[str] = None,              # e.g. "Europe/London"
    fields: Optional[Sequence[str]] = None,# subset of fields to keep
    long: bool = False                     # return long/“tidy” instead of wide
) -> pd.DataFrame:
    """
    Returns:
      - Wide: index=DatetimeIndex, columns=MultiIndex[(ticker, field)]
      - Long (if long=True): columns=['time','ticker','field','value']
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    out = []
    for tk in tickers:
        if interval == 0:
            # ---- Ticks
            df = con.bdit(tk, start, end, eventType=event_type)
            # Expected columns often: ['time','type','value','size'] (pdblp)
            if 'time' not in df.columns:
                raise ValueError(f"Unexpected tick schema for {tk}: {df.columns.tolist()}")
            df = df.set_index('time')
            # Standardize names; keep only known numeric fields
            rename_map = {'value': 'PRICE', 'size': 'SIZE'}
            df = df.rename(columns=rename_map)
            keep = [c for c in ['PRICE', 'SIZE'] if c in df.columns]
            if fields is not None:
                keep = [c for c in keep if c in set(fields)]
            df = df[keep]
        else:
            # ---- Bars
            # pdblp typically accepts datetimes directly
            df = con.bdib(tk, start, end, interval, eventType=event_type)
            # Expected columns include: open, high, low, close, volume, ...
            if 'time' not in df.columns and 'datetime' in df.columns:
                df = df.rename(columns={'datetime': 'time'})
            if 'time' not in df.columns:
                raise ValueError(f"Unexpected bar schema for {tk}: {df.columns.tolist()}")
            df = df.set_index('time')
            rename_map = {
                'open': 'OPEN', 'high': 'HIGH', 'low': 'LOW', 'close': 'CLOSE',
                'volume': 'VOLUME', 'numEvents': 'NUM_EVENTS', 'value': 'VALUE'
            }
            df = df.rename(columns=rename_map)
            keep = [c for c in ['OPEN','HIGH','LOW','CLOSE','VOLUME','NUM_EVENTS','VALUE'] if c in df.columns]
            if fields is not None:
                keep = [c for c in keep if c in set(fields)]
            df = df[keep]

        # Timezone handling
        if tz is not None:
            if df.index.tz is None:
                df.index = df.index.tz_localize(tz)  # assume provided times are in tz
            else:
                df.index = df.index.tz_convert(tz)

        # Add ticker level
        df.columns = pd.MultiIndex.from_product([[tk], df.columns], names=['ticker','field'])
        out.append(df)

    # Align on time; outer join keeps all timestamps across tickers
    wide = pd.concat(out, axis=1).sort_index()

    if long:
        long_df = (
            wide.stack('ticker')
                .stack('field')
                .rename('value')
                .reset_index()
                .rename(columns={'level_0':'time'})
        )
        return long_df

    return wide
