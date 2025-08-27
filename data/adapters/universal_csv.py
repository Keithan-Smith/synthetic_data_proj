# data/adapters/universal_csv.py
import pandas as pd
import numpy as np
import yaml
from typing import Optional, Dict, List

def _load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _apply_column_map(df: pd.DataFrame, cm: Dict[str, str]) -> pd.DataFrame:
    rename = {src: dst for src, dst in cm.items() if src in df.columns and src != dst}
    return df.rename(columns=rename)

def _rename_with_aliases(df: pd.DataFrame, alias_pack: dict) -> pd.DataFrame:
    cols_lower = {c.lower(): c for c in df.columns}

    def find_any(aliases: List[str]) -> Optional[str]:
        for a in aliases:
            if a in df.columns:
                return a
            al = a.lower()
            if al in cols_lower:
                return cols_lower[al]
        return None

    # unify field names
    for canon, aliases in (alias_pack.get("fields") or {}).items():
        if canon not in df.columns:
            src = find_any(aliases)
            if src:
                df = df.rename(columns={src: canon})

    # unify target names (don’t force existence)
    for canon, aliases in (alias_pack.get("targets") or {}).items():
        if canon not in df.columns:
            src = find_any(aliases)
            if src:
                df = df.rename(columns={src: canon})
    return df

def _maybe_parse_dates(df: pd.DataFrame):
    for c in df.columns:
        if any(k in c.lower() for k in ["date", "time", "timestamp", "datetime"]):
            try:
                df[c] = pd.to_datetime(df[c])
            except Exception:
                pass
    return df

def normalize_df(
    df: pd.DataFrame,
    column_map_path: Optional[str] = None,
    alias_pack_path: Optional[str] = None,
    parse_dates: bool = True,
) -> pd.DataFrame:
    """
    Apply optional column mapping YAML and alias pack YAML to a DataFrame that
    is already in memory (e.g., after fetching from ucimlrepo).
    - column_map YAML format: { map: { source_col: target_col, ... } }
    - alias_pack YAML format: { fields: {...}, targets: {...} }
    Also coerces common stringy binary targets to 0/1 and parses date-ish columns.
    """
    out = df.copy()

    if column_map_path:
        cm_yaml = _load_yaml(column_map_path)
        if isinstance(cm_yaml, dict) and "map" in cm_yaml:
            out = _apply_column_map(out, cm_yaml["map"])

    if alias_pack_path:
        alias_pack = _load_yaml(alias_pack_path)
        if isinstance(alias_pack, dict):
            out = _rename_with_aliases(out, alias_pack)
            # coerce any stringy binary targets to 0/1
            for t in (alias_pack.get("targets") or {}).keys():
                if t in out.columns and out[t].dtype == object:
                    sl = out[t].astype(str).str.lower()
                    if sl.isin(["good", "bad"]).any():
                        out[t] = (sl == "bad").astype(int)
                    elif sl.isin(["yes", "no"]).any():
                        out[t] = (sl == "yes").astype(int)
                    elif sl.isin(["true", "false"]).any():
                        out[t] = (sl == "true").astype(int)

    if parse_dates:
        out = _maybe_parse_dates(out)

    return out

def load_any_csv(
    path: str,
    column_map_path: Optional[str] = None,
    alias_pack_path: Optional[str] = None,
    parse_dates: bool = True,
    read_csv_kwargs: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Load a CSV from local disk or URL, then apply the same normalization
    (column map, alias pack, date parsing) as normalize_df.
    """
    read_csv_kwargs = read_csv_kwargs or {}
    df = pd.read_csv(path, **read_csv_kwargs)
    return normalize_df(
        df,
        column_map_path=column_map_path,
        alias_pack_path=alias_pack_path,
        parse_dates=parse_dates,
    )
