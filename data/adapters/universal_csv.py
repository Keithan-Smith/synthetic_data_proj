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

    for canon, aliases in (alias_pack.get("fields") or {}).items():
        if canon not in df.columns:
            src = find_any(aliases)
            if src:
                df = df.rename(columns={src: canon})

    for canon, aliases in (alias_pack.get("targets") or {}).items():
        if canon not in df.columns:
            src = find_any(aliases)
            if src:
                df = df.rename(columns={src: canon})
    return df

def _apply_enum_map(df: pd.DataFrame, enum_map: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    for col, mapping in (enum_map or {}).items():
        if col in df.columns:
            s = df[col].astype(str).str.strip()
            # normalize keys
            m = {str(k).strip(): v for k, v in (mapping or {}).items()}
            m.update({str(k).strip().lower(): v for k, v in (mapping or {}).items()})
            mapped = s.map(m)
            mask = mapped.isna()
            mapped.loc[mask] = s.loc[mask].str.lower().map(m)
            mapped = mapped.fillna(df[col])
            df[col] = pd.Categorical(mapped)
    return df

def _maybe_parse_dates(df: pd.DataFrame):
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["date", "time", "timestamp", "datetime"]):
            try:
                df[c] = pd.to_datetime(df[c])
            except Exception:
                pass
    return df

def normalize_df(df: pd.DataFrame,
                 column_map_path: Optional[str] = None,
                 alias_pack_path: Optional[str] = None,
                 parse_dates: bool = True) -> pd.DataFrame:
    # 1) optional flat column map (explicit rename)
    if column_map_path:
        cm_yaml = _load_yaml(column_map_path)
        if isinstance(cm_yaml, dict) and "map" in cm_yaml:
            df = _apply_column_map(df, cm_yaml["map"])

    # 2) alias pack: field/target renames + enum mapping + target normalization
    alias_pack = None
    if alias_pack_path:
        alias_pack = _load_yaml(alias_pack_path)
        if isinstance(alias_pack, dict):
            # (a) rename columns by aliases (fields + targets)
            df = _rename_with_aliases(df, alias_pack)

            # (b) categorical code -> human label mapping
            if "enum_map" in alias_pack and isinstance(alias_pack["enum_map"], dict):
                df = _apply_enum_map(df, alias_pack["enum_map"])

            # (c) normalize common binary targets to {0,1}
            for t in (alias_pack.get("targets") or {}).keys():
                if t in df.columns:
                    s = df[t]

                    # string targets
                    if s.dtype == object:
                        sl = s.astype(str).str.strip().str.lower()
                        u = set(sl.dropna().unique())
                        if u <= {"good", "bad"}:
                            df[t] = (sl == "bad").astype("Int8")
                            continue
                        if u <= {"yes", "no"}:
                            df[t] = (sl == "yes").astype("Int8")
                            continue
                        if u <= {"true", "false"}:
                            df[t] = (sl == "true").astype("Int8")
                            continue

                    # numeric {1,2} → {0,1}; keep {0,1} as-is; preserve NA with nullable ints
                    s_num = pd.to_numeric(s, errors="coerce")
                    vals = set(pd.unique(s_num.dropna()))
                    if vals <= {1, 2}:
                        df[t] = s_num.map({1: 0, 2: 1}).astype("Int8")
                    elif vals <= {0, 1}:
                        df[t] = s_num.astype("Int8")

    # 3) auto-parse date-like columns
    if parse_dates:
        df = _maybe_parse_dates(df)

    return df

def load_any_csv(path: str, column_map_path: Optional[str] = None,
                 alias_pack_path: Optional[str] = None,
                 parse_dates: bool = True,
                 read_csv_kwargs: Optional[dict] = None) -> pd.DataFrame:
    read_csv_kwargs = {"low_memory": False, **(read_csv_kwargs or {})}
    df = pd.read_csv(path, **read_csv_kwargs)
    return normalize_df(df, column_map_path, alias_pack_path, parse_dates)
