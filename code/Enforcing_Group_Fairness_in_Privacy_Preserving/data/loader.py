# data/loader.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


@dataclass
class DatasetBundle:
    X_num: np.ndarray   # [N, d_num] float
    X_cat: np.ndarray   # [N, d_cat] int (label-encoded)
    y: np.ndarray       # [N] int {0,1}
    g: np.ndarray       # [N] int {0,1} (unpriv=0, priv=1)


# ----------------------------
# Helpers
# ----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def _read_csv_robust(path: str) -> pd.DataFrame:
    """
    Read CSV robustly:
    - Try comma first
    - If only 1 column -> try semicolon
    - Strip column names
    """
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=";")
    df.columns = [c.strip() for c in df.columns]
    return df


def scale_num(X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return X.astype(float)
    scaler = StandardScaler()
    return scaler.fit_transform(X.astype(float))


def encode_cat(df_cat: pd.DataFrame) -> np.ndarray:
    if df_cat.shape[1] == 0:
        return np.empty((len(df_cat), 0), dtype=int)

    cols = []
    for c in df_cat.columns:
        le = LabelEncoder()
        vals = df_cat[c].astype(str).fillna("NA").str.strip()
        cols.append(le.fit_transform(vals).astype(int))
    return np.vstack(cols).T


def split_num_cat(df_features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numeric columns: pandas number dtypes
    Categorical columns: everything else
    """
    num_cols = df_features.select_dtypes(include=["number", "int64", "float64"]).columns.tolist()
    cat_cols = [c for c in df_features.columns if c not in num_cols]

    X_num = df_features[num_cols].values if num_cols else np.empty((len(df_features), 0), dtype=float)
    X_cat = encode_cat(df_features[cat_cols]) if cat_cols else np.empty((len(df_features), 0), dtype=int)

    X_num = scale_num(X_num)
    return X_num, X_cat


def _to01_from_yesno(series: pd.Series) -> np.ndarray:
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["yes", "1", "true", "t"]).astype(int).values


# ----------------------------
# Adult
# y = income
# sensitive = sex
# ----------------------------

def load_adult() -> DatasetBundle:
    path = os.path.join(DATA_DIR, "adult", "adult.csv")
    df = _read_csv_robust(path)

    # Normalize common column name variants
    df.columns = [c.strip() for c in df.columns]

    # Target income column: try common names
    target_candidates = ["income", "Income", "class", "target"]
    ycol = next((c for c in target_candidates if c in df.columns), None)
    if ycol is None:
        raise KeyError(f"[ADULT] Target column not found. Columns: {df.columns.tolist()}")

    # Sensitive sex column
    scol = "sex" if "sex" in df.columns else ("SEX" if "SEX" in df.columns else None)
    if scol is None:
        raise KeyError(f"[ADULT] Sensitive column 'sex' not found. Columns: {df.columns.tolist()}")

    # y mapping: allow strings like '>50K', '<=50K', '1/0'
    y_raw = df[ycol].astype(str).str.strip()
    y = y_raw.isin([">50K", ">50K.", "1", "True", "true"]).astype(int).values
    # If it looks numeric, fall back to numeric threshold
    if y.sum() == 0 and pd.to_numeric(df[ycol], errors="coerce").notna().any():
        y_num = pd.to_numeric(df[ycol], errors="coerce").fillna(0)
        y = (y_num > 0).astype(int).values

    # g: priv=Male(1), unpriv=Female(0)
    sex_raw = df[scol].astype(str).str.strip().str.lower()
    g = (sex_raw == "male").astype(int).values

    X_df = df.drop(columns=[ycol])
    # Keep sensitive column as feature? In your pipeline usually YES (like GIFAIR-FL). If not wanted, drop it:
    # X_df = X_df.drop(columns=[scol])

    X_num, X_cat = split_num_cat(X_df)
    return DatasetBundle(X_num=X_num, X_cat=X_cat, y=y, g=g)


# ----------------------------
# Bank
# y = y (yes/no)
# sensitive = age (<40 vs >=40)
# ----------------------------

def load_bank() -> DatasetBundle:
    path = os.path.join(DATA_DIR, "bank", "bank.csv")
    df = _read_csv_robust(path)

    if "y" not in df.columns:
        raise KeyError(f"[BANK] Target column 'y' not found. Columns: {df.columns.tolist()}")
    if "age" not in df.columns:
        raise KeyError(f"[BANK] Column 'age' not found. Columns: {df.columns.tolist()}")

    y = _to01_from_yesno(df["y"])
    g = (pd.to_numeric(df["age"], errors="coerce").fillna(0).astype(float) >= 40).astype(int).values

    X_df = df.drop(columns=["y"])
    X_num, X_cat = split_num_cat(X_df)
    return DatasetBundle(X_num=X_num, X_cat=X_cat, y=y, g=g)


# ----------------------------
# Census Income KDD (dein File)
# y = income  (bei dir offenbar -50000 / 50000 als Zahl)
# sensitive = ASEX
# ----------------------------

def load_census() -> DatasetBundle:
    """
    KDD Census Income (dein CSV-Format)
    - Target: income in {-50000, "50000+"} (Strings gemischt möglich)
    - Sensitive: ASEX (Male/Female)
    """
    path = os.path.join(DATA_DIR, "census_income_kdd", "census_income_kdd.csv")
    df = _read_csv_robust(path)
    df.columns = [c.strip() for c in df.columns]

    # Required columns (based on your header)
    if "income" not in df.columns:
        raise KeyError(f"[CENSUS] Target column 'income' not found. Columns: {df.columns.tolist()}")
    if "ASEX" not in df.columns:
        raise KeyError(f"[CENSUS] Sensitive column 'ASEX' not found. Columns: {df.columns.tolist()}")

    # ----- TARGET y -----
    # Your file contains values like "-50000" and "50000+"
    inc_raw = df["income"].astype(str).str.strip()

    # Positive if contains "50000+"
    # (robust to variants like "50000+.", whitespace, etc.)
    y = inc_raw.str.contains(r"50000\+", regex=True).astype(int).values

    # Sanity: must contain both classes
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        raise ValueError(
            f"[CENSUS] Degenerate target after parsing income. pos={pos} neg={neg}. "
            f"Sample income values: {inc_raw.head(10).tolist()}"
        )

    # ----- SENSITIVE GROUP g -----
    # priv=Male(1), unpriv=Female(0)
    sex_raw = df["ASEX"].astype(str).str.strip().str.lower()
    g = (sex_raw == "male").astype(int).values

    # ----- FEATURES -----
    # Drop target; keep ASEX as feature (like GIFAIR-FL often does).
    X_df = df.drop(columns=["income"])

    # If you want to REMOVE the sensitive attribute from the features, uncomment:
    # X_df = X_df.drop(columns=["ASEX"])

    X_num, X_cat = split_num_cat(X_df)
    return DatasetBundle(X_num=X_num, X_cat=X_cat, y=y, g=g)



# ----------------------------
# Communities & Crime
# y = ViolentCrimesPerPop -> binarize (>= median)
# sensitive = racepctblack (>= median -> priv=1)
# ----------------------------

def load_communities() -> DatasetBundle:
    path = os.path.join(DATA_DIR, "communities_crime", "communities_crime.csv")
    df = _read_csv_robust(path)

    # Column names vary wildly; use your specified names if present
    if "ViolentCrimesPerPop" not in df.columns:
        raise KeyError(f"[COMMUNITIES] Target 'ViolentCrimesPerPop' not found. Columns: {df.columns.tolist()}")
    if "racepctblack" not in df.columns:
        raise KeyError(f"[COMMUNITIES] Sensitive 'racepctblack' not found. Columns: {df.columns.tolist()}")

    y_val = pd.to_numeric(df["ViolentCrimesPerPop"], errors="coerce").fillna(0).astype(float)
    thr = float(np.median(y_val))
    y = (y_val >= thr).astype(int).values

    s_val = pd.to_numeric(df["racepctblack"], errors="coerce").fillna(0).astype(float)
    s_thr = float(np.median(s_val))
    g = (s_val >= s_thr).astype(int).values  # priv=1 is higher black percentage (your choice; consistent)

    X_df = df.drop(columns=["ViolentCrimesPerPop"])
    # If you want to drop sensitive attribute from features, uncomment:
    # X_df = X_df.drop(columns=["racepctblack"])

    X_num, X_cat = split_num_cat(X_df)
    return DatasetBundle(X_num=X_num, X_cat=X_cat, y=y, g=g)


# ----------------------------
# Public API
# ----------------------------

def load_dataset(name: str) -> DatasetBundle:
    name = name.lower()
    if name == "adult":
        return load_adult()
    if name == "bank":
        return load_bank()
    if name == "census":
        return load_census()
    if name == "communities":
        return load_communities()
    raise ValueError(f"Unknown dataset: {name}")
