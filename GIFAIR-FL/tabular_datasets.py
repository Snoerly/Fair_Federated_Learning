#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """Simple Dataset wrapper for tabular (feature, label) tensors.

    Optionally stores a sensitive group attribute per sample for
    fairness evaluation. The group labels are *not* returned by
    __getitem__ to keep compatibility with existing training code;
    instead they are accessed via the .groups attribute.
    """

    def __init__(self, features: torch.Tensor, labels: torch.Tensor,
                 groups: torch.Tensor = None):
        assert features.size(0) == labels.size(0)
        if groups is not None:
            assert features.size(0) == groups.size(0)
        self.features = features.float()
        self.labels = labels.long()
        self.groups = groups.long() if groups is not None else None

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def _standardize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize numeric columns to zero mean / unit variance.

    Keeps one-hot encoded columns (0/1) as they are.
    """

    numeric_cols = df.select_dtypes(include=["number"]).columns
    # ensure float dtype to avoid pandas dtype warnings
    df_num = df[numeric_cols].astype("float32")
    df_num = (df_num - df_num.mean()) / (df_num.std().replace(0, 1.0))
    df.loc[:, numeric_cols] = df_num
    return df


def load_adult_dataset(data_dir: str, sensitive_attr: str = "sex") -> Tuple[Dataset, Dataset, int]:
    """Load Adult income dataset from a single CSV file.

    Expected:
        - CSV file at ``os.path.join(data_dir, "adult.csv")``
        - One column named "label" or "target" with binary labels.
          If the original UCI strings (">50K", "<=50K") are present
          under a column named "income" or "class", they will be
          converted to 0/1.

    Returns:
        train_dataset, test_dataset, num_classes
    """

    csv_path = os.path.join(data_dir, "adult.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "Adult CSV not found. Expected file at: {}".format(csv_path)
        )

    df = pd.read_csv(csv_path)

    # Try to infer label column
    label_col = None
    for cand in ["label", "target", "income", "class"]:
        if cand in df.columns:
            label_col = cand
            break

    if label_col is None:
        raise ValueError(
            "Could not infer label column for Adult dataset. "
            "Please include one of: 'label', 'target', 'income', 'class'."
        )

    y_raw = df[label_col]

    # Map common string labels to {0, 1}
    if y_raw.dtype == object:
        y = y_raw.str.strip().replace({
            "<=50K": 0,
            "<=50K.": 0,
            ">50K": 1,
            ">50K.": 1,
        })
        if not set(y.unique()).issubset({0, 1}):
            raise ValueError(
                "Adult labels could not be mapped to {0,1}. "
                "Please pre-process the label column to be 0/1."
            )
    else:
        y = y_raw

    # Extract sensitive attribute for groups before dropping columns
    if sensitive_attr not in df.columns:
        raise ValueError(
            "Sensitive attribute '{}' not found in Adult dataset columns.".format(
                sensitive_attr
            )
        )

    group_raw = df[sensitive_attr]

    # Map common binary sensitive attributes to {0,1}
    if group_raw.dtype == object:
        # Example: 'Male'/'Female' or similar
        gr = group_raw.str.strip()
        uniques = sorted(gr.unique())
        if len(uniques) == 2:
            # map second value to 1, first to 0 (arbitrary but deterministic)
            mapping = {uniques[0]: 0, uniques[1]: 1}
            group = gr.replace(mapping).astype(int)
        else:
            # Fallback: treat first value as 0, all others as 1
            mapping = {u: (0 if i == 0 else 1) for i, u in enumerate(uniques)}
            group = gr.replace(mapping).astype(int)
    else:
        # already numeric; if more than 2 unique values, binarize at median
        vals = group_raw.astype(float)
        uniques = sorted(vals.unique())
        if len(uniques) <= 2:
            group = vals.astype(int)
        else:
            thresh = float(vals.median())
            group = (vals > thresh).astype(int)

    df = df.drop(columns=[label_col, sensitive_attr])

    # Basic preprocessing: one-hot encode categoricals, cast to numeric, standardize
    df = pd.get_dummies(df, drop_first=True)
    # force all remaining columns to numeric; non-convertible values become NaN
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(0.0)
    df = _standardize_features(df)

    # keep only numeric columns and ensure float32 dtype before tensor conversion
    df_numeric = df.select_dtypes(include=["number"]).astype("float32")

    X = torch.tensor(df_numeric.to_numpy(), dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)
    g_tensor = torch.tensor(group.values, dtype=torch.long)

    # Simple 80/20 train/test split
    n = X.size(0)
    split = int(0.8 * n)
    train_features, test_features = X[:split], X[split:]
    train_labels, test_labels = y_tensor[:split], y_tensor[split:]
    train_groups, test_groups = g_tensor[:split], g_tensor[split:]

    train_dataset = TabularDataset(train_features, train_labels, train_groups)
    test_dataset = TabularDataset(test_features, test_labels, test_groups)

    num_classes = int(y_tensor.unique().numel())
    return train_dataset, test_dataset, num_classes


def load_bank_dataset(data_dir: str, sensitive_attr: str = "age") -> Tuple[Dataset, Dataset, int]:
    """Load Bank Marketing dataset from a single CSV file.

    Expected:
        - CSV file at ``os.path.join(data_dir, "bank.csv")``
        - Label column "y" with values like "yes"/"no".

    The sensitive attribute can be any existing column; by default
    we use the numeric column "age" and binarize it at the median
    (<= median -> 0, > median -> 1).
    """

    csv_path = os.path.join(data_dir, "bank.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "Bank CSV not found. Expected file at: {}".format(csv_path)
        )

    df = pd.read_csv(csv_path)

    if "y" not in df.columns:
        raise ValueError("Expected label column 'y' in Bank dataset.")

    y_raw = df["y"]

    # Map 'yes'/'no' style labels to {0,1}
    if y_raw.dtype == object:
        y = y_raw.str.strip().replace({
            "no": 0,
            "yes": 1,
        })
        if not set(y.unique()).issubset({0, 1}):
            raise ValueError(
                "Bank labels could not be mapped to {0,1}. "
                "Please pre-process the 'y' column to be 'yes'/'no' or 0/1."
            )
    else:
        y = y_raw

    # Extract sensitive attribute for groups before dropping columns
    if sensitive_attr not in df.columns:
        raise ValueError(
            "Sensitive attribute '{}' not found in Bank dataset columns.".format(
                sensitive_attr
            )
        )

    group_raw = df[sensitive_attr]

    # Same mapping logic as in load_adult_dataset
    if group_raw.dtype == object:
        gr = group_raw.str.strip()
        uniques = sorted(gr.unique())
        if len(uniques) == 2:
            mapping = {uniques[0]: 0, uniques[1]: 1}
            group = gr.replace(mapping).astype(int)
        else:
            mapping = {u: (0 if i == 0 else 1) for i, u in enumerate(uniques)}
            group = gr.replace(mapping).astype(int)
    else:
        vals = group_raw.astype(float)
        uniques = sorted(vals.unique())
        if len(uniques) <= 2:
            group = vals.astype(int)
        else:
            thresh = float(vals.median())
            group = (vals > thresh).astype(int)

    df = df.drop(columns=["y", sensitive_attr])

    # Basic preprocessing: one-hot encode categoricals, cast to numeric, standardize
    df = pd.get_dummies(df, drop_first=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(0.0)
    df = _standardize_features(df)

    df_numeric = df.select_dtypes(include=["number"]).astype("float32")

    X = torch.tensor(df_numeric.to_numpy(), dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)
    g_tensor = torch.tensor(group.values, dtype=torch.long)

    # Simple 80/20 train/test split
    n = X.size(0)
    split = int(0.8 * n)
    train_features, test_features = X[:split], X[split:]
    train_labels, test_labels = y_tensor[:split], y_tensor[split:]
    train_groups, test_groups = g_tensor[:split], g_tensor[split:]

    train_dataset = TabularDataset(train_features, train_labels, train_groups)
    test_dataset = TabularDataset(test_features, test_labels, test_groups)

    num_classes = int(y_tensor.unique().numel())
    return train_dataset, test_dataset, num_classes


def load_census_income_kdd_dataset(data_dir: str, sensitive_attr: str = "ASEX") -> Tuple[Dataset, Dataset, int]:
    """Load Census Income KDD dataset from a single CSV file.

    Expected:
        - CSV file at ``os.path.join(data_dir, "census_income_kdd.csv")``
        - Label column "income" with values like ">50K", "<=50K".

    The sensitive attribute can be any existing column; by default
    we use the categorical column "asex" (sex).
    """

    csv_path = os.path.join(data_dir, "census_income_kdd.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "Census Income KDD CSV not found. Expected file at: {}".format(
                csv_path
            )
        )

    df = pd.read_csv(csv_path)

    if "income" not in df.columns:
        raise ValueError("Expected label column 'income' in Census Income KDD dataset.")

    y_raw = df["income"]

    # Map "-50000" / "50000+" to {0,1}
    if y_raw.dtype == object:
        y = y_raw.str.strip().replace({
            "-50000": 0,
            "50000+": 1,
            "50000+.": 1,
        })
        if not set(y.unique()).issubset({0, 1}):
            bad_vals = sorted(v for v in y.unique() if v not in {0, 1})
            raise ValueError(
                "Census Income KDD labels could not be mapped to {{0,1}}. "
                "Bitte prüfe die Spalte 'income'. Unerwartete Werte: {}".format(
                    bad_vals
                )
            )
    else:
        y = y_raw

    # Extract sensitive attribute for groups before dropping columns
    if sensitive_attr not in df.columns:
        raise ValueError(
            "Sensitive attribute '{}' not found in Census Income KDD columns.".format(
                sensitive_attr
            )
        )

    group_raw = df[sensitive_attr]

    # Mapping logic analogous to Adult/Bank
    if group_raw.dtype == object:
        gr = group_raw.str.strip()
        uniques = sorted(gr.unique())
        if len(uniques) == 2:
            mapping = {uniques[0]: 0, uniques[1]: 1}
            group = gr.replace(mapping).astype(int)
        else:
            mapping = {u: (0 if i == 0 else 1) for i, u in enumerate(uniques)}
            group = gr.replace(mapping).astype(int)
    else:
        vals = group_raw.astype(float)
        uniques = sorted(vals.unique())
        if len(uniques) <= 2:
            group = vals.astype(int)
        else:
            thresh = float(vals.median())
            group = (vals > thresh).astype(int)

    # Drop label and sensitive attribute from features
    df = df.drop(columns=["income", sensitive_attr])

    # One-hot encode categoricals, cast to numeric, standardize
    df = pd.get_dummies(df, drop_first=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(0.0)
    df = _standardize_features(df)

    df_numeric = df.select_dtypes(include=["number"]).astype("float32")

    X = torch.tensor(df_numeric.to_numpy(), dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)
    g_tensor = torch.tensor(group.values, dtype=torch.long)

    # Simple 80/20 train/test split
    n = X.size(0)
    split = int(0.8 * n)
    train_features, test_features = X[:split], X[split:]
    train_labels, test_labels = y_tensor[:split], y_tensor[split:]
    train_groups, test_groups = g_tensor[:split], g_tensor[split:]

    train_dataset = TabularDataset(train_features, train_labels, train_groups)
    test_dataset = TabularDataset(test_features, test_labels, test_groups)

    num_classes = int(y_tensor.unique().numel())
    return train_dataset, test_dataset, num_classes
    

def load_communities_crime_dataset(data_dir: str, sensitive_attr: str = "racepctblack") -> Tuple[Dataset, Dataset, int]:
    """Load Communities and Crime dataset from a single CSV file.

    Expected:
        - CSV file at ``os.path.join(data_dir, "communities_crime.csv")``
        - Label column "ViolentCrimesPerPop" with numeric values.

    The sensitive attribute can be any existing column; by default
    we use the numeric column "racepctblack" and binarize it at the median
    (<= median -> 0, > median -> 1).

    The prediction target "ViolentCrimesPerPop" is binarized into
    high-crime (1) vs. low-crime (0) communities using its median.
    """

    csv_path = os.path.join(data_dir, "communities_crime.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "Communities & Crime CSV not found. Expected file at: {}".format(
                csv_path
            )
        )

    df = pd.read_csv(csv_path)

    if "ViolentCrimesPerPop" not in df.columns:
        raise ValueError(
            "Expected label column 'ViolentCrimesPerPop' in Communities & Crime dataset."
        )

    y_raw = df["ViolentCrimesPerPop"].astype(float)

    # Binarize target by its median: <= median -> 0 (low crime), > median -> 1 (high crime)
    median_y = float(y_raw.median())
    y = (y_raw > median_y).astype(int)

    # Extract sensitive attribute for groups before dropping columns
    if sensitive_attr not in df.columns:
        raise ValueError(
            "Sensitive attribute '{}' not found in Communities & Crime columns.".format(
                sensitive_attr
            )
        )

    group_raw = df[sensitive_attr].astype(float)
    median_g = float(group_raw.median())
    # <= median -> 0, > median -> 1
    group = (group_raw > median_g).astype(int)

    # Drop label and sensitive attribute from features
    df = df.drop(columns=["ViolentCrimesPerPop", sensitive_attr])

    # One-hot encode categoricals, cast to numeric, standardize
    df = pd.get_dummies(df, drop_first=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(0.0)
    df = _standardize_features(df)

    df_numeric = df.select_dtypes(include=["number"]).astype("float32")

    X = torch.tensor(df_numeric.to_numpy(), dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)
    g_tensor = torch.tensor(group.values, dtype=torch.long)

    # Simple 80/20 train/test split
    n = X.size(0)
    split = int(0.8 * n)
    train_features, test_features = X[:split], X[split:]
    train_labels, test_labels = y_tensor[:split], y_tensor[split:]
    train_groups, test_groups = g_tensor[:split], g_tensor[split:]

    train_dataset = TabularDataset(train_features, train_labels, train_groups)
    test_dataset = TabularDataset(test_features, test_labels, test_groups)

    num_classes = int(y_tensor.unique().numel())
    return train_dataset, test_dataset, num_classes