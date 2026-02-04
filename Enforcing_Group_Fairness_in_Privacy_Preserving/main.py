#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
main.py - GFL (paper-inspired) runner with CSV output compatible to GIFAIR-FL plots.

This version writes CSV columns compatible with plot_result.py:
- round
- test_accuracy
- eop_gap
- di_ratio
and required metadata columns:
- dataset, model, num_users, frac, iid, tabular_noniid, sensitive_attr

It also writes extra columns that existed in your older CSVs:
- train_loss, train_accuracy
- tpr_priv, tpr_unpriv
- p_pos_unpriv, p_pos_priv
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Local imports (your project modules)
from data.loader import load_dataset, DatasetBundle
from clients.statistics import ClientStatistics
from server.server import GFLServer, GFLServerConfig


########################################
# Utils
########################################

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_mean(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def train_test_split_stratified(X_num, X_cat, y, g, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n0_test = int(len(idx0) * test_size)
    n1_test = int(len(idx1) * test_size)

    test_idx = np.concatenate([idx0[:n0_test], idx1[:n1_test]])
    train_idx = np.concatenate([idx0[n0_test:], idx1[n1_test:]])

    rng.shuffle(test_idx)
    rng.shuffle(train_idx)

    return (
        X_num[train_idx], X_cat[train_idx], y[train_idx], g[train_idx],
        X_num[test_idx],  X_cat[test_idx],  y[test_idx],  g[test_idx],
    )


########################################
# Federated partitioning
########################################

def partition_iid(n: int, num_users: int, seed: int = 42) -> Dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    idxs = np.arange(n)
    rng.shuffle(idxs)
    splits = np.array_split(idxs, num_users)
    return {i: splits[i] for i in range(num_users)}


def partition_label_skew(y: np.ndarray, num_users: int, shards_per_user: int = 2, seed: int = 42) -> Dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(y))
    idxs = idxs[np.argsort(y)]

    num_shards = num_users * shards_per_user
    shards = np.array_split(idxs, num_shards)
    rng.shuffle(shards)

    user_groups: Dict[int, np.ndarray] = {}
    for u in range(num_users):
        assigned = shards[u * shards_per_user:(u + 1) * shards_per_user]
        user_groups[u] = np.concatenate(assigned)
    return user_groups

def partition_feature_skew(X_num: np.ndarray, num_users: int, seed: int = 42) -> Dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = X_num.shape[0]
    if n == 0:
        return {i: np.array([], dtype=int) for i in range(num_users)}

    # choose the numeric column with highest variance
    variances = np.var(X_num, axis=0)
    j = int(np.argmax(variances))

    idxs = np.arange(n)
    idxs = idxs[np.argsort(X_num[:, j])]  # sort by selected feature

    # split into contiguous chunks (strong feature-skew)
    splits = np.array_split(idxs, num_users)

    # optional: shuffle within each client chunk (keeps distribution but removes ordering)
    user_groups = {}
    for i in range(num_users):
        chunk = splits[i].copy()
        rng.shuffle(chunk)
        user_groups[i] = chunk

    return user_groups



########################################
# Tabular model (PyTorch)
########################################

class TabularNet(nn.Module):
    """
    Simple numeric + categorical embedding MLP for binary classification.

    Model expects:
      forward(x_num, x_cat) -> logits
    """

    def __init__(self, d_num: int, cat_cardinalities: List[int], emb_dim: int = 8, hidden: int = 128):
        super().__init__()
        self.d_num = d_num
        self.cat_cardinalities = cat_cardinalities

        self.embeddings = nn.ModuleList()
        for card in cat_cardinalities:
            card = max(int(card), 2)
            self.embeddings.append(nn.Embedding(card, emb_dim))

        in_dim = d_num + emb_dim * len(cat_cardinalities)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        embs = []
        if x_cat.numel() > 0 and len(self.embeddings) > 0:
            for j, emb in enumerate(self.embeddings):
                # Robustness: clamp indices to the embedding range.
                # Some datasets/splits may contain unseen or sentinel values (e.g. -1)
                # in validation/test partitions.
                idx = x_cat[:, j]
                if idx.dtype != torch.long:
                    idx = idx.long()
                idx = idx.clamp(min=0, max=int(emb.num_embeddings) - 1)
                embs.append(emb(idx))
        if embs:
            x = torch.cat([x_num] + embs, dim=1)
        else:
            x = x_num
        logits = self.net(x).squeeze(1)
        return logits


class TorchModelWrapper:
    """
    Wrapper to satisfy server/server.py expectations:
    - clone()
    - state_dict() / load_state_dict()
    - predict_score(X_full) returns probability of y=1
    - train_on_dataset(X_full, y)
    """

    def __init__(self, model: nn.Module, device: torch.device, lr: float = 1e-3, batch_size: int = 256):
        self.model = model
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.model.to(self.device)

    def clone(self):
        return TorchModelWrapper(copy.deepcopy(self.model), self.device, lr=self.lr, batch_size=self.batch_size)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd)

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X = np.asarray(X)

        d_num = getattr(self.model, "d_num", None)
        if d_num is None:
            raise ValueError("Underlying model must have attribute d_num")

        X_num = torch.tensor(X[:, :d_num], dtype=torch.float32, device=self.device)
        X_cat = X[:, d_num:]
        if X_cat.size == 0:
            X_cat_t = torch.empty((X.shape[0], 0), dtype=torch.long, device=self.device)
        else:
            X_cat_t = torch.tensor(X_cat, dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model(X_num, X_cat_t)
            probs = torch.sigmoid(logits)
        return probs.detach().cpu().numpy().reshape(-1)

    def train_on_dataset(self, X: np.ndarray, y: np.ndarray, epochs: int = 1) -> Tuple[float, float]:
        """
        Train and return (avg_loss, avg_accuracy) on that dataset for logging.
        """
        self.model.train()
        X = np.asarray(X)
        y = np.asarray(y).astype(np.float32)

        d_num = getattr(self.model, "d_num", None)
        X_num = torch.tensor(X[:, :d_num], dtype=torch.float32)
        X_cat = X[:, d_num:]
        if X_cat.size == 0:
            X_cat_t = torch.empty((X.shape[0], 0), dtype=torch.long)
        else:
            X_cat_t = torch.tensor(X_cat, dtype=torch.long)
        y_t = torch.tensor(y, dtype=torch.float32)

        ds = TensorDataset(X_num, X_cat_t, y_t)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        self.model.to(self.device)

        total_loss = 0.0
        total_correct = 0
        total_n = 0

        for _ in range(epochs):
            for xb_num, xb_cat, yb in dl:
                xb_num = xb_num.to(self.device)
                xb_cat = xb_cat.to(self.device)
                yb = yb.to(self.device)

                opt.zero_grad()
                logits = self.model(xb_num, xb_cat)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    preds = (probs >= 0.5).float()
                    total_correct += int((preds == yb).sum().item())
                    total_loss += float(loss.item()) * int(yb.shape[0])
                    total_n += int(yb.shape[0])

        avg_loss = total_loss / max(total_n, 1)
        avg_acc = total_correct / max(total_n, 1)
        return avg_loss, avg_acc


########################################
# Local client training
########################################

def local_train_one_client(
    global_wrapper: TorchModelWrapper,
    X_num: np.ndarray,
    X_cat: np.ndarray,
    y: np.ndarray,
    local_epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    local = global_wrapper.clone()
    local.lr = lr
    local.batch_size = batch_size

    X_full = np.hstack([X_num, X_cat])
    local.train_on_dataset(X_full, y, epochs=local_epochs)
    return local.state_dict()


########################################
# Evaluation (compat fields)
########################################

def compute_metrics_compat(wrapper: TorchModelWrapper, X_num, X_cat, y, g) -> Dict[str, float]:
    """
    Compute metrics compatible with your existing CSV schema:
    - test_accuracy
    - eop_gap   (we compute Equal Opportunity difference)
    - di_ratio  (P(yhat=1|unpriv) / P(yhat=1|priv))
    - tpr_priv, tpr_unpriv
    - p_pos_unpriv, p_pos_priv
    """
    X = np.hstack([X_num, X_cat])
    y = np.asarray(y).reshape(-1).astype(int)
    g = np.asarray(g).reshape(-1).astype(int)

    y_score = wrapper.predict_score(X)
    y_pred = (y_score >= 0.5).astype(int)

    test_accuracy = float((y_pred == y).mean()) if y.size else 0.0

    # Define privileged/unprivileged groups (consistent)
    # priv = g==1, unpriv = g==0
    priv = (g == 1)
    unpriv = (g == 0)

    p_pos_priv = _safe_mean(y_pred[priv])     # P(yhat=1 | g=1)
    p_pos_unpriv = _safe_mean(y_pred[unpriv]) # P(yhat=1 | g=0)

    eps = 1e-12
    di_ratio = float(p_pos_unpriv / (p_pos_priv + eps))

    # True positive rates by group (condition on y=1)
    pos = (y == 1)
    tpr_priv = _safe_mean(y_pred[priv & pos]) if np.any(priv & pos) else 0.0
    tpr_unpriv = _safe_mean(y_pred[unpriv & pos]) if np.any(unpriv & pos) else 0.0

    # Equal Opportunity gap (EOp gap) = |TPR_priv - TPR_unpriv|
    eop_gap = float(abs(tpr_priv - tpr_unpriv))

    return {
        "test_accuracy": test_accuracy,
        "eop_gap": eop_gap,
        "di_ratio": di_ratio,
        "tpr_priv": float(tpr_priv),
        "tpr_unpriv": float(tpr_unpriv),
        "p_pos_unpriv": float(p_pos_unpriv),
        "p_pos_priv": float(p_pos_priv),
    }


########################################
# Sensitive attr mapping (for CSV metadata)
########################################

def sensitive_attr_for_dataset(name: str) -> str:
    name = name.lower()
    if name == "adult":
        return "sex"
    if name == "bank":
        return "age"
    if name == "census":
        return "ASEX"
    if name == "communities":
        return "racepctblack"
    return ""


########################################
# Args
########################################

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, choices=["adult", "bank", "census", "communities"], default="adult")
    p.add_argument("--num_users", type=int, default=10)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--frac", type=float, default=1)

    p.add_argument("--iid", action="store_true")
    p.add_argument(
        "--tabular_noniid",
        type=str,
        choices=["label-skew", "feature-skew"],
        default="label-skew",
        help="Non-IID split type for tabular data (label-skew or feature-skew)"
    )


    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Local training
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--local_lr", type=float, default=1e-3)
    p.add_argument("--local_bs", type=int, default=256)

    # Model
    p.add_argument("--emb_dim", type=int, default=8)
    p.add_argument("--hidden", type=int, default=128)

    # Global/server training (on synthetic dataset)
    p.add_argument("--global_lr", type=float, default=1e-3)
    p.add_argument("--global_bs", type=int, default=256)
    p.add_argument("--global_train_epochs", type=int, default=1)

    # GFL config
    p.add_argument("--fairness_mode", choices=["eo", "dp"], default="eo")
    p.add_argument("--use_gbcfss", action="store_true")
    p.add_argument("--gbcfss_budget", type=int, default=5000)
    p.add_argument("--gbcfss_group_balance", type=float, default=0.2)

    # Output
    p.add_argument("--results_dir", type=str, default=os.path.join("save", "results"))
    p.add_argument("--results_name", type=str, default="")  # if empty, auto-name

    return p.parse_args()


########################################
# Main
########################################

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # 1) Load dataset
    bundle: DatasetBundle = load_dataset(args.dataset)
    X_num, X_cat, y, g = bundle.X_num, bundle.X_cat, bundle.y, bundle.g

    # 2) Train/test split
    Xn_tr, Xc_tr, y_tr, g_tr, Xn_te, Xc_te, y_te, g_te = train_test_split_stratified(
        X_num, X_cat, y, g, test_size=0.2, seed=args.seed
    )

    # 3) Determine categorical cardinalities for embeddings
    # Use the full dataset (train+test) so test-time categories don't exceed the embedding size.
    cat_cardinalities: List[int] = []
    if X_cat.size:
        for j in range(X_cat.shape[1]):
            # Handle potential sentinel values like -1 by clamping at 0.
            mx = int(np.max(X_cat[:, j]))
            cat_cardinalities.append(max(mx + 1, 2))

    # 4) Create global model + wrapper
    model = TabularNet(
        d_num=Xn_tr.shape[1],
        cat_cardinalities=cat_cardinalities,
        emb_dim=args.emb_dim,
        hidden=args.hidden,
    )
    global_wrapper = TorchModelWrapper(model, device=device, lr=args.global_lr, batch_size=args.global_bs)

    # 5) Partition clients
    if args.iid:
        user_groups = partition_iid(len(y_tr), args.num_users, seed=args.seed)
        split_name = "iid"
        iid_val = 1
    else:
        iid_val = 0
        split_name = args.tabular_noniid

        if args.tabular_noniid == "feature-skew":
            user_groups = partition_feature_skew(Xn_tr, args.num_users, seed=args.seed)
        else:
            user_groups = partition_label_skew(y_tr, args.num_users, shards_per_user=2, seed=args.seed)


    # 6) Server config + server
    server_cfg = GFLServerConfig(
        fairness_mode=args.fairness_mode,
        use_gbcfss=args.use_gbcfss,
        gbcfss_budget=args.gbcfss_budget,
        gbcfss_group_balance=args.gbcfss_group_balance,
    )
    server = GFLServer(server_cfg, global_wrapper)

    # 7) CSV path & header (GIFAIR-FL compatible)
    os.makedirs(args.results_dir, exist_ok=True)

    if args.results_name.strip():
        csv_path = os.path.join(args.results_dir, args.results_name.strip())
        if not csv_path.lower().endswith(".csv"):
            csv_path += ".csv"
    else:
        # Auto-name similar to your existing naming style
        csv_path = os.path.join(
            args.results_dir,
            f"gfl_{args.dataset}_users[{args.num_users}]_iid[{iid_val}]_"
            f"C[{args.frac}]_E[{args.rounds}]_localE[{args.local_epochs}]_B[{args.local_bs}]_"
            f"split[{split_name}]_sens[{sensitive_attr_for_dataset(args.dataset)}]_seed[{args.seed}].csv"
        )

    header = [
        # --- required by your plot_result.py and your old CSVs ---
        "round",
        "dataset",
        "model",
        "num_users",
        "frac",
        "iid",
        "tabular_noniid",
        "epochs",
        "local_ep",
        "local_bs",
        "seed",
        "train_loss",
        "train_accuracy",
        "test_accuracy",
        "eop_gap",
        "di_ratio",
        "tpr_priv",
        "tpr_unpriv",
        "p_pos_unpriv",
        "p_pos_priv",
        "sensitive_attr",
        # --- optional extras (won't break plot_result.py) ---
        "fairness_mode",
        "use_gbcfss",
        "gbcfss_budget",
        "gbcfss_group_balance",
        "synth_eod",
        "synth_dpd",
        "alpha_min",
        "alpha_max",
        "alpha_entropy",
    ]

    write_header = not os.path.exists(csv_path)

    # 8) Training rounds
    m = max(int(args.frac * args.num_users), 1)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)

        for r in range(1, args.rounds + 1):
            selected = np.random.choice(np.arange(args.num_users), m, replace=False)

            client_state_dicts: Dict[int, Dict[str, torch.Tensor]] = {}
            client_stats: List[Dict[int, Dict[str, np.ndarray]]] = []

            for cid in selected:
                idx = user_groups[int(cid)]

                # Local train
                sd = local_train_one_client(
                    global_wrapper,
                    Xn_tr[idx], Xc_tr[idx], y_tr[idx],
                    local_epochs=args.local_epochs,
                    lr=args.local_lr,
                    batch_size=args.local_bs,
                    device=device,
                )
                client_state_dicts[int(cid)] = sd

                # SIC stats (numerical only)
                sic = ClientStatistics(Xn_tr[idx], y_tr[idx], g_tr[idx])
                client_stats.append(sic.compute())

            # Server round
            global_metrics, alphas, global_stats = server.run_round(
                client_state_dicts=client_state_dicts,
                client_stats=client_stats,
                real_X_num=Xn_tr,
                real_X_cat=Xc_tr,
                real_g=g_tr,
                label_strategy="pseudo",
                real_y=y_tr,
            )

            # Compute train metrics on full train set (for compatibility)
            train_metrics = compute_metrics_compat(global_wrapper, Xn_tr, Xc_tr, y_tr, g_tr)
            # Train loss/acc: we can approximate by running one pass BCE loss on train
            # For compatibility we keep it simple:
            # -> Use current model predictions to compute log loss mean
            X_train_full = np.hstack([Xn_tr, Xc_tr])
            y_score_tr = global_wrapper.predict_score(X_train_full)
            y_tr_f = y_tr.astype(np.float64)
            eps = 1e-12
            train_loss = float(
                -np.mean(y_tr_f * np.log(np.clip(y_score_tr, eps, 1 - eps)) +
                         (1 - y_tr_f) * np.log(np.clip(1 - y_score_tr, eps, 1 - eps)))
            )
            train_accuracy = float(train_metrics["test_accuracy"])

            # Compute test metrics on real test set (for plot_result.py)
            test_metrics = compute_metrics_compat(global_wrapper, Xn_te, Xc_te, y_te, g_te)

            # AFW diagnostics
            alpha_vals = np.array(list(alphas.values()), dtype=np.float64) if alphas else np.array([], dtype=np.float64)
            alpha_min = float(alpha_vals.min()) if alpha_vals.size else 0.0
            alpha_max = float(alpha_vals.max()) if alpha_vals.size else 0.0
            alpha_entropy = float(-(alpha_vals * np.log(alpha_vals + 1e-12)).sum()) if alpha_vals.size else 0.0

            # Console output
            print(
                f"Round {r:03d} | test_acc={test_metrics['test_accuracy']:.4f} "
                f"eop_gap={test_metrics['eop_gap']:.4f} di={test_metrics['di_ratio']:.4f} "
                f"| synth_eod={global_metrics.eod:.4f} synth_dpd={global_metrics.dpd:.4f}"
            )

            writer.writerow([
                # --- required / legacy-compatible columns ---
                r,
                args.dataset,
                "gfl_tabularnet",
                args.num_users,
                args.frac,
                iid_val,
                split_name,
                args.rounds,                 # epochs field (global rounds) to match legacy
                args.local_epochs,
                args.local_bs,
                args.seed,
                train_loss,
                train_accuracy,
                test_metrics["test_accuracy"],
                test_metrics["eop_gap"],
                test_metrics["di_ratio"],
                test_metrics["tpr_priv"],
                test_metrics["tpr_unpriv"],
                test_metrics["p_pos_unpriv"],
                test_metrics["p_pos_priv"],
                sensitive_attr_for_dataset(args.dataset),
                # --- optional extras ---
                args.fairness_mode,
                int(args.use_gbcfss),
                args.gbcfss_budget,
                args.gbcfss_group_balance,
                global_metrics.eod,
                global_metrics.dpd,
                alpha_min,
                alpha_max,
                alpha_entropy,
            ])

    print(f"[OK] Wrote results to: {csv_path}")


if __name__ == "__main__":
    main()
