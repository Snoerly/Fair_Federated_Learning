"""server/server.py

GFL Server Orchestrator.

This module ties together:
- SIC outputs from clients
- AGDG: approximate global numeric dataset generation
- LSH-CADC: categorical completion to obtain a full synthetic global dataset
- Global fairness metrics (EOD, DPD)
- AFW aggregation to update the global model
- Optional GBCFSS sample selection for training on the synthetic dataset

Notes:
- We intentionally omit cryptographic mechanisms (FuncEnc) in this educational implementation.
- DP (if desired) should be applied on the client side (e.g. DP-SGD), outside this module.

Expected external pieces:
- A model object that supports:
  - state_dict() and load_state_dict(...)
  - predict_proba(X) or predict_score(X) for synthetic dataset scoring
  - train_on_dataset(X, y) for optional server-side training

For PyTorch users, you can implement these as wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from server.agdg import AGDG
from server.lsh_cadc import LSH_CADC
from server.fairness import FairnessMetrics, metrics_from_scores, client_fairness_bias
from server.aggregation import compute_alphas, aggregate_state_dicts
from server.gbcfss import GBCFSSConfig, greedy_select_gbcfss


@dataclass
class GFLServerConfig:
    fairness_mode: str = "eo"  # "eo" or "dp" for fairness contribution / monitoring focus
    bias_scalar_mode: str = "sum"  # "sum" or "max" to convert (EOD,DPD) -> Fi
    use_additive_afw: bool = False

    # GBCFSS settings
    use_gbcfss: bool = True
    gbcfss_budget: int = 5000
    gbcfss_clean_w: float = 1.0
    gbcfss_fair_w: float = 1.0
    gbcfss_group_balance: Optional[float] = 0.2

    # LSH-CADC settings
    lsh_k: int = 7


class GFLServer:
    def __init__(
        self,
        config: GFLServerConfig,
        global_model,
    ):
        self.cfg = config
        self.global_model = global_model

    ############################################
    # Helpers
    ############################################

    @staticmethod
    def _stack_synth_numeric(synth_data: Dict[int, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        Xn = []
        gg = []
        for g, d in synth_data.items():
            Xn.append(d["X_num"])
            gg.append(d["g"])
        return np.vstack(Xn), np.concatenate(gg)

    ############################################
    # Main round
    ############################################

    def run_round(
        self,
        client_state_dicts: Dict[int, Dict[str, object]],
        client_stats: List[Dict[int, Dict[str, np.ndarray]]],
        # Real validation dataset for categorical completion
        real_X_num: np.ndarray,
        real_X_cat: np.ndarray,
        real_g: np.ndarray,
        # Synthetic labels generation (supervised setting)
        # In many setups, you can approximate labels by sampling from server validation distribution
        # or by using the current global model pseudo-labels.
        label_strategy: str = "pseudo",  # "pseudo" or "sample"
        real_y: Optional[np.ndarray] = None,
    ):
        """Execute one GFL round (server side).

        Parameters
        ----------
        client_state_dicts:
            {client_id: local_model_state_dict}
        client_stats:
            list of SIC outputs for each client
        real_X_num, real_X_cat, real_g:
            server-side validation/auxiliary dataset used for LSH-CADC completion
        label_strategy:
            "pseudo": label synthetic samples with current global model
            "sample": sample labels from real_y distribution (requires real_y)
        real_y:
            needed if label_strategy == "sample"

        Returns
        -------
        global_metrics: FairnessMetrics computed on synthetic dataset
        alphas: aggregation weights per client
        """

        # 1) AGDG: synthesize numeric global dataset
        agdg = AGDG(client_stats)
        synth_data, global_stats = agdg.run()

        # 2) LSH-CADC: reconstruct categorical attributes
        # Use group-wise covariance from global_stats for Mahalanobis
        covs = {g: global_stats[g]["sigma"] for g in global_stats}

        lsh = LSH_CADC(k=self.cfg.lsh_k)
        lsh.fit(real_X_num, real_X_cat, real_g, covs)

        synth_numeric = {g: synth_data[g]["X_num"] for g in synth_data}
        synth_groups = {g: synth_data[g]["g"] for g in synth_data}
        synth_cat = lsh.complete(synth_numeric, synth_groups)

        # 3) Build full synthetic feature matrix
        X_num_synth, g_synth = self._stack_synth_numeric(synth_data)
        X_cat_synth = np.vstack([synth_cat[g] for g in synth_cat]) if real_X_cat.shape[1] > 0 else np.empty((X_num_synth.shape[0], 0))
        X_synth = np.hstack([X_num_synth, X_cat_synth])

        # 4) Generate labels for synthetic dataset
        if label_strategy == "sample":
            if real_y is None:
                raise ValueError("real_y is required for label_strategy='sample'")
            p = float(np.mean(real_y))
            y_synth = (np.random.rand(X_synth.shape[0]) < p).astype(int)
        else:
            # pseudo-label using current global model
            y_score = self.global_model.predict_score(X_synth)
            y_synth = (y_score >= 0.5).astype(int)

        # 5) Score synthetic dataset with current global model
        y_score_synth = self.global_model.predict_score(X_synth)

        # 6) Optional: GBCFSS subset selection
        if self.cfg.use_gbcfss:
            cfg = GBCFSSConfig(
                budget=self.cfg.gbcfss_budget,
                cleanliness_weight=self.cfg.gbcfss_clean_w,
                fairness_weight=self.cfg.gbcfss_fair_w,
                group_balance=self.cfg.gbcfss_group_balance,
            )
            sel_idx = greedy_select_gbcfss(
                y_true=y_synth,
                g=g_synth,
                y_score=y_score_synth,
                cfg=cfg,
                fairness_mode=self.cfg.fairness_mode,
            )
            X_train = X_synth[sel_idx]
            y_train = y_synth[sel_idx]
        else:
            X_train, y_train = X_synth, y_synth

        # 7) Train / debias global model on synthetic dataset (server-side step)
        self.global_model.train_on_dataset(X_train, y_train)

        # 8) Compute global fairness metrics on synthetic dataset
        y_score_after = self.global_model.predict_score(X_synth)
        global_metrics = metrics_from_scores(y_score_after, y_synth, g_synth)

        # 9) Compute client fairness bias Fi for AFW weighting
        # Here we approximate each client's bias as the same global metric; in a fuller implementation,
        # evaluate per-client model on the *same* synthetic dataset to get Fi per client.
        fairness_bias: Dict[int, float] = {}
        for cid, w in client_state_dicts.items():
            # Temporarily evaluate client's model on synthetic dataset
            m = self.global_model.clone()
            m.load_state_dict(w)
            y_score_client = m.predict_score(X_synth)
            met = metrics_from_scores(y_score_client, y_synth, g_synth)
            Fi = client_fairness_bias(met, mode=self.cfg.bias_scalar_mode)
            fairness_bias[cid] = Fi

        alphas = compute_alphas(fairness_bias)

        # 10) AFW aggregate local client models into new global state
        new_state = aggregate_state_dicts(
            client_state_dicts,
            alphas,
            use_additive=self.cfg.use_additive_afw,
            global_state_dict=self.global_model.state_dict() if self.cfg.use_additive_afw else None,
        )
        self.global_model.load_state_dict(new_state)

        return global_metrics, alphas, global_stats
