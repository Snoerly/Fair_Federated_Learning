from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


def _ensure_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a


def _safe_pinv_cov(cov: np.ndarray, d: int) -> np.ndarray:
    """
    Robust pseudo-inverse for covariance matrices.
    If cov is None / wrong shape / singular -> fall back to identity.
    """
    if cov is None:
        return np.eye(d, dtype=float)

    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] != d:
        return np.eye(d, dtype=float)

    # Add tiny ridge for numerical stability
    cov = cov + 1e-6 * np.eye(d, dtype=float)
    try:
        return np.linalg.pinv(cov)
    except Exception:
        return np.eye(d, dtype=float)


@dataclass
class LSH_CADC:
    """
    LSH-CADC: Categorical Attribute Data Completion using LSH + distance in numeric space.

    This implementation:
    - Fits group-specific hashed buckets using numeric features (LSH-like random hyperplanes).
    - For each synthetic numeric sample, retrieves candidate real samples from same group bucket.
    - Chooses nearest candidate by (group-specific) Mahalanobis distance.
    - Copies that candidate's categorical attributes.

    Robustness improvements:
    - Normalizes group keys (0/1) for covariance dict.
    - If a group covariance is missing -> falls back to available covariance or identity.
    - If bucket empty -> falls back to searching within the same group (or globally if needed).
    """

    k: int = 7  # number of random projections (hash bits)

    # fitted state
    _W: Optional[np.ndarray] = None  # [k, d_num]
    _X_num: Optional[np.ndarray] = None
    _X_cat: Optional[np.ndarray] = None
    _g: Optional[np.ndarray] = None
    _cov_inv: Optional[Dict[int, np.ndarray]] = None  # {group: VI}
    _buckets: Optional[Dict[int, Dict[int, np.ndarray]]] = None  # {group: {hash: idxs}}

    def fit(
        self,
        X_num: np.ndarray,
        X_cat: np.ndarray,
        g: np.ndarray,
        cov_matrices: Dict,
        seed: int = 42,
    ) -> "LSH_CADC":
        """
        Parameters
        ----------
        X_num : real numeric features, shape [N, d_num]
        X_cat : real categorical features (encoded as ints), shape [N, d_cat]
        g     : real group labels (0/1), shape [N]
        cov_matrices : dict mapping group -> covariance matrix (d_num x d_num)
        seed  : random seed for projections
        """
        X_num = _ensure_2d(np.asarray(X_num, dtype=float))
        X_cat = _ensure_2d(np.asarray(X_cat, dtype=int)) if np.asarray(X_cat).size else np.empty((X_num.shape[0], 0), dtype=int)
        g = np.asarray(g).reshape(-1)

        if X_num.shape[0] != g.shape[0]:
            raise ValueError("LSH_CADC.fit: X_num and g must have same length")
        if X_cat.shape[0] != X_num.shape[0]:
            raise ValueError("LSH_CADC.fit: X_cat and X_num must have same length")

        N, d = X_num.shape

        # Normalize covariance dict keys to int (fixes KeyError np.int64(0) vs 0 vs '0')
        cov_norm: Dict[int, np.ndarray] = {}
        for kk, vv in cov_matrices.items():
            try:
                cov_norm[int(kk)] = np.asarray(vv, dtype=float)
            except Exception:
                # if key is weird, skip; we will fallback later
                continue

        # Precompute inverse covariance per group, with fallbacks
        groups_present = sorted(set(int(x) for x in np.unique(g)))
        cov_inv: Dict[int, np.ndarray] = {}

        for grp in groups_present:
            cov = cov_norm.get(grp, None)
            if cov is None and len(cov_norm) > 0:
                # fallback to any available covariance
                cov = next(iter(cov_norm.values()))
            cov_inv[grp] = _safe_pinv_cov(cov, d)

        # If still empty (shouldn't happen), set identity for both groups
        if not cov_inv:
            cov_inv = {0: np.eye(d, dtype=float), 1: np.eye(d, dtype=float)}

        # Random hyperplanes for hashing (LSH-like)
        rng = np.random.default_rng(seed)
        W = rng.normal(size=(self.k, d)).astype(float)

        def hash_vec(x: np.ndarray) -> int:
            bits = (W @ x) >= 0.0
            h = 0
            for b in bits.astype(int):
                h = (h << 1) | int(b)
            return int(h)

        # Build buckets: group -> hash -> indices
        buckets: Dict[int, Dict[int, List[int]]] = {}
        for i in range(N):
            grp = int(g[i])
            h = hash_vec(X_num[i])
            if grp not in buckets:
                buckets[grp] = {}
            if h not in buckets[grp]:
                buckets[grp][h] = []
            buckets[grp][h].append(i)

        # Convert lists to arrays for speed
        buckets_np: Dict[int, Dict[int, np.ndarray]] = {}
        for grp, bd in buckets.items():
            buckets_np[grp] = {h: np.asarray(idxs, dtype=int) for h, idxs in bd.items()}

        self._W = W
        self._X_num = X_num
        self._X_cat = X_cat
        self._g = g
        self._cov_inv = cov_inv
        self._buckets = buckets_np
        return self

    def _hash_vec(self, x: np.ndarray) -> int:
        if self._W is None:
            raise RuntimeError("LSH_CADC: call fit() first")
        W = self._W
        bits = (W @ x) >= 0.0
        h = 0
        for b in bits.astype(int):
            h = (h << 1) | int(b)
        return int(h)

    def _mahalanobis(self, x: np.ndarray, Xcand: np.ndarray, VI: np.ndarray) -> np.ndarray:
        """
        Compute squared Mahalanobis distance from x to each row in Xcand.
        """
        diff = Xcand - x.reshape(1, -1)
        # dist_i = diff_i^T VI diff_i
        return np.einsum("ij,jk,ik->i", diff, VI, diff)

    def complete(
        self,
        synth_numeric_by_group: Dict[int, np.ndarray],
        synth_groups_by_group: Dict[int, np.ndarray],
    ) -> Dict[int, np.ndarray]:
        """
        For each group g: take synthetic numeric samples, and impute categorical
        attrs by nearest neighbor (within LSH bucket) from real data.
        Returns dict: g -> X_cat_synth (N_g x d_cat)
        """
        if self._X_num is None or self._X_cat is None or self._g is None or self._buckets is None or self._cov_inv is None:
            raise RuntimeError("LSH_CADC: call fit() before complete()")

        X_num_real = self._X_num
        X_cat_real = self._X_cat
        g_real = self._g
        buckets = self._buckets
        cov_inv = self._cov_inv

        d_cat = X_cat_real.shape[1]

        out: Dict[int, np.ndarray] = {}

        for grp_key, Xn_s in synth_numeric_by_group.items():
            grp = int(grp_key)
            Xn_s = _ensure_2d(np.asarray(Xn_s, dtype=float))

            if Xn_s.shape[0] == 0:
                out[grp] = np.empty((0, d_cat), dtype=int)
                continue

            # choose VI for this group (fallback to any VI)
            VI = cov_inv.get(grp, next(iter(cov_inv.values())))

            Xc_s = np.empty((Xn_s.shape[0], d_cat), dtype=int) if d_cat > 0 else np.empty((Xn_s.shape[0], 0), dtype=int)

            # Precompute indices in the same group
            same_group_idxs = np.where(g_real.astype(int) == grp)[0]
            if same_group_idxs.size == 0:
                # fallback: use all real samples if group not present
                same_group_idxs = np.arange(X_num_real.shape[0])

            for i in range(Xn_s.shape[0]):
                x = Xn_s[i]
                h = self._hash_vec(x)

                # candidate indices: bucket within group
                cand_idxs = None
                if grp in buckets and h in buckets[grp]:
                    cand_idxs = buckets[grp][h]

                # fallback: if empty bucket, use all same-group samples
                if cand_idxs is None or cand_idxs.size == 0:
                    cand_idxs = same_group_idxs

                Xcand = X_num_real[cand_idxs]
                dists = self._mahalanobis(x, Xcand, VI)
                j = int(np.argmin(dists))
                best_idx = int(cand_idxs[j])

                if d_cat > 0:
                    Xc_s[i] = X_cat_real[best_idx]

            out[grp] = Xc_s

        return out
