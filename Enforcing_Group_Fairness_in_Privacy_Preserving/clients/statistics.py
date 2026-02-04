import numpy as np
from typing import Dict

#########################################
# SIC - Statistical Information Collection
#########################################

class ClientStatistics:
    """
    Computes group-wise statistics required for AGDG:
    - num: number of samples per group
    - mu: mean vector of numerical features per group
    - sigma: covariance matrix of numerical features per group

    This is the SIC (Statistical Information Collection) module
    described in the GFL paper.
    """

    def __init__(self, X_num: np.ndarray, y: np.ndarray, g: np.ndarray):
        """
        Parameters
        ----------
        X_num : np.ndarray
            Numerical features [N, d]
        y : np.ndarray
            Labels (not directly used here, but kept for extension)
        g : np.ndarray
            Sensitive group labels (binary: 0/1)
        """
        self.X_num = X_num
        self.y = y
        self.g = g

    def compute(self) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Returns
        -------
        stats : dict
            {
                group_id: {
                    "num": int,
                    "mu": np.ndarray [d],
                    "sigma": np.ndarray [d, d]
                }
            }
        """
        stats: Dict[int, Dict[str, np.ndarray]] = {}

        groups = np.unique(self.g)

        for group in groups:
            idx = (self.g == group)
            Xg = self.X_num[idx]

            # Safety for empty groups
            if Xg.shape[0] == 0:
                continue

            num = Xg.shape[0]
            mu = Xg.mean(axis=0)

            # Rowvar=False => each column is a variable
            sigma = np.cov(Xg, rowvar=False)

            # Numerical stability (diagonal jitter)
            eps = 1e-6
            sigma = sigma + eps * np.eye(sigma.shape[0])

            stats[int(group)] = {
                "num": int(num),
                "mu": mu,
                "sigma": sigma
            }

        return stats

#########################################
# Helper function for multiple clients
#########################################


def collect_all_client_stats(clients_data):
    """
    Collects SIC stats from all clients.

    Parameters
    ----------
    clients_data : list of tuples
        [(X_num, y, g), ...]

    Returns
    -------
    all_stats : list
        [stats_client_1, stats_client_2, ...]
    """
    all_stats = []

    for (X_num, y, g) in clients_data:
        sic = ClientStatistics(X_num, y, g)
        stats = sic.compute()
        all_stats.append(stats)

    return all_stats
