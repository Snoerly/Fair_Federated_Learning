import numpy as np
from typing import Dict, List

#########################################
# AGDG - Approximate Global Dataset Generation
#########################################

class AGDG:
    """
    Implements the AGDG module from the GFL paper.

    Steps:
    1) Aggregate client group statistics
    2) Build global group distributions
    3) Generate synthetic numerical dataset using
       multivariate Gaussian sampling (MVG)
    """

    def __init__(self, client_stats: List[Dict[int, Dict[str, np.ndarray]]]):
        """
        Parameters
        ----------
        client_stats : list of dict
            Each element is output of SIC for one client:
            {
                group_id: {
                    "num": int,
                    "mu": np.ndarray,
                    "sigma": np.ndarray
                }
            }
        """
        self.client_stats = client_stats

    #########################################
    # Step 1: Aggregate statistics
    #########################################

    def aggregate_statistics(self) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Aggregates statistics across all clients per group.

        Returns
        -------
        global_stats : dict
            {
                group_id: {
                    "num": int,
                    "mu": np.ndarray,
                    "sigma": np.ndarray
                }
            }
        """
        global_stats: Dict[int, Dict[str, np.ndarray]] = {}

        # All groups (assumed consistent across clients)
        groups = set()
        for cs in self.client_stats:
            groups.update(cs.keys())

        for g in groups:
            total_num = sum(cs[g]["num"] for cs in self.client_stats if g in cs)

            # Weighted mean
            mu = sum(
                (cs[g]["num"] / total_num) * cs[g]["mu"]
                for cs in self.client_stats if g in cs
            )

            # Weighted covariance
            sigma = sum(
                (cs[g]["num"] / total_num) * cs[g]["sigma"]
                for cs in self.client_stats if g in cs
            )

            # Numerical stability
            eps = 1e-6
            sigma = sigma + eps * np.eye(sigma.shape[0])

            global_stats[g] = {
                "num": int(total_num),
                "mu": mu,
                "sigma": sigma
            }

        return global_stats

    #########################################
    # Step 2: MVG Sampling
    #########################################

    def generate_synthetic_numeric_data(self, global_stats: Dict[int, Dict[str, np.ndarray]]):
        """
        Generates synthetic numerical samples per group using
        multivariate Gaussian distribution.

        Returns
        -------
        synth_data : dict
            {
                group_id: {
                    "X_num": np.ndarray [num, d],
                    "g": np.ndarray [num]
                }
            }
        """
        synth_data = {}

        for g, s in global_stats.items():
            num = s["num"]
            mu = s["mu"]
            sigma = s["sigma"]

            Xg = np.random.multivariate_normal(mean=mu, cov=sigma, size=num)
            gg = np.full(num, g)

            synth_data[g] = {
                "X_num": Xg,
                "g": gg
            }

        return synth_data

    #########################################
    # Full pipeline
    #########################################

    def run(self):
        """
        Full AGDG pipeline:
        client_stats -> global_stats -> synthetic numeric dataset

        Returns
        -------
        synth_data : dict
            Output of generate_synthetic_numeric_data
        global_stats : dict
            Aggregated statistics
        """
        global_stats = self.aggregate_statistics()
        synth_data = self.generate_synthetic_numeric_data(global_stats)
        return synth_data, global_stats

#########################################
# Helper function
#########################################


def run_agdg(client_stats):
    agdg = AGDG(client_stats)
    return agdg.run()
