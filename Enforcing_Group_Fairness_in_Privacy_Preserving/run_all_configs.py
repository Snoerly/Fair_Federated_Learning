#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys


SUPPORTED_DATASETS = ['adult', 'bank', 'census', 'communities']


def main():
    parser = argparse.ArgumentParser(description='Run a small grid of GFL configurations (Enforcing Group Fairness).')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['adult'],
        choices=SUPPORTED_DATASETS + ['all'],
        help="Datasets to run. Default: adult. Use 'all' to run every supported dataset.",
    )
    args = parser.parse_args()

    # Fixed knobs for all runs (adjust here if needed):
    rounds = 10         # global rounds
    local_epochs = 10     # local training epochs per client
    local_bs = 100       # local batch size
    frac = 1           # fraction of clients sampled each round
    seed = 42            # random seed

    # GFL-specific knobs
    fairness_mode = 'eo'  # 'eo' or 'dp'
    use_gbcfss = False
    gbcfss_budget = 5000
    gbcfss_group_balance = 0.2

    datasets = SUPPORTED_DATASETS if 'all' in args.datasets else args.datasets

    # Exactly the 6 configurations requested:
    # - IID (random split): 5 clients, 10 clients
    # - Non-IID: 3 clients, 5 clients × (label-skew, feature-skew)
    iid_configs = [5, 10]
    noniid_configs = [(3, 'label-skew'), (3, 'feature-skew'), (5, 'label-skew'), (5, 'feature-skew')]

    all_runs = []

    for dataset in datasets:
        base_cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), 'main.py'),
            '--dataset', dataset,
            '--rounds', str(rounds),
            '--local_epochs', str(local_epochs),
            '--local_bs', str(local_bs),
            '--frac', str(frac),
            '--seed', str(seed),
            '--fairness_mode', fairness_mode,
        ]

        # Optional GBCFSS
        if use_gbcfss:
            base_cmd += [
                '--use_gbcfss',
                '--gbcfss_budget', str(gbcfss_budget),
                '--gbcfss_group_balance', str(gbcfss_group_balance),
            ]

        # IID runs (note: --iid is a flag)
        for num_users in iid_configs:
            cmd = base_cmd + ['--num_users', str(num_users), '--iid']
            all_runs.append(cmd)

        # Non-IID runs
        for num_users, split in noniid_configs:
            cmd = base_cmd + [
                '--num_users', str(num_users),
                '--tabular_noniid', split,
            ]
            all_runs.append(cmd)

    print('Datasets:', ', '.join(datasets))
    print('Starting {} total configurations across datasets.'.format(len(all_runs)))

    for i, cmd in enumerate(all_runs, 1):
        print('\n[{0:02d}/{1}] Command:'.format(i, len(all_runs)))
        print(' ', ' '.join(cmd))
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print('  -> run failed (return code {}). Stop.'.format(res.returncode))
            break
        else:
            print('  -> run completed successfully.')
            print('  -> finished run {0}/{1}'.format(i, len(all_runs)))


if __name__ == '__main__':
    main()
