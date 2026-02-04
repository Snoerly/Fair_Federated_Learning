#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import sys
import subprocess
import argparse


SUPPORTED_DATASETS = ['adult', 'bank', 'census_income_kdd', 'communities_crime']


def main():
    parser = argparse.ArgumentParser(description='Run a small grid of FairTrade-FL configurations.')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['adult'],
        choices=SUPPORTED_DATASETS + ['all'],
        help="Datasets to run. Default: adult. Use 'all' to run every supported dataset.",
    )
    args = parser.parse_args()

    # Fixed knobs for all runs (adjust here if needed):
    epochs = 10            # global rounds
    local_ep = 10          # local epochs per client
    local_bs = 10          # local batch size
    frac = 1               # fraction of clients sampled each round
    seed = 1               # random seed
    fairness_notion = 'stat_parity'  # or 'ate'
    fairness_lambda = 1.0  # weight for fairness regularizer
    use_gpu = False        # set True to pass --gpu

    datasets = SUPPORTED_DATASETS if 'all' in args.datasets else args.datasets

    # Exactly the 6 configurations requested:
    # - IID (random split): 5 clients, 10 clients
    # - Non-IID: 3 clients, 5 clients × (label-skew, feature-skew)
    iid_configs = [5, 10]
    noniid_configs = [(3, 'label-skew'), (3, 'feature-skew'), (5, 'label-skew'), (5, 'feature-skew')]

    all_runs = []

    for dataset in datasets:
        if dataset == 'adult':
            sensitive_attr = 'sex'
        elif dataset == 'bank':
            sensitive_attr = 'age'
        elif dataset == 'census_income_kdd':
            sensitive_attr = 'ASEX'
        elif dataset == 'communities_crime':
            sensitive_attr = 'racepctblack'
        else:
            raise ValueError('Unsupported dataset: {}'.format(dataset))

        base_cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), 'fairtrade_federated.py'),
            '--dataset', dataset,
            '--model', 'mlp',
            '--num_classes', '2',
            '--epochs', str(epochs),
            '--local_ep', str(local_ep),
            '--local_bs', str(local_bs),
            '--frac', str(frac),
            '--seed', str(seed),
            '--sensitive_attr', sensitive_attr,
            '--fairness_notion', fairness_notion,
            '--fairness_lambda', str(fairness_lambda),
        ]
        if use_gpu:
            base_cmd.append('--gpu')

        for num_users in iid_configs:
            cmd = base_cmd + ['--num_users', str(num_users), '--iid', '1']
            all_runs.append(cmd)

        for num_users, split in noniid_configs:
            cmd = base_cmd + [
                '--num_users', str(num_users),
                '--iid', '0',
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
