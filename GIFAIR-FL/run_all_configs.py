#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import sys
import subprocess
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['adult', 'bank', 'census_income_kdd', 'communities_crime'],
                        help='Dataset name (adult, bank, census_income_kdd, communities_crime).')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Global rounds (E in den Plots).')
    parser.add_argument('--local_ep', type=int, default=10,
                        help='Lokale Epochen pro Client.')
    parser.add_argument('--local_bs', type=int, default=10,
                        help='Lokale Batchgröße.')
    parser.add_argument('--frac', type=float, default=0.1,
                        help='Fraktion ausgewählter Clients pro Round.')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random Seed.')
    args = parser.parse_args()

    dataset = args.dataset

    # Sensitives Attribut je Datensatz (kannst du bei Bedarf im federated_main
    # Aufruf unten anpassen)
    if dataset == 'adult':
        sensitive_attr = 'sex'
    elif dataset == 'bank':
        sensitive_attr = 'age'
    elif dataset == 'census_income_kdd':
        sensitive_attr = 'ASEX'  
    elif dataset == 'communities_crime':
        sensitive_attr = 'racepctblack'

    base_cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), 'federated_main.py'),
        '--dataset', dataset,
        '--model', 'mlp',
        '--num_classes', '2',
        '--epochs', str(args.epochs),
        '--local_ep', str(args.local_ep),
        '--local_bs', str(args.local_bs),
        '--frac', str(args.frac),
        '--seed', str(args.seed),
        '--sensitive_attr', sensitive_attr,
    ]

    # Konfigurationen wie von dir gefordert:
    # Random splits (IID): 5 Clients, 10 Clients
    iid_random_configs = [
        {'num_users': 5},
        {'num_users': 10},
    ]

    # Non-IID splits: 3 Clients, 5 Clients -> label-skew und feature-skew
    noniid_configs = []
    for num_users in [3, 5]:
        for split in ['label-skew', 'feature-skew']:
            noniid_configs.append({
                'num_users': num_users,
                'tabular_noniid': split,
            })

    all_runs = []

    # IID Läufe
    for cfg in iid_random_configs:
        cmd = base_cmd + [
            '--num_users', str(cfg['num_users']),
            '--iid', '1',
        ]
        all_runs.append(cmd)

    # Non-IID Läufe
    for cfg in noniid_configs:
        cmd = base_cmd + [
            '--num_users', str(cfg['num_users']),
            '--iid', '0',
            '--tabular_noniid', cfg['tabular_noniid'],
        ]
        all_runs.append(cmd)

    print('Starte {} Konfigurationen für Dataset={}'.format(len(all_runs), dataset))

    for i, cmd in enumerate(all_runs, 1):
        print('\n[{:02d}/{}] Kommando:'.format(i, len(all_runs)))
        print(' ', ' '.join(cmd))
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print('  -> Lauf abgebrochen (return code {}). Stop.'.format(res.returncode))
            break
        else:
            print('  -> Lauf erfolgreich abgeschlossen.')


if __name__ == '__main__':
    main()
