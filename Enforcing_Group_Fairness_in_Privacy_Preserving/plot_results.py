#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import glob

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_csv(csv_path, out_dir):
    """Create a PNG plot with three curves (test_accuracy, eop_gap, di_ratio)
    over global rounds for a single CSV file.
    """

    df = pd.read_csv(csv_path)

    # require per-round CSV with a 'round' column
    if 'round' not in df.columns:
        print("[WARN] Skipping {} (no 'round' column)".format(csv_path))
        return

    if len(df) == 0:
        print("[WARN] Skipping {} (empty)".format(csv_path))
        return

    # sort by round just to be safe
    df = df.sort_values('round')

    # Extract series (coerce to numeric where needed)
    rounds = df['round'].values
    test_acc = pd.to_numeric(df.get('test_accuracy', []), errors='coerce')
    eop_gap = pd.to_numeric(df.get('eop_gap', []), errors='coerce')
    di_ratio = pd.to_numeric(df.get('di_ratio', []), errors='coerce')

    # Basic config info (from first row)
    row0 = df.iloc[0]
    dataset = row0.get('dataset', '')
    model = row0.get('model', '')
    num_users = row0.get('num_users', '')
    frac = row0.get('frac', '')
    iid = row0.get('iid', '')
    tabular_noniid = row0.get('tabular_noniid', '')
    sensitive_attr = row0.get('sensitive_attr', '')

    # Build title string with config
    title_lines = [
        "Dataset: {}  Model: {}".format(dataset, model),
        "Users: {}  frac: {}  iid: {}  split: {}".format(
            num_users, frac, iid, tabular_noniid
        ),
    ]
    if sensitive_attr:
        title_lines.append("Sensitive attr: {}".format(sensitive_attr))
    title = "\n".join(title_lines)

    # Create figure with three curves on two y-axes for readability
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_acc = 'tab:blue'
    color_eop = 'tab:red'
    color_di = 'tab:green'

    # Accuracy on left y-axis
    ax1.set_xlabel('Global round')
    ax1.set_ylabel('Accuracy', color=color_acc)
    if not test_acc.isnull().all():
        ax1.plot(rounds, test_acc, color=color_acc, label='Test Accuracy')
    ax1.tick_params(axis='y', labelcolor=color_acc)
    # fix accuracy axis to [0, 1]
    ax1.set_ylim(0.0, 1.0)

    # Second y-axis for fairness metrics
    ax2 = ax1.twinx()
    ax2.set_ylabel('Fairness metrics', color='black')
    has_eop = not eop_gap.isnull().all()
    has_di = not di_ratio.isnull().all()
    if has_eop:
        ax2.plot(rounds, eop_gap, color=color_eop, linestyle='--', label='EOp gap')
    if has_di:
        ax2.plot(rounds, di_ratio, color=color_di, linestyle='-.', label='DI ratio')
    ax2.tick_params(axis='y')

    # Combine legends from both axes
    lines, labels = [], []
    for ax in [ax1, ax2]:
        l, lab = ax.get_legend_handles_labels()
        lines.extend(l)
        labels.extend(lab)
    if lines:
        ax1.legend(lines, labels, loc='best')

    plt.title(title)
    plt.tight_layout()

    # Build output path
    base = os.path.splitext(os.path.basename(csv_path))[0]
    out_path = os.path.join(out_dir, base + '_metrics.png')
    plt.savefig(out_path)
    plt.close(fig)
    print("[OK] Saved plot for {} -> {}".format(csv_path, out_path))


def main():
    results_dir = os.path.join('save', 'results')
    if not os.path.isdir(results_dir):
        print("Results directory not found:", results_dir)
        return

    out_dir = os.path.join(results_dir, 'plots')
    os.makedirs(out_dir, exist_ok=True)

    pattern = os.path.join(results_dir, '*.csv')
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        print("No CSV files found in:", results_dir)
        return

    for csv_path in csv_files:
        # Optionally skip summary.csv if it exists
        if os.path.basename(csv_path).lower() == 'summary.csv':
            continue
        plot_csv(csv_path, out_dir)


if __name__ == '__main__':
    main()
