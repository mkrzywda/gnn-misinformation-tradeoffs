#!/usr/bin/env python3
import os
import re
import pandas as pd

# Root directory containing all experiment subfolders
root_dir = '.'

# List of metric columns to extract
metrics = [
    'test_accuracy', 'test_precision', 'test_recall', 'test_f1_score',
    'test_mcc', 'test_aucroc', 'test_aucpr', 'test_sensitivity',
    'test_specificity', 'test_ppv', 'test_npv', 'test_ARI', 'test_NMI', 'Inference Time (ms)'
]

rows = []

# Regexps to capture config parameters
re_pretrain     = re.compile(r'pretrain=([^_]+)')
re_neighbours   = re.compile(r'neighbours=([^_]+)')
re_dataset_size = re.compile(r'dataset_size=([^_]+)')

for dirpath, dirnames, filenames in os.walk(root_dir):
    parts = dirpath.split(os.sep)
    # szukamy segmentu zaczynającego się od "pretrain="
    config_folders = [p for p in parts if p.startswith('pretrain=')]
    if not config_folders:
        continue
    config = config_folders[0]                # np. "pretrain=False_neighbours=2_dataset_size=0.1"
    dataset = parts[-1]                       # np. "welfake", "covid-19" itd.

    csv_files = [f for f in filenames if f.endswith('_crossval_results.csv')]
    if not csv_files:
        continue

    # Wyciągamy wartości
    m_pre = re_pretrain.search(config)
    m_nei = re_neighbours.search(config)
    m_sz  = re_dataset_size.search(config)
    pretrain_val     = m_pre.group(1) if m_pre else None
    neighbours_val   = m_nei.group(1) if m_nei else None
    dataset_size_val = m_sz.group(1)  if m_sz  else None

    for fn in csv_files:
        model_name = fn.replace('adjustmentGNN-', '').replace('_gnn_crossval_results.csv', '')
        df = pd.read_csv(os.path.join(dirpath, fn))

        # bierzemy tylko wiersze Mean i Std
        mean_row = df[df['Dataset'] == 'Mean'].iloc[0]
        std_row  = df[df['Dataset'] == 'Std'].iloc[0]

        entry = {
            'pretrain': pretrain_val,
            'neighbours': neighbours_val,
            'dataset_size': dataset_size_val,
            'dataset': dataset,
            'model': model_name
        }
        for m in metrics:
            entry[f'{m}_mean'] = mean_row.get(m, pd.NA)
            entry[f'{m}_std']  = std_row.get(m, pd.NA)

        rows.append(entry)

# zapisujemy do CSV
summary_df = pd.DataFrame(rows)
summary_df.to_csv('summary.csv', index=False)
print("Saved summary.csv with mean+std metrics, w tym dataset_size!") 

