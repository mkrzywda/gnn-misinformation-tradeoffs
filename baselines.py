#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from loguru import logger

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    matthews_corrcoef, confusion_matrix,
    adjusted_rand_score, normalized_mutual_info_score,
    log_loss
)

# ============================================================
BASE_PATH = "/net/pr2/projects/plgrid/plggphdgnn/FAKENEWS"
RESULTS_BASE = os.path.join(BASE_PATH, "final-results-baselines")
os.makedirs(RESULTS_BASE, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ============================================================
# METRICS
# ============================================================
def binary_metrics_from_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    ppv = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    return sensitivity, specificity, ppv, npv


def calculate_metrics(y_true, y_pred, y_probs=None, multiclass=False):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted")
    mcc = matthews_corrcoef(y_true, y_pred)

    if multiclass:
        sensitivity = specificity = ppv = npv = np.nan
    else:
        sensitivity, specificity, ppv, npv = binary_metrics_from_confusion(y_true, y_pred)

    try:
        if y_probs is not None:
            if multiclass:
                roc_auc = roc_auc_score(y_true, y_probs, multi_class="ovr")
                aucpr = average_precision_score(y_true, y_probs, average="weighted")
            else:
                roc_auc = roc_auc_score(y_true, y_probs[:, 1])
                aucpr = average_precision_score(y_true, y_probs[:, 1])
        else:
            roc_auc = aucpr = np.nan
    except Exception:
        roc_auc = aucpr = np.nan

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mcc": mcc,
        "aucroc": roc_auc,
        "aucpr": aucpr,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
    }

# ============================================================
# kNN FEATURE SMOOTHING (NEIGH)
# ============================================================
def knn_smooth_features(X, k):
    if k <= 0:
        return X
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1)
    nn.fit(X)
    idx = nn.kneighbors(X, return_distance=False)
    X_new = np.zeros_like(X)
    for i, ids in enumerate(idx):
        X_new[i] = X[ids[1:]].mean(axis=0)
    return X_new

# ============================================================
# MODELS
# ============================================================
def get_model(model_type):
    if model_type == "logreg":
        return LogisticRegression(max_iter=5000, n_jobs=-1, random_state=SEED)
    if model_type == "svm":
        return CalibratedClassifierCV(LinearSVC(random_state=SEED))
    if model_type == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            max_iter=200,
            early_stopping=True,
            random_state=SEED
        )
    raise ValueError(model_type)

# ============================================================
# DATA LOADING — 1:1 Z GNN
# ============================================================
def load_and_prepare_data(dataset_name, dataset_size=None):
    logger.info(f"Loading and preparing data for dataset: {dataset_name} (dataset_size={dataset_size})")
    path = os.path.join(BASE_PATH, dataset_name.lower())
    os.makedirs(path, exist_ok=True)
    name = dataset_name.lower()

    # Unified-DF datasets
    if name in ['kaggle','fakenewsnet','welfake','click-id']:
        # Load full DF
        if name=='kaggle':
            fake = pd.read_csv(os.path.join(path,'Fake.csv'))
            true = pd.read_csv(os.path.join(path,'True.csv'))
            fake['label'], true['label'] = 0,1
            df = pd.concat([fake,true],ignore_index=True)
            text_col, lab_col = 'text','label'
        elif name=='fakenewsnet':
            gf = pd.read_csv(os.path.join(path,'gossipcop_fake.csv'))
            gr = pd.read_csv(os.path.join(path,'gossipcop_real.csv'))
            pf = pd.read_csv(os.path.join(path,'politifact_fake.csv'))
            pr = pd.read_csv(os.path.join(path,'politifact_real.csv'))
            for d in (gf,pf): d['label']=1
            for d in (gr,pr): d['label']=0
            df = pd.concat([gf,gr,pf,pr],ignore_index=True).dropna(subset=['title']).reset_index(drop=True)
            text_col, lab_col = 'title','label'
        elif name=='welfake':
            df = pd.read_csv(os.path.join(path,'WELFake_Dataset.csv')).dropna(subset=['text']).reset_index(drop=True)
            text_col, lab_col = 'text','label'
        else:  # click-id
            df = pd.read_csv(os.path.join(path,'main.csv'))
            df['label']=df['label'].map({'clickbait':1,'non-clickbait':0})
            text_col, lab_col = 'title','label'

        # Original 70/20/10
        orig_trval, orig_test = train_test_split(df, test_size=0.1, random_state=42, stratify=df[lab_col])
        orig_train, orig_val  = train_test_split(orig_trval, test_size=0.2222, random_state=42, stratify=orig_trval[lab_col])
        logger.info(f"Original splits (100%) for {name}: train={len(orig_train)}, val={len(orig_val)}, test={len(orig_test)}")

        # Stratified sample to dataset_size
        if dataset_size is not None:
            df, _ = train_test_split(df, train_size=dataset_size, random_state=42, stratify=df[lab_col])
            pct = int(dataset_size*100)
            logger.info(f"Cutted to {pct}% for {name}: df_total={len(df)}")
            trval, test = train_test_split(df, test_size=0.1, random_state=42, stratify=df[lab_col])
            train, val = train_test_split(trval, test_size=0.2222, random_state=42, stratify=trval[lab_col])
        else:
            pct = 100
            train, val, test = orig_train, orig_val, orig_test

        logger.info(f"Cutted to {pct}% for {name}: train={len(train)}, val={len(val)}, test={len(test)}")

    else:
        # Pre-split: liar, covid-19, mpid
        if name=='liar':
            cols = ["id","label","statement","subject","speaker","job","state","party",
                    "barely_true_counts","false_counts","half_true_counts","mostly_true_counts",
                    "pants_fire_counts","context"]
            train = pd.read_csv(os.path.join(path,'train.tsv'),sep='\t',quoting=3,names=cols,usecols=['label','statement'])
            val   = pd.read_csv(os.path.join(path,'valid.tsv'),sep='\t',quoting=3,names=cols,usecols=['label','statement'])
            test  = pd.read_csv(os.path.join(path,'test.tsv'), sep='\t',quoting=3,names=cols,usecols=['label','statement'])
            def ml(l): return 1 if l in {'true','mostly-true','half-true'} else 0
            for d in (train,val,test): d['bin_label']=d['label'].apply(ml)
            text_col, lab_col = 'statement','bin_label'
        elif name=='covid-19':
            train = pd.read_csv(os.path.join(path,'Constraint_Train.csv'))
            val   = pd.read_csv(os.path.join(path,'Constraint_Val.csv'))
            test  = pd.read_csv(os.path.join(path,'english_test_with_labels.csv'))
            for d in (train,val,test): d['label']=d['label'].map({'fake':0,'real':1})
            text_col, lab_col = 'tweet','label'
        elif name=='mpid':
            train = pd.read_csv(os.path.join(path,'train.csv'))
            val   = pd.read_csv(os.path.join(path,'validation.csv'))
            test  = pd.read_csv(os.path.join(path,'test.csv'))
            mp = {l:i for i,l in enumerate(train['disinformation'].unique())}
            for d in (train,val,test): d['label']=d['disinformation'].map(mp)
            text_col, lab_col = 'article','label'
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        logger.info(f"Original splits (100%) for {name}: train={len(train)}, val={len(val)}, test={len(test)}")

        if dataset_size is not None:
            pct = int(dataset_size*100)
            train, _ = train_test_split(train, train_size=dataset_size, random_state=42, stratify=train[lab_col])
            val,   _ = train_test_split(val,   train_size=dataset_size, random_state=42, stratify=val[lab_col])
            test,  _ = train_test_split(test,  train_size=dataset_size, random_state=42, stratify=test[lab_col])
        else:
            pct = 100

        logger.info(f"Cutted to {pct}% for {name}: train={len(train)}, val={len(val)}, test={len(test)}")

    # TF-IDF vectorization
    vec = TfidfVectorizer(max_features=5000)
    vec.fit(pd.concat([train[text_col], val[text_col]], ignore_index=True))
    X_train, y_train = vec.transform(train[text_col]).toarray(), train[lab_col].values
    X_val,   y_val   = vec.transform(val[text_col]).toarray(),   val[lab_col].values
    X_test,  y_test  = vec.transform(test[text_col]).toarray(),  test[lab_col].values
    logger.info(f"Vectorized splits for {name}: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    return {'train': (X_train, y_train),
            'validation': (X_val,   y_val),
            'test':       (X_test,  y_test)}

# ============================================================
# MAIN CV LOOP — 1:1 Z GNN
# ============================================================
def run_cross_validation(dataset_name, model_type, neigh, fraction):
    # fraction -> dataset_size
    data = load_and_prepare_data(dataset_name, dataset_size=fraction)

    X_train_full, y_train_full = data['train']
    X_val_full,   y_val_full   = data['validation']
    X_test,       y_test       = data['test']

    # Train + Val for CV (1:1 jak w GNN)
    X_all = np.vstack([X_train_full, X_val_full])
    y_all = np.concatenate([y_train_full, y_val_full])

    multiclass = len(np.unique(y_all)) > 2
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    metrics_results = []

    train_losses, val_losses, test_losses = [], [], []
    test_accs, test_f1s, test_mccs, test_aucrocs, test_aucprs = [], [], [], [], []

    out_dir = os.path.join(RESULTS_BASE, dataset_name.lower())
    os.makedirs(out_dir, exist_ok=True)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_all, y_all), 1):
        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        X_va, y_va = X_all[va_idx], y_all[va_idx]

        # neigh smoothing
        X_tr = knn_smooth_features(X_tr, neigh)
        X_va = knn_smooth_features(X_va, neigh)
        X_te = knn_smooth_features(X_test, neigh)

        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", get_model(model_type))
        ])

        start = time.time()
        pipe.fit(X_tr, y_tr)
        inference_time = (time.time() - start) / len(y_test)

        # Probabilities
        y_tr_p = pipe.predict_proba(X_tr)
        y_va_p = pipe.predict_proba(X_va)
        y_te_p = pipe.predict_proba(X_te)

        # Losses (jak w baseline’ach)
        train_losses.append(log_loss(y_tr, y_tr_p))
        val_losses.append(log_loss(y_va, y_va_p))
        test_losses.append(log_loss(y_test, y_te_p))

        y_pred = np.argmax(y_te_p, axis=1)
        m = calculate_metrics(y_test, y_pred, y_te_p, multiclass)

        m['Inference Time (ms)'] = inference_time * 1000
        m['Fold'] = fold

        metrics_results.append(m)

        test_accs.append(m['accuracy'])
        test_f1s.append(m['f1_score'])
        test_mccs.append(m['mcc'])
        test_aucrocs.append(m['aucroc'])
        test_aucprs.append(m['aucpr'])

    # ============================
    # PLOTS (fold-wise)
    # ============================
    folds = range(1, len(train_losses) + 1)

    def plot(vals, name):
        plt.figure(figsize=(10, 6))
        plt.plot(folds, vals)
        plt.xlabel("Fold")
        plt.ylabel(name)
        plt.title(f"{dataset_name} | {model_type} | k={neigh} | frac={fraction}")
        plt.savefig(os.path.join(out_dir, f"{dataset_name}_{model_type}_k{neigh}_f{fraction}_{name}.png"))
        plt.close()

    plot(train_losses, "Train_Loss")
    plot(val_losses,   "Val_Loss")
    plot(test_losses,  "Test_Loss")
    plot(test_accs,    "Test_Accuracy")
    plot(test_f1s,     "Test_F1")
    plot(test_mccs,    "Test_MCC")
    plot(test_aucrocs, "Test_AUCROC")
    plot(test_aucprs,  "Test_AUCPR")

    # ============================
    # SAVE CSV
    # ============================
    df = pd.DataFrame(metrics_results)

    #mean_row = df.mean(numeric_only=True)
    #std_row  = df.std(numeric_only=True)

    #mean_row['Fold'] = 'Mean'
    #std_row['Fold']  = 'Std'

    #df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    mean_dict = df.mean(numeric_only=True).to_dict()
    std_dict  = df.std(numeric_only=True).to_dict()

    mean_dict['Fold'] = 'Mean'
    std_dict['Fold']  = 'Std'

    df = pd.concat(
        [df, pd.DataFrame([mean_dict, std_dict])],
        ignore_index=True
    )


    out_csv = os.path.join(
        out_dir,
        f"{dataset_name}_{model_type}_k{neigh}_f{fraction}.csv"
    )
    df.to_csv(out_csv, index=False)

    logger.info(f"Saved results to {out_csv}")



# ============================================================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True, choices=["logreg", "svm", "mlp"])
    p.add_argument("--neighbors", type=int, required=True)
    p.add_argument("--fraction", type=str, required=True)
    args = p.parse_args()
    
    if args.fraction.lower() == "none":
        fraction = None
    else:
        fraction = float(args.fraction)

    run_cross_validation(
        args.dataset,
        args.model,
        args.neighbors,
        fraction
    )

