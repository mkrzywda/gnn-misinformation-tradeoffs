#!/usr/bin/env python3
import os
import argparse
import time
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv, GINConv,
    ChebConv, APPNP, SGConv, EdgeConv, TransformerConv, TAGConv, FeaStConv,
    JumpingKnowledge, knn_graph
)
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    adjusted_rand_score, normalized_mutual_info_score, matthews_corrcoef,
    average_precision_score, confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt

# Use cuda:1 if available
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

BASE_PATH = '.'

def binary_metrics_from_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP/(TP+FN) if (TP+FN)>0 else 0.0
    specificity = TN/(TN+FP) if (TN+FP)>0 else 0.0
    ppv = TP/(TP+FP) if (TP+FP)>0 else 0.0
    npv = TN/(TN+FN) if (TN+FN)>0 else 0.0
    return sensitivity, specificity, ppv, npv

def calculate_metrics(y_true, y_pred, y_probs=None, multiclass=False):
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1        = f1_score(y_true, y_pred, average='weighted')
    mcc       = matthews_corrcoef(y_true, y_pred)

    if multiclass and len(np.unique(y_true))>2:
        sensitivity = specificity = ppv = npv = float('nan')
    else:
        sensitivity, specificity, ppv, npv = binary_metrics_from_confusion(y_true, y_pred)

    if y_probs is not None and len(np.unique(y_true))>1:
        try:
            if multiclass and len(np.unique(y_true))>2:
                roc_auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
                aucpr   = average_precision_score(y_true, y_probs, average='weighted')
            else:
                if y_probs.shape[1]==2:
                    roc_auc = roc_auc_score(y_true, y_probs[:,1])
                    aucpr   = average_precision_score(y_true, y_probs[:,1])
                else:
                    roc_auc = roc_auc_score(y_true, y_probs)
                    aucpr   = average_precision_score(y_true, y_probs)
        except:
            roc_auc = aucpr = float('nan')
    else:
        roc_auc = aucpr = float('nan')

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    return {
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'f1_score': f1,     'mcc': mcc,
        'aucroc': roc_auc,   'aucpr': aucpr,
        'sensitivity': sensitivity, 'specificity': specificity,
        'ppv': ppv, 'npv': npv,
        'ARI': ari, 'NMI': nmi
    }

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

class GNN(torch.nn.Module):
    def __init__(self, model_type='gin', input_dim=5000,
                 hidden_dim=64, output_dim=2, **kwargs):
        super().__init__()
        mt = model_type.lower()
        self.model_type = mt

        # [Initialize all conv layers exactly as before]
        if mt=='gcn':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif mt=='graphsage':
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        elif mt=='gat':
            self.conv1 = GATConv(input_dim, hidden_dim, heads=8, dropout=0.6)
            self.conv2 = GATConv(hidden_dim*8, hidden_dim, heads=1, concat=False, dropout=0.6)
        elif mt=='gin':
            self.conv1 = GINConv(torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            ))
            self.conv2 = GINConv(torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            ))
        elif mt=='chebnet':
            K = kwargs.get('K',3)
            self.conv1 = ChebConv(input_dim, hidden_dim, K=K)
            self.conv2 = ChebConv(hidden_dim, hidden_dim, K=K)
        elif mt=='appnp':
            K, alpha = kwargs.get('K',10), kwargs.get('alpha',0.1)
            self.lin1 = torch.nn.Linear(input_dim, hidden_dim)
            self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.prop = APPNP(K=K, alpha=alpha)
        elif mt=='sgc':
            K = kwargs.get('K',2)
            self.conv = SGConv(input_dim, output_dim, K=K, cached=False)
        elif mt=='edgeconv':
            self.nn_layer = torch.nn.Sequential(
                torch.nn.Linear(2*input_dim, hidden_dim), torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )
            self.conv1 = EdgeConv(self.nn_layer)
            self.k = kwargs.get('k',6)
        elif mt=='transformerconv':
            heads, concat = kwargs.get('heads',4), kwargs.get('concat',False)
            self.conv1 = TransformerConv(input_dim, hidden_dim, heads=heads, concat=concat)
            self.conv2 = TransformerConv(hidden_dim, hidden_dim, heads=heads, concat=concat)
        elif mt=='tagconv':
            K = kwargs.get('K',3)
            self.conv1 = TAGConv(input_dim, hidden_dim, K=K)
            self.conv2 = TAGConv(hidden_dim, hidden_dim, K=K)
        elif mt=='jknet':
            nl, mode = kwargs.get('num_layers',3), kwargs.get('mode','cat')
            self.convs = torch.nn.ModuleList()
            self.convs.append(GINConv(torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )))
            for _ in range(nl-1):
                self.convs.append(GINConv(torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim)
                )))
            self.jump = JumpingKnowledge(mode)
        elif mt=='feastconv':
            self.conv1 = FeaStConv(input_dim, hidden_dim)
            self.conv2 = FeaStConv(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Heads
        if mt not in ['sgc','appnp','edgeconv','jknet']:
            self.fc = torch.nn.Linear(hidden_dim, output_dim)
        else:
            self.fc = torch.nn.Identity()

        if mt=='sgc':
            pre_in = output_dim
        elif mt in ['appnp','edgeconv','feastconv','gcn','graphsage','gat','gin','chebnet','tagconv','transformerconv']:
            pre_in = hidden_dim
        else:  # jknet
            nl = kwargs.get('num_layers',3)
            mode = kwargs.get('mode','cat')
            pre_in = nl * hidden_dim if mode=='cat' else hidden_dim

        self.pretrain_fc = torch.nn.Linear(pre_in, 1)

    def forward(self, data, pretrain=False):
        x, edge_index = data.x, data.edge_index
        mt = self.model_type

        if mt in ['gcn','graphsage','gat','gin','chebnet','transformerconv','tagconv','feastconv']:
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            out = self.pretrain_fc(x.mean(0, keepdim=True)) if pretrain else self.fc(x)

        elif mt=='appnp':
            x = F.relu(self.lin1(x))
            x = F.dropout(x, training=self.training)
            x = self.lin2(x)
            x = self.prop(x, edge_index)
            out = self.pretrain_fc(x.mean(0, keepdim=True)) if pretrain else self.fc(x)

        elif mt=='sgc':
            x = self.conv(x, edge_index)
            out = self.pretrain_fc(x.mean(0, keepdim=True)) if pretrain else self.fc(x)

        elif mt=='edgeconv':
            batch = data.batch if hasattr(data,'batch') else None
            if edge_index.numel()==0:
                ei = knn_graph(x.cpu(), k=self.k)
                edge_index = ei.to(x.device)
            x = F.relu(self.conv1(x, edge_index))
            out = self.pretrain_fc(x.mean(0, keepdim=True)) if pretrain else self.fc(x)

        elif mt=='jknet':
            xs = []
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
                xs.append(x)
            x = self.jump(xs)
            out = self.pretrain_fc(x.mean(0, keepdim=True)) if pretrain else self.fc(x)

        else:
            raise ValueError(f"No forward for {mt}")

        return out

def pretrain(model, loader, optimizer, device, epochs=5):
    logger.info("Starting pre-training phase.")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Pretrain {epoch+1}/{epochs}"):
            b = batch.to(device)
            optimizer.zero_grad()
            out = model(b, pretrain=True).view(-1)
            labels = torch.randint(0,2,(out.size(0),),device=device).float()
            loss = F.mse_loss(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"Pretrain Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

def run_cross_validation(
    dataset_name, model_type='gin',
    without_pretrain=False, neighbours=5, dataset_size=None,
    max_epochs=500, pretrain_epochs=5, patience=10, learning_rate=1e-3,
    **model_kwargs
):
    pre_str = str(without_pretrain)
    folder = f"pretrain={pre_str}_neighbours={neighbours}_dataset_size={dataset_size}"
    out_root = os.path.join(BASE_PATH, 'GNN-Mixed', folder)
    os.makedirs(out_root, exist_ok=True)
    ds_dir = os.path.join(out_root, dataset_name.lower())
    os.makedirs(ds_dir, exist_ok=True)

    logger.info(f"CV start: ds={dataset_name}, model={model_type}, pretrain={not without_pretrain}, k={neighbours}, size={dataset_size}")

    splits = load_and_prepare_data(dataset_name, dataset_size)
    X_tr, y_tr = splits['train']
    X_val, y_val = splits['validation']
    X_te, y_te = splits['test']

    Xc = np.vstack((X_tr, X_val))
    yc = np.concatenate((y_tr, y_val))
    multiclass = len(np.unique(np.concatenate((yc, y_te)))) > 2

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics_results = []
    fold = 1

    for train_idx, _ in kf.split(Xc, yc):
        tr_idx, vl_idx = train_test_split(train_idx, test_size=0.2,
                                          random_state=42, stratify=yc[train_idx])
        xt  = torch.tensor(Xc[tr_idx], dtype=torch.float)
        yt  = torch.tensor(yc[tr_idx], dtype=torch.long)
        xv  = torch.tensor(Xc[vl_idx], dtype=torch.float)
        yv  = torch.tensor(yc[vl_idx], dtype=torch.long)
        xts = torch.tensor(X_te, dtype=torch.float)
        yts = torch.tensor(y_te, dtype=torch.long)

        e_tr = knn_graph(xt,  k=neighbours)
        e_vl = knn_graph(xv,  k=neighbours)
        e_ts = knn_graph(xts, k=neighbours)

        train_loader = DataLoader([Data(x=xt, edge_index=e_tr, y=yt)], batch_size=1, shuffle=True)
        val_loader   = DataLoader([Data(x=xv, edge_index=e_vl, y=yv)], batch_size=1)
        test_loader  = DataLoader([Data(x=xts,edge_index=e_ts,y=yts)], batch_size=1)

        model = GNN(
            model_type=model_type,
            input_dim=Xc.shape[1],
            hidden_dim=model_kwargs.get('hidden_dim',64),
            output_dim=len(np.unique(yc)),
            **model_kwargs
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        if not without_pretrain:
            pretrain(model, train_loader, optimizer, device, epochs=pretrain_epochs)
        else:
            logger.info("Skipping pre-training")

        best_val = float('inf')
        counter = 0
        best_state = None

        train_losses, val_losses, test_losses = [], [], []
        accs, f1s, mccs, aucs, aprs = [], [], [], [], []

        for epoch in range(max_epochs):
            model.train()
            lt = 0
            for batch in train_loader:
                b = batch.to(device)
                optimizer.zero_grad()
                out = model(b, pretrain=False)
                loss = F.cross_entropy(out, b.y)
                loss.backward()
                optimizer.step()
                lt += loss.item()
            train_losses.append(lt/len(train_loader))

            model.eval()
            lv = 0
            with torch.no_grad():
                for batch in val_loader:
                    b = batch.to(device)
                    lv += F.cross_entropy(model(b, pretrain=False), b.y).item()
            val_losses.append(lv/len(val_loader))

            ys, ps, pr = [], [], []
            with torch.no_grad():
                for batch in test_loader:
                    b = batch.to(device)
                    out = model(b, pretrain=False)
                    prob = out.softmax(dim=1).cpu().numpy()
                    pr.extend(prob.argmax(1))
                    ps.extend(prob)
                    ys.extend(b.y.cpu().numpy())
            test_losses.append(
                sum(F.cross_entropy(model(b.to(device), pretrain=False), b.y.to(device)).item()
                    for b in test_loader)/len(test_loader)
            )
            m = calculate_metrics(np.array(ys), np.array(pr), np.array(ps), multiclass=multiclass)
            accs.append(m['accuracy']); f1s.append(m['f1_score'])
            mccs.append(m['mcc']); aucs.append(m['aucroc']); aprs.append(m['aucpr'])

            if val_losses[-1] < best_val:
                best_val = val_losses[-1]
                best_state = model.state_dict()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        ys, ps, pr = [], [], []
        t0 = time.time()
        with torch.no_grad():
            for batch in test_loader:
                b = batch.to(device)
                out = model(b, pretrain=False)
                prob = out.softmax(dim=1).cpu().numpy()
                pr.extend(prob.argmax(1)); ps.extend(prob); ys.extend(b.y.cpu().numpy())
        inference_time = (time.time() - t0)/len(test_loader)
        final_metrics = calculate_metrics(np.array(ys), np.array(pr), np.array(ps), multiclass=multiclass)
        logger.info(f"Fold {fold} final ACC: {final_metrics['accuracy']:.4f}, F1: {final_metrics['f1_score']:.4f}")

        epochs_range = list(range(1, len(train_losses)+1))
        def save_plot(vals,label,suf,ylabel):
            plt.figure(); plt.plot(epochs_range, vals, label=label)
            plt.xlabel('Epoch'); plt.ylabel(ylabel); plt.legend()
            plt.savefig(os.path.join(ds_dir, f'adjustmentGNN-{dataset_name.lower()}_{model_type}_fold{fold}_{suf}.png'))
            plt.close()

        save_plot(train_losses,'Train Loss','train_loss','Loss')
        save_plot(val_losses,'Val Loss','val_loss','Loss')
        save_plot(test_losses,'Test Loss','test_loss','Loss')
        save_plot(accs,'Test ACC','accuracy','Accuracy')
        save_plot(f1s,'Test F1','f1_score','F1-Score')
        save_plot(aucs,'Test AUCROC','aucroc','AUCROC')
        save_plot(aprs,'Test AUCPR','aucpr','AUCPR')
        save_plot(mccs,'Test MCC','mcc','MCC')

        metrics_results.append({
            'Dataset': dataset_name.lower(),
            'Model': model_type,
            'Fold': fold,
            **{f"test_{k}": final_metrics[k] for k in [
                'accuracy','precision','recall','f1_score','mcc','aucroc','aucpr',
                'sensitivity','specificity','ppv','npv','ARI','NMI'
            ]},
            'Inference Time (ms)': inference_time*1000,
            'Stop Epoch': epoch+1,
            'Early Stopping': counter>=patience
        })

        fold += 1

    df = pd.DataFrame(metrics_results)
    mean = df.mean(numeric_only=True)
    std  = df.std(numeric_only=True)
    mean_row = mean.to_dict(); mean_row.update({'Dataset':'Mean','Model':'','Fold':''})
    std_row  = std.to_dict();  std_row.update({'Dataset':'Std','Model':'','Fold':''})
    df = pd.concat([df, pd.DataFrame([mean_row,std_row])], ignore_index=True)
    out_csv = os.path.join(ds_dir, f'adjustmentGNN-{model_type}_gnn_crossval_results.csv')
    df.to_csv(out_csv, index=False)
    logger.info(f"Results saved to {out_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNN Mixed Experiments')
    parser.add_argument('--max_epochs',      type=int,   default=500)
    parser.add_argument('--pretrain_epochs', type=int,   default=5)
    parser.add_argument('--patience',        type=int,   default=10)
    parser.add_argument('--learning_rate',   type=float, default=1e-3)
    args = parser.parse_args()

    dataset_list = ['liar','kaggle','fakenewsnet','welfake','covid-19','click-id','mpid']
    model_list = [
        {'model_type':'sgc','K':2},
        {'model_type':'appnp','K':10,'alpha':0.1},
        {'model_type':'gcn'},
        {'model_type':'graphsage'},
        {'model_type':'gat'},
        {'model_type':'gin'},
        {'model_type':'chebnet','K':3},
        {'model_type':'jknet','num_layers':3,'mode':'cat'},
        {'model_type':'tagconv','K':3},
        {'model_type':'feastconv'},
    ]

    for pretrain_flag in [False, True]:
        for neighbours in [2,3,4,5,8]:
            for ds_size in [0.2]:
                for ds in dataset_list:
                    for m in model_list:
                        mk = m.copy()
                        mt = mk.pop('model_type')
                        logger.info(f"\n=== Running {ds} | model={mt} | pretrain={pretrain_flag} | neighbours={neighbours} | dataset_size={ds_size} ===")
                        run_cross_validation(
                            ds,
                            model_type=mt,
                            neighbours=neighbours,
                            without_pretrain=True,
                            dataset_size=ds_size,
                            max_epochs=args.max_epochs,
                            pretrain_epochs=args.pretrain_epochs,
                            patience=args.patience,
                            learning_rate=args.learning_rate,
                            **mk
                        )
