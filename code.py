import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv, GINConv,
    ChebConv, APPNP, SGConv, EdgeConv, TransformerConv, TAGConv, FeaStConv,
    JumpingKnowledge
)
from torch_geometric.nn import knn_graph
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
import time
from loguru import logger
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

BASE_PATH = '.'

def binary_metrics_from_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN) if (TP+FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN+FP) > 0 else 0.0
    ppv = TP / (TP + FP) if (TP+FP) > 0 else 0.0
    npv = TN / (TN + FN) if (TN+FN) > 0 else 0.0
    return sensitivity, specificity, ppv, npv

def calculate_metrics(y_true, y_pred, y_probs=None, multiclass=False):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)

    if multiclass and len(np.unique(y_true)) > 2:
        sensitivity = float('nan')
        specificity = float('nan')
        ppv = float('nan')
        npv = float('nan')
    else:
        sensitivity, specificity, ppv, npv = binary_metrics_from_confusion(y_true, y_pred)

    if y_probs is not None and len(np.unique(y_true)) > 1:
        try:
            if multiclass and len(np.unique(y_true)) > 2:
                roc_auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
                aucpr = average_precision_score(y_true, y_probs, average='weighted')
            else:
                if y_probs.shape[1] == 2:
                    roc_auc = roc_auc_score(y_true, y_probs[:,1])
                    aucpr = average_precision_score(y_true, y_probs[:,1])
                else:
                    roc_auc = roc_auc_score(y_true, y_probs)
                    aucpr = average_precision_score(y_true, y_probs)
        except:
            roc_auc = float('nan')
            aucpr = float('nan')
    else:
        roc_auc = float('nan')
        aucpr = float('nan')

    ARI = adjusted_rand_score(y_true, y_pred)
    NMI = normalized_mutual_info_score(y_true, y_pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mcc': mcc,
        'aucroc': roc_auc,
        'aucpr': aucpr,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'ARI': ARI,
        'NMI': NMI
    }

def load_and_prepare_data(dataset_name):
    logger.info(f"Loading and preparing data for dataset: {dataset_name}")
    dataset_path = os.path.join(BASE_PATH, dataset_name.lower())
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        logger.info(f"Created directory: {dataset_path}")

    if dataset_name.lower() == 'kaggle':
        fake_csv = os.path.join(dataset_path, 'Fake.csv')
        true_csv = os.path.join(dataset_path, 'True.csv')
        fake_df = pd.read_csv(fake_csv)
        true_df = pd.read_csv(true_csv)
        fake_df['label'] = 0
        true_df['label'] = 1
        df = pd.concat([fake_df, true_df], ignore_index=True)
        text_column = 'text'
        label_column = 'label'
        train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df[label_column])
        train_df, val_df = train_test_split(train_val_df, test_size=0.2222, random_state=42, stratify=train_val_df[label_column])
    elif dataset_name.lower() == 'fakenewsnet':
        gossipcop_fake_csv = os.path.join(dataset_path, 'gossipcop_fake.csv')
        gossipcop_real_csv = os.path.join(dataset_path, 'gossipcop_real.csv')
        politifact_fake_csv = os.path.join(dataset_path, 'politifact_fake.csv')
        politifact_real_csv = os.path.join(dataset_path, 'politifact_real.csv')
        gossipcop_fake = pd.read_csv(gossipcop_fake_csv)
        gossipcop_real = pd.read_csv(gossipcop_real_csv)
        politifact_fake = pd.read_csv(politifact_fake_csv)
        politifact_real = pd.read_csv(politifact_real_csv)
        gossipcop_fake['label'] = 1
        gossipcop_real['label'] = 0
        politifact_fake['label'] = 1
        politifact_real['label'] = 0
        df = pd.concat([gossipcop_fake, gossipcop_real, politifact_fake, politifact_real], ignore_index=True)
        df = df.dropna(subset=['title']).reset_index(drop=True)
        text_column = 'title'
        label_column = 'label'
        train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df[label_column])
        train_df, val_df = train_test_split(train_val_df, test_size=0.2222, random_state=42, stratify=train_val_df[label_column])
    elif dataset_name.lower() == 'welfake':
        welfake_csv = os.path.join(dataset_path, 'WELFake_Dataset.csv')
        df = pd.read_csv(welfake_csv)
        df = df.dropna(subset=['text']).reset_index(drop=True)
        text_column = 'text'
        label_column = 'label'
        train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df[label_column])
        train_df, val_df = train_test_split(train_val_df, test_size=0.1111, random_state=42, stratify=train_val_df[label_column])
    elif dataset_name.lower() == 'mpid':
        train_csv = os.path.join(dataset_path, 'train.csv')
        validation_csv = os.path.join(dataset_path, 'validation.csv')
        test_csv = os.path.join(dataset_path, 'test.csv')
        train_df = pd.read_csv(train_csv)
        validation_df = pd.read_csv(validation_csv)
        test_df = pd.read_csv(test_csv)
        label_mapping = {label: idx for idx, label in enumerate(train_df['disinformation'].unique())}
        train_df['label'] = train_df['disinformation'].map(label_mapping)
        validation_df['label'] = validation_df['disinformation'].map(label_mapping)
        test_df['label'] = test_df['disinformation'].map(label_mapping)
        text_column = 'article'
        label_column = 'label'
    elif dataset_name.lower() == 'covid-19':
        train_csv = os.path.join(dataset_path, 'Constraint_Train.csv')
        val_csv = os.path.join(dataset_path, 'Constraint_Val.csv')
        test_csv = os.path.join(dataset_path, 'english_test_with_labels.csv')
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)
        label_mapping = {'fake': 0, 'real': 1}
        train_df['label'] = train_df['label'].map(label_mapping)
        val_df['label'] = val_df['label'].map(label_mapping)
        test_df['label'] = test_df['label'].map(label_mapping)
        text_column = 'tweet'
        label_column = 'label'
    elif dataset_name.lower() == 'click-id':
        data_csv = os.path.join(dataset_path, 'main.csv')
        df = pd.read_csv(data_csv)
        label_mapping = {'clickbait': 1, 'non-clickbait': 0}
        df['label'] = df['label'].map(label_mapping)
        text_column = 'title'
        label_column = 'label'
        train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df[label_column])
        train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=42, stratify=train_val_df[label_column])
    elif dataset_name.lower() == 'liar':
        train_tsv = os.path.join(dataset_path, 'train.tsv')
        val_tsv = os.path.join(dataset_path, 'valid.tsv')
        test_tsv = os.path.join(dataset_path, 'test.tsv')
        col_names = [
            "id", "label", "statement", "subject", "speaker", "job", "state",
            "party", "barely_true_counts", "false_counts", "half_true_counts",
            "mostly_true_counts", "pants_fire_counts", "context"
        ]
        use_cols = ['label', 'statement']
        train_df = pd.read_csv(train_tsv, sep='\t', quoting=3, names=col_names, usecols=use_cols)
        val_df = pd.read_csv(val_tsv, sep='\t', quoting=3, names=col_names, usecols=use_cols)
        test_df = pd.read_csv(test_tsv, sep='\t', quoting=3, names=col_names, usecols=use_cols)

        true_set = {'true', 'mostly-true', 'half-true'}
        false_set = {'false', 'barely-true', 'pants-fire'}

        def map_label(label):
            if label in true_set:
                return 1
            else:
                return 0

        train_df['bin_label'] = train_df['label'].apply(map_label)
        val_df['bin_label'] = val_df['label'].apply(map_label)
        test_df['bin_label'] = test_df['label'].apply(map_label)

        text_column = 'statement'
        label_column = 'bin_label'
    else:
        logger.error('Unknown dataset name provided.')
        raise ValueError('Unknown dataset name')

    if dataset_name.lower() == 'mpid':
        combined_train_val_text = pd.concat([train_df[text_column], validation_df[text_column]], ignore_index=True)
        vectorizer = TfidfVectorizer(max_features=5000)
        vectorizer.fit(combined_train_val_text)
        node_features_train = vectorizer.transform(train_df[text_column]).toarray()
        y_train = train_df[label_column].values
        node_features_val = vectorizer.transform(validation_df[text_column]).toarray()
        y_val = validation_df[label_column].values
        node_features_test = vectorizer.transform(test_df[text_column]).toarray()
        y_test = test_df[label_column].values
        logger.info(f"MPID Data loaded with {len(y_train)} training samples, {len(y_val)} validation samples, and {len(y_test)} test samples.")
        return {
            'train': (node_features_train, y_train),
            'validation': (node_features_val, y_val),
            'test': (node_features_test, y_test)
        }
    else:
        combined_train_val_df = pd.concat([train_df, val_df], ignore_index=True)
        vectorizer = TfidfVectorizer(max_features=5000)
        vectorizer.fit(combined_train_val_df[text_column])
        node_features_train = vectorizer.transform(train_df[text_column]).toarray()
        y_train = train_df[label_column].values
        node_features_val = vectorizer.transform(val_df[text_column]).toarray()
        y_val = val_df[label_column].values
        node_features_test = vectorizer.transform(test_df[text_column]).toarray()
        y_test = test_df[label_column].values
        logger.info(f"{dataset_name.upper()} Data loaded with {len(y_train)} training samples, {len(y_val)} validation samples, and {len(y_test)} test samples.")
        return {
            'train': (node_features_train, y_train),
            'validation': (node_features_val, y_val),
            'test': (node_features_test, y_test)
        }

class GNN(torch.nn.Module):
    def __init__(
        self, model_type='gin', input_dim=5000,
        hidden_dim=64, output_dim=2, pretrain_output_dim=1, **kwargs
    ):
        super(GNN, self).__init__()
        logger.debug(f"Initializing GNN model with type: {model_type}")
        self.model_type = model_type.lower()

        if self.model_type == 'gcn':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif self.model_type == 'graphsage':
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        elif self.model_type == 'gat':
            self.conv1 = GATConv(input_dim, hidden_dim, heads=8, dropout=0.6)
            self.conv2 = GATConv(hidden_dim * 8, hidden_dim, heads=1, concat=False, dropout=0.6)
        elif self.model_type == 'gin':
            self.conv1 = GINConv(torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            ))
            self.conv2 = GINConv(torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            ))
        elif self.model_type == 'chebnet':
            K = kwargs.get('K', 3)
            self.conv1 = ChebConv(input_dim, hidden_dim, K=K)
            self.conv2 = ChebConv(hidden_dim, hidden_dim, K=K)
        elif self.model_type == 'appnp':
            K = kwargs.get('K', 10)
            alpha = kwargs.get('alpha', 0.1)
            self.lin1 = torch.nn.Linear(input_dim, hidden_dim)
            self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.prop = APPNP(K=K, alpha=alpha)
        elif self.model_type == 'sgc':
            K = kwargs.get('K', 2)
            self.conv = SGConv(input_dim, output_dim, K=K, cached=False)
        elif self.model_type == 'edgeconv':
            self.nn = torch.nn.Sequential(
                torch.nn.Linear(2 * input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )
            self.conv1 = EdgeConv(self.nn)
            self.fc = torch.nn.Linear(hidden_dim, output_dim)
            self.k = kwargs.get('k', 6)
        elif self.model_type == 'transformerconv':
            heads = kwargs.get('heads', 4)
            concat = kwargs.get('concat', False)
            self.conv1 = TransformerConv(input_dim, hidden_dim, heads=heads, concat=concat)
            self.conv2 = TransformerConv(hidden_dim, hidden_dim, heads=heads, concat=concat)
        elif self.model_type == 'tagconv':
            K = kwargs.get('K', 3)
            self.conv1 = TAGConv(input_dim, hidden_dim, K=K)
            self.conv2 = TAGConv(hidden_dim, hidden_dim, K=K)
        elif self.model_type == 'jknet':
            num_layers = kwargs.get('num_layers', 3)
            mode = kwargs.get('mode', 'cat')
            self.convs = torch.nn.ModuleList()
            self.convs.append(GINConv(torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )))
            for _ in range(num_layers - 1):
                self.convs.append(GINConv(torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim)
                )))
            self.jump = JumpingKnowledge(mode)
            if mode == 'cat':
                self.fc = torch.nn.Linear(num_layers * hidden_dim, output_dim)
            else:
                self.fc = torch.nn.Linear(hidden_dim, output_dim)
        elif self.model_type == 'feastconv':
            self.conv1 = FeaStConv(input_dim, hidden_dim)
            self.conv2 = FeaStConv(hidden_dim, hidden_dim)
        else:
            logger.error(f"Unknown model type: {model_type}")
            raise ValueError(f"Unknown model type: {model_type}")

        if self.model_type not in ['sgc', 'appnp', 'edgeconv', 'jknet']:
            self.fc = torch.nn.Linear(hidden_dim, output_dim)
            self.pretrain_fc = torch.nn.Linear(hidden_dim, 1)
        elif self.model_type == 'appnp':
            self.pretrain_fc = torch.nn.Linear(hidden_dim, 1)
        elif self.model_type == 'sgc':
            self.pretrain_fc = torch.nn.Linear(output_dim, 1)
        elif self.model_type == 'edgeconv':
            self.pretrain_fc = torch.nn.Linear(output_dim, 1)
        elif self.model_type == 'jknet':
            mode = kwargs.get('mode', 'cat')
            num_layers = kwargs.get('num_layers', 3)
            if mode == 'cat':
                self.pretrain_fc = torch.nn.Linear(num_layers * hidden_dim, 1)
            else:
                self.pretrain_fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data, pretrain=False):
        x = data.x
        edge_index = data.edge_index

        if self.model_type in ['gcn', 'graphsage', 'gat', 'gin', 'chebnet', 'transformerconv', 'tagconv', 'feastconv']:
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            if pretrain:
                x = self.pretrain_fc(torch.mean(x, dim=0, keepdim=True))
            else:
                x = self.fc(x)
        elif self.model_type == 'appnp':
            x = F.relu(self.lin1(x))
            x = F.dropout(x, training=self.training)
            x = self.lin2(x)
            x = self.prop(x, edge_index)
            if pretrain:
                x = self.pretrain_fc(torch.mean(x, dim=0, keepdim=True))
        elif self.model_type == 'sgc':
            x = self.conv(x, edge_index)
            if pretrain:
                x = self.pretrain_fc(torch.mean(x, dim=0, keepdim=True))
        elif self.model_type == 'edgeconv':
            batch = data.batch if hasattr(data, 'batch') else None
            if edge_index is None or edge_index.numel() == 0:
                edge_index = knn_graph(x, k=self.k, batch=batch)
            x = F.relu(self.conv1(x, edge_index))
            if pretrain:
                x = self.pretrain_fc(torch.mean(x, dim=0, keepdim=True))
            else:
                x = self.fc(x)
        elif self.model_type == 'jknet':
            xs = []
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
                xs.append(x)
            x = self.jump(xs)
            if pretrain:
                x = self.pretrain_fc(torch.mean(x, dim=0, keepdim=True))
            else:
                x = self.fc(x)
        else:
            logger.error(f"Unknown model type: {self.model_type}")
            raise ValueError(f"Unknown model type: {self.model_type}")
        return x

def pretrain(model, data_loader, optimizer, device, epochs=5):
    logger.info("Starting pre-training phase.")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in tqdm(data_loader, desc=f"Pre-training Epoch {epoch+1}/{epochs}"):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data, pretrain=True)
            output = output.view(-1)
            edge_labels = torch.randint(0, 2, (output.size(0),)).float().to(device)
            loss = F.mse_loss(output, edge_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        logger.info(f"Pre-training Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

def run_cross_validation(dataset_name, model_type='gin', **model_kwargs):
    logger.info(f"Running cross-validation for dataset: {dataset_name} with model: {model_type}")
    data_splits = load_and_prepare_data(dataset_name)
    node_features_train, y_train = data_splits['train']
    node_features_val, y_val = data_splits['validation']
    node_features_test, y_test = data_splits['test']

    combined_features = np.vstack((node_features_train, node_features_val))
    combined_labels = np.concatenate((y_train, y_val))

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics_results = []

    multiclass = True if len(np.unique(np.concatenate([combined_labels, y_test]))) > 2 else False

    final_results_dir = os.path.join(BASE_PATH, 'final-results')
    if not os.path.exists(final_results_dir):
        os.makedirs(final_results_dir)
    dataset_dir = os.path.join(final_results_dir, dataset_name.lower())
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    fold_number = 1
    for train_idx, test_idx_split in kf.split(combined_features, combined_labels):
        logger.info(f"Starting Fold {fold_number}")
        train_idx_split, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42, stratify=combined_labels[train_idx])

        x_train = torch.tensor(combined_features[train_idx_split], dtype=torch.float)
        y_train_split = torch.tensor(combined_labels[train_idx_split], dtype=torch.long)

        x_val = torch.tensor(combined_features[val_idx], dtype=torch.float)
        y_val_split = torch.tensor(combined_labels[val_idx], dtype=torch.long)

        x_test = torch.tensor(node_features_test, dtype=torch.float)
        y_test_split = torch.tensor(y_test, dtype=torch.long)

        edge_index_train = knn_graph(x_train, k=5, batch=None, loop=False)
        edge_index_val = knn_graph(x_val, k=5, batch=None, loop=False)
        edge_index_test = knn_graph(x_test, k=5, batch=None, loop=False)

        train_data = Data(x=x_train, edge_index=edge_index_train, y=y_train_split)
        val_data = Data(x=x_val, edge_index=edge_index_val, y=y_val_split)
        test_data = Data(x=x_test, edge_index=edge_index_test, y=y_test_split)

        train_loader = DataLoader([train_data], batch_size=1, shuffle=True)
        val_loader = DataLoader([val_data], batch_size=1, shuffle=False)
        test_loader = DataLoader([test_data], batch_size=1, shuffle=False)

        model = GNN(
            model_type=model_type,
            input_dim=combined_features.shape[1],
            hidden_dim=model_kwargs.get('hidden_dim', 64),
            output_dim=len(np.unique(combined_labels)),
            **model_kwargs
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=model_kwargs.get('learning_rate', 0.001))

        logger.info("Starting pre-training")
        pretrain(model, train_loader, optimizer, device, epochs=model_kwargs.get('pretrain_epochs',5))

        max_epochs = model_kwargs.get('max_epochs', 500)
        patience = model_kwargs.get('patience', 10)
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping = False
        stop_epoch = max_epochs
        best_model_state = None

        # ZMIANA: Dodajemy listy do śledzenia metryk w czasie
        train_losses = []
        val_losses = []
        test_losses = []
        test_accuracies = []
        test_f1_scores = []
        test_mccs = []
        test_aucrocs = []
        test_aucprs = []

        logger.info("Starting training phase with Early Stopping.")
        for epoch in range(max_epochs):
            model.train()
            total_loss = 0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data, pretrain=False)
                loss = F.cross_entropy(out, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_loader)

            # Val loss
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    out = model(data, pretrain=False)
                    loss = F.cross_entropy(out, data.y)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)

            # ZMIANA: Test loss i metryki
            test_loss = 0
            test_y_true_ep, test_y_pred_ep, test_y_probs_ep = [], [], []
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    out = model(data, pretrain=False)
                    loss = F.cross_entropy(out, data.y)
                    test_loss += loss.item()
                    pred = out.argmax(dim=1).cpu().numpy()
                    test_y_pred_ep.extend(pred)
                    test_y_true_ep.extend(data.y.cpu().numpy())
                    test_y_probs_ep.extend(out.softmax(dim=1).cpu().numpy())
            avg_test_loss = test_loss / len(test_loader)
            test_metrics_epoch = calculate_metrics(np.array(test_y_true_ep), np.array(test_y_pred_ep), np.array(test_y_probs_ep), multiclass=multiclass)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            test_losses.append(avg_test_loss)
            test_accuracies.append(test_metrics_epoch['accuracy'])
            test_f1_scores.append(test_metrics_epoch['f1_score'])
            test_mccs.append(test_metrics_epoch['mcc'])
            test_aucrocs.append(test_metrics_epoch['aucroc'])
            test_aucprs.append(test_metrics_epoch['aucpr'])

            logger.info(
                f"Epoch {epoch+1}/{max_epochs}, "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Test Loss: {avg_test_loss:.4f}, "  # ZMIANA: Test Loss
                f"Test ACC: {test_metrics_epoch['accuracy']:.4f}, "
                f"Test F1: {test_metrics_epoch['f1_score']:.4f}, "
                f"Test MCC: {test_metrics_epoch['mcc']:.4f}, "
                f"Test AUCROC: {test_metrics_epoch['aucroc']:.4f}, "
                f"Test AUCPR: {test_metrics_epoch['aucpr']:.4f}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    stop_epoch = epoch + 1
                    early_stopping = True
                    logger.info(f"Early Stopping triggered at epoch {stop_epoch}")
                    break

        if best_model_state:
            model.load_state_dict(best_model_state)

        model.eval()
        train_y_true, train_y_pred, train_y_probs = [], [], []
        with torch.no_grad():
            for data in train_loader:
                data = data.to(device)
                out = model(data, pretrain=False)
                pred = out.argmax(dim=1).cpu().numpy()
                train_y_pred.extend(pred)
                train_y_true.extend(data.y.cpu().numpy())
                train_y_probs.extend(out.softmax(dim=1).cpu().numpy())
        train_metrics = calculate_metrics(np.array(train_y_true), np.array(train_y_pred), np.array(train_y_probs), multiclass=multiclass)

        test_y_true, test_y_pred, test_y_probs = [], [], []
        start_time = time.time()
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out = model(data, pretrain=False)
                pred = out.argmax(dim=1).cpu().numpy()
                test_y_pred.extend(pred)
                test_y_true.extend(data.y.cpu().numpy())
                test_y_probs.extend(out.softmax(dim=1).cpu().numpy())
        inference_time = (time.time() - start_time) / len(test_loader)
        test_metrics = calculate_metrics(np.array(test_y_true), np.array(test_y_pred), np.array(test_y_probs), multiclass=multiclass)

        logger.info(
            f"Fold {fold_number} training metrics - "
            f"Accuracy: {train_metrics['accuracy']:.4f}, "
            f"F1-Score: {train_metrics['f1_score']:.4f}"
        )
        logger.info(
            f"Fold {fold_number} test metrics - "
            f"Accuracy: {test_metrics['accuracy']:.4f}, "
            f"F1-Score: {test_metrics['f1_score']:.4f}, "
            f"ROC-AUC: {test_metrics['aucroc']:.4f}, "
            f"MCC: {test_metrics['mcc']:.4f}, "
            f"AUCPR: {test_metrics['aucpr']:.4f}, "
            f"ARI: {test_metrics['ARI']:.4f}, "
            f"NMI: {test_metrics['NMI']:.4f}"
        )

        metrics = {
            'Dataset': dataset_name.lower(),
            'Model': model_type,
            'Fold': fold_number,
            'train_accuracy': train_metrics['accuracy'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'train_f1': train_metrics['f1_score'],
            'train_mcc': train_metrics['mcc'],
            'train_aucroc': train_metrics['aucroc'],
            'train_aucpr': train_metrics['aucpr'],
            'train_sensitivity': train_metrics['sensitivity'],
            'train_specificity': train_metrics['specificity'],
            'train_ppv': train_metrics['ppv'],
            'train_npv': train_metrics['npv'],
            'train_ARI': train_metrics['ARI'],
            'train_NMI': train_metrics['NMI'],

            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1_score'],
            'test_mcc': test_metrics['mcc'],
            'test_aucroc': test_metrics['aucroc'],
            'test_aucpr': test_metrics['aucpr'],
            'test_sensitivity': test_metrics['sensitivity'],
            'test_specificity': test_metrics['specificity'],
            'test_ppv': test_metrics['ppv'],
            'test_npv': test_metrics['npv'],
            'test_ARI': test_metrics['ARI'],
            'test_NMI': test_metrics['NMI'],

            'Inference Time (ms)': inference_time * 1000,
            'Stop Epoch': stop_epoch,
            'Early Stopping': early_stopping
        }

        metrics_results.append(metrics)

        epochs_range = range(1, len(train_losses)+1)
        plt.figure(figsize=(10,6))
        plt.plot(epochs_range, train_losses, label='Train Loss')
        plt.plot(epochs_range, val_losses, label='Val Loss')
        plt.plot(epochs_range, test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{dataset_name} - {model_type} - Fold {fold_number} Loss')
        plt.legend()
        loss_plot_path = os.path.join(dataset_dir, f'{dataset_name.lower()}_{model_type}_fold{fold_number}_loss_plot.png')
        plt.savefig(loss_plot_path)
        plt.close()

        plt.figure(figsize=(10,6))
        plt.plot(epochs_range, test_accuracies, label='Test Accuracy', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'{dataset_name} - {model_type} - Fold {fold_number} Test Accuracy')
        plt.legend()
        acc_plot_path = os.path.join(dataset_dir, f'{dataset_name.lower()}_{model_type}_fold{fold_number}_accuracy_plot.png')
        plt.savefig(acc_plot_path)
        plt.close()

        plt.figure(figsize=(10,6))
        plt.plot(epochs_range, test_f1_scores, label='Test F1-Score', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('F1-Score')
        plt.title(f'{dataset_name} - {model_type} - Fold {fold_number} Test F1-Score')
        plt.legend()
        f1_plot_path = os.path.join(dataset_dir, f'{dataset_name.lower()}_{model_type}_fold{fold_number}_f1_plot.png')
        plt.savefig(f1_plot_path)
        plt.close()

        plt.figure(figsize=(10,6))
        plt.plot(epochs_range, test_aucrocs, label='Test AUCROC', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('AUCROC')
        plt.title(f'{dataset_name} - {model_type} - Fold {fold_number} Test AUCROC')
        plt.legend()
        aucroc_plot_path = os.path.join(dataset_dir, f'{dataset_name.lower()}_{model_type}_fold{fold_number}_aucroc_plot.png')
        plt.savefig(aucroc_plot_path)
        plt.close()

        plt.figure(figsize=(10,6))
        plt.plot(epochs_range, test_aucprs, label='Test AUCPR', color='purple')
        plt.xlabel('Epoch')
        plt.ylabel('AUCPR')
        plt.title(f'{dataset_name} - {model_type} - Fold {fold_number} Test AUCPR')
        plt.legend()
        aucpr_plot_path = os.path.join(dataset_dir, f'{dataset_name.lower()}_{model_type}_fold{fold_number}_aucpr_plot.png')
        plt.savefig(aucpr_plot_path)
        plt.close()

        plt.figure(figsize=(10,6))
        plt.plot(epochs_range, test_mccs, label='Test MCC', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('MCC')
        plt.title(f'{dataset_name} - {model_type} - Fold {fold_number} Test MCC')
        plt.legend()
        mcc_plot_path = os.path.join(dataset_dir, f'{dataset_name.lower()}_{model_type}_fold{fold_number}_mcc_plot.png')
        plt.savefig(mcc_plot_path)
        plt.close()

        fold_number += 1

    results_df = pd.DataFrame(metrics_results)

    mean_results = results_df.mean(numeric_only=True)
    std_results = results_df.std(numeric_only=True)

    mean_metrics = mean_results.to_dict()
    mean_metrics['Dataset'] = 'Mean'
    mean_metrics['Model'] = ''
    mean_metrics['Fold'] = ''
    std_metrics = std_results.to_dict()
    std_metrics['Dataset'] = 'Std'
    std_metrics['Model'] = ''
    std_metrics['Fold'] = ''
    results_df = pd.concat([results_df, pd.DataFrame([mean_metrics, std_metrics])], ignore_index=True)

    logger.info("\nCross-Validation Results (Mean ± Std):")
    metrics_to_display = [
        'train_accuracy', 'train_precision', 'train_recall', 'train_f1',
        'test_accuracy', 'test_precision', 'test_recall', 'test_f1',
        'test_aucroc', 'test_aucpr', 'test_mcc', 'test_sensitivity', 'test_specificity',
        'test_ppv', 'test_npv', 'test_ARI', 'test_NMI', 'Inference Time (ms)', 'Stop Epoch'
    ]
    for metric in metrics_to_display:
        if metric in mean_results:
            mean = mean_results[metric]
            std = std_results[metric]
            logger.info(f"{metric}: {mean:.4f} ± {std:.4f}")

    save_path = os.path.join(dataset_dir, f'{model_type}_gnn_crossval_results.csv')
    results_df.to_csv(save_path, index=False)
    logger.info(f"Results saved to {save_path}")


if __name__ == '__main__':
    dataset_list = ['liar','kaggle','fakenewsnet','welfake','covid-19','click-id','mpid']

    model_list = [
        {'model_type': 'sgc', 'K': 2},
        {'model_type': 'appnp', 'K': 10, 'alpha': 0.1},
        {'model_type': 'gcn'},
        {'model_type': 'graphsage'},
        {'model_type': 'gat'},
        {'model_type': 'gin'},
        {'model_type': 'chebnet', 'K': 3},
        {'model_type': 'jknet', 'num_layers': 3, 'mode': 'cat'},
        {'model_type': 'tagconv', 'K': 3},
        {'model_type': 'feastconv'},
    ]
    for dataset_name in dataset_list:
        for model_info in model_list:
            model_kwargs = model_info.copy()
            model_type = model_kwargs.pop('model_type')
            logger.info(f"\nRunning Dataset: {dataset_name.lower()}, Model: {model_type}")
            run_cross_validation(dataset_name, model_type=model_type, **model_kwargs)

