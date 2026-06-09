import os
import random
import argparse
import re

os.environ['PYTHONHASHSEED'] = '3407'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch

EXPECTED_TORCH_VERSION = "2.5.1"
EXPECTED_CUDA_RUNTIME = "12.1"


def validate_torch_build():
    torch_version = torch.__version__.split("+", 1)[0]
    if (
        torch_version != EXPECTED_TORCH_VERSION
        or torch.version.cuda != EXPECTED_CUDA_RUNTIME
    ):
        raise RuntimeError(
            "Incompatible PyTorch build. scHetGTL requires "
            f"torch=={EXPECTED_TORCH_VERSION}+cu121, but found "
            f"torch=={torch.__version__} with CUDA {torch.version.cuda}. "
            "Activating an existing Conda environment does not update it. "
            "Remove and recreate the environment from environment.yml."
        )


validate_torch_build()

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_SPARSE
import scanpy as sc

from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.metrics import adjusted_rand_score, precision_score, recall_score, f1_score
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
from torch_geometric.utils import negative_sampling, to_undirected

from settinng import available_datasets, get_dataset_config

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except RuntimeError:
        pass

    sc.settings.seed = seed
    torch_geometric.seed_everything(seed)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train scHetGTL using the fixed settings in settinng.py."
    )
    parser.add_argument("--dataset", choices=available_datasets(), default="pbmc")
    parser.add_argument("--list-datasets", action="store_true")
    return parser.parse_args()


def build_config(args):
    config = get_dataset_config(args.dataset)

    if len(config["num_neighbors"]) != config["num_layers"]:
        raise ValueError(
            "num_neighbors must contain one value for each GNN layer "
            f"({config['num_layers']} layers configured)."
        )

    return config


def normalised_average_precision(y_true, y_pred):
    try:
        from sklearn.metrics.ranking import _binary_clf_curve
    except ImportError:
        from sklearn.metrics._ranking import _binary_clf_curve

    fps, tps, thresholds = _binary_clf_curve(y_true, y_pred, pos_label=None, sample_weight=None)
    n_pos = np.array(y_true).sum()
    n_neg = (1 - np.array(y_true)).sum()
    precision = tps * n_pos / (tps * n_pos + fps * n_neg)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    precision, recall, thresholds = np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]
    return -np.sum(np.diff(recall) * np.array(precision)[:-1])

def closed_set_acc(preds, labels):
    return accuracy_score(labels, preds)

def compute_auroc(open_set_preds, open_set_labels):
    return roc_auc_score(open_set_labels, open_set_preds)

def compute_aupr(open_set_preds, open_set_labels, normalised_ap=False):
    if normalised_ap:
        return normalised_average_precision(open_set_labels, open_set_preds)
    else:
        return average_precision_score(open_set_labels, open_set_preds)

def compute_oscr(x1, x2, pred, labels):
    x1, x2 = -x1, -x2
    correct = (pred == labels)
    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)
    CCR = [0 for x in range(n + 2)]
    FPR = [0 for x in range(n + 2)]
    idx = predict.argsort()
    s_k_target = k_target[idx]
    s_u_target = u_target[idx]
    for k in range(n - 1):
        CC = s_k_target[k + 1:].sum()
        FP = s_u_target[k:].sum()
        CCR[k] = float(CC) / float(len(x1))
        FPR[k] = float(FP) / float(len(x2))
    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n + 1] = 1.0
    FPR[n + 1] = 1.0
    ROC = sorted(zip(FPR, CCR), reverse=True)
    OSCR = 0
    for j in range(n + 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0
        OSCR = OSCR + h * w
    return OSCR

def osr_evaluator(kn_data_closed_pr, kn_data_closed_gt, kn_data_open_pr, unk_data_open_pr=None):
    n_kn = len(kn_data_closed_pr)
    kn_data_acc = closed_set_acc(kn_data_closed_pr, kn_data_closed_gt)
    auroc, aupr, oscr = -1, -1, -1

    if (unk_data_open_pr is not None) and len(unk_data_open_pr) >= 1:
        n_unk = len(unk_data_open_pr)
        open_set_pred = list(kn_data_open_pr) + list(unk_data_open_pr)
        open_set_gt = list(np.zeros(n_kn)) + list(np.ones(n_unk))
        open_set_pred = np.array(open_set_pred)
        open_set_gt = np.array(open_set_gt)

        auroc = compute_auroc(open_set_pred, open_set_gt)
        aupr = compute_aupr(open_set_pred, open_set_gt, normalised_ap=False)

        open_set_preds_known_cls = open_set_pred[~open_set_gt.astype('bool')]
        open_set_preds_unknown_cls = open_set_pred[open_set_gt.astype('bool')]
        closed_set_preds_pred_cls = kn_data_closed_pr
        labels_known_cls = np.array(kn_data_closed_gt)

        oscr = compute_oscr(
            open_set_preds_known_cls,
            open_set_preds_unknown_cls,
            closed_set_preds_pred_cls,
            labels_known_cls
        )

    return kn_data_acc, auroc, aupr, oscr

def read_labels(adata, cfg, modality):
    label_column = cfg.get(f"{modality}_label_column", "cell_type")

    if label_column not in adata.obs.columns:
        raise ValueError(
            f"Label column '{label_column}' was not found in {modality} AnnData."
        )
    labels = adata.obs[label_column]

    labels = labels.astype(str).to_numpy()
    if len(labels) != adata.n_obs:
        raise ValueError(
            f"{modality} label count ({len(labels)}) does not match "
            f"the number of cells ({adata.n_obs})."
        )
    return labels


def load_real_data(cfg):
    print(f"Loading dataset: {cfg['dataset']}")
    adata_rna = sc.read_h5ad(cfg["rna_path"])
    adata_atac = sc.read_h5ad(cfg["atac_path"])

    rna_types_str = read_labels(adata_rna, cfg, "rna")
    atac_types_str = read_labels(adata_atac, cfg, "atac")

    unique_labels = np.unique(rna_types_str)
    label_to_id = {label: i for i, label in enumerate(unique_labels)}

    rna_labels_int = np.array([label_to_id[l] for l in rna_types_str])
    atac_labels_int = np.array([label_to_id.get(l, -1) for l in atac_types_str])

    print(f"Loaded {adata_rna.n_obs} RNA cells, {adata_atac.n_obs} ATAC cells.")
    print(f"RNA reference classes: {len(unique_labels)}")
    print(f"ATAC unknown cells: {int((atac_labels_int < 0).sum())}")

    return adata_rna, adata_atac, rna_labels_int, atac_labels_int, unique_labels

def preprocess(adata_rna, adata_atac, cfg):
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.normalize_total(adata_atac, target_sum=1e4)
    sc.pp.log1p(adata_atac)

    sc.pp.highly_variable_genes(adata_rna, n_top_genes=cfg["n_top_genes"], flavor='seurat')
    hvgs_rna = adata_rna.var[adata_rna.var.highly_variable].index.tolist()

    common_genes = [gene for gene in hvgs_rna if gene in adata_atac.var_names]
    common_genes = sorted(common_genes)
    print(f"Target top genes: {cfg['n_top_genes']}, found {len(common_genes)} common highly variable genes.")

    if len(common_genes) == 0:
        raise ValueError("No common genes found between RNA and ATAC.")

    adata_rna = adata_rna[:, common_genes].copy()
    adata_atac = adata_atac[:, common_genes].copy()

    sc.pp.scale(adata_rna)
    sc.pp.scale(adata_atac)

    def to_tensor(x):
        if scipy.sparse.issparse(x):
            return torch.FloatTensor(x.toarray())
        return torch.FloatTensor(x)

    rna_feat = to_tensor(adata_rna.X)
    atac_feat = to_tensor(adata_atac.X)

    print(f"Running PCA (n={cfg['n_comps']}) for RNA intra-modal KNN graphs...")
    pca_rna = PCA(n_components=cfg["n_comps"], random_state=cfg["seed"])
    rna_pca_feat = torch.FloatTensor(pca_rna.fit_transform(rna_feat.numpy()))

    return rna_feat, atac_feat, rna_pca_feat

def build_knn_graph(
    feat,
    k=10,
    metric='cosine',
    algorithm='brute',
    undirected=True,
    edge_direction="neighbor_to_cell",
):
    if feat.shape[0] <= k:
        raise ValueError(f"knn_k={k} must be smaller than the number of cells.")

    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm=algorithm)
    nn.fit(feat)
    indices = nn.kneighbors(feat, return_distance=False)[:, 1:]
    num_nodes = feat.shape[0]
    row = np.repeat(np.arange(num_nodes), k)
    col = indices.flatten()

    if edge_direction == "neighbor_to_cell":
        edge_index = torch.LongTensor(np.vstack([col, row]))
    elif edge_direction == "cell_to_neighbor":
        edge_index = torch.LongTensor(np.vstack([row, col]))
    else:
        raise ValueError(f"Unsupported knn_edge_direction: {edge_direction}")

    if undirected:
        edge_index = to_undirected(edge_index)
    return edge_index


def normalize_cell_id(value):
    value = str(value).split("#")[-1]
    return re.sub(r"-\d+$", "", value)


def map_cell_ids(values, obs_names, modality):
    exact_map = {str(name): i for i, name in enumerate(obs_names)}
    mapped = pd.Series(values).astype(str).map(exact_map)

    if mapped.notna().all():
        return mapped

    normalized_map = {}
    for i, name in enumerate(obs_names):
        key = normalize_cell_id(name)
        if key in normalized_map:
            raise ValueError(f"Non-unique normalized {modality} cell ID: {key}")
        normalized_map[key] = i

    fallback = pd.Series(values).astype(str).map(normalize_cell_id).map(normalized_map)
    return mapped.fillna(fallback)


def filter_anchors(df, cfg):
    if "score" not in df.columns:
        return df.copy()

    threshold_effector = cfg.get("score_threshold_effector")
    threshold_others = cfg.get("score_threshold_others")
    if (
        threshold_effector is not None
        and threshold_others is not None
        and "rna_clean" in df.columns
    ):
        effector_labels = {label.lower() for label in cfg["effector_labels"]}
        is_effector = df["rna_clean"].astype(str).str.lower().isin(effector_labels)
        keep = (
            (is_effector & (df["score"] >= threshold_effector))
            | (~is_effector & (df["score"] >= threshold_others))
        )
        return df[keep].copy()

    return df[df["score"] >= cfg["score_threshold"]].copy()


def build_cross_edges_from_anchors(anchor_path, cfg, rna_obs_names, atac_obs_names):
    if not os.path.exists(anchor_path):
        raise FileNotFoundError(f"Anchor file not found: {anchor_path}")

    df = pd.read_csv(anchor_path)
    df_filtered = filter_anchors(df, cfg)

    cols = df.columns.tolist()
    if 'ref_index' in cols and 'query_index' in cols:
        rna_indices = pd.to_numeric(df_filtered['ref_index'], errors="coerce") - 1
        atac_indices = pd.to_numeric(df_filtered['query_index'], errors="coerce") - 1
    else:
        col_ref, col_query = cols[0], cols[1]
        rna_indices = map_cell_ids(df_filtered[col_ref], rna_obs_names, "RNA")
        atac_indices = map_cell_ids(df_filtered[col_query], atac_obs_names, "ATAC")

    valid_mask = (
        rna_indices.notna()
        & atac_indices.notna()
        & rna_indices.between(0, len(rna_obs_names) - 1)
        & atac_indices.between(0, len(atac_obs_names) - 1)
    )
    rna_indices = rna_indices[valid_mask].to_numpy(dtype=np.int64)
    atac_indices = atac_indices[valid_mask].to_numpy(dtype=np.int64)

    edge_index_r2a = np.vstack([rna_indices, atac_indices])
    edge_index_a2r = np.vstack([atac_indices, rna_indices])

    return torch.LongTensor(edge_index_r2a), torch.LongTensor(edge_index_a2r)


def align_neighbor_features(adata, atac_obs_names):
    candidates = [adata.obs_names.astype(str)]
    for column in ("original_obs_name", "barcode"):
        if column in adata.obs.columns:
            candidates.append(adata.obs[column].astype(str).to_numpy())

    target = pd.Series(atac_obs_names).astype(str)
    for names in candidates:
        mapped = map_cell_ids(target, names, "ATAC neighbor")
        if mapped.notna().all():
            indices = mapped.to_numpy(dtype=np.int64)
            return adata[indices, :].X

    raise ValueError("Could not align ATAC neighbor features to ATAC cell IDs.")


def load_atac_neighbor_features(cfg, atac_obs_names):
    path = cfg["atac_neighbor_path"]
    feature_format = cfg.get("atac_neighbor_format", "h5ad")
    print(f"Loading ATAC neighbor features from {path}")

    if feature_format == "h5ad":
        adata = sc.read_h5ad(path)
        features = align_neighbor_features(adata, atac_obs_names)
    elif feature_format == "npz":
        features = scipy.sparse.load_npz(path)
    else:
        raise ValueError(f"Unsupported ATAC neighbor format: {feature_format}")

    if scipy.sparse.issparse(features):
        features = features.toarray()
    features = np.asarray(features)

    if features.shape[0] != len(atac_obs_names):
        raise ValueError(
            "ATAC neighbor feature rows do not match the number of ATAC cells."
        )
    return features


def build_hetero_graph(rna_feat, atac_feat, rna_pca_feat, rna_obs_names, atac_obs_names, cfg):
    data = HeteroData()
    data['rna'].x = rna_feat
    data['atac'].x = atac_feat

    data['rna', 'knn', 'rna'].edge_index = build_knn_graph(
        rna_pca_feat.numpy(),
        k=cfg["knn_k"],
        metric='cosine',
        undirected=cfg.get("knn_undirected", True),
        edge_direction=cfg.get("knn_edge_direction", "neighbor_to_cell"),
    )

    atac_neighbor_features = load_atac_neighbor_features(cfg, atac_obs_names)
    data['atac', 'knn', 'atac'].edge_index = build_knn_graph(
        atac_neighbor_features,
        k=cfg["knn_k"],
        metric='cosine',
        algorithm='brute',
        undirected=cfg.get("knn_undirected", True),
        edge_direction=cfg.get("knn_edge_direction", "neighbor_to_cell"),
    )

    edge_index_r2a, edge_index_a2r = build_cross_edges_from_anchors(
        cfg["anchor_path"],
        cfg,
        rna_obs_names,
        atac_obs_names
    )

    data['rna', 'knn', 'atac'].edge_index = edge_index_r2a
    data['atac', 'knn', 'rna'].edge_index = edge_index_a2r

    return data

class HeteroGNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.in_lin = nn.Linear(in_dim, hidden_dim)
        self.modality_emb = nn.Embedding(2, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('rna', 'knn', 'rna'): SAGEConv(hidden_dim, hidden_dim),
                ('atac', 'knn', 'atac'): SAGEConv(hidden_dim, hidden_dim),
                ('rna', 'knn', 'atac'): SAGEConv(hidden_dim, hidden_dim),
                ('atac', 'knn', 'rna'): SAGEConv(hidden_dim, hidden_dim),
            }, aggr='sum')
            self.convs.append(conv)

        self.out_lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {key: self.in_lin(x) for key, x in x_dict.items()}
        x_dict['rna'] = x_dict['rna'] + self.modality_emb(
            torch.zeros(x_dict['rna'].size(0), dtype=torch.long, device=x_dict['rna'].device)
        )
        x_dict['atac'] = x_dict['atac'] + self.modality_emb(
            torch.ones(x_dict['atac'].size(0), dtype=torch.long, device=x_dict['atac'].device)
        )

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        return {key: self.out_lin(x) for key, x in x_dict.items()}

class Classifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, num_classes)
        )

    def forward(self, z):
        return self.fc(z)

class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, z):
        return self.fc(z)

class HeteroGAE(nn.Module):
    def __init__(self, encoder, num_classes, use_decoder, feat_dim):
        super().__init__()
        self.encoder = encoder
        self.classifier = Classifier(encoder.out_lin.out_features, num_classes)

        self.centers = nn.Parameter(torch.randn(num_classes, encoder.out_lin.out_features))

        self.use_decoder = use_decoder
        if use_decoder:
            self.decoder_rna = Decoder(encoder.out_lin.out_features, feat_dim)
            self.decoder_atac = Decoder(encoder.out_lin.out_features, feat_dim)

    def forward(self, x_dict, edge_index_dict):
        return self.encoder(x_dict, edge_index_dict)

    def classify(self, z_rna):
        return self.classifier(z_rna)

    def decode(self, z_dict):
        if not self.use_decoder:
            return None
        return {
            'rna': self.decoder_rna(z_dict['rna']),
            'atac': self.decoder_atac(z_dict['atac'])
        }

def filter_perfect_edges(graph_data):
    for edge_type in graph_data.edge_types:
        src_type, rel, dst_type = edge_type
        if not (src_type == 'rna' and dst_type == 'rna'):
            continue
        edge_index = graph_data[edge_type].edge_index
        if edge_index.shape[1] == 0:
            continue
        src_labels = graph_data[src_type].y[edge_index[0]]
        dst_labels = graph_data[dst_type].y[edge_index[1]]
        mask = (src_labels == dst_labels)
        filtered_edge_index = edge_index[:, mask]
        graph_data[edge_type].edge_index = filtered_edge_index
    return graph_data

def info_nce_loss(z1, z2, temperature=0.2):
    z1 = F.normalize(z1, p=2, dim=-1)
    z2 = F.normalize(z2, p=2, dim=-1)
    sim_matrix = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(sim_matrix.size(0), device=z1.device)
    loss_z1_to_z2 = F.cross_entropy(sim_matrix, labels)
    loss_z2_to_z1 = F.cross_entropy(sim_matrix.T, labels)
    return (loss_z1_to_z2 + loss_z2_to_z1) / 2.0

def train_step(loader, model, optimizer, cfg, current_w_cls, device, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss, total_geo, total_cls, total_recon = 0, 0, 0, 0
    total_cl_intra, total_cl_inter, total_center = 0, 0, 0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            batch = batch.to(device)
            if is_train:
                optimizer.zero_grad()

            z_dict = model(batch.x_dict, batch.edge_index_dict)

            pos_loss = torch.tensor(0.0, device=device)
            neg_loss = torch.tensor(0.0, device=device)
            cl_intra_loss = torch.tensor(0.0, device=device)
            cl_inter_loss = torch.tensor(0.0, device=device)

            sorted_edge_types = sorted(list(batch.edge_index_dict.keys()), key=lambda x: str(x))

            for edge_type in sorted_edge_types:
                edge_index = batch.edge_index_dict[edge_type]
                if edge_index.numel() == 0:
                    continue
                src_type, _, dst_type = edge_type

                z_src = z_dict[src_type][edge_index[0]]
                z_dst = z_dict[dst_type][edge_index[1]]

                pos_score = (z_src * z_dst).sum(dim=-1)
                pos_loss += F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))

                num_src, num_dst = batch[src_type].num_nodes, batch[dst_type].num_nodes
                neg_edge_index = negative_sampling(
                    edge_index,
                    num_nodes=(num_src, num_dst),
                    num_neg_samples=edge_index.size(1)
                ).to(device)
                z_src_neg = z_dict[src_type][neg_edge_index[0]]
                z_dst_neg = z_dict[dst_type][neg_edge_index[1]]
                neg_score = (z_src_neg * z_dst_neg).sum(dim=-1)
                neg_loss += F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))

                max_cl_samples = 2048
                if edge_index.size(1) > max_cl_samples:
                    idx = torch.randperm(edge_index.size(1), device=device)[:max_cl_samples]
                    z_src_cl = z_src[idx]
                    z_dst_cl = z_dst[idx]
                else:
                    z_src_cl = z_src
                    z_dst_cl = z_dst

                cl_loss_edge = info_nce_loss(z_src_cl, z_dst_cl, temperature=cfg.get("tau", 0.2))

                if src_type == dst_type:
                    cl_intra_loss += cl_loss_edge
                else:
                    cl_inter_loss += cl_loss_edge

            geo_loss = pos_loss + neg_loss

            cls_loss = torch.tensor(0.0, device=device)
            center_loss_val = torch.tensor(0.0, device=device)

            if hasattr(batch['rna'], 'y'):
                supervision_scope = cfg.get("rna_supervision_scope", "seed")
                if supervision_scope == "seed":
                    batch_size = getattr(batch['rna'], 'batch_size', 0)
                    z_rna_supervised = z_dict['rna'][:batch_size]
                    y_rna_supervised = batch['rna'].y[:batch_size]
                elif supervision_scope == "all":
                    z_rna_supervised = z_dict['rna']
                    y_rna_supervised = batch['rna'].y
                else:
                    raise ValueError(
                        f"Unsupported rna_supervision_scope: {supervision_scope}"
                    )

                if z_rna_supervised.size(0) > 0:
                    logits = model.classify(z_rna_supervised)
                    cls_loss = F.cross_entropy(logits, y_rna_supervised)

                    center_loss_val = F.mse_loss(
                        z_rna_supervised,
                        model.centers[y_rna_supervised],
                    )

            recon_loss = torch.tensor(0.0, device=device)
            if model.use_decoder:
                recon_dict = model.decode(z_dict)
                recon_loss = (
                    F.mse_loss(recon_dict['rna'], batch['rna'].x) +
                    F.mse_loss(recon_dict['atac'], batch['atac'].x)
                )

            loss = (
                (cfg["w_geo"] * geo_loss) +
                (current_w_cls * cls_loss) +
                (cfg["w_recon"] * recon_loss) +
                (cfg["w_cl_intra"] * cl_intra_loss) +
                (cfg["w_cl_inter"] * cl_inter_loss) +
                (cfg["w_center"] * center_loss_val)
            )

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_geo += geo_loss.item()
            total_cls += cls_loss.item() if isinstance(cls_loss, torch.Tensor) else 0
            total_recon += recon_loss.item() if isinstance(recon_loss, torch.Tensor) else 0
            total_cl_intra += cl_intra_loss.item() if isinstance(cl_intra_loss, torch.Tensor) else 0
            total_cl_inter += cl_inter_loss.item() if isinstance(cl_inter_loss, torch.Tensor) else 0
            total_center += center_loss_val.item() if isinstance(center_loss_val, torch.Tensor) else 0

    N = max(len(loader), 1)
    return (
        total_loss / N,
        total_geo / N,
        total_cls / N,
        total_recon / N,
        total_cl_intra / N,
        total_cl_inter / N,
        total_center / N
    )

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def validate_config(cfg):
    required_paths = [
        "rna_path",
        "atac_path",
        "atac_neighbor_path",
        "anchor_path",
    ]
    missing = [
        cfg[key]
        for key in required_paths
        if cfg.get(key) and not os.path.isfile(cfg[key])
    ]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Required dataset files are missing. Download and extract the data "
            f"before training:\n{formatted}"
        )

    if cfg["epochs"] < 1:
        raise ValueError("epochs must be at least 1.")
    if cfg["num_workers"] < 0:
        raise ValueError("num_workers cannot be negative.")


def validate_runtime():
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not (WITH_PYG_LIB or WITH_TORCH_SPARSE):
        raise ImportError(
            "NeighborLoader requires pyg-lib or torch-sparse. For the provided "
            "PyTorch 2.5.1 + CUDA 12.1 environment, install the official PyG "
            "wheels with:\n\n"
            "python -m pip install --no-index pyg_lib torch_scatter "
            "torch_sparse -f "
            "https://data.pyg.org/whl/torch-2.5.0+cu121.html"
        )

    backend = "pyg-lib" if WITH_PYG_LIB else "torch-sparse"
    print(f"PyG sampling backend: {backend}")


def create_loader(data, node_type, node_mask, cfg, generator):
    if cfg.get("num_neighbors_by_edge_type", True):
        num_neighbors = {
            edge_type: cfg["num_neighbors"]
            for edge_type in data.edge_types
        }
    else:
        num_neighbors = cfg["num_neighbors"]

    loader_kwargs = {
        "data": data,
        "num_neighbors": num_neighbors,
        "batch_size": cfg["batch_size"],
        "input_nodes": (node_type, node_mask),
        "shuffle": True,
        "num_workers": cfg["num_workers"],
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if cfg["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
    return NeighborLoader(**loader_kwargs)


def evaluate_model(model, data, device, cfg):
    model.eval()
    with torch.no_grad():
        x_dict = {key: value.to(device) for key, value in data.x_dict.items()}
        edge_index_dict = {
            key: value.to(device)
            for key, value in data.edge_index_dict.items()
        }
        z_dict = model(x_dict, edge_index_dict)

        rna_y = data["rna"].y.to(device)
        rna_preds = model.classify(z_dict["rna"]).argmax(dim=1)
        rna_acc = (rna_preds == rna_y).float().mean().item()

        atac_y = data["atac"].y.to(device)
        atac_probs = F.softmax(model.classify(z_dict["atac"]), dim=1)
        max_probs, atac_preds = atac_probs.max(dim=1)

        atac_y_np = atac_y.cpu().numpy()
        atac_preds_np = atac_preds.cpu().numpy()
        open_set_scores = 1.0 - max_probs.cpu().numpy()
        is_known = atac_y_np >= 0

        if is_known.any():
            closed_acc, open_auroc, open_aupr, oscr = osr_evaluator(
                atac_preds_np[is_known],
                atac_y_np[is_known],
                open_set_scores[is_known],
                open_set_scores[~is_known],
            )
            ari = adjusted_rand_score(atac_y_np[is_known], atac_preds_np[is_known])
            ami = adjusted_mutual_info_score(atac_y_np[is_known], atac_preds_np[is_known])
            precision = precision_score(
                atac_y_np[is_known],
                atac_preds_np[is_known],
                average="macro",
                zero_division=0,
            )
            recall = recall_score(
                atac_y_np[is_known],
                atac_preds_np[is_known],
                average="macro",
                zero_division=0,
            )
            f1 = f1_score(
                atac_y_np[is_known],
                atac_preds_np[is_known],
                average="macro",
                zero_division=0,
            )
        else:
            closed_acc = open_auroc = open_aupr = oscr = -1.0
            ari = ami = precision = recall = f1 = 0.0

        rna_embeds = z_dict["rna"].cpu().numpy()
        atac_embeds = z_dict["atac"].cpu().numpy()
        combined = np.concatenate([rna_embeds, atac_embeds], axis=0)
        modalities = np.concatenate([
            np.zeros(len(rna_embeds)),
            np.ones(len(atac_embeds)),
        ])
        sample_size = min(len(combined), 10000)
        raw_silhouette = silhouette_score(
            combined,
            modalities,
            sample_size=sample_size if sample_size < len(combined) else None,
            random_state=cfg["seed"],
        )

    metrics = {
        "rna_acc": rna_acc,
        "closed_acc": closed_acc,
        "open_auroc": open_auroc,
        "open_aupr": open_aupr,
        "oscr": oscr,
        "ari": ari,
        "ami": ami,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "modality_silhouette": 1.0 - raw_silhouette,
    }
    return metrics, rna_embeds, atac_embeds


def main():
    args = parse_args()
    if args.list_datasets:
        print("\n".join(available_datasets()))
        return

    cfg = build_config(args)
    validate_runtime()
    validate_config(cfg)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    seed_everything(cfg["seed"])

    print(f"Outputs: {cfg['output_dir']}")
    adata_rna, adata_atac, rna_labels, atac_labels, unique_labels = load_real_data(cfg)
    rna_feat, atac_feat, rna_pca_feat = preprocess(adata_rna, adata_atac, cfg)
    data = build_hetero_graph(
        rna_feat,
        atac_feat,
        rna_pca_feat,
        adata_rna.obs_names,
        adata_atac.obs_names,
        cfg,
    )

    data["rna"].y = torch.LongTensor(rna_labels)
    data["atac"].y = torch.LongTensor(atac_labels)
    data["rna"].train_mask = torch.ones(data["rna"].num_nodes, dtype=torch.bool)
    data["atac"].train_mask = torch.ones(data["atac"].num_nodes, dtype=torch.bool)
    data = filter_perfect_edges(data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    encoder = HeteroGNNEncoder(
        in_dim=rna_feat.shape[1],
        hidden_dim=cfg["hidden_dim"],
        out_dim=cfg["out_dim"],
        num_layers=cfg["num_layers"],
    )
    model = HeteroGAE(
        encoder,
        len(unique_labels),
        use_decoder=cfg["use_decoder"],
        feat_dim=rna_feat.shape[1],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    generator = torch.Generator()
    generator.manual_seed(cfg["seed"])
    rna_loader = create_loader(
        data,
        "rna",
        data["rna"].train_mask,
        cfg,
        generator,
    )
    atac_loader = create_loader(
        data,
        "atac",
        data["atac"].train_mask,
        cfg,
        generator,
    )

    print("Starting training loop...")
    rna_embeds = atac_embeds = None
    for epoch in range(cfg["epochs"]):
        epoch_seed = cfg["seed"] + epoch
        seed_everything(epoch_seed)

        rna_stats = train_step(
            rna_loader,
            model,
            optimizer,
            cfg,
            cfg["w_cls"],
            device,
        )
        atac_stats = train_step(
            atac_loader,
            model,
            optimizer,
            cfg,
            cfg["w_cls"],
            device,
        )
        total_loss = rna_stats[0] + atac_stats[0]
        cl_inter = rna_stats[5] + atac_stats[5]

        if epoch == cfg["epochs"] - 1:
            metrics, rna_embeds, atac_embeds = evaluate_model(
                model,
                data,
                device,
                cfg,
            )
            print(
                f"Epoch {epoch:03d}, Loss: {total_loss:.4f} "
                f"| CL_Inter: {cl_inter:.4f} "
                f"| RNA Acc: {metrics['rna_acc'] * 100:.2f}% "
                f"| Closed Acc: {metrics['closed_acc']:.4f} "
                f"| Open AUROC: {metrics['open_auroc']:.4f} "
                f"| OSCR: {metrics['oscr']:.4f} "
                f"| ARI: {metrics['ari']:.4f} "
                f"| AMI: {metrics['ami']:.4f}",
                flush=True,
            )
        else:
            print(
                f"Epoch {epoch:03d}, Loss: {total_loss:.4f} "
                f"| CL_Inter: {cl_inter:.4f}",
                flush=True,
            )

    np.save(os.path.join(cfg["output_dir"], "rna_embeddings.npy"), rna_embeds)
    np.save(os.path.join(cfg["output_dir"], "atac_embeddings.npy"), atac_embeds)
    print(f"Finished. RNA embeddings: {rna_embeds.shape}")
    print(f"Finished. ATAC embeddings: {atac_embeds.shape}")


if __name__ == "__main__":
    main()
