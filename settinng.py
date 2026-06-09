from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"


DEFAULT_TRAINING = {
    "seed": 0,
    "score_threshold": 0.1,
    "score_threshold_effector": None,
    "score_threshold_others": None,
    "effector_labels": ["effector cd4 t", "effector cd8 t"],
    "knn_k": 20,
    "knn_undirected": True,
    "knn_edge_direction": "neighbor_to_cell",
    "n_comps": 50,
    "n_top_genes": 2000,
    "hidden_dim": 128,
    "out_dim": 64,
    "num_layers": 2,
    "use_decoder": True,
    "epochs": 20,
    "lr": 1e-3,
    "batch_size": 256,
    "num_neighbors": [10, 5],
    "num_neighbors_by_edge_type": True,
    "num_workers": 0,
    "log_interval": 1,
    "w_geo": 1.0,
    "w_cls": 0.05,
    "w_recon": 0.5,
    "w_center": 0.1,
    "w_cl_intra": 4.0,
    "w_cl_inter": 1.0,
    "tau": 2.0,
    "atac_neighbor_metric": "cosine",
    "rna_supervision_scope": "seed",
}


DATASETS = {
    "bmmc1": {
        "rna_path": DATA_ROOT / "data_bmmc1" / "batch_s1d1_rna_common.h5ad",
        "atac_path": DATA_ROOT / "data_bmmc1" / "atac_gene_common.h5ad",
        "atac_neighbor_path": DATA_ROOT / "data_bmmc1" / "atac_lsi_features.h5ad",
        "atac_neighbor_format": "h5ad",
        "anchor_path": DATA_ROOT / "data_bmmc1" / "anchors.csv",
    },
    "bmmc2": {
        "rna_path": DATA_ROOT / "data_bmmc2" / "batch_s2d1_rna_common.h5ad",
        "atac_path": DATA_ROOT / "data_bmmc2" / "atac_gene_common.h5ad",
        "atac_neighbor_path": DATA_ROOT / "data_bmmc2" / "atac_lsi_features.h5ad",
        "atac_neighbor_format": "h5ad",
        "anchor_path": DATA_ROOT / "data_bmmc2" / "anchors.csv",
    },
    "bmmc3": {
        "rna_path": DATA_ROOT / "data_bmmc3" / "batch_s3d3_rna_common.h5ad",
        "atac_path": DATA_ROOT / "data_bmmc3" / "atac_gene_common.h5ad",
        "atac_neighbor_path": DATA_ROOT / "data_bmmc3" / "atac_lsi_features.h5ad",
        "atac_neighbor_format": "h5ad",
        "anchor_path": DATA_ROOT / "data_bmmc3" / "anchors.csv",
    },
    "bmmc4": {
        "rna_path": DATA_ROOT / "data_bmmc4" / "batch_s4d1_rna_common.h5ad",
        "atac_path": DATA_ROOT / "data_bmmc4" / "atac_gene_common.h5ad",
        "atac_neighbor_path": DATA_ROOT / "data_bmmc4" / "atac_lsi_features.h5ad",
        "atac_neighbor_format": "h5ad",
        "anchor_path": DATA_ROOT / "data_bmmc4" / "anchors.csv",
    },
    "cite_asap": {
        "rna_path": DATA_ROOT / "data_cite-asap" / "citeseq_control_rna_labeled.h5ad",
        "atac_path": DATA_ROOT / "data_cite-asap" / "asapseq_control_atac_labeled.h5ad",
        "atac_neighbor_path": DATA_ROOT / "data_cite-asap" / "asapseq_control_adt.npz",
        "atac_neighbor_format": "npz",
        "anchor_path": DATA_ROOT / "data_cite-asap" / "anchors.csv",
    },
    "ms": {
        "rna_path": DATA_ROOT / "data_ms" / "rna.h5ad",
        "atac_path": DATA_ROOT / "data_ms" / "atac.h5ad",
        "atac_neighbor_path": DATA_ROOT / "data_ms" / "atac_lsi_features.h5ad",
        "atac_neighbor_format": "h5ad",
        "anchor_path": DATA_ROOT / "data_ms" / "anchors.csv",
    },
    "pbmc": {
        "rna_path": DATA_ROOT / "data_pbmc" / "rna_labeled.h5ad",
        "atac_path": DATA_ROOT / "data_pbmc" / "atac_labeled.h5ad",
        "atac_neighbor_path": DATA_ROOT / "data_pbmc" / "atac_lsi_features.h5ad",
        "atac_neighbor_format": "h5ad",
        "anchor_path": DATA_ROOT / "data_pbmc" / "anchors.csv",
    },
    "pbmc_drop3": {
        "rna_path": DATA_ROOT / "data_pbmc_drop3" / "rna_labeled.h5ad",
        "atac_path": DATA_ROOT / "data_pbmc_drop3" / "atac_labeled.rm3mid.h5ad",
        "atac_neighbor_path": DATA_ROOT / "data_pbmc_drop3" / "atac_lsi_features.h5ad",
        "atac_neighbor_format": "h5ad",
        "anchor_path": DATA_ROOT / "data_pbmc_drop3" / "anchors.csv",
    },
}


def available_datasets():
    return tuple(sorted(DATASETS))


def get_dataset_config(name):
    if name not in DATASETS:
        choices = ", ".join(available_datasets())
        raise KeyError(f"Unknown dataset '{name}'. Available datasets: {choices}")

    config = deepcopy(DEFAULT_TRAINING)
    config.update(deepcopy(DATASETS[name]))
    config["dataset"] = name
    if "output_dir" not in config:
        config["output_dir"] = RESULTS_ROOT / name

    for key, value in list(config.items()):
        config[key] = stringify_paths(value)

    return config


def stringify_paths(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [stringify_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(stringify_paths(item) for item in value)
    if isinstance(value, dict):
        return {key: stringify_paths(item) for key, item in value.items()}
    return value
