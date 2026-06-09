import os
import warnings

import scanpy as sc
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

warnings.filterwarnings("ignore")

CONFIG = {
    "atac_path": "/mnt/data0/GZH/GAE/data_case3_all/case3_ATAC.h5ad",
    "output_dir": "/mnt/data0/GZH/GAE/data_case3_all",
    "n_comps": 50,
}


def process_and_get_lsi(adata, n_comps):
    tfidf = TfidfTransformer(norm="l2", sublinear_tf=True)
    x_tfidf = tfidf.fit_transform(adata.X)

    svd = TruncatedSVD(n_components=n_comps + 1, random_state=42)
    x_svd = svd.fit_transform(x_tfidf)

    return x_svd[:, 1:]


if __name__ == "__main__":
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print("Loading ATAC dataset...")
    adata_atac = sc.read_h5ad(CONFIG["atac_path"])

    print("Generating LSI features...")
    feat_atac_lsi = process_and_get_lsi(adata_atac, CONFIG["n_comps"])

    adata_lsi = sc.AnnData(X=feat_atac_lsi, obs=adata_atac.obs.copy())
    adata_lsi.var_names = [
        f"LSI_{index + 1}"
        for index in range(feat_atac_lsi.shape[1])
    ]

    lsi_h5ad_path = os.path.join(CONFIG["output_dir"], "atac_lsi_features.h5ad")
    adata_lsi.write_h5ad(lsi_h5ad_path)

    print(f"Saved LSI features to {lsi_h5ad_path}")
    print(f"LSI feature shape: {feat_atac_lsi.shape}")
