import os
import subprocess

import pandas as pd
import scanpy as sc
import scipy.sparse as sps

THRESHOLDS = [0.0]
THRESHOLDS_R_STR = "c(" + ", ".join(map(str, THRESHOLDS)) + ")"

DATA_DIR = "/mnt/data0/GZH/GAE/data_case3_all/rna_atac_gam_common_genes/"

RNA_H5 = os.path.join(DATA_DIR, "RNA_common_genes.h5ad")
ATAC_H5 = os.path.join(DATA_DIR, "ATAC_GAM_common_genes.h5ad")

TEMP_RNA_CSV = os.path.join(DATA_DIR, "temp_rna_matrix.csv")
TEMP_ATAC_RAW_CSV = os.path.join(DATA_DIR, "temp_atac_raw_matrix.csv")

r_pipeline = f"""
suppressPackageStartupMessages({{
    library(Seurat)
    library(Matrix)
}})

print(">>> [R] Starting Seurat anchor pipeline")

setup_seurat <- function(mat_csv, assay_name) {{
    print(paste("   -> Loading:", mat_csv))
    counts <- read.csv(mat_csv, row.names = 1, check.names = FALSE)

    obj <- CreateSeuratObject(counts = as(as.matrix(counts), "sparseMatrix"), assay = assay_name)

    obj <- NormalizeData(obj, verbose = FALSE)
    if (assay_name == "RNA") {{
        obj <- FindVariableFeatures(obj, selection.method = "vst", nfeatures = 2000, verbose = FALSE)
    }}
    obj <- ScaleData(obj, verbose = FALSE)
    return(obj)
}}

print(">>> [R] Preparing Reference (RNA)...")
rna <- setup_seurat("{TEMP_RNA_CSV}", "RNA")

print(">>> [R] Preparing Query (RAW ATAC)...")
atac_raw <- setup_seurat("{TEMP_ATAC_RAW_CSV}", "ACTIVITY")

run_anchors <- function(query_obj, method_name) {{
    print(paste("\\n=================================================="))
    print(paste(">>> [R] Executing Anchor Search for:", method_name))
    print(paste("=================================================="))

    common_features <- intersect(rownames(rna), rownames(query_obj))
    shared_var_features <- intersect(VariableFeatures(rna), common_features)

    anchors <- FindTransferAnchors(
        reference = rna,
        query = query_obj,
        reference.assay = "RNA",
        query.assay = "ACTIVITY",
        features = shared_var_features,
        reduction = "cca",
        dims = 1:30,
        k.anchor = 20,
        verbose = FALSE
    )

    df <- data.frame(
        ref_barcode = colnames(rna)[anchors@anchors[, "cell1"]],
        query_barcode = colnames(query_obj)[anchors@anchors[, "cell2"]],
        score = anchors@anchors[, "score"]
    )

    thresholds <- {THRESHOLDS_R_STR}

    for (thresh in thresholds) {{
        df_filtered <- df[df$score >= thresh, ]
        print(paste("--- Threshold >=", thresh, "---"))
        print(paste("Total Anchors:", nrow(df_filtered)))

        output_file <- paste0("{DATA_DIR}anchors.csv")
        write.csv(df_filtered, file = output_file, row.names = FALSE)
    }}
}}

run_anchors(atac_raw, "RAW ATAC")

print(">>> [R] All tasks completed successfully.")
"""


def extract_h5ad_to_csv(h5_path, matrix_csv):
    print(f"[PYTHON] Extracting Matrix from: {os.path.basename(h5_path)}")
    adata = sc.read_h5ad(h5_path)
    x = adata.X.toarray() if sps.issparse(adata.X) else adata.X
    df = pd.DataFrame(x.T, index=adata.var_names, columns=adata.obs_names)
    df.to_csv(matrix_csv)
    print(f"   -> Saved. Matrix shape: {df.shape}")


if __name__ == "__main__":
    print("========== Pipeline Start (Anchoring) ==========")

    extract_h5ad_to_csv(
        h5_path=RNA_H5,
        matrix_csv=TEMP_RNA_CSV,
    )

    extract_h5ad_to_csv(
        h5_path=ATAC_H5,
        matrix_csv=TEMP_ATAC_RAW_CSV,
    )

    r_script_path = os.path.join(DATA_DIR, "run_anchor.R")
    with open(r_script_path, "w") as f:
        f.write(r_pipeline)

    print("[PYTHON] Data prep done. Launching R Seurat pipeline...\n")
    try:
        subprocess.run(["Rscript", r_script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n!!! R script execution failed with error code {e.returncode}.")
        exit(1)

    print("\n========== Pipeline Finished ==========")
