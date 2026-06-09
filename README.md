# scHetGTL

## Repository Layout

```text
.
|-- train.py
|-- settinng.py
|-- examples/
|   `-- pbmc_example.ipynb
|-- process/
|   |-- get_lsi.py
|   `-- get_anchor.py
|-- requirements.txt
`-- environment.yml
```

## Data

- PBMC: [CSUBioGroup/scNCL-release demo_data](https://github.com/CSUBioGroup/scNCL-release/tree/main/Examples/demo_data)
- CITE: [SydneyBioX/scJoint](https://github.com/SydneyBioX/scJoint)
- MS: [caokai1073/uniPort](https://github.com/caokai1073/uniPort)
- BMMC: [GSE194122](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122)

```text
data/
`-- data_pbmc/
    |-- rna_labeled.h5ad
    |-- atac_labeled.h5ad
    |-- atac_lsi_features.h5ad
    `-- anchors.csv
```

## Installation

```bash
conda env create -f environment.yml
conda activate scHetGTL
python -m ipykernel install --user --name scHetGTL --display-name "Python (scHetGTL)"
```

## Verify

```bash
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_SPARSE; print(WITH_PYG_LIB, WITH_TORCH_SPARSE)"
```

```text
2.5.1+cu121 12.1 True
```

## Training

```bash
python train.py --dataset pbmc
```

```text
results/pbmc/rna_embeddings.npy
results/pbmc/atac_embeddings.npy
```

Evaluation metrics are computed and printed only in the final epoch.

## Notebook

```bash
jupyter lab examples/pbmc_example.ipynb
```

## Configuration

- `DEFAULT_TRAINING`
- `DATASETS`
- `get_dataset_config(name)`

## Preprocessing

- `process/get_lsi.py`: writes `atac_lsi_features.h5ad`
- `process/get_anchor.py`: writes `anchors.csv`
