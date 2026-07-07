# Reproducing paper figures

This directory contains all code to reproduce the preprocessing and figures.

---

## Directory layout

```
examples/paper/
├── data/                   # raw and computed data (auto-populated by preprocessing notebooks)
│   ├── binding/
│   ├── cath/
│   ├── conservation/
│   ├── deeploc/
│   └── pla2g2/
├── preprocessing/          # preprocessing notebooks
├── figures/
│   ├── figure3/
│   ├── figure4/
│   ├── figure5/
│   └── figure6/
```

---

## Datasets and sources

| # | Dataset | Task | Source |
|---|---------|------|--------|
| 1 | **DeepLoc** | Subcellular localization (10 classes, 11,231 proteins) | Almagro Armenteros et al. DeepLoc: prediction of protein subcellular localization using deep learning. *Bioinformatics* (2017). Raw data: <https://services.healthtech.dtu.dk/services/DeepLoc-1.0/> |
| 2 | **ConSurf10k** | Residue conservation (9 ordinal classes, ~2.4 M residues) | Marquet et al. Embeddings from protein language models predict conservation and variant effect. *Human Genetics* (2022). Raw data: <https://zenodo.org/records/5238537> |
| 3 | **DevSet1014** | Residue binding site binary (156,687 non-binding / 13,999 binding) | Littmann et al. Protein embeddings and deep learning predict binding residues for various ligand classes. *Scientific Reports* (2021). Raw data: <https://github.com/Rostlab/bindPredict> |
| 4 | **LigandBindingSite** | Ligand binding site (three-class) | Data preprocessing and ML training scripts: <https://github.com/skrhakv/emb-space-analysis> |
| 5 | **CATH test300** | CATH structural class (5 classes, 300 domains) | Orengo et al. CATH - a hierarchic classification of protein domain structures. *Structure* (1997); subset defined by Heinzinger et al. Contrastive learning on protein embeddings enlightens midnight zone. *NAR Genomics and Bioinformatics* (2022). Raw data: <https://github.com/Rostlab/EAT> |
| 5 | **PLA2G2** | Enzyme family (11 classes, 446 proteins) | Koludarov et al. Reconstructing the evolutionary history of a functionally diverse gene family reveals complexity at the genetic origins of novelty. bioRxiv (2020); formatted via ProtSpace: <https://github.com/tsenoner/protspace/tree/main/data/Pla2g2> |

---

## Workflow

### Step 1: Preprocessing
Run `preprocessing/0N_<dataset>.ipynb`. Raw data files are downloaded automatically on first run and a backup is stored in `examples/paper/data/<dataset>/raw/`. Processed outputs are written to `examples/paper/data/<dataset>/processed/`.

### Step 2: Generating embeddings
Use [`plm_embeddings/get_embeddings.py`](../../plm_embeddings/get_embeddings.py) to generate embeddings from each PLM and place them in `data/<dataset>/embeddings/<ModelName>/`.

**CATH only:** Three embedding spaces are used — ProtT5 (via `plm_embeddings/`), ProtTucker (via the [ProtTucker repository](https://github.com/Rostlab/EAT)), and AlphaFold2.

### Step 3: Figures
Run the scripts in `figures/figureX/`. Each script reads from `data/` paths defined at the top of the file and saves PDF/PNG output to `figures/figureX/`.

---

## Notes

- **Conservation ANNOY indices:** On first run, figure scripts will build ANNOY approximate nearest-neighbour indices for the ~2.4 M-residue conservation dataset and cache them to `data/conservation/annoy_indices/`. Subsequent runs load these directly.

- **AlphaFold2 embeddings** for the CATH benchmark were generated with ColabFold v1.5.2 using default parameters (MMseqs2 for MSA generation). The single representation from the final evoformer layer was extracted and mean-pooled across residue positions.

- **Ligand binding site (three-class):** Data preprocessing and ML training scripts are maintained in a companion repository: <https://github.com/skrhakv/emb-space-analysis>.
