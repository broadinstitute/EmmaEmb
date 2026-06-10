"""
Figure 4A — Subcellular localization: per-class recall vs KNN alignment + mixing matrix
=========================================================================================
Reproduces Figure 4A

This script produces two files:
  1. figure4a_scatter_deeploc.pdf  — plot_metric_vs_knn (literature LA vs KNN, bubble scatter)
  2. figure4a_heatmap_deeploc.png  — plot_knn_ranking_heatmap (mixing matrix + confusion overlay)

PLM used for both panels: ProtT5 (mean-centred, cosine, k=100)
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import plotly.express as px

from emmaemb.core import Emma
from emmaemb.functions import get_knn_alignment_scores, get_class_mixing_in_neighborhood

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from shared_figures import plot_metric_vs_knn, plot_knn_ranking_heatmap

# ── Path placeholders — fill in before running ─────────────────────────────────
EMBEDDINGS_DIR    = "examples/paper/data/deeploc/embeddings/"       # subdirs: Ankh/, ESM2/, ProstT5/, ProtT5/
FEATURES_CSV      = "examples/paper/data/deeploc/processed/deeploc_train_features.csv"
CONFUSION_XLSX    = "examples/deeploc/confusion-martix-prott5.xlsx"
OUTPUT_DIR        = "examples/paper/figures/figure4/"
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

sub_loc_abbreviation_mapping = {
    "Cell.membrane": "Mem",
    "Cytoplasm": "Cyt",
    "Endoplasmic.reticulum": "End",
    "Golgi.apparatus": "Gol",
    "Lysosome/Vacuole": "Lys",
    "Mitochondrion": "Mit",
    "Nucleus": "Nuc",
    "Peroxisome": "Per",
    "Plastid": "Pla",
    "Extracellular": "Ext",
}

class_order = ["Nuc", "Cyt", "Ext", "Mit", "Mem", "End", "Pla", "Gol", "Lys", "Per"]

# Literature performance per class (from Stärk et al. / ProtT5 paper, DeepLoc-1 benchmarks)
literature_performance_per_class = {
    "Cell.membrane":        {"la": 0.81, "label_size": 1067, "text_label": "Mem"},
    "Cytoplasm":            {"la": 0.81, "label_size": 2180, "text_label": "Cyt"},
    "Endoplasmic.reticulum":{"la": 0.70, "label_size": 689,  "text_label": "End"},
    "Golgi.apparatus":      {"la": 0.33, "label_size": 286,  "text_label": "Gol"},
    "Lysosome/Vacuole":     {"la": 0.12, "label_size": 257,  "text_label": "Lys"},
    "Mitochondrion":        {"la": 0.88, "label_size": 1208, "text_label": "Mit"},
    "Nucleus":              {"la": 0.90, "label_size": 3235, "text_label": "Nuc"},
    "Peroxisome":           {"la": 0.17, "label_size": 124,  "text_label": "Per"},
    "Plastid":              {"la": 0.92, "label_size": 605,  "text_label": "Pla"},
    "Extracellular":        {"la": 0.95, "label_size": 1580, "text_label": "Ext"},
}

# Load metadata and abbreviate labels
metadata = pd.read_csv(FEATURES_CSV)
metadata["Subcellular Localization"] = metadata["subcellular_location"].map(
    sub_loc_abbreviation_mapping
)

# Load ProtT5 embeddings, mean-center, compute pairwise distances
ema = Emma(feature_data=metadata)
ema.add_emb_space(
    embeddings_source=EMBEDDINGS_DIR + "ProtT5",
    emb_space_name="ProtT5",
)
ema.mean_center(["ProtT5"])
ema.calculate_pairwise_distances(emb_space="ProtT5", metric="cosine")

# Kaleido warm-up
fig_dummy = px.scatter(x=[0, 1], y=[0, 1])
fig_dummy.write_image(OUTPUT_DIR + "_warmup.png", format="png")
time.sleep(2)

# ── Panel 1: Scatter — per-class literature recall vs KNN alignment score ──────
knn_1b = get_knn_alignment_scores(ema, feature="Subcellular Localization", k=100, metric="cosine")
proTt5_per_class = (
    knn_1b[knn_1b["Embedding"] == "ProtT5"]
    .groupby("Class")["Fraction"]
    .mean()
)

df = pd.DataFrame(literature_performance_per_class).T.reset_index()
df.columns = ["subcellular_localisation", "la", "label_size", "text_label"]
df["label_size"] = df["label_size"].astype(float)
df["cosine"] = df["text_label"].map(proTt5_per_class)
df_sorted = df.sort_values(by="label_size", ascending=False).reset_index(drop=True)

plot_metric_vs_knn(
    df=df_sorted,
    x_col="la",
    x_title="Recall as reported by Stärk et al.",
    y_title="Mean KNN feature alignment score",
    label_col="text_label",
    size_col="label_size",
    knn_col="cosine",
    output_path=OUTPUT_DIR + "figure4a_scatter_deeploc.pdf",
    width=600,
    height=720,
)
print(f"Saved figure4a_scatter_deeploc.pdf to {OUTPUT_DIR}")

# ── Panel 2: Heatmap — KNN ranking matrix + ML confusion overlay ───────────────
neighbor_class_counts, unique_classes = get_class_mixing_in_neighborhood(
    ema,
    emb_space="ProtT5",
    feature="subcellular_location",
    k=100,
    metric="cosine",
)

if isinstance(neighbor_class_counts, (np.ndarray, list)):
    mixing_matrix = pd.DataFrame(neighbor_class_counts, index=unique_classes, columns=unique_classes)
else:
    mixing_matrix = neighbor_class_counts

ranked_matrix = mixing_matrix.rank(axis=0, method="min", ascending=False).astype(int)
ranked_matrix.rename(index=sub_loc_abbreviation_mapping, columns=sub_loc_abbreviation_mapping, inplace=True)
ranked_matrix = ranked_matrix.reindex(index=class_order, columns=class_order)

# Build ML top-misclassification sets from the confusion matrix
conf_matrix = pd.read_excel(CONFUSION_XLSX, index_col=0).T.fillna(0)
conf_matrix.rename(index=sub_loc_abbreviation_mapping, columns=sub_loc_abbreviation_mapping, inplace=True)
conf_matrix = conf_matrix.reindex(index=class_order, columns=class_order)

def _ranked_set_at(col, target_rank):
    col = col.dropna()
    ranked = col.rank(method="min", ascending=False)
    if target_rank == 1:
        return set(ranked[ranked == ranked.min()].index)
    rest = ranked[ranked > ranked.min()]
    return set() if rest.empty else set(ranked[ranked == rest.min()].index)

top_ml_rank1 = {
    cls: _ranked_set_at(conf_matrix[cls].drop(cls, errors="ignore"), 1)
    for cls in class_order
}

fig_heatmap = plot_knn_ranking_heatmap(
    ranked_matrix=ranked_matrix,
    top_ml_rank1=top_ml_rank1,
    class_order=class_order,
    use_integer_coords=False,
    legend_max_label="Rank 10 (least common neighbor)",
    output_path=OUTPUT_DIR + "figure4a_heatmap_deeploc.png",
    width=620,
    height=750,
)
fig_heatmap.write_image(OUTPUT_DIR + "figure4a_heatmap_deeploc.pdf", format="pdf", width=620, height=750)
print(f"Saved figure4a_heatmap_deeploc.png/.pdf to {OUTPUT_DIR}")
