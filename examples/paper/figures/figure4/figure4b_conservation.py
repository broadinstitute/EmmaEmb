"""
Figure 4B — Residue conservation: per-class recall vs KNN alignment + mixing matrix
=====================================================================================
Reproduces Figure 4B

This script produces two files:
  1. figure4b_scatter_conservation.png   — plot_metric_vs_knn (recall vs KNN, by class)
  2. figure4b_heatmap_conservation.png   — plot_knn_ranking_heatmap (mixing + ML overlay)

PLM used for both panels: ProtT5 (mean-centred, cosine ANNOY, k=100)
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import plotly.express as px

from emmaemb.core import Emma
from emmaemb.functions import get_class_mixing_in_neighborhood, get_knn_alignment_scores

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from shared_figures import plot_knn_ranking_heatmap, plot_metric_vs_knn

# ── Path placeholders — fill in before running ─────────────────────────────────
FEATURES_CSV    = "examples/paper/data/conservation/processed/conservation_features.csv"
CONFUSION_CSV   = "examples/conservation/prott5cons-confusion-matrix.csv"
BACKUP_DIR      = "examples/paper/data/conservation/annoy_indices/" # cached ANNOY rank arrays
OUTPUT_DIR      = "examples/paper/figures/figure4/"
N_TREES         = 100
EMB_SPACE       = "ProtT5"
K               = 100
ANNOY_METRIC    = "cosine"
ANNOY_KEY       = "angular"
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_ORDER = [str(i) for i in range(1, 10)]
LABEL_SHORT = {f"{i}-conservation": str(i) for i in range(1, 10)}

metadata = pd.read_csv(FEATURES_CSV)
print(f"Residues: {len(metadata):,}")

ema = Emma(feature_data=metadata)

# Inject pre-computed ANNOY ranks (avoids loading multi-GB embedding matrix)
annoy_knn_path = os.path.join(BACKUP_DIR, f"{EMB_SPACE}_mc_{ANNOY_METRIC}_{N_TREES}_knn.npy")
if not os.path.exists(annoy_knn_path):
    raise FileNotFoundError(
        f"ANNOY rank array not found: {annoy_knn_path}\n"
        "Run the conservation diagnostics notebook first to build this cache."
    )

ema.emb[EMB_SPACE] = {
    "annoy_ranks": {
        ANNOY_KEY: {
            N_TREES: np.load(annoy_knn_path)
        }
    }
}
print(f"Loaded ANNOY ranks from {annoy_knn_path}")

# Kaleido warm-up
fig_dummy = px.scatter(x=[0, 1], y=[0, 1])
fig_dummy.write_image(OUTPUT_DIR + "_warmup.png", format="png")
time.sleep(2)

# ── Panel 1: Scatter — per-class conservation recall vs KNN alignment ──────────
print(f"Computing KNN alignment scores (k={K}, ANNOY cosine)...")
knn_scores = get_knn_alignment_scores(
    ema,
    feature="conservation",
    k=K,
    metric=ANNOY_METRIC,
    use_annoy=True,
    annoy_metric=ANNOY_METRIC,
    n_trees=N_TREES,
)
mean_knn_per_class = (
    knn_scores[knn_scores["Embedding"] == EMB_SPACE]
    .groupby("Class")["Fraction"]
    .mean()
)

# Load confusion matrix (rows = ground truth, cols = predictions)
conf_matrix = pd.read_csv(CONFUSION_CSV, header=None)
conf_matrix.index   = CLASS_ORDER
conf_matrix.columns = CLASS_ORDER

# Compute recall: diagonal / row sum
sample_counts = metadata["conservation"].value_counts()

recall_per_class = {
    cls: conf_matrix.loc[cls, cls] / conf_matrix.loc[cls].sum()
    if conf_matrix.loc[cls].sum() > 0 else float("nan")
    for cls in CLASS_ORDER
}

CLASS_LABELS_FULL = [f"{i}-conservation" for i in range(1, 10)]
df_recall = pd.DataFrame({
    "class":  CLASS_LABELS_FULL,
    "label":  CLASS_ORDER,
    "recall": [recall_per_class[c] for c in CLASS_ORDER],
    "knn":    [mean_knn_per_class.get(f"{c}-conservation", float("nan")) for c in CLASS_ORDER],
    "n":      [sample_counts.get(f"{c}-conservation", 0) for c in CLASS_ORDER],
})

fig_scatter = plot_metric_vs_knn(
    df=df_recall,
    x_col="recall",
    x_title="Recall as reported by Marquet et al.",
    y_title="Mean KNN feature alignment score*",
    label_col="label",
    size_col="n",
    knn_col="knn",
    output_path=OUTPUT_DIR + "figure4b_scatter_conservation.png",
    width=600,
    height=720,
)
fig_scatter.write_image(OUTPUT_DIR + "figure4b_scatter_conservation.pdf", format="pdf", width=600, height=720)
print(f"Saved figure4b_scatter_conservation.png/.pdf to {OUTPUT_DIR}")

# ── Panel 2: Heatmap — KNN ranking matrix + ML confusion overlay ───────────────
print(f"Computing class mixing in neighborhood (k={K})...")
neighbor_class_counts, unique_classes = get_class_mixing_in_neighborhood(
    ema,
    emb_space=EMB_SPACE,
    feature="conservation",
    k=K,
    metric=ANNOY_METRIC,
    use_annoy=True,
    annoy_metric=ANNOY_METRIC,
    n_trees=N_TREES,
)

mixing_matrix = pd.DataFrame(
    neighbor_class_counts,
    index=unique_classes,
    columns=unique_classes,
)

ranked_matrix = mixing_matrix.rank(axis=0, method="min", ascending=False).astype(int)
ranked_matrix.rename(index=LABEL_SHORT, columns=LABEL_SHORT, inplace=True)
ranked_matrix = ranked_matrix.reindex(index=CLASS_ORDER, columns=CLASS_ORDER)

def _ranked_set_at(col, target_rank):
    col = col.dropna()
    ranked = col.rank(method="min", ascending=False)
    if target_rank == 1:
        return set(ranked[ranked == ranked.min()].index)
    rest = ranked[ranked > ranked.min()]
    return set() if rest.empty else set(ranked[ranked == rest.min()].index)

# Top ML misclassification per class (row = ground truth, off-diagonal rank 1)
top_ml_rank1 = {
    cls: _ranked_set_at(conf_matrix.loc[cls].drop(cls, errors="ignore"), 1)
    for cls in CLASS_ORDER
}

n_classes = len(CLASS_ORDER)
fig_heatmap = plot_knn_ranking_heatmap(
    ranked_matrix=ranked_matrix,
    top_ml_rank1=top_ml_rank1,
    class_order=CLASS_ORDER,
    use_integer_coords=True,
    legend_max_label=f"Rank {n_classes} (least common neighbor)",
    output_path=OUTPUT_DIR + "figure4b_heatmap_conservation.png",
    width=560,
    height=680,
)
fig_heatmap.write_image(OUTPUT_DIR + "figure4b_heatmap_conservation.pdf", format="pdf", width=560, height=680)
print(f"Saved figure4b_heatmap_conservation.png/.pdf to {OUTPUT_DIR}")
