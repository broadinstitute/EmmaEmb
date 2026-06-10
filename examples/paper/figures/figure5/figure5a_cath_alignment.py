"""
Figure 5A — CATH structural classes: KNN alignment line plot (cosine only)
===========================================================================
Reproduces Figure 5A

Panel: KNN feature alignment score vs k (cosine, mean-centred) for 3 PLMs.
  - ProtTucker: structure-informed contrastive PLM (128 dims)
  - ProtT5:     sequence-based PLM (1024 dims)
  - AlphaFold2: structure-prediction model representations (256 dims)

Output saved to: examples/paper/figures/figure5/figure5a_cath_alignment.pdf
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from emmaemb.core import Emma
from emmaemb.functions import get_knn_alignment_scores

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# ── Path placeholders — fill in before running ─────────────────────────────────
EMBEDDINGS_DIR = "examples/paper/data/cath/embeddings/"           # subdirs: ProtTucker/, ProtT5/, AlphaFold/
FEATURES_CSV   = "examples/paper/data/cath/processed/test300_features.csv"
OUTPUT_DIR     = "examples/paper/figures/figure5/"
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

models = {
    "ProtTucker": "ProtTucker",
    "ProtT5":     "ProtT5",
    "AlphaFold2": "AlphaFold",
}

label_to_color = {
    "ProtTucker": "#245757",
    "ProtT5":     "#0F8B8D",
    "AlphaFold2": "#bb342f",
}
symbol_map = {model: "circle" for model in models}
sorted_embeddings = ["AlphaFold2", "ProtT5", "ProtTucker"]
sorted_labels     = list(reversed(sorted_embeddings))

distance_metric = "cosine"
k_values        = [3, 5, 10, 15, 20]
feature_col     = "class"

feature_data = pd.read_csv(FEATURES_CSV)

ema = Emma(feature_data=feature_data)
for model_alias, model_subdir in models.items():
    ema.add_emb_space(
        embeddings_source=EMBEDDINGS_DIR + model_subdir,
        emb_space_name=model_alias,
    )
ema.mean_center()

for model in models:
    ema.calculate_pairwise_distances(emb_space=model, metric=distance_metric)

protein_ids = ema.metadata["domain_id"]
knn_scores_df = pd.DataFrame(columns=["Sample", "Class", "Fraction", "Embedding", "k"])

for k_value in k_values:
    print(f"Computing KNN alignment (cosine, k={k_value})...")
    scores = get_knn_alignment_scores(
        ema,
        feature=feature_col,
        k=k_value,
        metric=distance_metric,
        adjust_for_imbalance=False,
    )
    scores["k"] = k_value
    scores["Sample"] = np.tile(protein_ids.values, len(models))
    knn_scores_df = pd.concat([knn_scores_df, scores], ignore_index=True)

df_grouped = (
    knn_scores_df
    .groupby(["k", "Embedding"], as_index=False)
    .agg({"Fraction": "mean"})
).sort_values(by=["Embedding", "k"])
df_grouped["Embedding"] = pd.Categorical(df_grouped["Embedding"], categories=sorted_labels, ordered=True)

# Build line plot manually to match reference style
fig = px.line(
    df_grouped,
    x="k",
    y="Fraction",
    color="Embedding",
    color_discrete_map=label_to_color,
    symbol="Embedding",
    symbol_map=symbol_map,
    category_orders={"Embedding": sorted_labels},
    markers=True,
)

for trace in fig.data:
    trace.line.dash = "solid"

fig.update_layout(
    template="plotly_white",
    font={"family": "Arial", "color": "black", "size": 20},
    legend_title_text="",
    width=600,
    height=500,
    title=dict(
        text="A. CATH class",
        x=0, xanchor="left",
        y=0.92, yanchor="top",
    ),
    legend=dict(
        orientation="h",
        x=-0.15, y=1.02,
        xanchor="left", yanchor="bottom",
        tracegroupgap=0,
    ),
    margin=dict(t=120),
    yaxis_title="Mean KNN feature<br>alignment score",
)

fig.update_traces(marker=dict(size=10), line=dict(width=4))

fig.update_xaxes(
    showgrid=False, linecolor="black", linewidth=3,
    ticks="outside", tickwidth=2, tickcolor="black", ticklen=6,
    tickformat=".0f", tick0=0, dtick=5, rangemode="tozero",
    title_text="k",
)
fig.update_yaxes(
    showgrid=False, linecolor="black", linewidth=3,
    ticks="outside", tickwidth=2, tickcolor="black", ticklen=6,
    tickformat=".2f", dtick=0.05,
)

# Kaleido warm-up
fig_dummy = px.scatter(x=[0, 1], y=[0, 1])
fig_dummy.write_image(OUTPUT_DIR + "_warmup.pdf", format="pdf")
time.sleep(2)

fig.write_image(OUTPUT_DIR + "figure5a_cath_alignment.pdf", format="pdf", width=600, height=500)
print(f"Saved figure5a_cath_alignment.pdf to {OUTPUT_DIR}")
