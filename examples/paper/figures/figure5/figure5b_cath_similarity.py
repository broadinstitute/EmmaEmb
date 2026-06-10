"""
Figure 5B — CATH structural classes: pairwise neighborhood similarity heatmap
===============================================================================
Reproduces Figure 5B

Panel: Lower-triangle pairwise neighborhood similarity between 3 PLMs at k=20 (cosine).
  - ProtTucker: structure-informed contrastive PLM
  - ProtT5:     sequence-based PLM
  - AlphaFold2: structure-prediction model representations

Output saved to: examples/paper/figures/figure5/figure5b_cath_similarity.png
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from emmaemb.core import Emma
from emmaemb.functions import get_neighborhood_similarity

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

distance_metric = "cosine"
k_values        = [3, 5, 10, 15, 20]
K_PLOT          = 20   # the k value shown in the final figure

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

model_list = list(models.keys())
nbsim_pairs = [
    ("ProtTucker", "ProtT5"),
    ("ProtTucker", "AlphaFold2"),
    ("ProtT5",     "AlphaFold2"),
]

# Compute pairwise neighborhood similarities across all k values
nbsim_records = []
for k_val in k_values:
    for model1, model2 in nbsim_pairs:
        sim = get_neighborhood_similarity(
            ema, emb_space_1=model1, emb_space_2=model2,
            k=k_val, metric=distance_metric,
        )
        nbsim_records.append({
            "k": k_val, "distance_metric": distance_metric,
            "model1": model1, "model2": model2,
            "similarity": float(np.mean(sim)),
        })
df_nbsim = pd.DataFrame(nbsim_records)


def _build_lower_tri_matrix(df_k, model_list):
    n = len(model_list)
    mat = np.full((n, n), np.nan)
    for _, entry in df_k.iterrows():
        i = model_list.index(entry["model1"])
        j = model_list.index(entry["model2"])
        lo, hi = min(i, j), max(i, j)
        mat[hi, lo] = entry["similarity"]
    y_display = model_list[::-1][:-1]
    x_display = model_list[:-1]
    z_display = mat[::-1, :][:-1, :-1]
    return mat, y_display, z_display, x_display


# Build the lower-triangle heatmap at K_PLOT
z_min = float(df_nbsim["similarity"].min())
z_max = float(df_nbsim["similarity"].max())
z_mid = (z_min + z_max) / 2

COLORSCALE = [(0, "#f0eff9"), (0.5, "#7b68c8"), (1, "#1c1255")]
FONT_SIZE  = 20

df_k = df_nbsim[df_nbsim["k"] == K_PLOT].copy()
mat, y_display, z_display, x_display = _build_lower_tri_matrix(df_k, model_list)

n = len(model_list)
fig = go.Figure()
fig.add_trace(go.Heatmap(
    z=z_display,
    x=x_display,
    y=y_display,
    colorscale=COLORSCALE,
    zmin=z_min, zmax=z_max,
    showscale=True,
    colorbar=dict(
        orientation="h",
        x=0.5, xanchor="center",
        y=1.12, yanchor="bottom",
        len=0.9, thickness=16,
        tickformat=".2f",
        tickfont=dict(size=FONT_SIZE, family="Arial"),
        title=dict(
            text="Mean Neighborhood Similarity",
            side="top",
            font=dict(size=FONT_SIZE, family="Arial"),
        ),
    ),
    zsmooth=False,
))

for i in range(n):
    for j in range(n):
        if i <= j or np.isnan(mat[i, j]):
            continue
        txt_color = "white" if mat[i, j] >= z_mid else "black"
        fig.add_annotation(
            x=model_list[j], y=model_list[i],
            text=f"{mat[i, j]:.2f}",
            showarrow=False,
            font=dict(size=FONT_SIZE, color=txt_color, family="Arial"),
        )

fig.update_xaxes(
    showgrid=False, linecolor="black", linewidth=2, tickangle=-30,
    tickfont=dict(size=FONT_SIZE), title_font=dict(size=FONT_SIZE),
    constrain="domain",
)
fig.update_yaxes(
    showgrid=False, linecolor="black", linewidth=2,
    tickfont=dict(size=FONT_SIZE), title_font=dict(size=FONT_SIZE),
    scaleanchor="x", scaleratio=1, constrain="domain",
)
fig.update_layout(
    template="plotly_white",
    font={"family": "Arial", "color": "black", "size": FONT_SIZE},
    title=dict(
        text="B.",
        x=0.06, xanchor="left",
        font=dict(size=30, family="Arial"),
    ),
    width=520, height=520,
    margin=dict(l=120, r=40, t=140, b=100),
)

# Kaleido warm-up
fig_dummy = px.scatter(x=[0, 1], y=[0, 1])
fig_dummy.write_image(OUTPUT_DIR + "_warmup.png", format="png")
time.sleep(2)

fig.write_image(
    OUTPUT_DIR + "figure5b_cath_similarity.png",
    format="png", width=520, height=520, scale=2,
)
fig.write_image(
    OUTPUT_DIR + "figure5b_cath_similarity.pdf",
    format="pdf", width=520, height=520,
)
print(f"Saved figure5b_cath_similarity.png/.pdf to {OUTPUT_DIR}")
