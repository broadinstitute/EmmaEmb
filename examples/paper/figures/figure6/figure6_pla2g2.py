"""
Figure 6 — PLA2G2 enzyme family: pairwise cosine distance scatter (ESMC vs ProtT5)
====================================================================================
Reproduces Figure 6

Panel layout (3 × 4 grid of scatter plots):
  Panel (1,1)  : all pairs overview, single neutral color, no Spearman annotation
  Panels (1,2)–(3,4) : 11 enzyme classes (A B C D D1 D2 D3 E F G V) individually
                        grey background = all pairs; colored = within-class pairs

Axes:
  X: pairwise cosine distances in ESMC (mean-centred)
  Y: pairwise cosine distances in ProtT5 (mean-centred)
  Both axes fixed to [0, 2]

Output saved to: examples/paper/figures/figure6/figure6_pla2g2.pdf
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from emmaemb.core import Emma

# ── Path placeholders — fill in before running ─────────────────────────────────
EMBEDDINGS_DIR = "examples/paper/data/pla2g2/embeddings/"         # subdirs: ProtT5/, ESMC/
FEATURES_CSV   = "examples/paper/data/pla2g2/processed/Pla2g2_features.csv"
OUTPUT_DIR     = "examples/paper/figures/figure6/"
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

NEUTRAL_COLOR = "#CCCCCC"
SINGLE_COLOR  = "#C6B89E"
distance_metric = "cosine"

models = {
    "ProtT5": "ProtT5",
    "ESMC":   "ESMC",
}

metadata = pd.read_csv(FEATURES_CSV)
ema = Emma(feature_data=metadata)
for model_alias, model_name in models.items():
    ema.add_emb_space(
        embeddings_source=EMBEDDINGS_DIR + model_name,
        emb_space_name=model_alias,
    )

ema.mean_center(emb_spaces=["ESMC", "ProtT5"])

for model_alias in models:
    ema.calculate_pairwise_distances(
        emb_space=model_alias, metric=distance_metric, store_distances=True
    )

n_samples = len(ema.sample_names)
idx_i, idx_j = np.triu_indices(n_samples, k=1)
x_dist = ema.emb["ESMC"]["pairwise_distances"][distance_metric][idx_i, idx_j]
y_dist = ema.emb["ProtT5"]["pairwise_distances"][distance_metric][idx_i, idx_j]

color_maps = ema.color_map
color_maps["enzyme_class"]["B"] = "#E07B39"
color_maps["enzyme_class"]["C"] = "#5B90D0"


def _ax_ref(subplot_num):
    return "" if subplot_num == 1 else str(subplot_num)


def _add_overall_panel(fig, row, col, snum):
    xr = f"x{_ax_ref(snum)} domain"
    yr = f"y{_ax_ref(snum)} domain"
    fig.add_trace(go.Scattergl(
        x=x_dist, y=y_dist, mode="markers",
        marker=dict(size=3, color=SINGLE_COLOR, opacity=0.2),
        showlegend=False, hoverinfo="skip",
    ), row=row, col=col)
    fig.add_annotation(
        xref=xr, yref=yr, x=0.97, y=0.95, text="All classes",
        showarrow=False, align="right",
        font=dict(size=11, color="black", family="Arial"),
        bgcolor="rgba(255,255,255,0.75)", borderpad=2,
    )


def _add_panel(fig, row, col, snum, fg_mask, color, label):
    xr = f"x{_ax_ref(snum)} domain"
    yr = f"y{_ax_ref(snum)} domain"
    fig.add_trace(go.Scattergl(
        x=x_dist[~fg_mask], y=y_dist[~fg_mask], mode="markers",
        marker=dict(size=3, color=NEUTRAL_COLOR, opacity=0.2),
        showlegend=False, hoverinfo="skip",
    ), row=row, col=col)
    fig.add_trace(go.Scattergl(
        x=x_dist[fg_mask], y=y_dist[fg_mask], mode="markers",
        marker=dict(size=4, color=color, opacity=0.8),
        showlegend=False,
    ), row=row, col=col)
    fig.add_annotation(
        xref=xr, yref=yr, x=0.97, y=0.95, text=label,
        showarrow=False, align="right",
        font=dict(size=11, color="black", family="Arial"),
        bgcolor="rgba(255,255,255,0.75)", borderpad=2,
    )


def _style_composite(fig):
    fig.update_xaxes(
        range=[0, 2], showgrid=False, showline=True,
        linecolor="black", linewidth=1, tickfont=dict(size=9),
    )
    fig.update_yaxes(
        range=[0, 2], showgrid=False, showline=True,
        linecolor="black", linewidth=1, tickfont=dict(size=9),
    )
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Arial", size=11, color="black"),
        margin=dict(l=75, r=10, t=10, b=75),
        showlegend=False,
    )
    fig.add_annotation(
        x=0.5, y=-0.07, xref="paper", yref="paper",
        text="Cosine distances ESMC", showarrow=False, font=dict(size=13),
    )
    fig.add_annotation(
        x=-0.07, y=0.5, xref="paper", yref="paper",
        text="Cosine distances ProtT5", showarrow=False,
        font=dict(size=13), textangle=-90,
    )


# Build 3×4 composite grid (show_corr=False as in paper figure)
ec_cmap   = color_maps["enzyme_class"]
ec_labels = ema.metadata["enzyme_class"].values
ec_sorted = sorted(ec_cmap.keys())   # 11 classes: A B C D D1 D2 D3 E F G V

NROWS, NCOLS = 3, 4
OVERALL_RC   = (1, 1)
overall_snum = (OVERALL_RC[0] - 1) * NCOLS + OVERALL_RC[1]

all_rc   = [(r, c) for r in range(1, NROWS + 1) for c in range(1, NCOLS + 1)]
class_rc = [rc for rc in all_rc if rc != OVERALL_RC]

fig = make_subplots(
    rows=NROWS, cols=NCOLS,
    horizontal_spacing=0.06, vertical_spacing=0.06,
)
_add_overall_panel(fig, *OVERALL_RC, overall_snum)
for (row, col), class_val in zip(class_rc, ec_sorted):
    snum = (row - 1) * NCOLS + col
    fg = (ec_labels[idx_i] == class_val) & (ec_labels[idx_j] == class_val)
    _add_panel(fig, row, col, snum, fg, ec_cmap[class_val], f"Class {class_val}")

_style_composite(fig)

# Kaleido warm-up
fig_dummy = px.scatter(x=[0], y=[0])
fig_dummy.write_image(OUTPUT_DIR + "_warmup.pdf", format="pdf")
time.sleep(2)

fig.write_image(
    OUTPUT_DIR + "figure6_pla2g2.pdf",
    format="pdf",
    width=NCOLS * 250,
    height=NROWS * 250,
)
print(f"Saved figure6_pla2g2.pdf to {OUTPUT_DIR}")
