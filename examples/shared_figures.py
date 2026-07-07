"""
Shared Plotly figure functions for Emma embedding analyses.

Three reusable figure types:
  1. plot_model_ranking     — cosine KNN alignment (line) + ML metric (bar), side by side
  2. plot_metric_vs_knn     — per-class ML metric vs mean KNN alignment score (bubble scatter)
  3. plot_knn_ranking_heatmap — KNN neighborhood ranking heatmap with ML misclassification overlay
"""

import math

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr


def _nice_round(x):
    """Round x to 1 significant figure (e.g. 83 500 → 80 000)."""
    if x <= 0:
        return 1
    exp = math.floor(math.log10(x))
    return int(round(x / 10 ** exp) * 10 ** exp)

# ---------------------------------------------------------------------------
# Shared color constants
# ---------------------------------------------------------------------------

#: Consistent model colors across all figures
MODEL_COLORS = {
    "ProtT5":  "#0F8B8D",
    "Ankh":    "#EC9A29",
    "ESM2":    "#A8201A",
    "ProstT5": "#00008B",
}

#: Heatmap teal palette (dark = rank 1, light = rank max)
TEAL_DARK  = "#084f51"
TEAL_MID   = "#0F8B8D"
TEAL_LIGHT = "#ceeaea"

#: Diagonal exclusion color (self-neighborhood)
DIAG_COLOR   = "rgba(240,240,240,1.0)"
#: ML misclassification border color
BORDER_COLOR = "#8b1a1a"


# ---------------------------------------------------------------------------
# 1. Embedding space ranking: cosine KNN alignment (line) + ML metric (bar)
# ---------------------------------------------------------------------------

def plot_model_ranking(
    df_knn,
    ml_performance,
    model_order,
    knn_y_title="Mean KNN alignment score",
    ml_y_title="ML performance",
    ml_y_range=None,
    knn_dtick=0.05,
    ml_text_fmt="{:.1f}%",
    title=None,
    model_colors=None,
    embedding_col="Embedding",
    output_path=None,
    width=1000,
    height=500,
):
    """
    Side-by-side figure: cosine KNN alignment curves (left) and ML metric bars (right).

    Parameters
    ----------
    df_knn : pd.DataFrame
        KNN alignment scores, already filtered for cosine distance.
        Required columns: [k, <embedding_col>, Fraction].
    ml_performance : dict
        {model_name: float} — ML metric value per model (e.g. accuracy, F1).
    model_order : list[str]
        Display order for models (determines bar order and line legend order).
    knn_y_title : str
        Y-axis title for the KNN line panel.
    ml_y_title : str
        Y-axis title for the ML bar panel.
    ml_y_range : list[float] or None
        [min, max] for the ML y-axis.  None = auto.
    knn_dtick : float
        Tick interval for the KNN y-axis.
    ml_text_fmt : str
        Format string for bar labels, e.g. "{:.1f}%" or "{:.2f}".
    title : str or None
        Optional figure title (top-left).
    model_colors : dict or None
        {model_name: hex_color}.  Defaults to MODULE-LEVEL MODEL_COLORS.
    embedding_col : str
        Column in df_knn that holds model names (default "Embedding").
    output_path : str or None
        Full path to save the figure (PDF recommended).  None = no save.
    width, height : int
        Figure dimensions in pixels.

    Returns
    -------
    plotly.graph_objs.Figure
    """
    colors = model_colors or MODEL_COLORS

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],
        horizontal_spacing=0.22,
        subplot_titles=["Rank by EmmaEmb", "Rank by transfer learning"],
    )

    # Left: KNN alignment lines
    for model in model_order:
        df_m = df_knn[df_knn[embedding_col] == model]
        fig.add_trace(
            go.Scatter(
                x=df_m["k"],
                y=df_m["Fraction"],
                mode="lines+markers",
                name=model,
                line=dict(color=colors[model], width=4),
                marker=dict(size=10, color=colors[model]),
                showlegend=False,
            ),
            row=1, col=1,
        )

    # Right: ML metric bars
    for model in model_order:
        val = ml_performance[model]
        fig.add_trace(
            go.Bar(
                x=[model],
                y=[val],
                marker_color=colors[model],
                showlegend=False,
                name=model,
                text=[ml_text_fmt.format(val)],
                textposition="outside",
                textfont=dict(size=20, color="black", family="Arial"),
            ),
            row=1, col=2,
        )

    layout_kw = dict(
        template="plotly_white",
        font={"family": "Arial", "color": "black", "size": 20},
        width=width,
        height=height,
        barmode="group",
        margin=dict(t=100),
    )
    if title:
        # yref="container" uses the full figure (0=bottom, 1=top), so we can place
        # the title near the figure top independently of the plot-area paper coords.
        layout_kw["title"] = dict(
            text=title, x=0, xanchor="left",
            yref="container", y=0.98, yanchor="top",
        )
    fig.update_layout(**layout_kw)

    # Subplot titles are layout annotations — set larger than global font (20) so they
    # visually match the weight of the y-axis tick/title labels.
    fig.for_each_annotation(lambda a: a.update(
        font=dict(size=24, family="Arial", color="black"),
    ))

    fig.update_xaxes(
        showgrid=False, linecolor="black", linewidth=3,
        ticks="outside", tickwidth=2, tickcolor="black", ticklen=6,
        tickformat=".0f", title_text="k",
        row=1, col=1,
    )
    fig.update_yaxes(
        showgrid=False, linecolor="black", linewidth=3,
        ticks="outside", tickwidth=2, tickcolor="black", ticklen=6,
        tickformat=".2f", dtick=knn_dtick,
        title_text=knn_y_title,
        row=1, col=1,
    )
    fig.update_xaxes(
        showgrid=False, linecolor="black", linewidth=3,
        ticks="outside", tickwidth=2, tickcolor="black", ticklen=6,
        row=1, col=2,
    )
    ml_y_kwargs = dict(
        showgrid=False, linecolor="black", linewidth=3,
        ticks="outside", tickwidth=2, tickcolor="black", ticklen=6,
        title_text=ml_y_title,
        row=1, col=2,
    )
    if ml_y_range is not None:
        ml_y_kwargs["range"] = ml_y_range
    fig.update_yaxes(**ml_y_kwargs)

    if output_path:
        fmt = "pdf" if output_path.endswith(".pdf") else "png"
        fig.write_image(output_path, format=fmt, width=width, height=height)
        print(f"Saved {output_path}")

    return fig


# ---------------------------------------------------------------------------
# 2. Per-class ML metric vs mean KNN alignment score (bubble scatter)
# ---------------------------------------------------------------------------

def plot_metric_vs_knn(
    df,
    x_col="x_val",
    x_title="ML metric per class",
    y_title="Mean KNN feature alignment score",
    color=TEAL_MID,
    label_col="label",
    size_col="n",
    knn_col="knn",
    size_max=40,
    output_path=None,
    width=560,
    height=560,
):
    """
    Bubble scatter of per-class ML metric (x) vs mean KNN alignment score (y).

    Includes a Spearman correlation trendline and overlap-free label placement
    using an iterative pairwise repulsion algorithm that avoids both label↔label
    and label↔dot collisions.

    Parameters
    ----------
    df : pd.DataFrame
        One row per class.  Required columns: [<x_col>, <knn_col>, <size_col>, <label_col>].
    x_col : str
        Column containing the x-axis metric (e.g. "recall" or "precision").
    x_title : str
        X-axis label.
    y_title : str
        Y-axis label.
    color : str
        Bubble fill color.  Defaults to TEAL_MID.
    label_col : str
        Column with short class labels shown on the plot.
    size_col : str
        Column with sample counts used as bubble size.
    knn_col : str
        Column with mean KNN alignment scores.
    size_max : int
        Maximum bubble diameter in pixels passed to px.scatter (default 40).
    output_path : str or None
        Full save path.  None = no save.
    width, height : int
        Figure dimensions in pixels.

    Returns
    -------
    plotly.graph_objs.Figure
    """
    dot_x = df[x_col].values.astype(float)
    dot_y = df[knn_col].values.astype(float)

    r_val, p_val = spearmanr(dot_x, dot_y)
    m, b = np.polyfit(dot_x, dot_y, 1)
    x_line = np.linspace(dot_x.min(), dot_x.max(), 100)
    y_line = m * x_line + b

    p_text = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"

    fig = px.scatter(df, x=x_col, y=knn_col, size=size_col, size_max=size_max)

    fig.update_traces(
        marker=dict(color=color, line=dict(width=1.5, color="black")),
    )

    # Trendline (no legend entry)
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines",
        line=dict(color="grey", width=1.5, dash="dash"),
        showlegend=False,
    ))

    # In-plot Spearman annotation — upper-left corner
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        text=f"ρ = {r_val:.2f}<br>{p_text}",
        showarrow=False,
        font=dict(size=22, color="black", family="Arial"),
        xanchor="left", yanchor="top",
        align="left",
    )

    # --- Overlap-free label placement via iterative repulsion ---
    # Estimate each dot's radius in data units so labels are repelled from dots too.
    # px.scatter sizemode="area": pixel_radius = sqrt(size / sizeref)
    # sizeref = 2 * max(size) / size_max^2
    sizeref = 2.0 * float(df[size_col].max()) / (size_max ** 2)
    # Margin constants (kept in sync with update_layout below)
    _ML, _MR, _MT, _MB = 20, 20, 120, 60
    # Approximate plot area (px) accounting for fixed margins in the layout
    plot_w_px = max(width  - _ML - _MR, 1)
    plot_h_px = max(height - _MT - _MB, 1)

    # Displayed axis range starts at 0 (range=[0, None]), so the data-to-pixel
    # conversion must use the full displayed range [0, max], not just max - min.
    x_display = max(dot_x.max() * 1.1, 0.1)   # approximate displayed x range
    y_display = max(dot_y.max() * 1.1, 0.1)   # approximate displayed y range

    # Dot radii in data units for each point (with a 1.3× safety margin so labels
    # are guaranteed to clear the bubble edge, not just its centre)
    dot_sizes = df[size_col].values.astype(float)
    dot_rx = np.array([1.3 * math.sqrt(s / sizeref) / plot_w_px * x_display for s in dot_sizes])
    dot_ry = np.array([1.3 * math.sqrt(s / sizeref) / plot_h_px * y_display for s in dot_sizes])

    lw = 0.030 * x_display   # label half-width  in data units
    lh = 0.045 * y_display   # label half-height in data units

    lbl_x = dot_x.copy()
    lbl_y = dot_y.copy()
    # Start labels to the right of their dot (clear of the dot edge)
    lbl_x += dot_rx + lw * 1.2

    n = len(lbl_x)
    for _ in range(400):
        # 1. Label ↔ label repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = lbl_x[i] - lbl_x[j]
                dy = lbl_y[i] - lbl_y[j]
                if abs(dx) < 2 * lw and abs(dy) < 2 * lh:
                    ox = (2 * lw - abs(dx)) / 2 + 1e-9
                    oy = (2 * lh - abs(dy)) / 2 + 1e-9
                    sx = np.sign(dx) if dx != 0 else 1.0
                    sy = np.sign(dy) if dy != 0 else 1.0
                    lbl_x[i] += ox * sx
                    lbl_x[j] -= ox * sx
                    lbl_y[i] += oy * sy
                    lbl_y[j] -= oy * sy

        # 2. Label ↔ dot repulsion (every label is pushed away from every dot)
        for i in range(n):
            for j in range(n):
                ex_x = lw + dot_rx[j]
                ex_y = lh + dot_ry[j]
                dx = lbl_x[i] - dot_x[j]
                dy = lbl_y[i] - dot_y[j]
                if abs(dx) < ex_x and abs(dy) < ex_y:
                    ox = (ex_x - abs(dx)) / 2 + 1e-9
                    oy = (ex_y - abs(dy)) / 2 + 1e-9
                    sx = np.sign(dx) if dx != 0 else 1.0
                    sy = np.sign(dy) if dy != 0 else 1.0
                    lbl_x[i] += ox * sx
                    lbl_y[i] += oy * sy

    for i, (_, row) in enumerate(df.iterrows()):
        moved = bool(abs(lbl_x[i] - dot_x[i]) > 1e-6 or abs(lbl_y[i] - dot_y[i]) > 1e-6)
        fig.add_annotation(
            x=dot_x[i], y=dot_y[i],
            ax=lbl_x[i], ay=lbl_y[i],
            axref="x", ayref="y",
            text=row[label_col],
            showarrow=moved,
            arrowwidth=1, arrowcolor="lightgrey",
            font=dict(size=20, color="black", family="Arial"),
            xanchor="center", yanchor="middle",
        )

    fig.update_layout(
        template="plotly_white",
        font={"family": "Arial", "color": "black", "size": 22},
        width=width, height=height,
        showlegend=False,
        xaxis=dict(
            showgrid=False, linecolor="black", linewidth=2,
            title=dict(text=x_title, font=dict(size=22)),
            ticks="outside", tickwidth=2, tickcolor="black", ticklen=6,
            tickfont=dict(size=20),
            range=[0, None],
        ),
        yaxis=dict(
            showgrid=False, linecolor="black", linewidth=2,
            title=dict(text=y_title, font=dict(size=22)),
            ticks="outside", tickwidth=2, tickcolor="black", ticklen=6,
            tickfont=dict(size=20),
            range=[0, None],
        ),
        margin=dict(l=_ML, r=_MR, t=_MT, b=_MB),
    )

    # --- Custom size legend drawn as shapes in the top margin ---
    # Use xsizemode/ysizemode="pixel" so the bounding box is specified as
    # pixel offsets from an anchor in paper coordinates.  This gives a
    # perfectly round 2r × 2r pixel bounding box regardless of the figure's
    # aspect ratio or Plotly's internal margin adjustments.
    n_max_val = int(df[size_col].max())
    legend_n_vals = [max(_nice_round(n_max_val // 4), 1),
                     max(_nice_round(n_max_val // 2), 1),
                     _nice_round(n_max_val)]
    # Pixel radii (same formula as plotly.js area mode)
    legend_radii_px = [math.sqrt(n / sizeref) for n in legend_n_vals]

    # Vertical centre of top margin in paper-space (y > 1 = above plot area)
    y_c = 1.0 + (_MT / 2.0) / plot_h_px

    # Lay circles out left-to-right; reserve ~85 px per label
    _label_gap_px, _label_w_px, _between_px = 10, 85, 20
    x_cur = 0.0   # cursor in plot-area pixels from left edge
    cx_list = []
    for r_px in legend_radii_px:
        cx_list.append((x_cur + r_px) / plot_w_px)   # centre in paper x
        x_cur += 2 * r_px + _label_gap_px + _label_w_px + _between_px

    for cx, r_px, n_val in zip(cx_list, legend_radii_px, legend_n_vals):
        # xsizemode/ysizemode="pixel": x0/x1 and y0/y1 are pixel offsets from
        # the anchor point (cx, y_c) in paper coordinates → always a square
        # bounding box → always a round circle.
        fig.add_shape(
            type="circle",
            xref="paper", yref="paper",
            xsizemode="pixel", ysizemode="pixel",
            xanchor=cx, yanchor=y_c,
            x0=-r_px, x1=r_px,
            y0=-r_px, y1=r_px,
            fillcolor=color,
            line=dict(color="black", width=1.5),
        )
        fig.add_annotation(
            xref="paper", yref="paper",
            x=cx + r_px / plot_w_px + _label_gap_px / plot_w_px,
            y=y_c,
            text=f"n = {n_val:,}",
            showarrow=False,
            font=dict(size=16, family="Arial", color="black"),
            xanchor="left", yanchor="middle",
        )

    if output_path:
        fmt = "pdf" if output_path.endswith(".pdf") else "png"
        fig.write_image(output_path, format=fmt, width=width, height=height,
                        scale=1 if fmt == "pdf" else 2)
        print(f"Saved {output_path}")

    return fig


# ---------------------------------------------------------------------------
# 3. Class-mixing / confusion matrix heatmap
# ---------------------------------------------------------------------------

def plot_class_heatmap(
    matrix,
    text_matrix=None,
    text_fmt="%{text}",
    text_annotation_fmt=None,
    text_white_threshold_frac=None,
    use_integer_coords=False,
    x_from_index=False,
    colorscale="Reds_r",
    zmin=None,
    zmax=None,
    showscale=False,
    legend_labels=None,
    xaxis_title="Neighbor classes",
    yaxis_title="Query classes",
    font_size=20,
    margin=None,
    output_path=None,
    width=600,
    height=600,
):
    """
    Heatmap for class-mixing or confusion matrices.

    Parameters
    ----------
    matrix : pd.DataFrame
        Values used for cell coloring.
        By default: index = y-axis labels, columns = x-axis labels.
        Set *x_from_index=True* to use index as x-axis instead (useful when
        the source DataFrame has the x categories in its index rather than its
        columns).
    text_matrix : pd.DataFrame or None
        Values displayed as cell text (same orientation as *matrix*).
        None = use *matrix* values.
    text_fmt : str
        Plotly texttemplate string used when *text_annotation_fmt* is None,
        e.g. ``"%{text}"`` or ``"%{text}%"``.
    text_annotation_fmt : str or None
        Python format spec (e.g. ``".2f"``) for per-cell annotation text.
        When set, explicit ``go.layout.Annotation`` objects are used instead
        of the heatmap texttemplate, allowing per-cell font colors.
    text_white_threshold_frac : float or None
        Fraction of the color-axis maximum above which cell text is drawn in
        white; below it in black.  Only used when *text_annotation_fmt* is set.
        E.g. ``0.6`` → cells with value > 0.6 × z_max get white text.
    use_integer_coords : bool
        Use 0-based integer indices for annotation x/y coordinates instead of
        category label strings.  Required when labels are numeric strings
        (e.g. "1"–"9") to avoid Plotly's numeric-string parsing offset.
    x_from_index : bool
        If True, use ``matrix.index`` as x-axis labels and ``matrix.columns``
        as y-axis labels (and ``z = matrix.values.T`` is still applied).
        Useful when the source DataFrame stores x categories in its index.
    colorscale : str or list
        Plotly colorscale name or list of ``(frac, color)`` tuples.
    zmin, zmax : float or None
        Color axis limits.  None = auto.
    showscale : bool
        Show continuous colorbar.
    legend_labels : list of (color, label) or None
        Categorical legend entries as ``(hex_color, label_string)`` pairs.
    xaxis_title, yaxis_title : str
        Axis labels.
    font_size : int
        Font size for all text elements.
    margin : dict or None
        Plotly margin dict, e.g. ``dict(l=70, r=70, t=20, b=60)``.
        None = automatic defaults.
    output_path : str or None
        Full save path (PDF or PNG).  None = no save.
    width, height : int
        Figure dimensions in pixels.

    Returns
    -------
    plotly.graph_objs.Figure
    """
    if x_from_index:
        x_labels = matrix.index.tolist()
        y_labels = matrix.columns.tolist()
    else:
        x_labels = matrix.columns.tolist()
        y_labels = matrix.index.tolist()

    # go.Heatmap: z[row][col] → y[row], x[col]
    z = matrix.values.T
    text_source = matrix if text_matrix is None else text_matrix

    use_annotations = text_annotation_fmt is not None

    heatmap_kw = dict(
        z=z, x=x_labels, y=y_labels,
        colorscale=colorscale, showscale=showscale,
        zsmooth=False,
    )
    if not use_annotations:
        heatmap_kw["text"] = text_source.values.T
        heatmap_kw["texttemplate"] = text_fmt
    if zmin is not None:
        heatmap_kw["zmin"] = zmin
    if zmax is not None:
        heatmap_kw["zmax"] = zmax

    fig = go.Figure(go.Heatmap(**heatmap_kw))

    if use_annotations:
        text_z = text_source.values.T
        z_max_data = float(np.nanmax(z)) if zmax is None else float(zmax)
        threshold = (z_max_data * text_white_threshold_frac
                     if text_white_threshold_frac is not None else None)
        annotations = []
        for yi, y_lbl in enumerate(y_labels):
            for xi, x_lbl in enumerate(x_labels):
                val = float(text_z[yi, xi])
                txt_color = (
                    "white" if (threshold is not None and val > threshold) else "black"
                )
                ann = dict(
                    text=f"{val:{text_annotation_fmt}}",
                    showarrow=False,
                    font=dict(color=txt_color, size=font_size, family="Arial"),
                    xref="x", yref="y",
                )
                if use_integer_coords:
                    ann["x"] = xi
                    ann["y"] = yi
                else:
                    ann["x"] = x_lbl
                    ann["y"] = y_lbl
                annotations.append(ann)
        fig.update_layout(annotations=annotations)
    else:
        fig.update_traces(textfont=dict(size=font_size, family="Arial"))

    if legend_labels:
        for color, label in legend_labels:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=15, color=color),
                showlegend=True, name=label,
            ))

    has_legend = bool(legend_labels)
    default_margin = dict(l=10, r=10, t=100 if has_legend else 20, b=10)

    fig.update_layout(
        title=None,
        font={"family": "Arial", "color": "black", "size": font_size},
        template="plotly_white",
        width=width,
        height=height,
        xaxis=dict(
            title=xaxis_title,
            tickfont=dict(size=font_size),
            tickangle=0,
            automargin=True,
            showgrid=False,
        ),
        yaxis=dict(
            title=yaxis_title,
            tickfont=dict(size=font_size),
            ticklabelstandoff=10,
            automargin=True,
            showgrid=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            borderwidth=0,
            font=dict(size=font_size),
        ) if has_legend else dict(visible=False),
        margin=margin if margin is not None else default_margin,
    )

    if output_path:
        fmt = "pdf" if output_path.endswith(".pdf") else "png"
        fig.write_image(output_path, format=fmt, width=width, height=height,
                        scale=1 if fmt == "pdf" else 2)
        print(f"Saved {output_path}")

    return fig


# ---------------------------------------------------------------------------
# 4. KNN neighborhood ranking heatmap with ML misclassification overlay
# ---------------------------------------------------------------------------


def plot_knn_ranking_heatmap(
    ranked_matrix,
    top_ml_rank1,
    class_order,
    use_integer_coords=False,
    rank_white_threshold=3,
    legend_max_label=None,
    ml_border_label="Top-1 ML misclassification",
    xaxis_title="Neighbor classes",
    yaxis_title="Query classes",
    output_path=None,
    width=560,
    height=680,
):
    """
    Ranked KNN neighborhood heatmap with optional ML misclassification borders.

    Cells are colored by sqrt-transformed rank (rank 1 = darkest teal = most
    common neighbor).  A light-grey diagonal overlay marks the excluded self.
    A dark-red solid border marks the top-1 ML misclassification per class.

    Parameters
    ----------
    ranked_matrix : pd.DataFrame
        Integer rank matrix; index = neighbor classes, columns = query classes.
        Rank 1 = most common neighbor in that query class's neighborhood.
    top_ml_rank1 : dict
        {query_class: set_of_neighbor_classes} — top ML confusion per class.
        Drawn as solid dark-red borders on the heatmap.
    class_order : list[str]
        Display order for both axes.
    use_integer_coords : bool
        Use 0-based integer indices for annotation x/y coordinates instead of
        string class labels.  Required when class labels are numeric strings
        (e.g. "1"–"9") to avoid Plotly's numeric-string parsing offset.
    rank_white_threshold : int
        Ranks <= this value receive white annotation text (dark cells).
        Ranks above it receive black text (light cells).
    legend_max_label : str or None
        Legend label for the lightest color (highest rank).
        Defaults to "Rank N (least common neighbor)" where N = len(class_order).
    ml_border_label : str
        Legend label for the ML misclassification border.
    xaxis_title, yaxis_title : str
        Axis titles.
    output_path : str or None
        Full save path.  None = no save.
    width, height : int
        Figure dimensions in pixels.

    Returns
    -------
    plotly.graph_objs.Figure
    """
    n_classes = len(class_order)
    if legend_max_label is None:
        legend_max_label = f"Rank {n_classes} (least common neighbor)"

    y_labels = ranked_matrix.index.tolist()
    x_labels = ranked_matrix.columns.tolist()

    # --- Shapes ---
    def _cell_shape(x_idx, y_idx, dash, width_px):
        return dict(
            type="rect",
            x0=x_idx - 0.5, x1=x_idx + 0.5,
            y0=y_idx - 0.5, y1=y_idx + 0.5,
            xref="x", yref="y",
            line=dict(color=BORDER_COLOR, width=width_px, dash=dash),
            fillcolor="rgba(0,0,0,0)",
            layer="above",
        )

    shapes = []

    # Diagonal overlay
    for cls in class_order:
        if cls in x_labels and cls in y_labels:
            xi = x_labels.index(cls)
            yi = y_labels.index(cls)
            shapes.append(dict(
                type="rect",
                x0=xi - 0.5, x1=xi + 0.5,
                y0=yi - 0.5, y1=yi + 0.5,
                xref="x", yref="y",
                fillcolor=DIAG_COLOR,
                line=dict(color="rgba(0,0,0,0)"),
                layer="above",
            ))

    # ML rank-1 misclassification borders
    for y_idx, y_class in enumerate(y_labels):
        for ml_class in top_ml_rank1.get(y_class, set()):
            if ml_class in x_labels:
                shapes.append(_cell_shape(x_labels.index(ml_class), y_idx, "solid", 4))

    # --- Color values (sqrt-transform) ---
    color_values = np.sqrt(ranked_matrix.values.T.astype(float))

    # --- Annotations ---
    text_matrix = ranked_matrix.values.T  # shape: (n_y, n_x)
    annotations = []
    for yi, y_cls in enumerate(y_labels):
        for xi, x_cls in enumerate(x_labels):
            rank_val = int(text_matrix[yi, xi])
            is_diag  = (xi == yi)
            text_color = (
                "#333333" if is_diag
                else ("white" if rank_val <= rank_white_threshold else "black")
            )
            ann = dict(
                text=str(rank_val),
                showarrow=False,
                font=dict(color=text_color, size=18, family="Arial"),
                xref="x", yref="y",
            )
            # Numeric string labels like "1"-"9" must use integer coords
            if use_integer_coords:
                ann["x"] = xi
                ann["y"] = yi
            else:
                ann["x"] = x_cls
                ann["y"] = y_cls
            annotations.append(ann)

    # --- Figure ---
    fig = go.Figure(go.Heatmap(
        z=color_values,
        x=x_labels,
        y=y_labels,
        colorscale=[(0.0, TEAL_DARK), (0.5, TEAL_MID), (1.0, TEAL_LIGHT)],
        zmin=1.0,
        zmax=float(np.sqrt(n_classes)),
        showscale=False,
        zsmooth=False,
    ))

    # Legend dummy traces
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="square", size=14, color=TEAL_DARK),
        name="Rank 1 (most common neighbor)", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="square", size=14, color=TEAL_LIGHT),
        name=legend_max_label, showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color=BORDER_COLOR, width=4, dash="solid"),
        name=ml_border_label, showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="square", size=14, color="rgba(185,185,185,0.75)"),
        name="Diagonal (self, excluded)", showlegend=True,
    ))

    fig.update_layout(
        title=None,
        font={"family": "Arial", "color": "black", "size": 18},
        template="plotly_white",
        width=width,
        height=height,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis=dict(
            tickfont=dict(size=18), tickangle=0,
            automargin=False, showgrid=False,
        ),
        yaxis=dict(
            tickfont=dict(size=18), automargin=False, showgrid=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            font=dict(size=16, family="Arial"), borderwidth=0,
        ),
        shapes=shapes,
        annotations=annotations,
        margin=dict(l=60, r=20, t=120, b=60),
    )

    if output_path:
        fmt = "pdf" if output_path.endswith(".pdf") else "png"
        fig.write_image(output_path, format=fmt, width=width, height=height,
                        scale=1 if fmt == "pdf" else 2)
        print(f"Saved {output_path}")

    return fig


# ---------------------------------------------------------------------------
# 4. KNN alignment score vs k — line plot across embedding spaces
# ---------------------------------------------------------------------------

def plot_knn_alignment_lines(
    df,
    color_map,
    category_order,
    color_col="Embedding",
    legend_title="",
    title=None,
    y_title="Mean KNN feature alignment score",
    y_dtick=0.01,
    facet_col="distance_metric",
    facet_aliases=None,
    facet_order=None,
    symbol_map=None,
    output_path=None,
    width=1200,
    height=500,
):
    """
    Line plot of mean KNN alignment score vs k, colored by embedding model.

    One line per model, one facet column per distance metric (or other column).
    Models are labelled as "performance (ModelName)" in the legend so the color
    encodes relative performance at a glance.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-grouped data with columns [k, <color_col>, Fraction, <facet_col>].
        The <color_col> should already contain the display labels (e.g.
        "64.9 (ProtT5)") and be a Categorical with the desired ordering.
    color_map : dict
        {display_label: hex_color} — passed to px.line as color_discrete_map.
    category_order : list[str]
        Display order for the color/legend entries.
    color_col : str
        Column in df used for color grouping (default "Embedding").
    legend_title : str
        Title shown above the legend (e.g. "Q9 as reported by Heinzinger et al.").
    title : str or None
        Figure title (upper-left).  None = no title.
    y_title : str
        Y-axis label.
    y_dtick : float
        Tick interval for the y-axis.
    facet_col : str
        Column used for facet columns (default "distance_metric").
    facet_aliases : dict or None
        {raw_value: display_label} — renames facet column titles.
    facet_order : list[str] or None
        Left-to-right order of facet values.  Values absent from the data are
        ignored; values in the data but not in the list are appended at the end.
        None = plotly default order.
    symbol_map : dict or None
        {display_label: symbol_name} — adds distinct marker symbols per model.
        None = all circles.
    output_path : str or None
        Full save path (PDF or PNG).  None = no save.
    width, height : int
        Figure dimensions in pixels.

    Returns
    -------
    plotly.graph_objs.Figure
    """
    cat_orders = {color_col: category_order}
    if facet_order is not None:
        present = df[facet_col].unique().tolist()
        ordered = [f for f in facet_order if f in present]
        ordered += [f for f in present if f not in ordered]
        cat_orders[facet_col] = ordered

    px_kwargs = dict(
        data_frame=df,
        x="k",
        y="Fraction",
        color=color_col,
        color_discrete_map=color_map,
        category_orders=cat_orders,
        facet_col=facet_col,
        markers=True,
    )
    if symbol_map is not None:
        px_kwargs["symbol"] = color_col
        px_kwargs["symbol_map"] = symbol_map

    fig = px.line(**px_kwargs)
    fig.update_traces(marker=dict(size=10), line=dict(width=4))

    # Axis styling — iterate over all x/y axes generated by facets
    for axis_key in fig.layout:
        if axis_key.startswith("xaxis"):
            fig.layout[axis_key].update(
                showgrid=False, linecolor="black", linewidth=3,
                ticks="outside", tickwidth=2, tickcolor="black", ticklen=6,
                tickformat=".0f", title_text="k",
            )
        if axis_key.startswith("yaxis"):
            fig.layout[axis_key].update(
                showgrid=False, linecolor="black", linewidth=3,
                ticks="outside", tickwidth=2, tickcolor="black", ticklen=6,
                tickformat=".2f", dtick=y_dtick,
            )

    # Rename facet strip annotations
    if facet_aliases:
        for ann in fig.layout.annotations:
            for key, alias in facet_aliases.items():
                if key in ann.text:
                    ann.text = ann.text.replace(key, alias)
            if "=" in ann.text:
                ann.text = ann.text.split("=")[1]

    layout_kw = dict(
        template="plotly_white",
        font={"family": "Arial", "color": "black", "size": 20},
        legend_title_text=legend_title,
        yaxis_title=y_title,
        width=width,
        height=height,
        legend=dict(
            orientation="h",
            x=0.5,
            y=1.05,
            xanchor="center",
            yanchor="bottom",
        ),
        margin=dict(t=120),
    )
    if title is not None:
        layout_kw["title"] = dict(text=title, x=0, xanchor="left")

    fig.update_layout(**layout_kw)

    if output_path:
        fmt = "pdf" if output_path.endswith(".pdf") else "png"
        fig.write_image(output_path, format=fmt, width=width, height=height)
        print(f"Saved {output_path}")

    return fig


# ---------------------------------------------------------------------------
# 5. KNN alignment lines — model-centric variant (uses MODEL_COLORS, no perf label)
# ---------------------------------------------------------------------------

_CANONICAL_METRIC_ORDER = ["cosine", "cityblock", "euclidean"]


def plot_knn_alignment_lines_model(
    df,
    color_col="Embedding",
    model_colors=None,
    legend_title="",
    title=None,
    y_title="Mean KNN feature alignment score",
    y_dtick=0.01,
    facet_col="distance_metric",
    facet_aliases=None,
    facet_order=None,
    symbol_map=None,
    output_path=None,
    width=1200,
    height=500,
):
    """
    Variant of plot_knn_alignment_lines that uses MODULE-LEVEL MODEL_COLORS
    automatically and shows only the model/embedding name in the legend
    (no downstream-performance prefix).

    Distance-metric facets are ordered cosine -> manhattan -> euclidean by default.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-grouped data with columns [k, <color_col>, Fraction, <facet_col>].
        The <color_col> must contain model names that are keys in MODEL_COLORS
        (or in *model_colors* if supplied).
    color_col : str
        Column in df used for color grouping (default "Embedding").
    model_colors : dict or None
        Override the module-level MODEL_COLORS mapping.  None = use MODEL_COLORS.
    legend_title : str
        Title shown above the legend.
    title : str or None
        Figure title (upper-left).  None = no title.
    y_title : str
        Y-axis label.
    y_dtick : float
        Tick interval for the y-axis.
    facet_col : str
        Column used for facet columns (default "distance_metric").
    facet_aliases : dict or None
        {raw_value: display_label} -- renames facet strip titles.
    facet_order : list[str] or None
        Left-to-right facet order.  Defaults to
        ["cosine", "manhattan", "euclidean"]; absent values are skipped.
    symbol_map : dict or None
        {model_name: symbol_name} -- adds distinct marker symbols per model.
    output_path : str or None
        Full save path (PDF or PNG).  None = no save.
    width, height : int
        Figure dimensions in pixels.

    Returns
    -------
    plotly.graph_objs.Figure
    """
    colors = model_colors or MODEL_COLORS

    # Keep only models that appear in the data, in MODEL_COLORS key order
    present = set(df[color_col].unique())
    category_order = [m for m in colors if m in present]
    # Append any models in the data but not in colors (fallback)
    category_order += [m for m in present if m not in colors]
    color_map = {m: colors[m] for m in category_order if m in colors}

    return plot_knn_alignment_lines(
        df=df,
        color_map=color_map,
        category_order=category_order,
        color_col=color_col,
        legend_title=legend_title,
        title=title,
        y_title=y_title,
        y_dtick=y_dtick,
        facet_col=facet_col,
        facet_aliases=facet_aliases,
        facet_order=facet_order if facet_order is not None else _CANONICAL_METRIC_ORDER,
        symbol_map=symbol_map,
        output_path=output_path,
        width=width,
        height=height,
    )
