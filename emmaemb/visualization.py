import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from scipy import stats

from emmaemb.core import Emma
from emmaemb.functions import *


# Canonical metric ordering and display labels for KNN alignment plots
_METRIC_ORDER = ["cosine", "cityblock", "euclidean"]
_METRIC_DISPLAY = {
    "cosine": "Cosine distance",
    "cityblock": "Manhattan distance",
    "euclidean": "Euclidean distance",
}


def _find_elbow(x: np.ndarray, y: np.ndarray) -> int:
    """Return the index of the elbow using the max-distance-from-chord method.

    Both axes are normalized to [0, 1] before computing the perpendicular
    distance from the line connecting the first and last points, so the result
    is scale-invariant.
    """
    x_n = (x - x[0]) / (x[-1] - x[0] + 1e-12)
    span = y.max() - y.min()
    y_n = (y - y[0]) / (span + 1e-12)
    dx, dy = x_n[-1] - x_n[0], y_n[-1] - y_n[0]
    norm = np.hypot(dx, dy) + 1e-12
    dist = np.abs(dy * x_n - dx * y_n + dx * y_n[0] - dy * x_n[0]) / norm
    return int(np.argmax(dist))


def update_fig_layout(fig: go.Figure) -> go.Figure:
    """Update the layout of a plotly figure to adjust the font, line,\
        and grid settings.

    Args:
        fig (go.Figure): Plotly figure object.

    Returns:
        go.Figurge : Plotly figure object with updated layout.
    """
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Arial", size=12, color="black"),
    )
    # show line at y=0 and x=0
    fig.update_xaxes(showline=True, linecolor="black", linewidth=2)
    fig.update_yaxes(showline=True, linecolor="black", linewidth=2)
    # hide gridlines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def plot_emb_space(
    emma: Emma,
    emb_space: str,
    method: str = "PCA",
    normalize: bool = True,
    color_by: str = None,
    logarithmic_colors: bool = False,
    verbose_tooltips: bool = False,
    random_state: int = 42,
    perplexity: int = 30,
    shuffle_umap: bool = True,
) -> go.Figure:
    """Function to plot the embeddings of a given embedding space in 2D. \
    Dimensionality reduction is performed using PCA, TSNE, or UMAP.\
    The dots are colored by a column in the metadata.

    Args:
        emma (Emma): An instance of the Emma class.
        emb_space (str): Name of an embedding space in the Emma instance.
        method (str, optional): Method for dimensionality reduction. \
            Either "PCA", "TSNE", or "UMAP". Defaults to "PCA".
        normalize (bool, optional): Whether to perform z-score normalisation \
            prior to dimensionality reduction. Defaults to True.
        color_by (str, optional): A column name from the metadata stored in \
            the Emma object, by which the dots are colored. Defaults to None.
        verbose_tooltips (bool, optional): Show all metadata on hover tooltips \
            rather than only the sample ID. Defaults to False.
        logarithmic_colors (bool, optional): Use a logarithmic scale to color by \
            a numerical column. Defaults to False.
        random_state (int, optional): Random state for UMAP or TSNE. Defaults \
            to 42.
        perplexity (int, optional): Perplexity, only applied to t-SNE.\
            Defaults to 30.
        shuffle_umap (bool, optional): Shuffle order of embeddings before \
            running UMAP. Defaults to True

    Returns:
        go.Figure: A scatter plot of the embeddings in 2D.
    """

    embeddings_2d = emma.get_2d(
            emb_space=emb_space,
            method=method,
            normalize=normalize,
            random_state=random_state,
            perplexity=perplexity,
            shuffle_umap=shuffle_umap
        )

    if verbose_tooltips:
        hover_data = emma.metadata.to_dict(orient='list')
    else:
        hover_data = {"Sample": emma.sample_names}

    # args for px.scatter
    scatter_args = {
        "x": embeddings_2d["2d"][:, 0],
        "y": embeddings_2d["2d"][:, 1],
        "title": f"{emb_space} embeddings after {method}",
        "hover_data": hover_data,
        "opacity": 0.5,
    }

    # categorical column
    try:
        emma._check_column_is_categorical(color_by)
        scatter_args["color_discrete_map"] = emma.color_map[color_by]
        scatter_args["color"] = emma.metadata[color_by]
        scatter_args["labels"] = {"color": color_by}
    except: pass

    # numeric column
    try:
        emma._check_column_is_numeric(color_by)
        if logarithmic_colors:
            scatter_args["color"] = np.log10(emma.metadata[color_by])
            scatter_args["labels"] = {"color": f"log({color_by})"}
        else:
            scatter_args["color"] = emma.metadata[color_by]
            scatter_args["labels"] = {"color": color_by}
    except: pass

    fig = px.scatter(**scatter_args)

    fig.update_layout(
        width=800,
        height=800,
        autosize=False,
        legend=dict(
            title=f"{color_by.capitalize() if color_by else 'Sample'}",
        ),
    )

    fig.update_traces(
        marker=dict(size=max(10, (1 / len(emma.sample_names)) * 400))
    )

    if method == "PCA" and "variance_explained" in embeddings_2d:
        variance_explained = embeddings_2d["variance_explained"]
        fig.update_layout(
            xaxis_title="PC1 ({}%)".format(
                round(variance_explained[0] * 100, 2)
            ),
            yaxis_title="PC2 ({}%)".format(
                round(variance_explained[1] * 100, 2)
            ),
        )
    fig = update_fig_layout(fig)
    return fig


def plot_pairwise_distance_heatmap(
    emma: Emma,
    emb_space: str,
    metric: str = "euclidean",
    group_by: str = None,
    sample_labels: bool = True,
    color_scale: str = "Greys",
) -> go.Figure:
    """Function to plot a heatmap of pairwise distances between samples in an \
    embedding space.
        
    Args:
        emma (Emma): An instance of the Emma class.
        emb_space (str): Name of an embedding space in the Emma instance.
        metric (str, optional): Distance metric to use. Defaults to "euclidean"
        group_by (str): Metadata column name to group and order the heatmap. \
            Default is None.
        sample_labels (bool, optional): Whether to show sample names on the \
            x and y axes. Defaults to True.
        color_scale (str, optional): Color scale for the heatmap. \
            Defaults to "Greys".
    Returns:

        go.Figure: A heatmap of pairwise distances between samples.
    """

    # Ensure pairwise distances are calculated
    emma._check_for_emb_space(emb_space)
    if metric not in emma.emb[emb_space].get("pairwise_distances", {}):
        raise ValueError(
            f"Pairwise distances for metric {metric} not found. \
            Run `calculate_pairwise_distances` first."
        )
    if group_by:
        if group_by not in emma.metadata.columns:
            raise ValueError(
                f"Group column '{group_by}' not found in metadata."
            )

    # retrieve pairwise distances and sample names
    pairwise_distances = emma.emb[emb_space]["pairwise_distances"][metric]
    sample_names = emma.sample_names if sample_labels else None

    if group_by is not None:
        group_labels = emma.metadata[group_by].values
        sorted_indices = np.argsort(group_labels)
        pairwise_distances = pairwise_distances[sorted_indices][
            :, sorted_indices
        ]
        if sample_labels:
            sample_names = np.array(emma.sample_names)[sorted_indices]
        else:
            sample_names = None
        group_labels = group_labels[sorted_indices]
    else:
        group_labels = None
        sample_names = np.array(emma.sample_names) if sample_labels else None

    median_value = np.median(pairwise_distances)
    reversed_color_scale = color_scale + "_r"

    hover_text = []
    for i in range(pairwise_distances.shape[0]):
        hover_row = []
        for j in range(pairwise_distances.shape[1]):
            distance = pairwise_distances[i, j]
            row_label = group_labels[i] if group_labels is not None else "N/A"
            col_label = group_labels[j] if group_labels is not None else "N/A"
            row_name = sample_names[i] if sample_labels else f"Sample {i}"
            col_name = sample_names[j] if sample_labels else f"Sample {j}"
            hover_info = (
                f"Row Sample: {row_name}<br>Col Sample: {col_name}<br>"
                f"Distance: {distance:.2f}<br>"
                f"Row Group: {row_label}<br>Col Group: {col_label}"
            )
            hover_row.append(hover_info)
        hover_text.append(hover_row)

    heatmap = go.Heatmap(
        z=pairwise_distances,
        x=sample_names,
        y=sample_names,
        text=hover_text,
        hoverinfo="text",
        colorscale=reversed_color_scale,
        zmid=median_value,
        colorbar=dict(title=f"{metric.capitalize()} Distance"),
    )

    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        title=(
            f"Pairwise Distance Heatmap ({metric.capitalize()}) in {emb_space}"
        ),
        xaxis=dict(title="Samples", tickangle=45),
        yaxis=dict(title="Samples"),
    )

    fig = update_fig_layout(fig)

    return fig


def plot_pairwise_distance_comparison(
    emma: Emma,
    emb_space_x: str,
    emb_space_y: str,
    metric: str = "euclidean",
    title: str = "Pairwise Distance Comparison",
    color: str = "blue",
    group_by: str = None,
    point_opacity: float = 0.5,
) -> go.Figure:
    """Function to plot a scatter plot comparing pairwise distances between \
    samples in two embedding spaces.
    
    Args:
        emma (Emma): An instance of the Emma class.
        emb_space_x (str): Name of the first embedding space in the \
            Emma instance.
        emb_space_y (str): Name of the second embedding space in the \
            Emma instance.
        metric (str, optional): Distance metric to use. Defaults to "euclidean".
        title (str, optional): Title of the plot. Defaults to \
            "Pairwise Distance Comparison".
        color (str, optional): Color of the plot elements. Defaults to "blue".
        group_by (str, optional): Metadata column name to group and color \
            the points. Defaults to None.
        point_opacity (float, optional): Opacity of the points. \
            Defaults to 0.5.
    
    Returns:
        go.Figure: A scatter plot comparing pairwise distances between \
        samples in two embedding spaces.
    """
    # Ensure both embedding spaces exist and pairwise distances are calculated
    for emb_space in [emb_space_x, emb_space_y]:
        emma._check_for_emb_space(emb_space)
        if metric not in emma.emb[emb_space].get("pairwise_distances", {}):
            raise ValueError(
                f"Pairwise distances for metric {metric} not found \
                    in {emb_space}. Run `calculate_pairwise_distances` first."
            )

    neutral_color: str = "#CCCCCC"

    emb_pwd_1 = emma.emb[emb_space_x]["pairwise_distances"][metric]
    emb_pwd_2 = emma.emb[emb_space_y]["pairwise_distances"][metric]

    group_labels = None
    if group_by:
        if group_by not in emma.metadata.columns:
            raise ValueError(
                f"Group column '{group_by}' not found in metadata."
            )
        group_labels = emma.metadata[group_by].values

    group_labels = None
    if group_by:
        if group_by not in emma.metadata.columns:
            raise ValueError(
                f"Group column '{group_by}' not found in metadata."
            )
        group_labels = emma.metadata[group_by].values

    n_samples = len(emma.sample_names)
    colors = []
    hover_samples = []
    legend_labels = []

    for i in range(n_samples):
        for j in range(i + 1, n_samples):

            sample_pair = f"{emma.sample_names[i]} - {emma.sample_names[j]}"
            hover_samples.append(sample_pair)

            if group_labels is not None:
                group_i = group_labels[i]
                group_j = group_labels[j]

                # If both samples belong to the same group, use group label
                # for color
                if group_i == group_j:
                    color = emma.color_map.get(group_i, neutral_color)
                    legend_labels.append(group_i)
                else:
                    # Use neutral color for different groups
                    color = neutral_color
                    legend_labels.append("Neutral")
            else:
                # If no group_by is specified, assign all points to neutral
                # color
                color = neutral_color
                legend_labels.append("Neutral")

            colors.append(color)

    x = emb_pwd_1[np.triu_indices(n_samples, k=1)]
    y = emb_pwd_2[np.triu_indices(n_samples, k=1)]

    # Create the scatter plot
    color_discrete_map = (
        {group: neutral_color for group in set(legend_labels)}
        if group_by is None else
        {
            "Neutral": neutral_color,
            **{
                group: emma.color_map[group_by].get(group, neutral_color)
                for group in set(legend_labels)
            },
        }
    )
    fig = px.scatter(
        x=x,
        y=y,
        title=title,
        opacity=point_opacity,
        color=legend_labels,
        color_discrete_map=color_discrete_map,
        hover_data={"Sample pair": hover_samples},
    )

    # Compute Spearman correlation between the distances of both
    # embedding spaces
    corr, p_value = stats.spearmanr(x, y)

    # Add correlation to title
    fig.update_layout(
        title=f"{title} <br> Spearman correlation: {corr:.2f} <br> \
            p-value: {p_value:.4f}"
    )

    # Adjust axes to have the same scale
    fig.update_xaxes(
        range=[0, max(x.max() * 1.1, y.max() * 1.1)],
        title=f"{emb_space_x} {metric.capitalize()} Distance",
    )
    fig.update_yaxes(
        range=[0, max(x.max() * 1.1, y.max() * 1.1)],
        title=f"{emb_space_y} {metric.capitalize()} Distance",
    )

    fig = update_fig_layout(fig)

    return fig


def plot_knn_alignment_across_embedding_spaces(
    emma: Emma,
    feature: str,
    k: int = 10,
    metric: str = "euclidean",
    emb_space_order: list = None,
    color: str = "#303496",
    use_annoy: bool = False,
    annoy_metric: str = None,
    n_trees: int = None,
    adjust_for_imbalance: bool = False,
):
    """
    Function to plot KNN alignment scores for a given feature \
    across multiple embedding spaces.
    
    Args:
        emma (Emma): An instance of the Emma class.
        feature (str): Name of the feature in the metadata.
        k (int, optional): Number of nearest neighbors to consider. \
            Defaults to 10.
        metric (str, optional): Distance metric to use. \
            Defaults to "euclidean".
        emb_space_order (list, optional): Order in which to display the \
            embedding spaces. Defaults to None.
        color (str, optional): Color of the plot elements. \
            Defaults to "#303496".
        use_annoy (bool, optional): Whether to use Annoy index. \
            Defaults to False.
        annoy_metric (str, optional): Annoy distance metric to use. \
            Defaults to None.
        n_trees (int, optional): Number of trees used to build the Annoy index. \
            Defaults to None.
        
    Returns:
        go.Figure: A box plot of KNN alignment scores across embedding spaces.
    """

    df = get_knn_alignment_scores(emma, feature, k, metric, use_annoy, annoy_metric, n_trees, adjust_for_imbalance)
    fig = px.box(
        df,
        x="Embedding",
        y="Fraction",
        title=f"KNN feature alignment scores for {feature}<br>k = {k}, {metric}",
        labels={
            "Embedding": "Embedding Space",
            "Fraction": "KNN feature alignment scores",
        },
        template="plotly_white",
        color_discrete_sequence=[color],
    )

    if emb_space_order:
        fig.update_xaxes(categoryorder="array", categoryarray=emb_space_order)

    fig = update_fig_layout(fig)

    return fig


def plot_knn_alignment_across_k(
    emma: Emma,
    feature: str,
    emb_spaces: list = None,
    k_values: list = None,
    metrics: list = None,
    color_discrete_map: dict = None,
    title: str = None,
    elbow_detection: bool = False,
    show_random_baselines: bool = True,
    return_data: bool = False,
    width: int = 1000,
    height: int = 420,
    use_annoy: bool = False,
    annoy_metric: str = None,
    n_trees: int = None,
) -> go.Figure:
    """Plot mean KNN feature alignment score vs k, faceted by distance metric.

    For each k and distance metric, computes the fraction of k nearest
    neighbors that share the same label as the query point, then averages
    over all samples.  Two random baselines are shown:

    - **Uniform random** – expected score if all classes were equal-sized
      (``1 / n_classes``).
    - **Distribution random** – expected score given the observed class
      proportions (``sum(p_i^2)``); always ≥ the uniform baseline.

    Facets are ordered Cosine → Manhattan → Euclidean when those metrics are
    present.  The facet column titles show only the metric name (e.g.
    "Cosine distance"), not the raw column label.

    Args:
        emma: Emma object with precomputed pairwise ranks for the requested
            metrics.
        feature: Categorical metadata column to use as the class label.
        k_values: k values to evaluate.  Defaults to ``[5, 10, 20, 30, 50]``.
        metrics: Distance metrics to show as facet columns.  Defaults to all
            metrics precomputed on the Emma object, in canonical order.
        emb_spaces: Embedding spaces to include.  Defaults to all spaces.
        color_discrete_map: Mapping from embedding-space name to color hex.
        title: Plot title.  Defaults to ``"KNN feature alignment — <feature>"``.
        elbow_detection: If ``True``, annotate each curve with the elbow k
            found via the max-distance-from-chord method.
        show_random_baselines: If ``True`` (default), draw the uniform and
            distribution-aware random baseline lines.
        return_data: If ``True``, return a ``(fig, df_mean)`` tuple instead of
            just the figure.  ``df_mean`` is the aggregated DataFrame with
            columns ``k``, ``Embedding``, ``distance_metric``, ``Fraction``.
            Defaults to ``False``.
        width: Figure width in pixels.  Defaults to 1000.
        height: Figure height in pixels.  Defaults to 420.
        use_annoy: If ``True``, use the prebuilt Annoy index instead of the
            precomputed rank matrix.  Defaults to ``False``.
        annoy_metric: Annoy metric name to use.  Required when
            ``use_annoy=True``.  Passed through to
            ``get_knn_alignment_scores``.
        n_trees: Number of Annoy trees.  Required when ``use_annoy=True``.

    Returns:
        go.Figure | tuple: Plotly figure, or ``(figure, df_mean)`` if
            ``return_data=True``.
    """
    if k_values is None:
        k_values = [5, 10, 20, 30, 50]
    if emb_spaces is None:
        emb_spaces = list(emma.emb.keys())

    # Determine metrics and sort in canonical order
    if metrics is None:
        all_metrics: set = set()
        for es in emb_spaces:
            all_metrics.update(emma.emb[es].get("ranks", {}).keys())
        metrics = [m for m in _METRIC_ORDER if m in all_metrics]
        metrics += sorted(m for m in all_metrics if m not in _METRIC_ORDER)
    else:
        metrics = sorted(
            metrics,
            key=lambda m: _METRIC_ORDER.index(m) if m in _METRIC_ORDER else 99,
        )

    # Compute KNN alignment scores
    rows = []
    for metric in metrics:
        for k in k_values:
            scores = get_knn_alignment_scores(
                emma, feature=feature, k=k, metric=metric,
                use_annoy=use_annoy,
                annoy_metric=annoy_metric if annoy_metric is not None else metric,
                n_trees=n_trees,
            )
            scores["k"] = k
            scores["distance_metric"] = metric
            rows.append(scores)

    knn_df = pd.concat(rows, ignore_index=True)
    df_mean = (
        knn_df
        .groupby(["k", "Embedding", "distance_metric"], as_index=False)
        .agg({"Fraction": "mean"})
        .sort_values(["Embedding", "k"])
    )

    # Map metric keys to display labels
    df_plot = df_mean.copy()
    df_plot["distance_metric"] = df_plot["distance_metric"].map(
        lambda m: _METRIC_DISPLAY.get(m, m)
    )
    metric_labels_ordered = [_METRIC_DISPLAY.get(m, m) for m in metrics]

    # Random baselines
    labels = emma.metadata[feature]
    n_classes = int(labels.nunique())
    uniform_baseline = 1.0 / n_classes
    probs = labels.value_counts(normalize=True)
    distribution_baseline = float((probs ** 2).sum())

    if title is None:
        title = f"KNN feature alignment — {feature}"

    fig = px.line(
        df_plot, x="k", y="Fraction",
        color="Embedding",
        facet_col="distance_metric",
        category_orders={"distance_metric": metric_labels_ordered},
        markers=True,
        title=title,
        labels={"Fraction": "Mean KNN feature<br>alignment score", "distance_metric": ""},
        color_discrete_map=color_discrete_map,
        template="plotly_white",
        width=width, height=height,
    )

    # Strip "distance_metric=" prefix from facet column annotations
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    # Compute y-axis range including baselines if shown
    y_vals = list(df_mean["Fraction"])
    if show_random_baselines:
        y_vals += [uniform_baseline, distribution_baseline]
    y_min = min(y_vals)
    y_max = max(y_vals)
    y_pad = (y_max - y_min) * 0.12 or 0.05
    y_range = [max(0.0, y_min - y_pad), min(1.0, y_max + y_pad)]

    if show_random_baselines:
        # Uniform random baseline
        fig.add_hline(
            y=uniform_baseline, line_dash="dot", line_color="grey",
            annotation_text=f"uniform random ({uniform_baseline:.2f})",
            annotation_position="bottom right",
        )
        # Distribution-aware baseline (only show when meaningfully different)
        if abs(distribution_baseline - uniform_baseline) > 0.005:
            fig.add_hline(
                y=distribution_baseline, line_dash="dash", line_color="lightgrey",
                annotation_text=f"distribution random ({distribution_baseline:.2f})",
                annotation_position="top right",
            )

    # Styling
    fig.update_traces(marker=dict(size=8), line=dict(width=3))
    fig.update_layout(font=dict(family="Arial", size=13))
    fig.update_xaxes(
        showline=True, linecolor="black", linewidth=2, showgrid=False,
        range=[0, max(k_values) * 1.08],
        tick0=0,
    )
    fig.update_yaxes(
        showline=True, linecolor="black", linewidth=2, showgrid=False,
        range=y_range,
    )

    # Elbow detection: print a summary table of elbow k per (embedding, metric)
    if elbow_detection:
        k_arr = np.array(sorted(k_values), dtype=float)
        elbow_rows = []
        for metric, metric_label in zip(metrics, metric_labels_ordered):
            for emb in emb_spaces:
                subset = df_mean[
                    (df_mean["Embedding"] == emb)
                    & (df_mean["distance_metric"] == metric)
                ].sort_values("k")
                if len(subset) < 3:
                    continue
                y_arr = subset["Fraction"].values
                elbow_idx = _find_elbow(k_arr, y_arr)
                elbow_k = int(k_arr[elbow_idx])
                elbow_score = float(y_arr[elbow_idx])
                elbow_rows.append({
                    "Embedding": emb,
                    "Metric": metric_label,
                    "Elbow k": elbow_k,
                    "Score at elbow": round(elbow_score, 3),
                })
        if elbow_rows:
            print("Elbow k (max-distance-from-chord method):")
            print(pd.DataFrame(elbow_rows).to_string(index=False))

    if return_data:
        return fig, df_mean
    return fig


def plot_knn_alignment_across_classes(
    emma: Emma,
    feature: str,
    k: int = 10,
    metric: str = "euclidean",
    emb_space_order: list = None,
    color: str = "#303496",
    use_annoy: bool = False,
    annoy_metric: str = None,
    n_trees: int = None,
    adjust_for_imbalance: bool = False,
) -> go.Figure:
    """Function to plot KNN alignment scores for a given feature across \
    multiple embedding spaces.
    
    Args:
        emma (Emma): An instance of the Emma class.
        feature (str): Name of the feature in the metadata.
        k (int, optional): Number of nearest neighbors to consider. \
            Defaults to 10.
        metric (str, optional): Distance metric to use. Defaults to "euclidean".
        emb_space_order (list, optional): Order in which to display the \
            embedding spaces. Defaults to None.
        color (str, optional): Color of the plot elements. \
            Defaults to "#303496".
        use_annoy (bool, optional): Whether to use Annoy index. \
            Defaults to False.
        annoy_metric (str, optional): Annoy distance metric to use. \
            Defaults to None.
        n_trees (int, optional): Number of trees used to build the Annoy index. \
            Defaults to None.
    
    Returns:
        go.Figure: A heatmap of KNN alignment scores across
    """
    df = get_knn_alignment_scores(emma, feature, k, metric, use_annoy, annoy_metric, n_trees, adjust_for_imbalance)

    heatmap_data = (
        df.groupby(["Class", "Embedding"])["Fraction"]
        .mean()
        .unstack()  # Reshape to have Classes as rows and Embeddings as columns
    )

    if emb_space_order:
        heatmap_data = heatmap_data.reindex(columns=emb_space_order)

    class_counts = df.groupby("Class").size()

    heatmap_data.index = [
        f"{feature_class} (n = {int(count / len(df['Embedding'].unique()))})"
        for feature_class, count in zip(
            heatmap_data.index, class_counts[heatmap_data.index]
        )
    ]

    fig = px.imshow(
        heatmap_data,
        labels=dict(
            x="Embedding Space",
            y="Feature Class (Samples)",
            color="Mean KNN feature alignment score",
        ),
        title=f"Mean KNN feature alignment scores for {feature} \
            across Embedding Spaces<br> \
            k = {k}, {metric}",
        color_continuous_scale=[
            (0.0, "lightblue"),
            (1.0, color),
        ],
        text_auto=".2f",
        aspect="auto",
    )

    # Update font settings for the heatmap
    fig.update_layout(font=dict(family="Arial"))

    return fig


def plot_knn_class_mixing_matrix(
    emma: Emma,
    emb_space: str,
    feature: str,
    k: int = 10,
    metric: str = "euclidean",
    use_annoy: bool = False,
    annoy_metric: str = None,
    n_trees: int = None,
) -> go.Figure:
    """Function to plot a matrix of class mixing in k \
    nearest neighbors for a given feature in an embedding space.
    
    Args:
        emma (Emma): An instance of the Emma class.
        emb_space (str): Name of the embedding space in the Emma instance.
        feature (str): Name of the feature in the metadata.
        k (int, optional): Number of nearest neighbors to consider. \
            Defaults to 10.
        metric (str, optional): Distance metric to use. Defaults to "euclidean".
        use_annoy (bool, optional): Whether to use Annoy index. \
            Defaults to False.
        annoy_metric (str, optional): Annoy distance metric to use. \
            Defaults to None.
        n_trees (int, optional): Number of trees used to build the Annoy index. \
            Defaults to None.
        
    Returns:
        go.Figure: A heatmap of class mixing in k nearest neighbors. \
            Rows represent the feature class of the sample, \
            columns represent the feature class of the neighbor. \
                Values represent the count of neighbors in each class.
    """
    mixing_counts, class_labels = get_class_mixing_in_neighborhood(
        emma, emb_space, feature, k, metric, use_annoy, annoy_metric, n_trees
    )

    mixing_df = pd.DataFrame(
        mixing_counts, index=class_labels, columns=class_labels
    )

    fig = px.imshow(
        mixing_df,
        labels=dict(
            x="Feature Class (Sample)",
            y="Feature Class (Neighbor)",
            color="Neighbor Count",
        ),
        title=f"Class Mixing in Neighborhoods (Embedding: {emb_space})",
        color_continuous_scale="Blues",
        text_auto=True,
        aspect="auto",
    )

    fig.update_traces(texttemplate="%{z:.0f}")

    fig.update_layout(font=dict(family="Arial", color="black"))

    return fig


def plot_low_similarity_distribution(
    emma: Emma,
    emb_space_1: str,
    emb_space_2: str,
    feature: str,
    metric: str = "euclidean",
    k: int = 10,
    similarity_threshold: float = 0.2,
    use_annoy: bool = False,
    annoy_metric: str = None,
    n_trees: int = None,
) -> go.Figure:

    for emb_space in [emb_space_1, emb_space_2]:
        emma._check_for_emb_space(emb_space)

    emma._check_column_is_categorical(feature)

    similarities = get_neighborhood_similarity(
        emma, emb_space_1, emb_space_2, k, metric, use_annoy, annoy_metric, n_trees
    )

    low_similarity_indices = np.where(similarities < similarity_threshold)[0]
    low_similarity_samples = emma.metadata.iloc[low_similarity_indices]

    # Compute distributions
    total_distribution = emma.metadata[feature].value_counts(
        normalize=True
    )  # Entire dataset
    low_similarity_distribution = low_similarity_samples[feature].value_counts(
        normalize=True
    )  # Low-similarity subset

    aligned_distributions = total_distribution.align(
        low_similarity_distribution, fill_value=0
    )

    # Prepare scatter plot data
    fractions_in_dataset = aligned_distributions[
        0
    ]  # Fraction in entire dataset
    fractions_in_subset = aligned_distributions[
        1
    ]  # Fraction in low-similarity subset
    class_labels = aligned_distributions[0].index

    fig = px.scatter(
        x=fractions_in_dataset,
        y=fractions_in_subset,
        color=class_labels,
        labels={
            "x": "Fraction in Dataset",
            "y": "Fraction in Subsample",
            "color": feature,
        },
        title=f"Comparison of {feature} Fractions (Similarity < {similarity_threshold} between {emb_space_1} and {emb_space_2}, Metric: {metric}, k: {k})",
        template="plotly_white",
    )
    fig.update_traces(
        marker=dict(size=10, line=None)
    )  # Remove rim around dots
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(color="LightGrey", dash="dash"),
    )

    fig.update_layout(
        template="plotly_white",
        font=dict(family="Arial", size=12, color="black"),
    )
    # show line at y=0 and x=0
    fig.update_xaxes(showline=True, linecolor="black", linewidth=2)
    fig.update_yaxes(showline=True, linecolor="black", linewidth=2)
    # hide gridlines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    # Update layout for font and legend
    fig.update_layout(
        font=dict(family="Arial", color="black"),
        # legend=dict(orientation="h", yanchor="bottom", y=-0.2),  # Move legend below plot
        showlegend=True,
    )

    return fig

def plot_knn_alignment_vs_class_balance(
    emma: Emma,
    feature: str,
    emb_spaces: list = None,
    k_values: list = None,
    metrics: list = None,
    n_balance_steps: int = 8,
    seed: int = 42,
    color_discrete_map: dict = None,
    show_random_baselines: bool = True,
    return_data: bool = False,
) -> go.Figure:
    """Plot mean k-NN feature alignment score as a function of class balance.

    Starting from the original (potentially imbalanced) dataset, each class is
    progressively downsampled so that the maximum class size decreases toward
    the size of the smallest class.  The x-axis shows the cap applied to each
    class; the leftmost point corresponds to the fully balanced dataset and the
    rightmost to the original.

    k-NN are computed within each balanced subset so that the metric is not
    inflated by samples excluded from the subset.  For each metric, neighbors
    are computed once for the largest k and reused for all smaller k values.

    **Precomputed ranks (fast path):** if pairwise distances for a metric have
    already been computed on the Emma object, the stored global rank matrix is
    reused.  For each subset, the rank list of each sample is filtered to only
    keep neighbors that belong to the subset — no distance recomputation is
    needed.  The filtering is fully vectorised using a stable argsort.

    **Fallback:** if ranks are not precomputed for a given metric the function
    computes k-NN within each subset using ``sklearn.NearestNeighbors``.
    For cosine distance the embeddings are L2-normalized before fitting so
    that a tree-based index can be used instead of brute-force pairwise.

    Args:
        emma (Emma): Emma object.
        feature (str): Categorical metadata column to use as the class label.
        emb_spaces (list, optional): Embedding spaces to compare.
            Defaults to all spaces in the Emma object.
        k_values (list of int, optional): k values to show as facet rows.
            Defaults to [5, 10, 20].
        metrics (list of str, optional): Distance metrics to show as facet
            columns.  Defaults to ["euclidean"].
        n_balance_steps (int, optional): Number of balance steps between the
            smallest and largest class size.  Defaults to 8.
        seed (int, optional): Random seed for reproducible downsampling.
            Defaults to 42.
        color_discrete_map (dict, optional): Mapping from embedding-space name
            to color hex.  Defaults to Plotly Set2 palette.
        show_random_baselines (bool, optional): If True (default), draw the
            uniform and distribution-aware random baseline lines.

    Returns:
        go.Figure: Line plot faceted by k (rows) and metric (columns), with
            one line per embedding space.
    """
    if k_values is None:
        k_values = [5, 10, 20]
    if metrics is None:
        metrics = ["euclidean"]
    if emb_spaces is None:
        emb_spaces = list(emma.emb.keys())

    emma._check_column_is_categorical(feature)
    for emb_space in emb_spaces:
        emma._check_for_emb_space(emb_space)

    # Ensure ranks are precomputed for every requested (emb_space, metric) pair.
    # calculate_pairwise_distances is idempotent — it skips if already cached.
    for emb_space in emb_spaces:
        for metric in metrics:
            if metric not in emma.emb[emb_space].get("ranks", {}):
                print(
                    f"Computing pairwise distances for '{emb_space}' / '{metric}' "
                    f"(will be cached for future calls)..."
                )
                emma.calculate_pairwise_distances(emb_space=emb_space, metric=metric)

    class_labels = emma.metadata[feature].values
    n_total = len(class_labels)
    unique_classes = np.unique(class_labels)
    class_indices = {cls: np.where(class_labels == cls)[0] for cls in unique_classes}
    class_sizes = {cls: len(idx) for cls, idx in class_indices.items()}
    min_class_size = min(class_sizes.values())
    max_class_size = max(class_sizes.values())
    max_k = max(k_values)
    uniform_baseline = 1.0 / len(unique_classes)
    probs = emma.metadata[feature].value_counts(normalize=True)
    distribution_baseline = float((probs ** 2).sum())

    # n_cap steps: small (balanced) → large (original)
    n_caps = np.unique(
        np.linspace(min_class_size, max_class_size, n_balance_steps).astype(int)
    )

    rng = np.random.default_rng(seed)
    rows = []

    for metric in metrics:
        for n_cap in n_caps:
            # Build subset indices: cap each class at n_cap
            subset_idx = np.concatenate([
                rng.choice(class_indices[cls], size=min(n_cap, class_sizes[cls]), replace=False)
                for cls in unique_classes
            ])
            subset_labels = class_labels[subset_idx]
            n_subset = len(subset_idx)
            actual_k = min(max_k, n_subset - 1)

            # Boolean mask over the full dataset: True = in current subset
            in_subset = np.zeros(n_total, dtype=bool)
            in_subset[subset_idx] = True

            for emb_space in emb_spaces:
                # ranks[i, 0] = i (self); neighbors start at col 1.
                # For n > 5000 the Emma object stores only the top-500 neighbors
                # (argpartition), so ranks.shape[1] may be < n_total.
                ranks = emma.emb[emb_space]["ranks"][metric]
                max_rank_col = ranks.shape[1] - 1   # last valid column index

                # Buffer: enough to hold actual_k valid in-subset neighbors after
                # filtering.  Expected valid ≈ buffer × n_subset/n_total, so
                # buffer = ceil(n_total/n_subset × actual_k × 5) is safe — capped
                # at the number of stored neighbors.
                k_buffer = min(
                    int(np.ceil(n_total / n_subset * actual_k * 5)) + actual_k,
                    max_rank_col,
                )
                # Slice: (n_subset, k_buffer) of global neighbor indices
                rank_slice = ranks[
                    subset_idx[:, None],
                    np.arange(1, k_buffer + 1)[None, :],
                ]
                valid_mask = in_subset[rank_slice]            # True = in subset

                # Stable argsort moves valid (0 after ~) before invalid (1),
                # preserving distance order within each group
                sorted_pos = np.argsort(~valid_mask, axis=1, kind="stable")

                # How many valid neighbors are guaranteed across all rows?
                # Use this to cap effective_k so we never pick invalid entries.
                n_valid_min = int(valid_mask.sum(axis=1).min())
                effective_k = min(actual_k, n_valid_min)

                knn_global = rank_slice[
                    np.arange(n_subset)[:, None],
                    sorted_pos[:, :effective_k],
                ]                                             # (n_subset, effective_k)

                for k in k_values:
                    if k > effective_k:
                        continue
                    neighbor_labels = class_labels[knn_global[:, :k]]
                    mean_score = float(
                        (neighbor_labels == subset_labels[:, None]).mean()
                    )
                    rows.append({
                        "Max samples per class": int(n_cap),
                        "Total samples": n_subset,
                        "Metric": _METRIC_DISPLAY.get(metric, metric),
                        "Embedding": emb_space,
                        "k": k,
                        "Mean alignment score": mean_score,
                    })

    df = pd.DataFrame(rows)

    metric_labels_ordered = [_METRIC_DISPLAY.get(m, m) for m in
                             sorted(metrics, key=lambda m: _METRIC_ORDER.index(m)
                             if m in _METRIC_ORDER else 99)]
    k_order = sorted(k_values)

    fig = px.line(
        df,
        x="Max samples per class",
        y="Mean alignment score",
        color="Embedding",
        facet_row="k",
        facet_col="Metric",
        markers=True,
        hover_data={"Total samples": True},
        title=f"k-NN alignment vs class balance — {feature}",
        labels={"Mean alignment score": "Mean KNN feature<br>alignment score"},
        color_discrete_map=color_discrete_map,
        color_discrete_sequence=None if color_discrete_map else px.colors.qualitative.Set2,
        category_orders={"Metric": metric_labels_ordered, "k": k_order},
        template="plotly_white",
    )

    # Fix facet annotations: "Metric=Cosine distance" → "Cosine distance"
    #                         "k=5" → "k = 5"
    fig.for_each_annotation(lambda a: a.update(
        text=a.text.split("=")[-1] if a.text.startswith("Metric=")
        else a.text.replace("k=", "k = ")
    ))

    if show_random_baselines:
        fig.add_hline(
            y=uniform_baseline, line_dash="dot", line_color="grey",
            annotation_text=f"uniform random ({uniform_baseline:.2f})",
            annotation_position="bottom right",
        )
        if abs(distribution_baseline - uniform_baseline) > 0.005:
            fig.add_hline(
                y=distribution_baseline, line_dash="dash", line_color="lightgrey",
                annotation_text=f"distribution random ({distribution_baseline:.2f})",
                annotation_position="top right",
            )

    fig = update_fig_layout(fig)
    fig.update_layout(height=max(300, 220 * len(k_values) + 80))
    return (fig, df) if return_data else fig


def plot_knn_alignment_vs_feature_noise(
    emma: Emma,
    feature: str,
    emb_spaces: list = None,
    k_values: list = None,
    metrics: list = None,
    n_noise_steps: int = 10,
    n_repeats: int = 5,
    seed: int = 42,
    color_discrete_map: dict = None,
    show_random_baselines: bool = True,
    return_data: bool = False,
) -> go.Figure:
    """Plot mean k-NN feature alignment score as a function of feature noise.

    At each noise level a fraction of the feature values are randomly permuted
    among the affected samples (i.e. features are swapped, not redrawn, so the
    overall class distribution is preserved).  The x-axis runs from 0 (original
    features) to 1 (all features permuted).  Multiple random permutations are run
    at each noise level and the mean ± std is shown as a shaded band.

    Because only features change — not the embedding geometry — distances never
    need to be recomputed.  The function reads the precomputed rank matrix once
    per (embedding space, metric) and slices it for each k, making the entire
    sweep very fast.  Ranks are auto-computed via
    ``emma.calculate_pairwise_distances`` if not already cached.

    Args:
        emma (Emma): Emma object.
        feature (str): Categorical metadata column to use as the class feature.
        emb_spaces (list, optional): Embedding spaces to compare.
            Defaults to all spaces in the Emma object.
        k_values (list of int, optional): k values to show as facet rows.
            Defaults to [5, 10, 20].
        metrics (list of str, optional): Distance metrics to show as facet
            columns.  Defaults to ["euclidean"].
        n_noise_steps (int, optional): Number of noise levels between 0 and 1
            inclusive.  Defaults to 10.
        n_repeats (int, optional): Number of independent random permutations
            per noise level.  Mean and std are shown.  Defaults to 5.
        seed (int, optional): Base random seed.  Defaults to 42.
        color_discrete_map (dict, optional): Mapping from embedding-space name
            to color hex.  Defaults to Plotly Set2 palette.
        show_random_baselines (bool, optional): If True (default), draw the
            uniform and distribution-aware random baseline lines.

    Returns:
        go.Figure: Line plot with shaded std band, faceted by k (rows) and
            metric (columns), with one line per embedding space.
    """
    if k_values is None:
        k_values = [5, 10, 20]
    if metrics is None:
        metrics = ["euclidean"]
    if emb_spaces is None:
        emb_spaces = list(emma.emb.keys())

    emma._check_column_is_categorical(feature)
    for emb_space in emb_spaces:
        emma._check_for_emb_space(emb_space)

    # Ensure ranks are cached for every (emb_space, metric) pair
    for emb_space in emb_spaces:
        for metric in metrics:
            if metric not in emma.emb[emb_space].get("ranks", {}):
                print(
                    f"Computing pairwise distances for '{emb_space}' / '{metric}' "
                    f"(will be cached for future calls)..."
                )
                emma.calculate_pairwise_distances(emb_space=emb_space, metric=metric)

    class_features = emma.metadata[feature].values
    n_total = len(class_features)
    n_classes = len(np.unique(class_features))
    uniform_baseline = 1.0 / n_classes
    probs = emma.metadata[feature].value_counts(normalize=True)
    distribution_baseline = float((probs ** 2).sum())
    max_k = max(k_values)

    noise_fracs = np.linspace(0, 1, n_noise_steps)
    rows = []

    for emb_space in emb_spaces:
        for metric in metrics:
            ranks = emma.emb[emb_space]["ranks"][metric]
            # ranks[:, 0] = self; cap k at the number of stored neighbors
            k_cap = ranks.shape[1] - 1
            # Pre-slice rank matrix for max usable k (reused across noise levels)
            k_use = min(max_k, k_cap)
            knn_full = ranks[:, 1 : k_use + 1]   # (n, k_use) — never changes

            for noise_frac in noise_fracs:
                n_permute = round(n_total * noise_frac)

                # noise_frac == 0: deterministic, no need for multiple repeats
                repeats = 1 if n_permute == 0 else n_repeats

                scores_per_k = {k: [] for k in k_values}

                for rep in range(repeats):
                    rng = np.random.default_rng(seed + rep)
                    noisy_features = class_features.copy()
                    if n_permute > 0:
                        perm_idx = rng.choice(n_total, size=n_permute, replace=False)
                        noisy_features[perm_idx] = rng.permutation(
                            class_features[perm_idx]
                        )

                    for k in k_values:
                        if k > k_use:
                            continue
                        neighbor_features = noisy_features[knn_full[:, :k]]  # (n, k)
                        score = float(
                            (neighbor_features == noisy_features[:, None]).mean()
                        )
                        scores_per_k[k].append(score)

                for k in k_values:
                    if not scores_per_k[k]:
                        continue
                    vals = scores_per_k[k]
                    rows.append({
                        "Noise fraction": float(noise_frac),
                        "Metric": _METRIC_DISPLAY.get(metric, metric),
                        "Embedding": emb_space,
                        "k": k,
                        "Mean alignment score": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                    })

    df = pd.DataFrame(rows)

    metric_labels_ordered = [_METRIC_DISPLAY.get(m, m) for m in
                             sorted(metrics, key=lambda m: _METRIC_ORDER.index(m)
                             if m in _METRIC_ORDER else 99)]
    k_order = sorted(k_values)

    fig = px.line(
        df,
        x="Noise fraction",
        y="Mean alignment score",
        error_y="std",
        color="Embedding",
        facet_row="k",
        facet_col="Metric",
        markers=True,
        title=f"k-NN alignment vs label noise — {feature}",
        labels={
            "Noise fraction": "Fraction of labels permuted",
            "Mean alignment score": "Mean KNN feature<br>alignment score",
        },
        color_discrete_map=color_discrete_map,
        color_discrete_sequence=None if color_discrete_map else px.colors.qualitative.Set2,
        category_orders={"Metric": metric_labels_ordered, "k": k_order},
        template="plotly_white",
    )

    fig.for_each_annotation(lambda a: a.update(
        text=a.text.split("=")[-1] if a.text.startswith("Metric=")
        else a.text.replace("k=", "k = ")
    ))

    if show_random_baselines:
        fig.add_hline(
            y=uniform_baseline, line_dash="dot", line_color="grey",
            annotation_text=f"uniform random ({uniform_baseline:.2f})",
            annotation_position="bottom right",
        )
        if abs(distribution_baseline - uniform_baseline) > 0.005:
            fig.add_hline(
                y=distribution_baseline, line_dash="dash", line_color="lightgrey",
                annotation_text=f"distribution random ({distribution_baseline:.2f})",
                annotation_position="top right",
            )

    fig = update_fig_layout(fig)
    fig.update_layout(height=max(300, 220 * len(k_values) + 80))
    fig.for_each_xaxis(lambda ax: ax.update(tickformat=".0%"))
    return (fig, df) if return_data else fig


def plot_within_between_distributions(emma: Emma, emb_space: str, metric: str,
                                      feature: str, feature_class: str = None
                                      ) -> go.Figure:
    """
    Plot distributions of within-class and between-class distances
    for a given feature category, optionally for a specific feature class.
    
    This function internally uses the compute_within_between_distances method to compute the distances.
    
    Args:
        emma (Emma): An instance of the Emma class.
        emb_space (str): Name of the embedding space to use.
        metric (str): The distance metric to use (e.g., "euclidean", "cosine").
        feature (str): The feature category (e.g., "age", "disease_status") for classification.
        feature_class (str, optional): Specific feature class to visualize. If None, all classes are included.
        
    Returns:
        go.Figure: A Plotly figure object containing the histogram of distances.
    """
    
    # Compute the within and between class distances using the compute_within_between_distances method
    distances = emma.compute_within_between_distances(
        emb_space=emb_space,
        metric=metric,
        feature=feature,
    )
    feature_classes = []
    types = []
    distances_flat = []

    for cls, dists in distances.items():
        for dist_type in ("within", "between"):
            dist_values = dists.get(dist_type, [])
            feature_classes.extend([cls] * len(dist_values))
            types.extend([dist_type] * len(dist_values))
            distances_flat.extend(dist_values)

    distances_df = pd.DataFrame({
        "feature_class": feature_classes,
        "type": types,
        "distance": distances_flat,
    })

    if feature_class is not None:
        if feature_class not in distances_df["feature_class"].unique():
            raise ValueError(f"Feature class '{feature_class}' not found.")
        distances_df = distances_df[distances_df["feature_class"] == feature_class]

    fig = px.histogram(
        distances_df,
        x="distance",
        color="type",
        facet_col="feature_class" if feature_class is None else None,
        marginal="box",
        nbins=50,
        title=f"Within vs. Between Class Distances for {feature}" + 
            (f" (Class: {feature_class})" if feature_class else ""),
        labels={"distance": "Distance", "type": "Type"},
        barmode="overlay",
    )
    
    fig.update_layout(
        bargap=0.1,
        template="simple_white",
        legend_title_text="Distance Type",
    )
    fig.update_traces(hoverinfo="skip", selector=dict(type="histogram"))
    fig.update_layout(dragmode=False)
    return fig
