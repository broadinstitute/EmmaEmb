import plotly.express as px
import plotly.graph_objects as go

from typing import List

import numpy as np
import pandas as pd

from scipy import stats

from emmaemb.core import Emma
from emmaemb.functions import *


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
    normalise: bool = True,
    color_by: str = None,
    logarithmic_colors: bool = False,
    verbose_tooltips: bool = False,
    random_state: int = 42,
    perplexity: int = 30,
    shuffle_umap: bool = True,
) -> go.Figure:
    """Function to plot the embeddings of a given embedding space in 2D. \
    Dimensionality reduction is performed using PCA, TSNE, or UMAP.\
    The dots are coloured by a column in the metadata.

    Args:
        emma (Emma): An instance of the Emma class.
        emb_space (str): Name of an embedding space in the Emma instance.
        method (str, optional): Method for dimensionality reduction. \
            Either "PCA", "TSNE", or "UMAP". Defaults to "PCA".
        normalise (bool, optional): Whether to perform z-score normalisation \
            prior to dimensionality reduction. Defaults to True.
        color_by (str, optional): A column name from the metadata stored in \
            the Emma object, by which the dots are coloured. Defaults to None.
        verbose_tooltips (bool, optional): Show all metadata on hover tooltips \
            rather than only the sample ID. Defaults to False.
        logarithmic_colors (bool, optional): Use a logarithmic scale to color by \
            a numerical column. Defaults to False.
        random_state (int, optional): Random state for UMAP or TSNE. Defaults \
            to 42.
        perplexity (int, optional): Perplexity, only applied to UMAP.\
            Defaults to 30.
        shuffle_umap (bool, optional): Shuffle order of embeddings before \
            running UMAP. Defaults to True

    Returns:
        go.Figure: A scatter plot of the embeddings in 2D.
    """

    embeddings_2d = emma.get_2d(
            emb_space=emb_space,
            method=method,
            normalise=normalise,
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
        color_scale (str, optional): Colour scale for the heatmap. \
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
        color (str, optional): Colour of the plot elements. Defaults to "blue".
        group_by (str, optional): Metadata column name to group and colour \
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
        k (int, optional): Number of nearest neighbours to consider. \
            Defaults to 10.
        metric (str, optional): Distance metric to use. \
            Defaults to "euclidean".
        emb_space_order (list, optional): Order in which to display the \
            embedding spaces. Defaults to None.
        color (str, optional): Colour of the plot elements. \
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
        k (int, optional): Number of nearest neighbours to consider. \
            Defaults to 10.
        metric (str, optional): Distance metric to use. Defaults to "euclidean".
        emb_space_order (list, optional): Order in which to display the \
            embedding spaces. Defaults to None.
        color (str, optional): Colour of the plot elements. \
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

def plot_knn_alignment_scores_across_k_and_distance_metrics(
    emma: Emma,
    feature: str,
    k_values: List[int] = [5, 10],
    metrics: List[str] = ["euclidean"],
    use_annoy: bool = False,
    annoy_metric: str = None,
    n_trees: int = None,
    adjust_for_imbalance: bool = False,
) -> go.Figure:

    knn_alignment_scores_df = pd.DataFrame(columns=["Sample", "Class", "Fraction", "Embedding", "k", "distance_metric"])
    sample_ids = emma.sample_names
    
    for distance_metric in metrics:
        for k_value in k_values:
            print(f"Calculating KNN alignment scores for {distance_metric} with k={k_value}...")
            knn_alignment_scores = get_knn_alignment_scores(
                emma,
                feature=feature,
                k=k_value,
                metric=distance_metric,
                use_annoy=use_annoy,
                annoy_metric=annoy_metric,
                n_trees=n_trees,
                adjust_for_imbalance=adjust_for_imbalance,
            )
            knn_alignment_scores["k"] = k_value
            knn_alignment_scores["distance_metric"] = distance_metric
            knn_alignment_scores["Sample"] = np.tile(sample_ids, len(emma.emb.keys()))
            # merge with the existing dataframe
            knn_alignment_scores_df = pd.concat([knn_alignment_scores_df, knn_alignment_scores], ignore_index=True)        
    
    df_grouped = (
        knn_alignment_scores_df
        .groupby(["k", "Embedding", "distance_metric"], as_index=False)
        .agg({"Fraction": "mean"})
    )
    df_grouped = df_grouped.sort_values(by=["Embedding", "k"])
    
    fig = px.line(
        df_grouped,
        x="k",
        y="Fraction",
        color="Embedding",
        facet_col="distance_metric",
        markers=True,
    )
    
    fig.update_layout(
        yaxis_title="Mean KNN feature<br>alignment score",
        template="plotly_white",
        font={"family": "Arial", "color": "black", "size": 12},
        legend_title_text="Embedding models",
        width=1200,
        height=500,
    )
    
    fig.update_traces(marker=dict(size=10), line=dict(width=4))
    
    for axis in fig.layout:
        if axis.startswith('xaxis'):
            fig.layout[axis].update(
                        showgrid=False,
                        linecolor='black',
                        linewidth=3,
                        ticks='outside',
                        tickwidth=2,
                        tickcolor='black',
                        ticklen=6,
                        tickformat=".0f"
                    )
        if axis.startswith('yaxis'):
            fig.layout[axis].update(
                showgrid=False,
                linecolor='black',
                linewidth=3,
                ticks='outside',
                tickwidth=2,
                tickcolor='black',
                ticklen=6,
                tickformat=".2f",
                dtick=0.05
            )    

    for annotation in fig.layout.annotations:
        if '=' in annotation.text:
            annotation.text = annotation.text.split('=')[1]
                
    fig.update_layout(
        title=dict(
            x=0,
            xanchor='left'
        ),
        legend=dict(
            orientation='h',
            x=0.48,
            y=1.05,
            xanchor='center',
            yanchor='bottom'
        ),
        margin=dict(t=120)
    )
    
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
        k (int, optional): Number of nearest neighbours to consider. \
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

    similarities = get_neighbourhood_similarity(
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

def plot_within_between_distributions(emma: Emma, emb_space: str, metric: str, 
                                      feature_category: str, feature_class: str = None
                                      ) -> go.Figure:
    """
    Plot distributions of within-class and between-class distances
    for a given feature category, optionally for a specific feature class.
    
    This function internally uses the compute_within_between_distances method to compute the distances.
    
    Args:
        emma (Emma): An instance of the Emma class.
        emb_space (str): Name of the embedding space to use.
        metric (str): The distance metric to use (e.g., "euclidean", "cosine").
        feature_category (str): The feature category (e.g., "age", "disease_status") for classification.
        feature_class (str, optional): Specific feature class to visualize. If None, all classes are included.
        
    Returns:
        go.Figure: A Plotly figure object containing the histogram of distances.
    """
    
    # Compute the within and between class distances using the compute_within_between_distances method
    distances = emma.compute_within_between_distances(
        emb_space=emb_space,
        metric=metric,
        feature_category=feature_category,
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
        title=f"Within vs. Between Class Distances for {feature_category}" + 
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
