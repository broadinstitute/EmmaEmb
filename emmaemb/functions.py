import warnings

import pandas as pd
import numpy as np

import textwrap
from collections import Counter
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder

from emmaemb.core import Emma


@dataclass
class DiagnosticResult:
    """Container for embedding diagnostic outputs.

    Attributes:
        name (str): Name of the diagnostic.
        data (object): Raw data — a DataFrame or dict of DataFrames.
        summary (dict): Per-embedding-space summary strings.
        context (str): Background text contextualising the metrics.
    """

    name: str
    data: object
    summary: dict
    context: str

    def __repr__(self):
        wrapped_context = textwrap.fill(self.context, width=80)
        lines = [f"=== {self.name} ===", "", wrapped_context, "", "Summary:"]
        for k, v in self.summary.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# get knn alignment scores
def get_knn_alignment_scores(
    emma: Emma,
    feature: str,
    k: int = 10,
    metric: str = "euclidean",
    use_annoy: bool = False, 
    annoy_metric: str = None, 
    n_trees: int = None,
    adjust_for_imbalance: bool = False,
) -> pd.DataFrame:
    """Function to calculate the alignment scores of k-nearest neighbors \
        across different embedding spaces.

    Args:
        emma (Emma): Emma object
        feature (str): Column name in the metadata DataFrame of \
            the Emma object.
        k (int, optional): Number of nearest neighbors to consider. \
            Defaults to 10.
        metric (str, optional): Distance metric to use. \
            Defaults to "euclidean".
        use_annoy (bool): Whether to use Annoy index. Default False.
        annoy_metric (str): Annoy distance metric to use. \
            Required if use_annoy is True.
        n_trees (int): Number of trees used to build the Annoy index. \
            Required if use_annoy is True.

    Returns:
        pd.DataFrame: DataFrame containing the alignment scores of \
            k-nearest neighbors across different embedding spaces.\
            Columns: Sample, Class (feature class name), \
                Fraction (KNN feature alignment score), \
                    Embedding (embedding space name)
    """

    # validate input
    embedding_spaces = emma.emb.keys()
    if embedding_spaces is None:
        raise ValueError("No embeddings found in Emma object")
    emma._check_column_is_categorical(feature)

    class_sizes = emma.metadata[feature].value_counts()
    n_min_class = int(class_sizes.min())
    if k >= n_min_class:
        warnings.warn(
            f"k={k} >= smallest class size (n_min={n_min_class}, "
            f"class='{class_sizes.idxmin()}'). KNN alignment scores will be "
            f"inflated for samples in that class. Use k < {n_min_class}.",
            UserWarning,
            stacklevel=2,
        )

    # check if metric is already calculated
    if not use_annoy:
        for emb_space in embedding_spaces:
            if metric not in emma.emb[emb_space]["ranks"]:
                raise ValueError(
                    f"Metric {metric} not calculated for embedding {emb_space}"
            )

    all_results = []
    feature_classes = emma.metadata[feature]
    feature_classes_array = feature_classes.values
    class_distribution = feature_classes.value_counts(normalize=True).to_dict()
    
    for emb_space in embedding_spaces:
        nearest_neighbors = emma.get_knn(
            emb_space=emb_space,
            k=k,
            metric=metric,
            use_annoy=use_annoy,
            annoy_metric=annoy_metric,
            n_trees=n_trees,
        )

        neighbor_classes = feature_classes_array[nearest_neighbors]  # shape: (n_samples, k)

        # Compare each neighbor class to the sample's class
        same_class_mask = neighbor_classes == feature_classes_array[:, None]  # shape: (n_samples, k)
        fractions = np.sum(same_class_mask, axis=1) / k
        
        if adjust_for_imbalance:
            class_probs = np.vectorize(class_distribution.get)(feature_classes_array)
            fractions = fractions / class_probs
        # fractions = []
        # for i in range(len(nearest_neighbors)):
        #     # Get the indices of the k-nearest neighbors (ranked by distance)
        #     neighbor_indices = nearest_neighbors[i]
            
        #     # Count how many of the k-nearest neighbors belong to
        #     # the same class
        #     same_class_count = np.sum(
        #         feature_classes.iloc[neighbor_indices].values
        #         == feature_classes.iloc[i]
        #     )
        #     fraction = same_class_count / k
        #     fractions.append(fraction)

        # Prepare results in a DataFrame for the current embedding space
        df = pd.DataFrame(
            {
                # "Sample": emma.sample_names,
                "Class": feature_classes_array,
                "Fraction": fractions,
                "Embedding": emb_space,
            }
        )
        all_results.append(df)

    return pd.concat(all_results, ignore_index=True)


def get_class_mixing_in_neighborhood(
    emma: Emma,
    emb_space: str,
    feature: str,
    k: int = 10,
    metric: str = "euclidean",
    use_annoy: bool = False, 
    annoy_metric: str = None, 
    n_trees: int = None,
):
    # validate input
    emma._check_for_emb_space(emb_space)
    emma._check_column_is_categorical(feature)
    # check if metric is already calculated
    if not use_annoy:
        if metric not in emma.emb[emb_space]["ranks"]:
            raise ValueError(
                f"Metric {metric} not calculated for embedding {emb_space}"
            )

    le = LabelEncoder()
    encoded_classes = le.fit_transform(emma.metadata[feature])
    unique_classes = le.classes_
    num_classes = len(unique_classes)

    neighbor_class_counts = np.zeros((num_classes, num_classes), dtype=int)

    neighboring_indices = emma.get_knn(
            emb_space=emb_space,
            k=k,
            metric=metric,
            use_annoy=use_annoy,
            annoy_metric=annoy_metric,
            n_trees=n_trees,
        )
    # neighboring_indices = rank_matrix[:, 1 : k + 1]

    for i, neighbors in enumerate(neighboring_indices):
        sample_class_idx = encoded_classes[i]
        neighbor_class_indices = encoded_classes[neighbors]

        class_counts = Counter(neighbor_class_indices)

        for neighbor_class_idx, count in class_counts.items():
            neighbor_class_counts[
                neighbor_class_idx, sample_class_idx
            ] += count

    return neighbor_class_counts, unique_classes


def get_neighborhood_similarity(
    emma: Emma,
    emb_space_1: str,
    emb_space_2: str,
    k: int = 10,
    metric: str = "euclidean",
    use_annoy: bool = False,
    annoy_metric: str = None,
    n_trees: int = None,
):
    for emb_space in [emb_space_1, emb_space_2]:
        emma._check_for_emb_space(emb_space)
        if not use_annoy:
            if metric not in emma.emb[emb_space]["ranks"]:
                raise ValueError(
                    f"Metric {metric} not calculated for embedding {emb_space_1}"
                )

    # Get the k-nearest neighbors for both embedding spaces
    knn_1 = emma.get_knn(
        emb_space=emb_space_1,
        k=k,
        metric=metric,
        use_annoy=use_annoy,
        annoy_metric=annoy_metric,
        n_trees=n_trees,
    )
    knn_2 = emma.get_knn(
        emb_space=emb_space_2,
        k=k,
        metric=metric,
        use_annoy=use_annoy,
        annoy_metric=annoy_metric,
        n_trees=n_trees,
    )
    
    # knn_1 = emma.emb[emb_space_1]["ranks"].get(metric)[:, 1 : k + 1]
    # knn_2 = emma.emb[emb_space_2]["ranks"].get(metric)[:, 1 : k + 1]

    similarity = np.zeros(len(knn_1))

    for i, (neighbors_1, neighbors_2) in enumerate(zip(knn_1, knn_2)):
        similarity[i] = len(set(neighbors_1).intersection(neighbors_2)) / k

    return similarity


def _avg_cosine_sim(X: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray) -> float:
    """Return average pairwise cosine similarity for the given index pairs."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / np.where(norms == 0, 1, norms)
    return float(np.sum(X_norm[idx_a] * X_norm[idx_b], axis=1).mean())


def get_anisotropy_diagnostics(
    emma: Emma,
    n_pairs: int = 10000,
    seed: int = 42,
    check_mean_centering: bool = True,
    max_samples: int = None,
) -> DiagnosticResult:
    """Compute anisotropy diagnostics for all embedding spaces.

    Args:
        emma (Emma): Emma object.
        n_pairs (int): Number of random pairs to sample for cosine similarity.
            Defaults to 10000.
        seed (int): Random seed for reproducibility. Defaults to 42.
        check_mean_centering (bool): If True, recompute AnisotropyScore and
            AvgPairwiseCosineSim on the mean-centred embeddings for any space
            where AnisotropyScore > 1. Columns *_MC are added to the summary
            DataFrame (NaN for spaces that are already isotropic).
            Defaults to True.
        max_samples (int, optional): If set, randomly subsample this many rows
            from the embedding matrix before computing diagnostics. Useful for
            very large datasets where loading the full matrix is feasible but
            computing statistics over all samples would be slow. Defaults to
            None (use all samples).

    Returns:
        DiagnosticResult: .data is a dict with keys:
            "summary"  — DataFrame with one row per embedding space;
            "variance_per_dim" — dict mapping embedding name → 1-D array of
                per-dimension variances (length = embedding dimensionality).
    """
    rng = np.random.default_rng(seed)
    rows = []

    for emb_space in emma.emb.keys():
        X = emma.emb[emb_space]["emb"].astype(np.float64)
        n = len(X)
        if max_samples is not None and n > max_samples:
            idx = rng.choice(n, size=max_samples, replace=False)
            X = X[idx]
            n = max_samples
        d = X.shape[1]
        expected_std = 1.0 / np.sqrt(d)

        idx_a = rng.integers(0, n, size=min(n_pairs, n * (n - 1) // 2))
        idx_b = rng.integers(0, n, size=min(n_pairs, n * (n - 1) // 2))

        avg_cos_sim = _avg_cosine_sim(X, idx_a, idx_b)
        var_per_dim = np.var(X, axis=0)
        anisotropy_score = avg_cos_sim / expected_std
        participation_ratio = float(var_per_dim.sum() ** 2 / (var_per_dim ** 2).sum())

        # Mean-centering check — only for anisotropic spaces
        avg_cos_sim_mc = np.nan
        anisotropy_score_mc = np.nan
        mc_helps = None
        # if check_mean_centering and anisotropy_score > 1:
        X_mc = X - X.mean(axis=0)
        avg_cos_sim_mc = _avg_cosine_sim(X_mc, idx_a, idx_b)
        anisotropy_score_mc = avg_cos_sim_mc / expected_std
        mc_helps = anisotropy_score_mc < anisotropy_score

        rows.append(
            {
                "Embedding": emb_space,
                "AvgPairwiseCosineSim": avg_cos_sim,
                "AnisotropyScore": anisotropy_score,
                "AvgPairwiseCosineSim_MC": avg_cos_sim_mc,
                "AnisotropyScore_MC": anisotropy_score_mc,
                "MeanCenteringHelps": mc_helps,
                "EmbeddingDim": d,
                "MeanDimVariance": float(var_per_dim.mean()),
                "StdDimVariance": float(var_per_dim.std()),
                "ParticipationRatio": participation_ratio,
                "_var_per_dim": var_per_dim,
            }
        )

    summary_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "_var_per_dim"} for r in rows]
    )
    var_per_dim_dict = {r["Embedding"]: r["_var_per_dim"] for r in rows}

    def _summary_line(r):
        line = f"cosine sim = {r['AvgPairwiseCosineSim']:.3f}"
        if r["MeanCenteringHelps"] is not None:
            verdict = "resolves" if r["AnisotropyScore_MC"] <= 1 else "reduces"
            line += (
                f"  |  mean-centred = {r['AvgPairwiseCosineSim_MC']:.3f}"
                f"  ({verdict} anisotropy)"
            )
        line += (
            f"  —  dim = {r['EmbeddingDim']}, "
            f"participation ratio = {r['ParticipationRatio']:.1f}, "
            f"mean var = {r['MeanDimVariance']:.4f} (±{r['StdDimVariance']:.4f})"
        )
        return line

    summary = {r["Embedding"]: _summary_line(r) for r in rows}

    context = (
        "Raw average cosine similarity between random pairs. "
        "Near 0 = isotropic. Above 0 = embeddings cluster in a narrow cone. "
        "Other metrics: AnisotropyScore (dimensionality-corrected, comparable across models), "
        "ParticipationRatio (effective number of dimensions in use), "
        "MeanDimVariance ± StdDimVariance (spread and unevenness across dimensions). "
        "Full table available in .data['summary']."
    )

    return DiagnosticResult(
        name="Anisotropy Diagnostics",
        data={"summary": summary_df, "variance_per_dim": var_per_dim_dict},
        summary=summary,
        context=context,
    )


def get_hubness_diagnostics(
    emma: Emma,
    k: int = 10,
    metric: str = "euclidean",
    use_annoy: bool = False,
    annoy_metric: str = None,
    n_trees: int = None,
) -> DiagnosticResult:
    """Compute hubness diagnostics for all embedding spaces.

    Two metrics are reported per embedding space:
    - k-occurrence distribution: how often each point appears as a k-nearest
      neighbor of any other point.
    - Robin Hood index (RHI): a scalar summary of inequality in the
      k-occurrence distribution (0 = uniform, ~1 = extreme hubness).

    Args:
        emma (Emma): Emma object.
        k (int): Number of nearest neighbors. Defaults to 10.
        metric (str): Distance metric. Defaults to "euclidean".
        use_annoy (bool): Whether to use Annoy index. Default False.
        annoy_metric (str): Annoy distance metric. Required if use_annoy True.
        n_trees (int): Number of Annoy trees. Required if use_annoy True.

    Returns:
        DiagnosticResult: .data is a dict with keys:
            "rhi"          — DataFrame with RHI per embedding space;
            "k_occurrence" — long-form DataFrame with columns KOccurrence
                and Embedding (one row per sample per embedding space).
    """
    embedding_spaces = list(emma.emb.keys())
    if not use_annoy:
        for emb_space in embedding_spaces:
            if metric not in emma.emb[emb_space].get("ranks", {}):
                raise ValueError(
                    f"Metric '{metric}' not calculated for embedding "
                    f"'{emb_space}'. Call emma.calculate_pairwise_distances() "
                    f"first, or set use_annoy=True."
                )

    rhi_rows = []
    kocc_frames = []

    for emb_space in embedding_spaces:
        knn = emma.get_knn(
            emb_space=emb_space,
            k=k,
            metric=metric,
            use_annoy=use_annoy,
            annoy_metric=annoy_metric,
            n_trees=n_trees,
        )                                           # (n, k)

        # Count how often each point index appears across all neighbor lists
        k_occ = np.bincount(knn.ravel(), minlength=len(knn))

        mean_occ = k_occ.mean()
        rhi = float(
            (k_occ[k_occ > mean_occ] - mean_occ).sum() / k_occ.sum()
        )

        rhi_rows.append({"Embedding": emb_space, "RobinHoodIndex": rhi})
        kocc_frames.append(
            pd.DataFrame({"KOccurrence": k_occ, "Embedding": emb_space})
        )

    rhi_df = pd.DataFrame(rhi_rows)
    kocc_df = pd.concat(kocc_frames, ignore_index=True)

    summary = {
        row["Embedding"]: f"Robin Hood index = {row['RobinHoodIndex']:.3f}"
        for _, row in rhi_df.iterrows()
    }

    context = (
        "Hubness is the tendency for a small number of points ('hubs') to "
        "appear disproportionately often as k-nearest neighbors of other "
        "points — a geometric phenomenon that grows with dimensionality. "
        "The k-occurrence distribution shows how many times each point was "
        "selected as a neighbor; a heavy right tail indicates hub points. "
        "The Robin Hood index (RHI) quantifies this inequality: "
        "0 = perfectly uniform occurrence, ~1 = extreme hubness."
    )

    return DiagnosticResult(
        name="Hubness Diagnostics",
        data={"rhi": rhi_df, "k_occurrence": kocc_df},
        summary=summary,
        context=context,
    )
