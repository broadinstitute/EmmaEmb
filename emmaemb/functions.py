import pandas as pd
import numpy as np

from collections import Counter
from sklearn.preprocessing import LabelEncoder

from emmaemb.core import Emma


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
    # check if metric is already calculated
    if not use_annoy:
        for emb_space in embedding_spaces:
            if metric not in emma.emb[emb_space]["ranks"]:
                raise ValueError(
                    f"Metric {metric} not calculated for embedding {emb_space}"
            )

    all_results = []
    feature_classes = emma.metadata[feature]
    feature_classes_array = feature_classes.values.to_numpy()
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

    rank_matrix = emma.emb[emb_space]["ranks"].get(metric)
    
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


def get_neighbourhood_similarity(
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
