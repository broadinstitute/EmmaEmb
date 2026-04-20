import os
import warnings

import math
import numpy as np
import pandas as pd
import plotly.express as px

import torch

from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from annoy import AnnoyIndex
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from emmaemb.config import EMB_SPACE_COLORS, DISTANCE_METRIC_ALIASES


GPU_BATCH_SIZE = 100


def _check_memory(required_bytes: int, label: str = "") -> None:
    """Warn if estimated free RAM is insufficient for an upcoming operation.

    Requires ``psutil``; silently skips the check if it is not installed.

    Args:
        required_bytes (int): Estimated bytes the operation will allocate.
        label (str): Short description shown in the warning message.
    """
    try:
        import psutil
        available = psutil.virtual_memory().available
        if required_bytes > available * 0.9:
            warnings.warn(
                f"{label}: operation needs ~{required_bytes / 1e9:.1f} GB but "
                f"only {available / 1e9:.1f} GB of RAM is free. "
                "Consider using store_distances=False or reducing dataset size "
                "to avoid an out-of-memory crash.",
                ResourceWarning,
                stacklevel=3,
            )
    except ImportError:
        pass  # psutil not installed — skip the check silently


class Emma:
    def __init__(self, feature_data: pd.DataFrame):

        # Metadata
        self.metadata = feature_data
        self.metadata_numeric_columns = self._get_numeric_columns()
        self.metadata_categorical_columns = self._get_categorical_columns()
        self.sample_names = self.metadata.iloc[:, 0].tolist()
        self.color_map = self._get_color_map_for_features()

        # Embedding spaces
        self.emb = dict()

        print(f"{len(self.sample_names)} samples loaded.")
        print(f"Categories in meta data: {self.metadata_categorical_columns}")
        print(
            f"Numerical columns in meta data: {self.metadata_numeric_columns}"
        )

        missing = feature_data.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            print("⚠ Missing values detected in metadata:")
            for col, n in missing.items():
                print(
                    f"  '{col}': {n} missing value{'s' if n > 1 else ''} "
                    f"({n / len(feature_data):.1%} of samples)"
                )

    # Metadata

    def _get_numeric_columns(self) -> list:
        """Identify numeric columns in the metadata.

        Returns:
        list: List of column names that are numeric.
        """
        numerical_columns = (
            self.metadata.iloc[:, 1:]
            .select_dtypes(include=["int64", "float64"])
            .columns.tolist()
        )

        return numerical_columns

    def _get_categorical_columns(self) -> list:
        """Identify categorical columns in the metadata.

        Returns:
        list: List of column names that are categorical.
        """
        categorical_columns = [
            col
            for col in self.metadata.columns[1:]
            if col not in self.metadata_numeric_columns
        ]

        return categorical_columns

    def _check_column_in_metadata(self, column: str):
        """Check if a column is in the metadata.

        Args:
        column (str): Column name.
        """
        if column not in self.metadata.columns:
            raise ValueError(f"Column {column} not found in metadata.")
        else:
            return True

    def _check_column_is_categorical(self, column: str):
        """Check if a column is categorical.

        Args:
        column (str): Column name.
        """
        if column not in self.metadata_categorical_columns:
            raise ValueError(f"Column {column} is not categorical.")
        else:
            return True
        
    def _check_column_is_numeric(self, column: str):
        """Check if a column is numeric.
        Args:
        column (str): Column name.
        """
        if column not in self.metadata_numeric_columns:
            raise ValueError(f"Column {column} is not numeric.")
        else:
            return True

    def _get_color_map_for_features(self) -> dict:
        """Generate a color map for categorical features
        in the metadata. The color map is used for plotting.
        The color map is generated based on the unique values
        in the categorical columns. Only defined for columns
        with less than 50 unique values."""

        if len(self.metadata_categorical_columns) == 0:
            print("No categorical columns found in metadata.")
            return {}

        color_map = {}

        for column in self.metadata_categorical_columns:
            column_values = self.metadata[column].unique()
            if len(column_values) > 50:
                print(
                    f"Skipping {column} as it has more than \
                        50 unique values."
                )
                continue
            
            # select smallest color set from list that fits or fall back to 24 colors
            color_set = next((
                set for set in [
                    px.colors.qualitative.Set2, # 8 colors
                    px.colors.qualitative.Pastel, # 11 colors
                    px.colors.qualitative.Set3, # 12 colors
                    px.colors.qualitative.Light24, # 24 colors
                ]
                if len(set) >= len(column_values)
            ), px.colors.qualitative.Alphabet)

            # repeat colors if we don't have enough
            
            colors = (
                color_set * math.ceil(len(column_values) / len(color_set))
            )[:len(column_values)]
            color_map[column] = dict(zip(column_values, colors))

            # check for specifial values
            if "True" in column_values:
                color_map[column]["True"] = "steelblue"

            if "False" in column_values:
                color_map[column]["True"] = "darkred"

        return color_map

    # Embeddings

    def _load_embeddings_from_dir(self, dir_path: str, file_extension: str):
        """Load embeddings from individual files in a directory.

        Args:
        dir_path (str): Path to the directory containing the individual files.
        file_extension (str): Extension of the embedding files. Default 'npy'.
        """

        # Load first file to determine shape and dtype, then preallocate.
        # This avoids the 2× peak memory of building a list then calling np.stack.
        first_file = os.path.join(dir_path, f"{self.sample_names[0]}.{file_extension}")
        if not os.path.isfile(first_file):
            raise ValueError(f"Embedding file '{first_file}' not found.")
        first_emb = np.load(first_file)
        _check_memory(
            first_emb.nbytes * len(self.sample_names),
            label="add_emb_space",
        )
        embeddings = np.empty(
            (len(self.sample_names),) + first_emb.shape, dtype=first_emb.dtype
        )
        embeddings[0] = first_emb
        for i, sample in enumerate(self.sample_names[1:], start=1):
            emb_file = os.path.join(dir_path, f"{sample}.{file_extension}")
            if not os.path.isfile(emb_file):
                raise ValueError(f"Embedding file '{emb_file}' not found.")
            embeddings[i] = np.load(emb_file)
        return embeddings

    def _assign_color_to_embedding_space(self, num_emb_spaces: int) -> str:
        """Assign a color to the embedding space."""
        return EMB_SPACE_COLORS[
            (num_emb_spaces - len(EMB_SPACE_COLORS)) % len(EMB_SPACE_COLORS)
        ]

    def add_emb_space(
        self,
        emb_space_name: str,
        embeddings_source: str,
        file_extension: str = "npy",
    ):
        """Add an embedding space to the Emma object.
        
        Args:
        embeddings_source (str): Path to either a .npy file or a \
            directory containing .npy files for each embedding.
        emb_space_name (str): Name of the embedding space. Must be unique.
        ext (str): Extension of the embedding files (default 'npy').
        
        If embeddings_source is a .npy file, it is loaded directly assuming \
            it contains all embeddings for the provided meta data in \
                respective order.
        If embedding_source is a directory, embeddings are loaded from files \
            in the directory corresponding to self.sample_names.
        """

        # Validate the embedding space name
        if not emb_space_name:
            raise ValueError("Embedding space name must be provided.")
        if emb_space_name in self.emb:
            raise ValueError(
                f"Embedding space '{emb_space_name}' already \
                exists."
            )

        # Load embeddings
        embeddings = None
        if embeddings_source.endswith(f".{file_extension}"):
            # Single .npy file
            if not os.path.isfile(embeddings_source):
                raise ValueError(
                    f"Embedding file '{embeddings_source}' not found."
                )
            embeddings = np.load(embeddings_source)
        elif os.path.isdir(embeddings_source):
            # Directory with .npy files
            embeddings = self._load_embeddings_from_dir(
                embeddings_source, file_extension
            )
        else:
            raise ValueError(
                (
                    "'embeddings_source' must be a .npy file or \
                        a directory path."
                )
            )

        # Validate the number of embeddings
        if embeddings.shape[0] != len(self.sample_names):
            raise ValueError(
                (
                    "Number of embeddings does not match the number \
                        of samples in the metadata."
                )
            )

        # Cast to float32 to avoid overflow in downstream numerical ops
        # (e.g. pairwise distances, variance) when embeddings are stored as float16
        if embeddings.dtype == np.float16:
            embeddings = embeddings.astype(np.float32)

        # Add the embedding space
        self.emb[emb_space_name] = {
            "emb": embeddings,
            "color": self._assign_color_to_embedding_space(len(self.emb)),
        }

        print(f"Embedding space '{emb_space_name}' added successfully.")
        print(f"Embeddings have {embeddings.shape[1]} features each.")

    def _check_for_emb_space(self, emb_space_name: str):
        """Check if an embedding space is available.

        Args:
        emb_space_name (str): Name of the embedding space.
        """
        if emb_space_name not in self.emb:
            raise ValueError(f"Embedding space {emb_space_name} not found.")

    def remove_emb_space(self, emb_space_name: str):
        """Remove an embedding space from the Emma object.

        Args:
        emb_space_name (str): Name of the embedding space.
        """
        self._check_for_emb_space(emb_space_name)
        del self.emb[emb_space_name]
        print(f"Embedding space '{emb_space_name}' removed.")

    # Pairwise distances
    def __compute_pairwise_distances(
        self, emb_space: str, metric: str, embeddings: np.ndarray
    ):
        """Calculate pairwise distances between samples in an embedding space.

        Args:
        emb_space (str): Name of the embedding space.
        metric (str): Distance metric to use.
        """
        if metric not in DISTANCE_METRIC_ALIASES:
            raise ValueError(f"Distance metric {metric} not supported.")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        emb = self.emb[emb_space]["emb"]
        
        if device == "cuda":
            print("Using GPU for distance calculation.")
            
            emb = torch.tensor(emb, device=device, dtype=torch.float32)
            
            if metric == "euclidean":
                batch_size = GPU_BATCH_SIZE
                n_samples = emb.size(0)
                results = []

                for start in tqdm(range(0, n_samples, batch_size), desc="Computing pairwise distances (euclidean)"):
                    end = min(start + batch_size, n_samples)
                    batch = emb[start:end]

                    # Compute pairwise distances between this batch and all samples
                    dists = torch.cdist(batch, emb, p=2)  # batch_size x n_samples
                    results.append(dists.cpu())  # move to CPU immediately

                emb_pwd = torch.cat(results, dim=0)
            
            elif metric == "cityblock":
                batch_size = GPU_BATCH_SIZE
                n_samples = emb.size(0)
                results = []
                
                for start in tqdm(range(0, n_samples, batch_size), desc="Computing pairwise distances (cityblock)"):
                    end = min(start + batch_size, n_samples)
                    batch = emb[start:end]

                    # Compute pairwise distances between this batch and all samples
                    dists = torch.cdist(batch, emb, p=1)
                    results.append(dists.cpu())  # move to CPU immediately
                emb_pwd = torch.cat(results, dim=0)
            
            elif metric == "cosine":
                emb_norm = torch.nn.functional.normalize(emb, p=2, dim=1)
                batch_size = GPU_BATCH_SIZE 
                n_samples = emb_norm.size(0)
                
                # Preallocate a CPU tensor to store the full similarity matrix
                cosine_sim = torch.empty((n_samples, n_samples), dtype=torch.float32, device='cpu')

                for start in tqdm(range(0, n_samples, batch_size), desc="Computing cosine similarities"):
                    end = min(start + batch_size, n_samples)
                    part = emb_norm[start:end] @ emb_norm.T  # (batch_size, n_samples)

                    # Copy the result directly into the correct slice of the preallocated matrix
                    cosine_sim[start:end] = part.cpu()

                emb_pwd = 1 - cosine_sim
            
            return emb_pwd.cpu().numpy()
            
        else:

            if metric == "sqeuclidean_normalized":
                # divide each row by its norm
                emb_norm = np.linalg.norm(emb, axis=1)
                emb = emb / emb_norm[:, None]  # divide each row by its norm
                emb_pwd = squareform(pdist(emb, metric="sqeuclidean"))
                return emb_pwd

            elif metric == "euclidean_normalized":
                # divide each row of the emb by its norm
                emb_norm = np.linalg.norm(emb, axis=1)
                emb = emb / emb_norm[:, None]  # divide each row by its norm
                emb_pwd = squareform(pdist(emb, metric="euclidean"))
                return emb_pwd

            elif metric == "cityblock_normalized":
                emb_pwd = squareform(
                    pdist(emb, metric="cityblock")
                )
                emb_pwd = emb_pwd / len(self.emb[emb_space]["emb"][1])
                return emb_pwd

            elif metric == "adjusted_cosine":
                # substract the mean of each column from each value
                emb = emb - np.median(emb, axis=0)  # emb.median(axis=0)
                emb_pwd = squareform(pdist(emb, metric="cosine"))
                return emb_pwd

            emb_pwd = squareform(pdist(embeddings, metric=metric))
            return emb_pwd

    def __compute_knn_ranks(
        self, emb_space: str, metric: str, k: int
    ) -> np.ndarray:
        """Compute top-k nearest-neighbor ranks in batches without building
        or storing the full N×N distance matrix.

        Supports euclidean, cityblock, cosine, and the normalized/adjusted
        variants by preprocessing the embedding matrix before batching.
        Falls back to the full-matrix path for metrics that require it
        (e.g. mahalanobis, sqeuclidean).

        Args:
            emb_space (str): Name of the embedding space.
            metric (str): Distance metric.
            k (int): Number of nearest neighbors to return.

        Returns:
            np.ndarray: Array of shape (n_samples, k) with sorted neighbor indices.
        """
        emb = self.emb[emb_space]["emb"].copy()

        # Apply any per-metric preprocessing to the embedding matrix.
        scipy_metric = metric
        if metric == "sqeuclidean_normalized":
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.where(norms == 0, 1, norms)
            scipy_metric = "sqeuclidean"
        elif metric == "euclidean_normalized":
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.where(norms == 0, 1, norms)
            scipy_metric = "euclidean"
        elif metric == "cityblock_normalized":
            scipy_metric = "cityblock"
        elif metric == "adjusted_cosine":
            emb = emb - np.median(emb, axis=0)
            scipy_metric = "cosine"

        n = len(emb)
        ranked = np.empty((n, k), dtype=np.int32)
        batch_size = GPU_BATCH_SIZE
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda" and metric in ("euclidean", "cityblock", "cosine",
                                            "sqeuclidean_normalized",
                                            "euclidean_normalized",
                                            "cityblock_normalized",
                                            "adjusted_cosine"):
            print(f"Using GPU for kNN computation ({metric}).")
            emb_t = torch.tensor(emb, dtype=torch.float32, device=device)
            if scipy_metric in ("cosine", "sqeuclidean"):
                emb_t = torch.nn.functional.normalize(emb_t, p=2, dim=1)

            for start in tqdm(range(0, n, batch_size), desc=f"Computing kNN ({metric})"):
                end = min(start + batch_size, n)
                batch = emb_t[start:end]

                if scipy_metric in ("euclidean", "euclidean_normalized"):
                    dists = torch.cdist(batch, emb_t, p=2)
                elif scipy_metric == "sqeuclidean":
                    dists = torch.cdist(batch, emb_t, p=2) ** 2
                elif scipy_metric == "cityblock":
                    dists = torch.cdist(batch, emb_t, p=1)
                    if metric == "cityblock_normalized":
                        dists = dists / emb.shape[1]
                else:  # cosine (including adjusted_cosine)
                    dists = 1.0 - batch @ emb_t.T

                # Exclude self by setting the diagonal to inf
                for local_i in range(end - start):
                    dists[local_i, start + local_i] = float("inf")

                topk = torch.topk(dists, k, largest=False).indices.cpu().numpy()
                ranked[start:end] = topk

        else:
            # CPU path: batch with scipy cdist to avoid the full N×N matrix.
            from scipy.spatial.distance import cdist as scipy_cdist

            for start in tqdm(range(0, n, batch_size), desc=f"Computing kNN ({metric})"):
                end = min(start + batch_size, n)
                dists = scipy_cdist(emb[start:end], emb, metric=scipy_metric)

                if metric == "cityblock_normalized":
                    dists /= emb.shape[1]

                # Exclude self
                for local_i in range(end - start):
                    dists[local_i, start + local_i] = np.inf

                # Partial sort: find k smallest per row
                part = np.argpartition(dists, k, axis=1)[:, :k]
                row_idx = np.arange(end - start)[:, None]
                ranked[start:end] = part[row_idx, np.argsort(dists[row_idx, part], axis=1)]

        return ranked

    def calculate_pairwise_distances(
        self, emb_space: str, metric: str = "euclidean",
        store_distances: bool = False,
    ):
        """Calculate pairwise distances between samples in an embedding space.
        Stores the top-k neighbor ranks in the Emma object.

        Args:
            emb_space (str): Name of the embedding space.
            metric (str): Distance metric to use. Default 'euclidean'.
            store_distances (bool): If True, also store the full N×N distance
                matrix. This uses O(N²) memory and should only be used when the
                matrix is explicitly needed (e.g. for within/between-class
                distance analysis). Default False.
        """
        self._check_for_emb_space(emb_space)
        if metric not in DISTANCE_METRIC_ALIASES:
            raise ValueError(f"Distance metric {metric} not supported.")

        # Early-exit if ranks are already cached.
        if metric in self.emb[emb_space].get("ranks", {}):
            print(f"Pairwise distances using {metric} already calculated.")
            return

        print(f"Calculating pairwise distances using {metric}...")
        n = len(self.sample_names)
        k = min(500, n - 1)

        if store_distances:
            # N×N float32 matrix — warn early if this will likely OOM.
            _check_memory(
                n * n * 4,
                label=f"calculate_pairwise_distances({metric}, store_distances=True)",
            )
            emb_pwd = self.__compute_pairwise_distances(
                emb_space, metric, self.emb[emb_space]["emb"]
            )

            # Compute ranks from the stored distance matrix, excluding self.
            # Self-distance is 0 so it always sorts to position 0; we request
            # k+1 neighbors and drop position 0 to get k non-self neighbors.
            if n > 5000:
                ranked_indices_list = []
                for start_idx in range(0, n, GPU_BATCH_SIZE):
                    end_idx = min(start_idx + GPU_BATCH_SIZE, n)
                    batch = emb_pwd[start_idx:end_idx]
                    part = np.argpartition(batch, kth=k, axis=1)[:, :k + 1]
                    row_idx = np.arange(batch.shape[0])[:, None]
                    sorted_part = part[row_idx, np.argsort(batch[row_idx, part], axis=1)]
                    ranked_indices_list.append(sorted_part[:, 1:])  # drop self at col 0
                ranked_indices = np.vstack(ranked_indices_list)
            else:
                # Full sort; self (distance 0) is always at column 0 — drop it.
                ranked_indices = np.argsort(emb_pwd, axis=1)[:, 1:]

            if "pairwise_distances" not in self.emb[emb_space]:
                self.emb[emb_space]["pairwise_distances"] = {}
            self.emb[emb_space]["pairwise_distances"][metric] = emb_pwd
        else:
            ranked_indices = self.__compute_knn_ranks(emb_space, metric, k)

        if "ranks" not in self.emb[emb_space]:
            self.emb[emb_space]["ranks"] = {}
        self.emb[emb_space]["ranks"][metric] = ranked_indices

    def mean_center(self, emb_spaces: list = None):
        """Apply mean-centering to embedding spaces in-place.

        Subtracts the per-dimension mean from each embedding space. The mean
        vector is stored internally so the operation can be reverted with
        revert_mean_centering(). Any cached pairwise distances, ranks, and 2-D
        projections are cleared, since they were computed on the original embeddings.

        Args:
            emb_spaces (list, optional): Names of embedding spaces to centre.
                Defaults to None, which centres all spaces.
        """
        targets = emb_spaces if emb_spaces is not None else list(self.emb.keys())
        for emb_space in targets:
            self._check_for_emb_space(emb_space)
            if self.emb[emb_space].get("_mean_centered", False):
                print(f"'{emb_space}' is already mean-centred, skipping.")
                continue
            mean = self.emb[emb_space]["emb"].mean(axis=0)
            self.emb[emb_space]["emb"] -= mean
            self.emb[emb_space]["_mean_centered"] = True
            self.emb[emb_space]["_emb_mean"] = mean
            for key in ("pairwise_distances", "ranks", "annoy_index", "annoy_ranks", "2d"):
                self.emb[emb_space].pop(key, None)
            print(f"'{emb_space}' mean-centred. Cached distances and projections cleared.")

    def revert_mean_centering(self, emb_spaces: list = None):
        """Revert mean-centred embedding spaces to their original values.

        Adds back the stored per-dimension mean. Any cached pairwise distances,
        ranks, and 2-D projections are cleared.

        Args:
            emb_spaces (list, optional): Names of embedding spaces to revert.
                Defaults to None, which reverts all mean-centred spaces.
        """
        targets = emb_spaces if emb_spaces is not None else list(self.emb.keys())
        for emb_space in targets:
            self._check_for_emb_space(emb_space)
            if not self.emb[emb_space].get("_mean_centered", False):
                print(f"'{emb_space}' is not mean-centred, skipping.")
                continue
            self.emb[emb_space]["emb"] += self.emb[emb_space].pop("_emb_mean")
            self.emb[emb_space]["_mean_centered"] = False
            for key in ("pairwise_distances", "ranks", "annoy_index", "annoy_ranks", "2d"):
                self.emb[emb_space].pop(key, None)
            print(f"'{emb_space}' reverted to original embeddings. Cached distances and projections cleared.")

    def get_pairwise_distances(
        self, emb_space: str, metric: str = "euclidean"
    ) -> np.ndarray:
        """Get pairwise distances between samples in an embedding space. \
            Will calculate the distances if not already done.

        Args:
        emb_space (str): Name of the embedding space.
        metric (str): Distance metric to use. Default 'euclidean'.

        Returns:
        np.ndarray: Pairwise distances.
        """
        self._check_for_emb_space(emb_space)
        if metric not in DISTANCE_METRIC_ALIASES:
            raise ValueError(f"Distance metric {metric} not supported.")

        if metric not in self.emb[emb_space].get("pairwise_distances", {}):
            # store_distances=True required here — we must return the full matrix.
            self.calculate_pairwise_distances(
                emb_space=emb_space, metric=metric, store_distances=True
            )

        return self.emb[emb_space]["pairwise_distances"][metric]

    def free_pairwise_distances(
        self, emb_space: str = None, metric: str = None
    ) -> None:
        """Release cached pairwise distance matrices to free memory.

        Neighbor ranks are kept; only the full N×N distance matrix is freed.

        Args:
            emb_space (str, optional): Specific embedding space to free.
                Defaults to None, which frees all spaces.
            metric (str, optional): Specific metric to free within the space.
                Defaults to None, which frees all metrics.
        """
        targets = [emb_space] if emb_space else list(self.emb.keys())
        for space in targets:
            self._check_for_emb_space(space)
            if metric:
                self.emb[space].get("pairwise_distances", {}).pop(metric, None)
            else:
                self.emb[space].pop("pairwise_distances", None)
        print("Freed pairwise distance matrices.")

    def get_knn(
        self, emb_space: str, k: int, metric: str = "euclidean", 
        use_annoy: bool = False, annoy_metric: str = None, n_trees: int = None,
    ) -> np.ndarray:
        """Get the k-nearest neighbors for each sample in an embedding space. \
            Will calculate the neighbors if not already done.

        Args:
        emb_space (str): Name of the embedding space.
        k (int): Number of neighbors to consider.
        metric (str): Distance metric to use. Default 'euclidean'.
        use_annoy (bool): Whether to use Annoy index. Default False.
        annoy_metric (str): Annoy distance metric to use. \
            Required if use_annoy is True.
        n_trees (int): Number of trees used to build the Annoy index. \
            Required if use_annoy is True.

        Returns:
        np.ndarray: Indices of the k-nearest neighbors.
        """

        # Validate input
        self._check_for_emb_space(emb_space)
        if k < 1:
            raise ValueError("k must be a positive integer.")
        if k > len(self.sample_names):
            raise ValueError("k must be less than the number of samples.")
        if metric not in DISTANCE_METRIC_ALIASES:
            raise ValueError(f"Distance metric {metric} not supported.")

        if use_annoy:
            # Validate Annoy-specific inputs
            if annoy_metric is None or n_trees is None:
                raise ValueError("annoy_metric and n_trees must be provided when use_annoy is True.")
            if "annoy_ranks" not in self.emb[emb_space]:
                raise ValueError(f"No Annoy indices found for embedding space '{emb_space}'.")
            if annoy_metric == "cosine":
                annoy_metric = "angular"  # Annoy uses 'angular' for cosine distance
            elif annoy_metric == "cityblock":
                annoy_metric = "manhattan"
            if annoy_metric not in self.emb[emb_space]["annoy_ranks"]:
                raise ValueError(f"No Annoy ranks found for metric '{annoy_metric}'.")
            if n_trees not in self.emb[emb_space]["annoy_ranks"][annoy_metric]:
                raise ValueError(f"No Annoy ranks with {n_trees} trees for metric '{annoy_metric}'.")

            # All stored annoy_ranks arrays exclude self (filtered in
            # build_annoy_index), so col 0 is the closest real neighbor.
            ranked_indices = self.emb[emb_space]["annoy_ranks"][annoy_metric][n_trees]

            print(f"Using Annoy index with {n_trees} trees and {annoy_metric} metric.")

            return ranked_indices[:, :k]

        try:
            ranked_indices = self.emb[emb_space]["ranks"][metric]
        except KeyError:
            self.calculate_pairwise_distances(emb_space, metric)
            ranked_indices = self.emb[emb_space]["ranks"][metric]

        # All stored ranks arrays exclude self (whether computed via
        # __compute_knn_ranks or the store_distances path), so col 0 is the
        # closest real neighbor.
        return ranked_indices[:, :k]
    
    def build_annoy_index(self, emb_space: str, n_trees: int = 50, 
                          metric: str = 'euclidean', random_seed: int = 42, max_k: int = 500):
        """Build the Annoy index for a given embedding space.
        
        Args:
        emb_space (str): Name of the embedding space.
        n_trees (int): Number of trees in the Annoy index. Default is 50.
        metric (str): Distance metric. Default is 'euclidean'.
        random_seed (int): Seed for reproducibility. Default is 42.
        max_k (int): Number of nearest neighbors to consider. Default is 500.
        """
        # Check if the embedding space exists
        if emb_space not in self.emb:
            raise ValueError(f"Embedding space {emb_space} not found.")
        if metric not in DISTANCE_METRIC_ALIASES:
            raise ValueError(f"Distance metric {metric} not supported.")
        if metric == "cosine":
            metric = "angular" # Annoy uses 'angular' for cosine distance
        elif metric == "cityblock":
            metric = "manhattan" # Annoy uses 'manhattan' for cityblock distance
        
        # Get the embeddings for the space
        embeddings = self.emb[emb_space]["emb"]

        # Create an Annoy index with the specified metric and dimensionality
        dim = embeddings.shape[1] 
        annoy_index = AnnoyIndex(dim, metric)
        annoy_index.set_seed(random_seed)  # Set a seed for reproducibility
        
        # Add embeddings to the index
        for i, emb in enumerate(embeddings):
            annoy_index.add_item(i, emb)

        # Build the index with n_trees
        print(f"Building Annoy index with {n_trees} trees...")
        annoy_index.build(n_trees)

        knn_indices = []
        for i in range(len(self.emb[emb_space]["emb"])):
            neighbors = annoy_index.get_nns_by_item(i, max_k + 1)
            neighbors = [n for n in neighbors if n != i][:max_k]
            knn_indices.append(neighbors)

        # Store only the compact ranks array; drop the index object to free memory.
        if "annoy_ranks" not in self.emb[emb_space]:
            self.emb[emb_space]["annoy_ranks"] = {}
        if metric not in self.emb[emb_space]["annoy_ranks"]:
            self.emb[emb_space]["annoy_ranks"][metric] = {}
        self.emb[emb_space]["annoy_ranks"][metric][n_trees] = np.array(knn_indices, dtype=int)
        del annoy_index

        print(f"Annoy index for {emb_space} built successfully with {n_trees} trees.")

    # Dimensionality reduction
    def get_2d(
        self,
        emb_space: str,
        method: str = "PCA",
        normalize: bool = True,
        random_state: int = 42,
        perplexity: int = 30,
        shuffle_umap: bool = True,
    ) -> dict:
        """Function to get the 2D reduction of a given embedding space. \
        Dimensionality reduction is performed using PCA, TSNE, or UMAP. \
        Uses cached values for recurring arguments.
        Args:
            emb_space (str): Name of an embedding space in the Emma instance.
            method (str, optional): Method for dimensionality reduction. \
                Either "PCA", "TSNE", or "UMAP". Defaults to "PCA".
            normalize (bool, optional): Whether to perform z-score normalisation \
                prior to dimensionality reduction. Defaults to True.
            random_state (int, optional): Random state for UMAP or TSNE. Defaults \
                to 42.
            perplexity (int, optional): Perplexity, only applied to UMAP.\
                Defaults to 30.
            shuffle_umap (bool, optional): Shuffle order of embeddings before \
                running UMAP. Defaults to True
        Returns:
            dict: A dictionary with key `"2d"` for the reduced embeddings and \
                optionally additional information.
        """
        self._check_for_emb_space(emb_space)

        self.emb[emb_space]["2d"] = self.emb[emb_space].get("2d", dict())
        key = "__".join(
            (str(arg) for arg in [method, normalize, random_state, perplexity, shuffle_umap])
        )
        # cache
        if key in self.emb[emb_space]["2d"]:
            return self.emb[emb_space]["2d"][key]

        embeddings = self.emb[emb_space]["emb"]
        result = {}

        if normalize:
            scaler = StandardScaler()
            embeddings = scaler.fit_transform(embeddings)

        if method == "PCA":
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)
            result["variance_explained"] = pca.explained_variance_ratio_
        elif method == "TSNE":
            tsne = TSNE(
                n_components=2, random_state=random_state, perplexity=perplexity
            )
            embeddings_2d = tsne.fit_transform(embeddings)
        elif method == "UMAP":
            umap = UMAP(n_components=2, random_state=random_state)
            if shuffle_umap:
                shuffled_i = np.random.permutation(len(embeddings))
                embeddings_2d = umap.fit_transform(embeddings[shuffled_i])
                unshuffled_i = np.argsort(shuffled_i)
                embeddings_2d = embeddings_2d[unshuffled_i]
            else:
                embeddings_2d = umap.fit_transform(embeddings)
        else:
            raise ValueError(f"Method {method} not implemented")

        result["2d"] = embeddings_2d
        self.emb[emb_space]["2d"][key] = result
        return result

    def compute_within_between_distances(self, emb_space: str, metric: str, feature: str):
        """Compute within-class and between-class distances for a feature category.

        Args:
            emb_space (str): Name of the embedding space.
            metric (str): Distance metric to use.
            feature (str): Name of the feature category in metadata.
            
        Returns:
            dict: {class_value: {"within": [...], "between": [...]}}
        """
        
        self._check_for_emb_space(emb_space)
        self._check_column_in_metadata(feature)
        self._check_column_is_categorical(feature)
        
        if metric not in DISTANCE_METRIC_ALIASES:
            raise ValueError(f"Distance metric {metric} not supported.")
        
        if metric not in self.emb[emb_space].get("pairwise_distances", {}):
            raise ValueError(
                f"Pairwise distances for {metric} not calculated. \
                    Please calculate them first."
            )
        
        emb_pwd = self.emb[emb_space]["pairwise_distances"][metric]
        labels = self.metadata[feature].values  # array of labels, one per sample
        
        unique_classes = np.unique(labels)
        results = {}

        for cls in unique_classes:
            mask_cls = labels == cls
            mask_other = labels != cls

            # Within-class distances
            within_distances = emb_pwd[np.ix_(mask_cls, mask_cls)]
            within_distances = within_distances[np.triu_indices_from(within_distances, k=1)]

            # Between-class distances
            between_distances = emb_pwd[np.ix_(mask_cls, mask_other)].flatten()

            results[cls] = {
                "within": within_distances,
                "between": between_distances
            }

        return results