import os

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

        embeddings = []
        for sample in self.sample_names:
            emb_file = os.path.join(dir_path, f"{sample}.{file_extension}")
            if not os.path.isfile(emb_file):
                raise ValueError(f"Embedding file '{emb_file}' not found.")
            embeddings.append(np.load(emb_file))
        return np.stack(embeddings)

    def _assign_colour_to_embedding_space(self, num_emb_spaces: int) -> str:
        """Assign a colour to the embedding space."""
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
            "colour": self._assign_colour_to_embedding_space(len(self.emb)),
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

            if metric == "sqeuclidean_normalised":
                # divide each row by its norm
                emb_norm = np.linalg.norm(emb, axis=1)
                emb = emb / emb_norm[:, None]  # divide each row by its norm
                emb_pwd = squareform(pdist(emb, metric="sqeuclidean"))
                return emb_pwd

            elif metric == "euclidean_normalised":
                # divide each row of the emb by its norm
                emb_norm = np.linalg.norm(emb, axis=1)
                emb = emb / emb_norm[:, None]  # divide each row by its norm
                emb_pwd = squareform(pdist(emb, metric="euclidean"))
                return emb_pwd

            elif metric == "cityblock_normalised":
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

    def calculate_pairwise_distances(
        self, emb_space: str, metric: str = "euclidean"
    ):
        """Calculate pairwise distances between samples in an embedding space.\
            Will store the distances in the Emma object.
            Will also calculate and store the ranks based on the distances.

        Args:
        emb_space (str): Name of the embedding space.
        metric (str): Distance metric to use. Default 'euclidean'.
        """
        self._check_for_emb_space(emb_space)
        if metric not in DISTANCE_METRIC_ALIASES:
            raise ValueError(f"Distance metric {metric} not supported.")

        if metric not in self.emb[emb_space].get("pairwise_distances", {}):
            print(f"Calculating pairwise distances using {metric}...")

            emb_pwd = self.__compute_pairwise_distances(
                emb_space, metric, self.emb[emb_space]["emb"]
            )
            
            # Compute ranks based on distances
            if emb_pwd.shape[0] > 5000:
                batch_size = GPU_BATCH_SIZE
                k = 500
                ranked_indices_list = []

                for start_idx in range(0, emb_pwd.shape[0], batch_size):
                    end_idx = min(start_idx + batch_size, emb_pwd.shape[0])
                    emb_pwd_batch = emb_pwd[start_idx:end_idx]

                    partitioned_indices = np.argpartition(emb_pwd_batch, kth=k, axis=1)[:, :k]
                    row_indices = np.arange(emb_pwd_batch.shape[0])[:, None]
                    topk_distances = emb_pwd_batch[row_indices, partitioned_indices]
                    sorted_topk_indices = np.argsort(topk_distances, axis=1)
                    ranked_indices_batch = partitioned_indices[row_indices, sorted_topk_indices]

                    ranked_indices_list.append(ranked_indices_batch)

                ranked_indices = np.vstack(ranked_indices_list)
            else:
                ranked_indices = np.argsort(emb_pwd, axis=1)

            if "pairwise_distances" not in self.emb[emb_space]:
                self.emb[emb_space]["pairwise_distances"] = {}
            if "ranks" not in self.emb[emb_space]:
                self.emb[emb_space]["ranks"] = {}

            self.emb[emb_space]["pairwise_distances"][metric] = emb_pwd
            self.emb[emb_space]["ranks"][metric] = ranked_indices

        else:
            print(f"Pairwise distances using {metric} already calculated.")

    def mean_center(self, emb_spaces: list = None):
        """Apply mean-centering to embedding spaces in-place.

        Subtracts the per-dimension mean from each embedding space.
        The original embeddings are preserved internally so the operation
        can be reverted with revert_mean_centering().
        Any cached pairwise distances, ranks, and 2-D projections are
        cleared, since they were computed on the original embeddings.

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
            X = self.emb[emb_space]["emb"]
            self.emb[emb_space]["_emb_original"] = X
            self.emb[emb_space]["emb"] = X - X.mean(axis=0)
            self.emb[emb_space]["_mean_centered"] = True
            # Clear caches that depend on the raw embeddings
            for key in ("pairwise_distances", "ranks", "annoy_index", "annoy_ranks", "2d"):
                self.emb[emb_space].pop(key, None)
            print(f"'{emb_space}' mean-centred. Cached distances and projections cleared.")

    def revert_mean_centering(self, emb_spaces: list = None):
        """Revert mean-centred embedding spaces to their original values.

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
            self.emb[emb_space]["emb"] = self.emb[emb_space].pop("_emb_original")
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
            self.calculate_pairwise_distances(
                emb_space=emb_space, metric=metric
            )

        return self.emb[emb_space]["pairwise_distances"][metric]

    def get_knn(
        self, emb_space: str, k: int, metric: str = "euclidean", 
        use_annoy: bool = False, annoy_metric: str = None, n_trees: int = None,
    ) -> np.ndarray:
        """Get the k-nearest neighbours for each sample in an embedding space. \
            Will calculate the neighbours if not already done.

        Args:
        emb_space (str): Name of the embedding space.
        k (int): Number of neighbours to consider.
        metric (str): Distance metric to use. Default 'euclidean'.
        use_annoy (bool): Whether to use Annoy index. Default False.
        annoy_metric (str): Annoy distance metric to use. \
            Required if use_annoy is True.
        n_trees (int): Number of trees used to build the Annoy index. \
            Required if use_annoy is True.

        Returns:
        np.ndarray: Indices of the k-nearest neighbours.
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

            # Get Annoy index
            ranked_indices = self.emb[emb_space]["annoy_ranks"][annoy_metric][n_trees]

            print(f"Using Annoy index with {n_trees} trees and {annoy_metric} metric.")

            #return np.array(knn_indices, dtype=int)
            return ranked_indices[:, 1 : k + 1]
        
        try:
            ranked_indices = self.emb[emb_space]["ranks"][metric]
        except KeyError:
            self.calculate_pairwise_distances(emb_space, metric)
            ranked_indices = self.emb[emb_space]["ranks"][metric]

        return ranked_indices[:, 1 : k + 1]
    
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

        # Store the built index in the emb_space
        if "annoy_index" not in self.emb[emb_space]:
            self.emb[emb_space]["annoy_index"] = {}
            self.emb[emb_space]["annoy_ranks"] = {}
        if metric not in self.emb[emb_space]["annoy_index"]:
            self.emb[emb_space]["annoy_index"][metric] = {}
            self.emb[emb_space]["annoy_ranks"][metric] = {}
        self.emb[emb_space]["annoy_index"][metric][n_trees] = annoy_index
        
        knn_indices = []

        for i in range(len(self.emb[emb_space]["emb"])):
            neighbors = annoy_index.get_nns_by_item(i, max_k + 1)  # fetch k+1 neighbors
            neighbors = [n for n in neighbors if n != i][:max_k] 
            knn_indices.append(neighbors)
        self.emb[emb_space]["annoy_ranks"][metric][n_trees] = np.array(knn_indices, dtype=int)
        
        print(f"Annoy index for {emb_space} built successfully with {n_trees} trees.")

    # Dimensionality reduction
    def get_2d(
        self,
        emb_space: str,
        method: str = "PCA",
        normalise: bool = True,
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
            normalise (bool, optional): Whether to perform z-score normalisation \
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
            (str(arg) for arg in [method, normalise, random_state, perplexity, shuffle_umap])
        )
        # cache
        if key in self.emb[emb_space]["2d"]:
            return self.emb[emb_space]["2d"][key]

        embeddings = self.emb[emb_space]["emb"]
        result = {}

        if normalise:
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

    def compute_within_between_distances(self, emb_space: str, metric: str, feature_category: str):
        """Compute within-class and between-class distances for a feature category.

        Args:
            emb_space (str): Name of the embedding space.
            metric (str): Distance metric to use.
            feature_category (str): Name of the feature category in metadata.
            
        Returns:
            dict: {class_value: {"within": [...], "between": [...]}}
        """
        
        self._check_for_emb_space(emb_space)
        self._check_column_in_metadata(feature_category)
        self._check_column_is_categorical(feature_category)
        
        if metric not in DISTANCE_METRIC_ALIASES:
            raise ValueError(f"Distance metric {metric} not supported.")
        
        if metric not in self.emb[emb_space].get("pairwise_distances", {}):
            raise ValueError(
                f"Pairwise distances for {metric} not calculated. \
                    Please calculate them first."
            )
        
        emb_pwd = self.emb[emb_space]["pairwise_distances"][metric]
        labels = self.metadata[feature_category].values  # array of labels, one per sample
        
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