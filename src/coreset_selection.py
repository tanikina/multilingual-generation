import random
import warnings

random.seed(42)

from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModel, AutoTokenizer

# Based on the following code:
# https://github.com/webis-de/small-text/blob/v1.3.2/small_text/query_strategies/coresets.py

classes = [
    "alarm_query",
    "audio_volume_down",
    "calendar_remove",
    "cooking_recipe",
    "datetime_convert",
    "email_sendemail",
    "play_audiobook",
    "recommendation_movies",
    "transport_ticket",
    "weather_query",
]

_DISTANCE_METRICS = ["cosine", "euclidean"]


def self_greedy_coreset(X, k):
    """Greedy coreset selection using Farthest Point Sampling.

    Parameters:
    - X: Data points (numpy array of shape [N, D])
    - k: Number of selected points

    Returns:
    - S: Indices of selected points
    """
    N = X.shape[0]
    selected_indices = []

    # Step 1: Randomly select the first point
    first_index = np.random.randint(N)
    selected_indices.append(first_index)

    # Step 2: Iteratively select points farthest from current selection
    for _ in range(k - 1):
        distances = pairwise_distances(X, X[selected_indices])
        min_distances = np.min(distances, axis=1)
        next_index = np.argmax(min_distances)
        selected_indices.append(next_index)

    return np.array(selected_indices)


def _check_coreset_size(x, n):
    if n > x.shape[0]:
        raise ValueError(
            f"n (n={n}) is greater the number of available samples (num_samples={x.shape[0]})"
        )


def _cosine_distance(a, b, normalized=False):
    sim = np.matmul(a, b.T)
    if not normalized:
        sim = sim / np.dot(
            np.linalg.norm(a, axis=1)[:, np.newaxis], np.linalg.norm(b, axis=1)[np.newaxis, :]
        )
    res = np.arccos(sim) / np.pi
    return res


def _euclidean_distance(a, b, normalized=False):
    _ = normalized
    return pairwise_distances(a, b, metric="euclidean")


def compute_density(X, k_neighbors=5):
    """Compute density for each point in X using k-Nearest Neighbors.

    Parameters:
    - X: Data points (numpy array of shape [N, D])
    - k_neighbors: Number of neighbors to consider for density estimation.

    Returns:
    - density: Array of density values for each point.
    """
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(X)
    distances, _ = nbrs.kneighbors(X)

    # Density is the inverse of the sum of distances to k-nearest neighbors
    density = 1.0 / (np.sum(distances, axis=1) + 1e-8)  # Add small value to avoid division by zero
    return density


def density_aware_greedy_coreset(X, k, indices_labeled=[], k_neighbors=5):
    """Density-Aware Greedy Coreset selection.

    Parameters:
    - X: Data points (numpy array of shape [N, D])
    - k: Number of selected points
    - indices_labeled: whether to initialize the cluster with gold samples
    - k_neighbors: Number of neighbors for density computation

    Returns:
    - selected_indices: Indices of selected points
    """
    N = X.shape[0]
    selected_indices = []

    # Step 1: Compute density for all points
    density = compute_density(X, k_neighbors=k_neighbors)
    median_density = np.median(density)

    # Step 2: Randomly select the first point if no gold data is available
    if len(indices_labeled) > 0:
        selected_indices = indices_labeled.tolist()
    else:
        first_index = np.random.randint(N)
        selected_indices.append(first_index)

    for _ in range(k - 1):
        # Compute distances from all points to the selected set
        distances = pairwise_distances(X, X[selected_indices])
        min_distances = np.min(distances, axis=1)

        # Compute selection score: Distance weighted by density
        scores = min_distances / (density + 1e-8)  # Avoid division by zero
        for i in range(len(scores)):
            if density[i] < median_density:
                scores[i] = 0

        # Select the point with the highest score
        next_index = np.argmax(scores)
        selected_indices.append(next_index)

    res = np.array(selected_indices[len(indices_labeled) :])

    if len(indices_labeled) > 0:
        res = res - len(indices_labeled)
    return res


def adaptive_greedy_coreset(X, k, indices_labeled=[], lambda_threshold=2.5):
    """Greedy Coreset selection with Adaptive Distance Thresholding to avoid outliers.

    Parameters:
    - X: Data points (numpy array of shape [N, D])
    - k: Number of selected points
    - indices_labeled: Labeled indices to initialize the selection.
    - lambda_threshold: Controls how strict the outlier filtering is (higher = stricter)

    Returns:
    - selected_indices: Indices of selected points
    """
    N = X.shape[0]
    selected_indices = []

    if len(indices_labeled) > 0:
        selected_indices = indices_labeled.tolist()
    else:
        # Step 1: Randomly select the first point
        first_index = np.random.randint(N)
        selected_indices.append(first_index)

    for _ in range(k - 1):
        # Compute distances from all points to the currently selected points
        distances = pairwise_distances(X, X[selected_indices])
        min_distances = np.min(distances, axis=1)

        # Compute mean and standard deviation of distances
        mean_dist = np.mean(min_distances)
        std_dist = np.std(min_distances)

        # Apply adaptive thresholding to avoid outliers
        candidate_indices = np.where(min_distances > (mean_dist + lambda_threshold * std_dist))[0]

        if len(candidate_indices) == 0:
            # If no candidates pass the threshold, use the max distance point
            next_index = np.argmax(min_distances)
        else:
            # Otherwise, select the farthest point among candidates
            next_index = candidate_indices[np.argmax(min_distances[candidate_indices])]

        selected_indices.append(next_index)

    res = np.array(selected_indices[len(indices_labeled) :])
    if len(indices_labeled) > 0:
        res = res - len(indices_labeled)
    return res


def greedy_coreset(
    x,
    indices_unlabeled,
    indices_labeled,
    n,
    distance_metric="euclidean",
    batch_size=100,
    normalized=False,
    selection_strategy="greedy",
):
    """Computes a greedy coreset [SS17]_ over `x` with size `n`.

    Parameters
    ----------
    x : np.ndarray
        A matrix of row-wise vector representations.
    indices_unlabeled : np.ndarray
        Indices (relative to `dataset`) for the unlabeled data.
    indices_labeled : np.ndarray
        Indices (relative to `dataset`) for the unlabeled data.
    n : int
        Size of the coreset (in number of instances).
    distance_metric : {'cosine', 'euclidean'}
        Distance metric to be used.
    batch_size : int
        Batch size.
    normalized : bool
        If `True` the data `x` is assumed to be normalized,
        otherwise it will be normalized where necessary.
    selection_strategy : str
        If `distance_thresholding` the adaptive distance thresholding is applied to avoid adding outliers. If `density_aware_coreset` we use kNN clusterig to perform density-aware coreset selection. `max_similarity` means that we select the samples with the smallest distance to the provided gold data.

    Returns
    -------
    indices : numpy.ndarray
        Indices relative to `x`.

    References
    ----------
    .. [SS17] Ozan Sener and Silvio Savarese. 2017.
       Active Learning for Convolutional Neural Networks: A Core-Set Approach.
       In International Conference on Learning Representations 2018 (ICLR 2018).
    """
    _check_coreset_size(x, n)

    num_batches = int(np.ceil(x.shape[0] / batch_size))
    ind_new = []

    if distance_metric == "cosine":
        dist_func = _cosine_distance
    elif distance_metric == "euclidean":
        dist_func = _euclidean_distance
    else:
        raise ValueError(
            f"Invalid distance metric: {distance_metric}. " f"Possible values: {_DISTANCE_METRICS}"
        )

    for _ in range(n):
        indices_s = np.concatenate([indices_labeled, ind_new]).astype(np.int64)
        dists = np.array([], dtype=np.float32)
        for batch in np.array_split(x[indices_unlabeled], num_batches, axis=0):

            dist = dist_func(batch, x[indices_s], normalized=normalized)
            sims_batch = np.amin(dist, axis=1)

            # Compute mean and standard deviation of distances
            # mean_dist = np.mean(sims_batch)
            # std_dist = np.std(sims_batch)
            dists = np.append(dists, sims_batch)

        if selection_strategy == "max_similarity":
            dists[ind_new] = np.inf
        else:
            dists[ind_new] = -np.inf

        # TODO: add selection strategy with similarity > avg_similarity between gold class examples but max distance from the existing ones?

        if selection_strategy == "distance_thresholding":
            return adaptive_greedy_coreset(x[indices_unlabeled], n, lambda_threshold=1.5)
        elif selection_strategy == "density_aware_coreset":
            # Select data points with density that is above average
            return density_aware_greedy_coreset(x[indices_unlabeled], n, k_neighbors=5)
        elif selection_strategy == "distance_thresholding_with_gold":
            return adaptive_greedy_coreset(x, n, indices_labeled, lambda_threshold=1.5)
        elif selection_strategy == "density_aware_coreset_with_gold":
            # Select data points with density that is above average
            return density_aware_greedy_coreset(x, n, indices_labeled, k_neighbors=5)
        elif selection_strategy == "max_similarity":
            index_new = np.argmin(dists)
        elif selection_strategy == "greedy":
            index_new = np.argmax(dists)
        elif selection_strategy == "self_greedy":
            return self_greedy_coreset(x[indices_unlabeled], n)
        else:
            raise ValueError(f"Unknown selection strategy: {selection_strategy}.")

        ind_new.append(index_new)

    return np.array(ind_new)


class GreedyCoreset:  # EmbeddingBasedQueryStrategy
    """Selects instances by constructing a greedy coreset [SS17]_ over document embeddings."""

    def __init__(self, distance_metric="euclidean", normalize=True, batch_size=100):
        """
        Parameters
        ----------
        distance_metric : {'cosine', 'euclidean'}
             Distance metric to be used.

             .. versionadded:: 1.2.0
        normalize : bool
            Embeddings will be normalized before the coreset construction if True.
        batch_size : int
            Batch size used for computing document distances.


        .. note::

           The default distance metric before v1.2.0 used to be cosine distance.

        .. seealso::

           Function :py:func:`.greedy_coreset`
              Docstrings of the underlying :py:func:`greedy_coreset` method.
        """
        if distance_metric not in set(_DISTANCE_METRICS):
            raise ValueError(
                f"Invalid distance metric: {distance_metric}. "
                f"Possible values: {_DISTANCE_METRICS}"
            )

        self.distance_metric = distance_metric
        self.normalize = normalize
        self.batch_size = batch_size

    def sample(
        self,
        indices_unlabeled,
        indices_labeled,
        y,
        n,
        embeddings,
        embeddings_proba=None,
        selection_strategy="greedy",
    ):  # clf, dataset,
        if self.normalize:
            from sklearn.preprocessing import normalize

            embeddings = normalize(embeddings, axis=1)
        return greedy_coreset(
            embeddings,
            indices_unlabeled,
            indices_labeled,
            n,
            distance_metric=self.distance_metric,
            normalized=self.normalize,
            selection_strategy=selection_strategy,
        )

    def __str__(self):
        return (
            f"GreedyCoreset(distance_metric={self.distance_metric}, "
            f"normalize={self.normalize}, batch_size={self.batch_size})"
        )


def lightweight_coreset(x, x_mean, n, normalized=False, proba=None):
    """Computes a lightweight coreset [BLK18]_ of `x` with size `n`.

    Parameters
    ----------
    x : np.ndarray
        2D array in which each row represents a sample.
    x_mean : np.ndarray
        Elementwise mean over the columns of `x`.
    n : int
        Coreset size.
    normalized : bool
        If `True` the data `x` is assumed to be normalized,
        otherwise it will be normalized where necessary.
    proba : np.ndarray or None
        A probability distribution over `x`, which makes up half of the probability mass
        of the sampling distribution. If `proba` is not `None` a uniform distribution is used.

    Returns
    -------
    indices : numpy.ndarray
        Indices relative to `x`.
    """
    _check_coreset_size(x, n)

    sim = x.dot(x_mean)
    if not normalized:
        sim = sim / (np.linalg.norm(x, axis=1) * np.linalg.norm(x_mean))

    dists = np.arccos(sim) / np.pi
    dists = np.square(dists)

    sum_dists = dists.sum()

    if proba is None:
        uniform = 0.5 * 1 / x.shape[0]
        proba = uniform + 0.5 * dists / sum_dists
    else:
        proba = 0.5 * proba / proba.sum() + 0.5 * dists / sum_dists

    proba = proba / np.linalg.norm(proba, ord=1)

    return np.random.choice(np.arange(x.shape[0]), n, replace=False, p=proba)


class LightweightCoreset:  # EmbeddingBasedQueryStrategy
    """Selects instances by constructing a lightweight coreset [BLK18]_ over document
    embeddings."""

    def __init__(self, normalize=True):
        """
        Parameters
        ----------
        normalize : bool
            Embeddings will be normalized before the coreset construction if True.
        """
        self.normalize = normalize

    def sample(
        self,
        indices_unlabeled,
        _indices_labeled,
        _y,
        n,
        embeddings,
        embeddings_proba=None,
        selection_strategy=None,
    ):  # clf, dataset,

        embeddings = embeddings[indices_unlabeled]

        embeddings_mean = np.mean(embeddings, axis=0)
        if self.normalize:
            from sklearn.preprocessing import normalize

            embeddings = normalize(embeddings)
            embeddings_mean = normalize(embeddings_mean[np.newaxis, :])

        embeddings_mean = embeddings_mean.ravel()

        return lightweight_coreset(embeddings, embeddings_mean, n, normalized=self.normalize)

    def __str__(self):
        return f"LightweightCoreset(normalize={self.normalize})"


def main():
    model_name = "google-bert/bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    for selection_strategy in [
        "distance_thresholding_with_gold",
        "density_aware_coreset_with_gold",
        "max_similarity",
        "density_aware_coreset",
        "distance_thresholding",
        "self_greedy",
        "greedy",
        "lightweight",
        "random",
    ]:
        # selection_strategy = "lightweight" #"max_similarity" #"density_aware_coreset" # "distance_thresholding" "greedy" "lightweight"

        gold_data_path = "ko-KR_train.csv"
        lang = gold_data_path.split("-")[0]
        generated_data_path = (
            f"{lang}_llama8b_summarized_intent_description_10demos_self_check_1000.csv"
        )

        out_path = f"{lang}_{selection_strategy}.csv"

        coreset: Union[LightweightCoreset, GreedyCoreset]
        if selection_strategy == "lightweight":
            coreset = LightweightCoreset()
        else:
            coreset = GreedyCoreset()

        # select random 10 samples from the gold training set
        df = pd.read_csv(gold_data_path, delimiter="\t")
        demo_texts = list(df["text"])
        demo_labels = list(df["intent"])

        num_gold_demos = 10

        class2demos: Dict[str, List[str]] = dict()
        for txt, lbl in zip(demo_texts, demo_labels):
            if lbl in classes:
                if lbl not in class2demos:
                    class2demos[lbl] = []
                class2demos[lbl].append(txt)

        # collect the generated data per class
        df = pd.read_csv(generated_data_path, delimiter=",", header=None)
        orig_generated_texts = list(df[0])
        orig_generated_labels = list(df[1])
        generated_texts = []
        generated_labels = []
        for idx in range(len(orig_generated_texts)):
            if len(orig_generated_texts[idx].strip().split()) > 1:
                generated_texts.append(orig_generated_texts[idx])
                generated_labels.append(orig_generated_labels[idx])

        class2generated: Dict[str, List[str]] = dict()
        for txt, lbl in zip(generated_texts, generated_labels):
            if lbl in classes:
                if lbl not in class2generated:
                    class2generated[lbl] = []
                class2generated[lbl].append(txt)

        new_labels = []
        new_texts = []
        for cls, class_demos in class2demos.items():
            random.shuffle(class_demos)
            gold_class_samples = class_demos[:num_gold_demos]
            generated_samples = class2generated[cls]
            input_texts = gold_class_samples + generated_samples
            inputs = tokenizer.batch_encode_plus(
                input_texts, return_tensors="pt", padding=True, truncation=True
            )
            embeddings = model(**inputs)
            embeddings = embeddings[1]  # pooled embeds
            labels = np.zeros(len(input_texts), dtype=np.int32)
            indices_unlabeled = np.array(
                [idx for idx in range(len(gold_class_samples), len(input_texts))], dtype=np.int32
            )
            indices_labeled = np.array(
                [idx for idx in range(0, len(gold_class_samples))], dtype=np.int32
            )
            coreset_size = 20  # how many samples per label we select
            if selection_strategy == "random":
                sampled = random.choices(
                    [i for i in range(len(generated_samples))], k=coreset_size
                )
            elif selection_strategy == "lightweight":
                sampled = coreset.sample(
                    indices_unlabeled,
                    indices_labeled,
                    labels,
                    coreset_size,
                    embeddings.detach().numpy(),
                )
            else:
                sampled = coreset.sample(
                    indices_unlabeled,
                    indices_labeled,
                    labels,
                    coreset_size,
                    embeddings.detach().numpy(),
                    selection_strategy=selection_strategy,
                )
            for s in sampled:
                new_texts.append(generated_samples[s])
                new_labels.append(cls)

        # write into file
        df = pd.DataFrame.from_dict({"texts": new_texts, "labels": new_labels})
        df.to_csv(out_path, index=False, header=False)


if __name__ == "__main__":
    main()
