"""Finding the most representative class samples (demos)"""

import argparse
import random
import warnings
from typing import List

import numpy as np
import pandas as pd
import torch

random.seed(42)

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

CLASS_LABELS10 = [
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


def find_demos(input_file, given_class_labels, threshold_per_class, method, print_selected):

    demo_texts = []
    demo_labels = []

    # Filter out relevant class labels
    df = pd.read_csv(input_file, sep="\t")
    texts = df.text  # list(df[0])
    labels = df.intent  # list(df[1])
    label2samples = dict()
    sample2label = dict()
    for txt, lbl in zip(texts, labels):
        # collect text per valid label
        if lbl in given_class_labels:
            if lbl not in label2samples:
                label2samples[lbl] = []
            label2samples[lbl].append(txt)
            sample2label[txt] = lbl
    # Prepare corpus
    corpus = list(sample2label.keys())
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    corpus_embeddings = embedder.encode(corpus)
    label_embs = embedder.encode([lbl.replace("_", " ") for lbl in given_class_labels])

    if method == "kmeans":
        # Perform kmean clustering
        num_clusters = len(given_class_labels)
        clustering_model = KMeans(n_clusters=num_clusters)
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        clustered_sentences = [[] for i in range(num_clusters)]
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clustered_sentences[cluster_id].append(corpus[sentence_id])

        for cluster in clustered_sentences:
            # Find the most likely label based on the similarity to clustered samples
            cluster_emb = embedder.encode(cluster)
            similarities = embedder.similarity(cluster_emb, label_embs)
            label_idx = torch.mode(torch.max(similarities, dim=1).indices).values.item()
            label_text = given_class_labels[label_idx]
            if label_text not in demo_labels:
                for demo in cluster[:threshold_per_class]:
                    demo_texts.append(demo)
                    demo_labels.append(label_text)
            else:
                warnings.warn(
                    f"Warning: more than one cluster was assigned label {label_text}, skipping..."
                )
            if print_selected:
                print(label_text, cluster[:threshold_per_class])
    elif method == "maxsim":
        # Compute max cossim between labels and examples
        corpus_embeddings = embedder.encode(corpus)
        similarities = embedder.similarity(label_embs, corpus_embeddings)
        for i in range(len(given_class_labels)):
            selected_sample_ids = torch.topk(
                similarities[i], k=threshold_per_class, dim=0, sorted=True
            ).indices
            selected_samples = [corpus[j] for j in selected_sample_ids]
            demo_texts.extend(selected_samples)
            label_text = given_class_labels[i]
            demo_labels.extend([label_text for _ in range(len(selected_samples))])
            if print_selected:
                print(label_text, selected_samples)
    elif method == "centroids":
        for label in label2samples:
            cluster_embs = embedder.encode(label2samples[label])
            centroid_emb = np.mean(cluster_embs, axis=0)
            similarities = embedder.similarity(centroid_emb, cluster_embs)
            max_centroid_sim_indices = torch.topk(
                similarities[0], k=threshold_per_class, dim=0
            ).indices
            selected_samples = [label2samples[label][j] for j in max_centroid_sim_indices]
            demo_texts.extend(selected_samples)
            demo_labels.extend([label for _ in range(len(selected_samples))])
            if print_selected:
                print(label, selected_samples)
    elif method == "dissimilar":
        for label in label2samples:
            cluster_embs = embedder.encode(label2samples[label])
            # select a random point for each class
            class_samples = label2samples[label]
            first_selected = class_samples[
                np.random.choice([i for i in range(len(class_samples))])
            ]
            class_samples.remove(first_selected)
            selected_samples = [first_selected]
            while len(selected_samples) < threshold_per_class:
                # select least similar point to already selected ones
                candidate_embs = embedder.encode(class_samples)
                selected_embs = embedder.encode(selected_samples)
                similarities = embedder.similarity(selected_embs, candidate_embs)
                min_sim_idx = torch.topk(
                    torch.mean(similarities, dim=0), largest=False, k=1, dim=0
                ).indices
                selected_samples.append(class_samples[min_sim_idx])
                class_samples.remove(class_samples[min_sim_idx])
            demo_texts.extend(selected_samples)
            demo_labels.extend([label for _ in range(len(selected_samples))])
            if print_selected:
                print(label, selected_samples)

    # Save to file
    df = pd.DataFrame.from_dict({"text": demo_texts, "labels": demo_labels})
    df.to_csv(input_file.replace(".csv", f"_{method}.csv"), sep=",", index=False, header=False)


if __name__ == "__main__":
    # input_file = "data/massive-en/en-US_train.csv"
    # find_demos(input_file)
    parser = argparse.ArgumentParser(description="Parameters for finding demonstrations")
    parser.add_argument("--input_file", help="Input file with gold annotated data.", type=str)
    parser.add_argument(
        "--given_class_labels", help="Valid class label.", type=List[str], default=CLASS_LABELS10
    )
    parser.add_argument(
        "--threshold_per_class", help="Number of demonstrations per class.", type=int, default=10
    )
    parser.add_argument(
        "--method",
        help="Method to select demonstrations: maxsim, kmeans, centroids, dissimilar",
        type=str,
        default="maxsim",
    )
    parser.add_argument(
        "--print_selected",
        help="Whether to print selected demonstrations.",
        type=bool,
        default=False,
    )

    args = parser.parse_args()
    print(args)
    find_demos(
        input_file=args.input_file,
        given_class_labels=args.given_class_labels,
        threshold_per_class=args.threshold_per_class,
        method=args.method,
        print_selected=args.print_selected,
    )
