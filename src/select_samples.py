import argparse
import random

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

# The code for selecting similar examples is based on:
# https://github.com/kinit-sk/selec-strats-for-aug/blob/e2996caadb5e634893023710aaa33ddcbeb6988c/mistral_collect_scripts/sample_selection_strategies.py


def get_embs_for_sents(label2samples, sent_model) -> dict:
    label2embeds = {}

    for label in label2samples.keys():
        label2embeds[label] = {
            "emb": sent_model.encode(label2samples[label], show_progress_bar=False),
            "sent": label2samples[label],
        }
    return label2embeds


def select_samples(
    selection_strategy, max_per_label, lang, input_path, output_path, seed, verbose=False
):
    # read input data (with generated samples)
    df = pd.read_csv(input_path, header=None)
    texts = list(df[0])
    labels = list(df[1])

    # load sentence transformers model
    sent_model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

    # create mapping from classes to samples
    label2samples = dict()
    for text, label in zip(texts, labels):
        if label not in label2samples:
            label2samples[label] = []
        label2samples[label].append(text)

    # choose the top k samples for each class based on the selection strategy
    new_labels = []
    new_samples = []
    for label, samples in label2samples.items():
        label2embeds = get_embs_for_sents(label2samples, sent_model)
        if selection_strategy in ["outliers", "reverse_outliers"]:
            selected_samples = get_outliers(
                samples,
                label2embeds[label],
                max_per_label,
                reverse=False if selection_strategy == "reverse_outliers" else True,
            )
        else:
            indices = np.arange(len(samples))
            np.random.seed(seed)
            np.random.shuffle(indices)
            indices = indices[:1].tolist()
            features = label2embeds[label]["emb"]
            for _ in range(max_per_label - 1):
                if selection_strategy == "most_similar":
                    sim = np.mean(cos_sim(features[indices], features), axis=0).argsort()[::-1]
                elif selection_strategy == "most_diverse":
                    sim = np.mean(cos_sim(features[indices], features), axis=0).argsort()

                for index in sim:
                    if index not in indices:
                        indices.append(index)
                        break
            selected_samples = [samples[idx] for idx in indices]
        if verbose:
            print(f"Label {label} with the following samples: {selected_samples}\n")
        new_labels.extend([label for _ in range(len(selected_samples))])
        new_samples.extend(selected_samples)

    # store the selected samples
    new_df = pd.DataFrame(data={"text": new_samples, "label": new_labels})
    new_df.to_csv(output_path, index=False, header=False)


def get_outliers(samples, embeds, max_per_label, reverse):
    dist_text_tuples = []
    # calculate mean vector per label
    mean_emb = embeds["emb"].mean(axis=0)
    for (sent_emb, sent) in zip(embeds["emb"], embeds["sent"]):
        dist = np.linalg.norm(mean_emb - sent_emb)
        dist_text_tuples.append((dist, sent))
    selected_outliers = sorted(dist_text_tuples, key=lambda x: x[0], reverse=reverse)[
        :max_per_label
    ]
    return [tpl[1] for tpl in selected_outliers]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for selecting generated samples.")
    parser.add_argument("--selection_strategy", type=str, default="most_similar")
    parser.add_argument("--max_per_label", type=int, default=20)
    parser.add_argument("--lang", type=str, default="de-DE")
    parser.add_argument(
        "--input_path", type=str, default="massive-10x-de/prepared/self_demos_10_100demos_80B.csv"
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    selection_strategy = args.selection_strategy
    max_per_label = args.max_per_label
    lang = args.lang
    input_path = args.input_path

    output_path = input_path.replace(".csv", "_" + selection_strategy + ".csv")
    seed = args.seed
    select_samples(selection_strategy, max_per_label, lang, input_path, output_path, seed)
