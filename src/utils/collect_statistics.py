import argparse
import os

import numpy as np
import pandas as pd

VALID_SETTINGS = [
    "summarized_intent",
    "english_demos",
    "target_lang_demos",
    "target_lang_demos_and_revision",
]

LANG_ORDER = ["az", "cy", "de", "en", "he", "id", "ro", "sl", "sw", "te", "th"]


def collect_statistics(input_dir_path, metric):
    lang2avg_scores = dict()
    lang2std = dict()
    for fname in os.listdir(input_dir_path):
        df = pd.read_csv(os.path.join(input_dir_path, fname))
        settings = list(df["setting"])
        if metric == "f1":
            scores = list(df["f1"])
        elif metric == "accuracy":
            scores = list(df["accuracy"])

        lang = fname.split("_")[0]
        lang2avg_scores[lang] = dict()
        lang2std[lang] = dict()
        for setting in VALID_SETTINGS:
            lang2avg_scores[lang][setting] = 0.0
            lang2std[lang][setting] = 0.0

        setting2score = dict()
        for setting, score in zip(settings, scores):
            if "only_summarized_intent" in setting:
                setting = "summarized_intent"
            elif "english_demos" in setting:
                setting = "english_demos"
            elif "target_lang_demos_and_revision" in setting:
                setting = "target_lang_demos_and_revision"
            elif "target_lang_demos" in setting:
                setting = "target_lang_demos"
            else:
                raise ValueError(f"Unknown setting: {setting}")

            if setting not in setting2score:
                setting2score[setting] = []

            setting2score[setting].append(float(score))

        for setting in VALID_SETTINGS:
            if setting not in setting2score:
                raise ValueError(f"Setting {setting} is missing in the file {fname}!")
            else:
                scores_per_setting = setting2score[setting]
                lang2avg_scores[lang][setting] = round(np.mean(scores_per_setting), 3)
                lang2std[lang][setting] = round(np.std(scores_per_setting), 5)

    df_avg_scores = pd.DataFrame.from_dict(lang2avg_scores)
    df_avg_scores = df_avg_scores.reindex(columns=LANG_ORDER)
    df_std = pd.DataFrame.from_dict(lang2std)
    df_std = df_std.reindex(columns=LANG_ORDER)

    md_avg_scores = df_avg_scores.to_markdown()
    md_std = df_std.to_markdown()

    print(f"*** Average {metric} scores ***")
    print(md_avg_scores)
    print()
    print("*** Standard deviation ***")
    print(md_std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for collecting statistics.")
    parser.add_argument("--input_dir_path", type=str, default="results/massive10/llama3_8b")
    parser.add_argument("--metric", type=str, default="f1", choices=["f1", "accuracy"])
    args = parser.parse_args()

    collect_statistics(args.input_dir_path, args.metric)
