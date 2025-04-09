import argparse
import os

import numpy as np
import pandas as pd

BASELINE_SETTINGS = ["gold"]
GENERATION_SETTINGS = [
    "summarized_intent",
    "english_demos",
    "target_lang_demos",
    "target_lang_demos_and_revision",
]
ALL_SETTINGS = BASELINE_SETTINGS + GENERATION_SETTINGS

LANG_ORDER = ["az", "cy", "de", "en", "he", "id", "ro", "sl", "sw", "te", "th"]


def process_results(input_dir_path, lang2avg_scores, lang2std, metric="f1", is_baseline=False):
    # if not os.path.isfile(input_dir_path):
    #    raise ValueError(f"Invalid path to the file: {input_dir_path}")
    for fname in os.listdir(input_dir_path):
        df = pd.read_csv(os.path.join(input_dir_path, fname))
        settings = list(df["setting"])
        if metric == "f1":
            scores = list(df["f1"])
        elif metric == "accuracy":
            scores = list(df["accuracy"])

        lang = fname.split("_")[0]
        if lang not in lang2avg_scores:
            lang2avg_scores[lang] = dict()
            lang2std[lang] = dict()
            for setting in ALL_SETTINGS:
                lang2avg_scores[lang][setting] = 0.0
                lang2std[lang][setting] = 0.0

        setting2score = dict()
        for setting, score in zip(settings, scores):
            if is_baseline:
                setting = "gold"
            elif "only_summarized_intent" in setting:
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

        if is_baseline:
            settings_to_check = BASELINE_SETTINGS
        else:
            settings_to_check = GENERATION_SETTINGS
        for setting in settings_to_check:
            if setting not in setting2score:
                raise ValueError(f"Setting {setting} is missing in the file {fname}!")
            else:
                scores_per_setting = setting2score[setting]
                lang2avg_scores[lang][setting] = round(np.mean(scores_per_setting), 3)
                lang2std[lang][setting] = round(np.std(scores_per_setting), 5)


def collect_statistics(input_dir_path, metric):
    lang2avg_scores = dict()
    lang2std = dict()
    # collecting baseline results with the gold data
    baseline_dir = os.path.join("/".join(input_dir_path.split("/")[:-2]), "baseline")

    # collecting results with the generated data, if baseline is not available, we skip it
    try:
        process_results(baseline_dir, lang2avg_scores, lang2std, metric, is_baseline=True)
    except Exception as e:
        print(e)
    process_results(input_dir_path, lang2avg_scores, lang2std, metric, is_baseline=False)

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
