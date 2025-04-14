import argparse
import os
from typing import List

import pandas as pd
from dataset_constants import (
    INTENT_MASSIVE,
    INTENT_SENTIMENT,
    INTENT_SIB200,
    LANG_MASSIVE,
    LANG_SENTIMENT,
    LANG_SIB200,
    SCENARIO_MASSIVE,
    sentiment_to_massive_lang_name,
    sib200_to_massive_lang_name,
)
from datasets import load_dataset


def save_as_csv(input_data: str, dataset_type: str, valid_intents: List[str], output_path: str):
    ids = []
    texts = []
    intents = []
    if dataset_type == "massive":
        # id, text, intent
        for item in input_data:
            assert isinstance(item, dict)
            intent = INTENT_MASSIVE[item["intent"]]
            if intent not in valid_intents:
                continue
            intents.append(intent)
            # scenario = SCENARIO_MASSIVE[item["scenario"]]  # TODO: shall we use this?
            ids.append(item["id"])
            texts.append(item["utt"])
    elif dataset_type == "sib200":
        # id, text, intent
        for item in input_data:
            assert isinstance(item, dict)
            intent = item["category"]
            if intent not in valid_intents:
                continue
            intents.append(intent)
            ids.append(item["index_id"])
            texts.append(item["text"])
    elif dataset_type == "sentiment":
        # label, text
        for idx, item in enumerate(input_data):
            assert isinstance(item, dict)
            intent = "positive" if item["label"] == 1 else "negative"
            if intent not in valid_intents:
                continue
            intents.append(intent)
            ids.append(idx)
            texts.append(item["text"])
    else:
        raise ValueError(f"Unknown dataset {dataset_type}.")
    df = pd.DataFrame.from_dict({"id": ids, "text": texts, "intent": intents})
    df.to_csv(output_path, index=False)


def prepare_massive(languages: List[str], intents: List[str]):
    # sanity check
    for lang in languages:
        if lang not in LANG_MASSIVE:
            raise ValueError(f"Incorrect language {lang}.")
    for intent in intents:
        if intent not in INTENT_MASSIVE:
            raise ValueError(f"Incorrect intent {intent}.")
    # download the data for each language
    for language in languages:
        lang_code = language.split("-")[0]
        lang_dir = os.path.dirname("data/" + lang_code + "-massive/")
        os.makedirs(lang_dir, exist_ok=True)
        lang_data = load_dataset("AmazonScience/massive", language)
        save_as_csv(
            lang_data["train"],
            valid_intents=intents,
            dataset_type="massive",
            output_path=os.path.join(lang_dir, language + "_train.csv"),
        )
        save_as_csv(
            lang_data["validation"],
            valid_intents=intents,
            dataset_type="massive",
            output_path=os.path.join(lang_dir, language + "_val.csv"),
        )
        save_as_csv(
            lang_data["test"],
            valid_intents=intents,
            dataset_type="massive",
            output_path=os.path.join(lang_dir, language + "_test.csv"),
        )


def prepare_sentiment(languages: List[str], intents: List[str]):
    # sanity check
    for lang in languages:
        if lang not in LANG_SENTIMENT:
            raise ValueError(f"Incorrect language {lang}.")
    for intent in intents:
        if intent not in INTENT_SENTIMENT:
            raise ValueError(f"Incorrect intent {intent}.")
    # download the data for each language
    for language in languages:
        normalized_lang_name = sentiment_to_massive_lang_name[language]
        lang_code = normalized_lang_name.split("-")[0]
        lang_dir = os.path.dirname("data/" + lang_code + "-sentiment/")
        os.makedirs(lang_dir, exist_ok=True)
        if normalized_lang_name in ["en-US", "de-DE"]:
            lang_data = load_dataset(f"sepidmnorozy/{language}")
        else:
            lang_data = load_dataset(f"DGurgurov/{language}")
        save_as_csv(
            lang_data["train"],
            valid_intents=intents,
            dataset_type="sentiment",
            output_path=os.path.join(lang_dir, normalized_lang_name + "_train.csv"),
        )
        save_as_csv(
            lang_data["validation"],
            valid_intents=intents,
            dataset_type="sentiment",
            output_path=os.path.join(lang_dir, normalized_lang_name + "_val.csv"),
        )
        save_as_csv(
            lang_data["test"],
            valid_intents=intents,
            dataset_type="sentiment",
            output_path=os.path.join(lang_dir, normalized_lang_name + "_test.csv"),
        )


def prepare_sib200(languages: List[str], intents: List[str]):
    # sanity check
    for lang in languages:
        if lang not in LANG_SIB200:
            raise ValueError(f"Incorrect language {lang}.")
    for intent in intents:
        if intent not in INTENT_SIB200:
            raise ValueError(f"Incorrect intent {intent}.")
    # download the data for each language
    for language in languages:
        normalized_lang_name = sib200_to_massive_lang_name[language]
        lang_code = normalized_lang_name.split("-")[0]
        lang_dir = os.path.dirname("data/" + lang_code + "-sib200/")
        os.makedirs(lang_dir, exist_ok=True)
        lang_data = load_dataset("Davlan/sib200", language)
        save_as_csv(
            lang_data["train"],
            valid_intents=intents,
            dataset_type="sib200",
            output_path=os.path.join(lang_dir, normalized_lang_name + "_train.csv"),
        )
        save_as_csv(
            lang_data["validation"],
            valid_intents=intents,
            dataset_type="sib200",
            output_path=os.path.join(lang_dir, normalized_lang_name + "_val.csv"),
        )
        save_as_csv(
            lang_data["test"],
            valid_intents=intents,
            dataset_type="sib200",
            output_path=os.path.join(lang_dir, normalized_lang_name + "_test.csv"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parameters for generating summarized intent descriptions."
    )
    parser.add_argument("--dataset", type=str, default="massive")
    parser.add_argument("--languages", nargs="+", default=[])
    parser.add_argument("--intents", nargs="+", default=[])

    args = parser.parse_args()

    dataset = args.dataset.lower()
    languages = args.languages
    intents = args.intents

    if dataset == "massive":
        prepare_massive(languages, intents)
    elif dataset == "sib200":
        prepare_sib200(languages, intents)
    elif dataset == "sentiment":
        prepare_sentiment(languages, intents)
    else:
        raise NotImplementedError(
            f"Dataset can be either massive, sentiment, or sib200, the provided dataset name {dataset} is invalid."
        )
