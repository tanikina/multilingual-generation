import json
import re
import sys

import jsonlines

massive_slot_pattern = re.compile(r"\[.*?]")

# Exatract dialog act and slot annotations for MWOZ and MASSIVE datasets
# Store in the CSV file with the following format:
# turn_id, turn_text, dialogue_act, slots

# Creates de, en, it splits for training, dev, test (csv files)
# 400 - test, 600 - train, 200 - dev


def preprocess_mwoz(input_data, output_dir):
    mwoz_de_files = ["woz_test_de.json", "woz_train_de.json", "woz_validate_de.json"]
    mwoz_it_files = ["woz_test_it.json", "woz_train_it.json", "woz_validate_it.json"]
    mwoz_en_files = ["woz_test_en.json", "woz_train_en.json", "woz_validate_en.json"]
    mwoz_files = mwoz_de_files + mwoz_it_files + mwoz_en_files
    for input_file in mwoz_files:
        output_file = output_dir + "/" + input_file.replace(".json", ".csv")
        input_file = input_data + "/" + input_file
        preprocess_mwoz_file(input_file, output_file)


# NB: system turns are not annotated with dialog acts (aka intents)
def preprocess_mwoz_file(input_file, output_file):
    f = open(input_file)
    data = json.load(f)
    f.close()
    total_count = len(data)
    turn_idx = 0
    for el in data:
        dialogue_idx = el["dialogue_idx"]
        for turn in el["dialogue"]:
            turn_acts = []
            turn_slot_types = []
            turn_slot_values = []
            bf_state = turn["belief_state"]
            for bf in bf_state:
                turn_acts.append(bf["act"])
                for slot in bf["slots"]:
                    turn_slot_types.append(slot[0])
                    turn_slot_values.append(slot[1])
            assert len(turn_slot_types) == len(turn_slot_values)
            if "transcript" in turn:
                turn_text = turn["transcript"]
            if "system_transcript" in turn:
                transcript_text = turn["system_transcript"]
            print(turn_text, ">>>", transcript_text)
            print(turn_acts)
            print(turn_slot_types)
            print(turn_slot_values)
            print()
            turn_idx += 1
    print(input_file, output_file, total_count)


def get_massive_slot_annotations(anno_text):
    slot_annotations = []
    tokens = []
    start_position = 0
    end_position = 0
    for match in massive_slot_pattern.finditer(anno_text):
        matched = match.group()
        start_position = match.start()
        if start_position > 0:
            pre_text = anno_text[end_position:start_position]
            for word in pre_text.split():
                slot_annotations.append("O")
                tokens.append(word)
        end_position = start_position + len(matched)
        res = matched[1:-1].split(" : ")
        assert len(res) == 2
        label = "_".join(res[0].split())
        text = res[1]
        for i, word in enumerate(text.split()):
            if i == 0:
                slot_annotations.append("B-" + label)
            else:
                slot_annotations.append("I-" + label)
            tokens.append(word)
    if end_position != len(anno_text):
        post_text = anno_text[end_position:]
        for word in post_text.split():
            slot_annotations.append("O")
            tokens.append(word)
    assert len(slot_annotations) == len(tokens)
    return slot_annotations, tokens


def preprocess_massive(input_data, output_dir):
    """
    MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages. MASSIVE is a parallel dataset of > 1M utterances across 51 languages with annotations for the Natural Language Understanding tasks of intent prediction and slot annotation. Utterances span 60 intents and include 55 slot types. MASSIVE was created by localizing the SLURP dataset, composed of general Intelligent Voice Assistant single-shot interactions.
    """

    langs = [
        "nl-NL",
        "el-GR",
        "ko-KR",
        "mn-MN",
        "vi-VN",
        "de-DE",
        "en-US",
        "pl-PL",
        "it-IT",
        "tr-TR",
        "is-IS",
    ]
    # more well-resourced with their own bert-base-uncased models
    # German, English, Polish, Italian, Turkish
    # less resourced, some w/o bert-base-uncased general purpose models
    # Icelandic, Mongolian, Korean (?)
    for lang in langs:
        out_prefix = output_dir + "/" + lang
        out_files = [out_prefix + "_train.csv", out_prefix + "_val.csv", out_prefix + "_test.csv"]
        train_samples = []
        dev_samples = []
        test_samples = []
        # read in the data
        in_file = input_data + "/" + lang + ".jsonl"
        with jsonlines.open(in_file) as reader:
            for sample in reader:
                partition = sample["partition"]
                intent = sample["intent"]
                anno_text = sample["annot_utt"]
                slot_annotations, tokens = get_massive_slot_annotations(anno_text)
                # print(slot_annotations, ">>>", tokens, ">>>", sample["annot_utt"])
                sample_line = (
                    sample["id"]
                    + "\t"
                    + " ".join(tokens)
                    + "\t"
                    + intent
                    + "\t"
                    + " ".join(slot_annotations)
                )
                if partition == "train":
                    train_samples.append(sample_line)
                elif partition == "dev":
                    dev_samples.append(sample_line)
                elif partition == "test":
                    test_samples.append(sample_line)
                else:
                    raise Exception("Unknown partition:", partition)
        # writing the annotations
        for out_file in out_files:
            if "train" in out_file:
                sample_lines = train_samples
            elif "val" in out_file:
                sample_lines = dev_samples
            elif "test" in out_file:
                sample_lines = test_samples
            else:
                raise Exception("Unknown partition", out_file)
            print("Writing to ...", out_file)
            with open(out_file, "w") as f:
                f.write("id\ttext\tintent\tslots\n")
                for sample_line in sample_lines:
                    print(sample_line)
                    f.write(sample_line + "\n")


def process_xsid_file(in_file, out_file):
    samples = []
    with open(in_file) as f:
        print("Reading ...", in_file)
        data = f.readlines()
        sample = None
        next_id = 0
        for line in data:
            if line.startswith("# id:") or (
                ("fixed" in in_file or "/en." in in_file) and line.startswith("# text:")
            ):
                if sample is not None:
                    sample_line = (
                        str(sample["id"])
                        + "\t"
                        + " ".join(sample["text"])
                        + "\t"
                        + sample["intent"]
                        + "\t"
                        + " ".join(sample["slots"])
                    )
                    assert len(sample["slots"]) == len(sample["text"])
                    samples.append(sample_line)
                sample = {"id": next_id, "text": [], "slots": []}
                next_id += 1
            elif line.startswith("# intent:"):
                sample["intent"] = line.strip().replace("# intent: ", "")
            elif not (line.startswith("#")) and len(line.strip()) > 0:
                split = line.strip().split("\t")
                slot = split[-1]
                sample["slots"].append(slot)
                word = split[1]
                sample["text"].append(word)
        if len(sample["text"]) > 0:
            sample_line = (
                str(sample["id"])
                + "\t"
                + " ".join(sample["text"])
                + "\t"
                + sample["intent"]
                + "\t"
                + " ".join(sample["slots"])
            )
            assert len(sample["slots"]) == len(sample["text"])
            samples.append(sample_line)
    # writing the annotations
    print("Writing to ...", out_file)
    with open(out_file, "w") as f:
        f.write("id\ttext\tintent\tslots\n")
        for sample in samples:
            # print(sample)
            f.write(sample + "\n")


def preprocess_xsid(input_data, output_dir):
    langs = ["de", "da", "en", "id", "it", "kk", "nl", "sr", "tr", "de-st"]
    for lang in langs:
        test_file = input_data + "/" + lang + ".test.conll"
        if lang == "en":
            train_file = input_data + "/" + lang + ".train.conll"
        elif lang == "de-st":
            train_file = input_data + "/de.projectedTrain.conll.fixed"
        else:
            train_file = input_data + "/" + lang + ".projectedTrain.conll.fixed"
        val_file = input_data + "/" + lang + ".valid.conll"
        test_data = process_xsid_file(test_file, output_dir + "/" + lang + "_test.csv")
        train_data = process_xsid_file(train_file, output_dir + "/" + lang + "_train.csv")
        val_data = process_xsid_file(val_file, output_dir + "/" + lang + "_val.csv")


if __name__ == "__main__":
    # preprocess_mwoz(input_data="mwoz/ontologies/data/woz", output_dir="data/mwoz")
    # preprocess_xsid(input_data="xsid/xSID-0.4", output_dir="data/xsid")
    preprocess_massive(
        input_data="massive/amazon-massive-dataset-1.1/1.1/data", output_dir="data/massive"
    )
