import argparse
import os
import random
from os.path import isfile
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score

device = "cuda" if torch.cuda.is_available() else "cpu"

from string import punctuation

import torch.nn as nn
from datasets import Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from class_labels import (
    MASSIVE10_LABELS,
    MASSIVE60_LABELS,
    SENTIMENT_LABELS,
    SIB200_LABELS,
)


def write_into_file(texts, labels, fname):
    with open(fname, "w") as f:
        f.write("text\tlabels\n")
        for text, label in zip(texts, labels):
            f.write(text + "\t" + str(label) + "\n")


def remove_invalid_samples(in_texts, in_labels):
    # since the gold data sometimes has only punctuation and no text
    # (e.g. https://huggingface.co/datasets/DGurgurov/romanian_sa/viewer/default/test?p=2&row=223)
    # we remove such cases from training/evaluation
    out_texts = []
    out_labels = []
    for txt, lbl in zip(in_texts, in_labels):
        if len(txt) > 0:
            out_texts.append(txt)
            out_labels.append(lbl)
    return out_texts, out_labels


def balance_data(texts, labels, max_per_class):
    new_texts_with_labels = []
    label2texts = dict()
    for txt, lbl in zip(texts, labels):
        if lbl not in label2texts:
            label2texts[lbl] = []
        label2texts[lbl].append(txt)
    # collect the defined max_per_class number of samples
    # and shuffle the training data
    for lbl in label2texts:
        added_texts = label2texts[lbl][:max_per_class]
        for txt in added_texts:
            new_texts_with_labels.append((txt, lbl))
    random.shuffle(new_texts_with_labels)

    new_texts = []
    new_labels = []
    for txt, lbl in new_texts_with_labels:
        new_texts.append(txt)
        new_labels.append(lbl)

    return new_texts, new_labels


def normalize_text(input_texts):
    normalized_texts = []
    for input_text in input_texts:
        normalized_text = input_text.lower()
        normalized_text = "".join([ch for ch in normalized_text if ch not in punctuation])
        normalized_texts.append(normalized_text)
    return normalized_texts


def create_data(
    train_data_path,
    val_data_path,
    test_data_path,
    intent_labels,
    lang,
    num_labels=10,
    max_per_class=100,
    val_proportion=0.10,
    balanced=True,
    normalized=True,
):
    # we do not modify the test data distribution
    test_df = pd.read_csv(test_data_path)
    test_texts = list(test_df["text"])
    if normalized:
        test_texts = normalize_text(test_texts)
    test_labels = [intent_labels.index(lbl) for lbl in list(test_df["intent"])]
    test_texts, test_labels = remove_invalid_samples(test_texts, test_labels)

    train_df = pd.read_csv(train_data_path)
    train_texts = list(train_df["text"])
    if normalized:
        train_texts = normalize_text(train_texts)
    train_labels = [intent_labels.index(lbl) for lbl in list(train_df["intent"])]
    train_texts, train_labels = remove_invalid_samples(train_texts, train_labels)
    if balanced:
        train_texts, train_labels = balance_data(train_texts, train_labels, max_per_class)

    if len(val_data_path) > 0:
        val_df = pd.read_csv(val_data_path)
        val_texts = list(val_df["text"])
        if normalized:
            val_texts = normalize_text(val_texts)
        val_labels = [intent_labels.index(lbl) for lbl in list(val_df["intent"])]
    else:  # empty string, no validation set provided
        train_val_texts = list(train_df["text"])
        if normalized:
            train_val_texts = normalize_text(train_val_texts)
        train_val_labels = [intent_labels.index(lbl) for lbl in list(train_df["intent"])]
        if balanced:
            train_val_texts, train_val_labels = balance_data(
                train_val_texts, train_val_labels, max_per_class
            )
        # split into training and validation sets
        val_limit = round(len(train_val_texts) * val_proportion)
        train_texts = train_val_texts[val_limit:]
        train_labels = train_val_labels[val_limit:]
        val_texts = train_val_texts[:val_limit]
        val_labels = train_val_labels[:val_limit]

    # write into files
    os.makedirs("data/prepared_data_tmp", exist_ok=True)

    df = pd.DataFrame.from_dict({"text": test_texts, "labels": test_labels})
    df.to_csv("data/prepared_data_tmp/test_" + lang + ".tsv", sep="\t")

    df = pd.DataFrame.from_dict({"text": train_texts, "labels": train_labels})
    df.to_csv("data/prepared_data_tmp/train_" + lang + ".tsv", sep="\t")

    df = pd.DataFrame.from_dict({"text": val_texts, "labels": val_labels})
    df.to_csv("data/prepared_data_tmp/dev_" + lang + ".tsv", sep="\t")


def tokenize_function(data, tokenizer):
    return tokenizer(
        [doc_tokens for doc_i, doc_tokens in enumerate(data["text"])],
        pad_to_max_length=True,
        padding="max_length",
        max_length=128,  # TODO: adjust this depending on the dataset?
        truncation=True,
        add_special_tokens=True,
    )


def train(model, model_name, optimizer, train_loader, dev_loader, criterion, num_epochs):
    model.train()
    min_dev_loss = None
    no_improvement = 0
    patience = 5
    for epoch in range(num_epochs):
        total_loss = 0
        total_dev_loss = 0
        # training loop
        for batch in train_loader:
            for k, v in batch.items():
                batch[k] = v.to(device)
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # validation set evaluation
        with torch.no_grad():
            for batch in dev_loader:
                for k, v in batch.items():
                    batch[k] = v.to(device)
                outputs = model(**batch)
                loss = outputs.loss
                total_dev_loss += loss.item()

            if min_dev_loss is None or min_dev_loss > total_dev_loss:
                min_dev_loss = total_dev_loss
                no_improvement = 0
                torch.save(model.state_dict(), "saved_models/" + model_name + ".pt")
            elif min_dev_loss is not None:
                no_improvement += 1
            print(f"epoch {epoch} with validation loss {total_dev_loss}")
            if no_improvement > patience:
                return


def evaluate(
    model,
    tokenizer,
    test_loader,
    test_set,
    intent_labels,
    id2label,
    setting,
    language,
    eval_results_file,
    seed,
):
    model.eval()
    total_loss = 0
    total_acc = 0

    predictions_list = []
    expected_list = []

    with torch.no_grad():
        for batch in test_loader:
            for k, v in batch.items():
                batch[k] = v.to(device)
            outputs = model(**batch)
            labels = batch["labels"]
            loss = outputs.loss
            total_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)
            expected = batch["labels"].float()
            predictions_list.append(predictions)
            expected_list.append(expected)

            decoded = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            for sent, pred, gold in zip(decoded, predictions, labels):
                pred = intent_labels[pred.item()]
                gold = intent_labels[gold.item()]
                # if pred != gold:
                #    print(sent, pred, "gold:", gold)
            total_acc += (predictions == labels).sum().item()

    expected_list = torch.flatten(torch.cat(expected_list)).cpu().numpy()
    predictions_list = torch.flatten(torch.cat(predictions_list)).cpu().numpy()
    cm = confusion_matrix(
        y_true=[id2label[item] for item in expected_list],
        y_pred=[id2label[item] for item in predictions_list],
        labels=intent_labels,
    )
    shortened_labels = []
    for lbl in intent_labels:
        shortened_labels.append("_".join([el[:5] for el in lbl.split("_")]))

    cmd = ConfusionMatrixDisplay(cm, display_labels=shortened_labels)
    cmd.plot(xticks_rotation="vertical")
    fig = cmd.figure_
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig_fname = "figures/" + language + "_" + setting.split("/")[-1]
    fig.savefig(fig_fname.replace(".csv", ".png"))

    # print(pd.crosstab([id2label[item] for item in expected_list], [id2label[item] for item in predictions_list], rownames=["True"], colnames=["Predicted"], margins=True))
    print("Setting: " + setting)
    acc = round(total_acc / len(test_set) * 100, 2)
    print(f"Test loss: {round(total_loss/len(test_loader),5)}\nTest acc: {acc}%")
    f1 = f1_score(expected_list, predictions_list, average="macro")
    f1 = round(f1 * 100, 2)
    print(f"F1 score: {f1}%")

    # saving evaluation results into file
    if isfile(eval_results_file):
        header = False
    else:
        header = True
    Path(eval_results_file).parent.absolute().mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data={"setting": [setting], "seed": [seed], "f1": [f1], "accuracy": [acc]})
    df.to_csv(eval_results_file, mode="a", header=header, index=False)


def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)

    num_epochs = args.num_epochs  # 50
    batch_size = args.batch_size  # 16
    learning_rate = args.learning_rate  # 2e-5
    finetuned_model_name = args.finetuned_model_name  # e.g. "gold_baseline"
    base_model_name = args.base_model_name  # e.g.

    num_labels = args.num_labels  # e.g. 10 for MASSIVE
    max_per_class = args.max_per_class  # 100
    balanced = args.balanced  # True
    normalized = args.normalized  # True

    lang = args.lang  # "de-DE"
    lang_prefix = lang.split("-")[0]
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    val_data_path = args.val_data_path
    eval_results_file = args.eval_results_file

    if args.dataset == "massive10":
        intent_labels = MASSIVE10_LABELS
    elif args.dataset == "massive60":
        intent_labels = MASSIVE60_LABELS
    elif args.dataset == "sib200":
        intent_labels = SIB200_LABELS
    elif args.dataset == "sentiment":
        intent_labels = SENTIMENT_LABELS
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    id2label = dict()
    for idx, label in enumerate(intent_labels):
        id2label[idx] = label

    # we always specify the language as part of the saved model name
    if not finetuned_model_name.startswith(f"{lang_prefix}_"):
        finetuned_model_name = lang_prefix + "_" + finetuned_model_name

    # saving subsampled data in a temporary directory
    create_data(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        test_data_path=test_data_path,
        intent_labels=intent_labels,
        lang=lang,
        num_labels=num_labels,
        max_per_class=max_per_class,
        balanced=balanced,
        normalized=normalized,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    train_set = Dataset.from_csv("data/prepared_data_tmp/train_" + lang + ".tsv", delimiter="\t")
    train_set = train_set.map(
        lambda x: tokenize_function(x, tokenizer), batched=True, batch_size=batch_size
    )
    train_set.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    dev_set = Dataset.from_csv("data/prepared_data_tmp/dev_" + lang + ".tsv", delimiter="\t")
    dev_set = dev_set.map(
        lambda x: tokenize_function(x, tokenizer), batched=True, batch_size=batch_size
    )
    dev_set.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    test_set = Dataset.from_csv("data/prepared_data_tmp/test_" + lang + ".tsv", delimiter="\t")
    test_set = test_set.map(
        lambda x: tokenize_function(x, tokenizer), batched=True, batch_size=batch_size
    )
    test_set.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=num_labels
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)  # 1e-5

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    os.makedirs("saved_models", exist_ok=True)

    train(model, finetuned_model_name, optimizer, train_loader, dev_loader, criterion, num_epochs)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=num_labels
    )

    model.load_state_dict(torch.load("saved_models/" + finetuned_model_name + ".pt"))
    model.to(device)
    evaluate(
        model=model,
        tokenizer=tokenizer,
        test_loader=test_loader,
        test_set=test_set,
        intent_labels=intent_labels,
        id2label=id2label,
        setting=train_data_path,
        language=lang_prefix,
        eval_results_file=eval_results_file,
        seed=seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training or evaluation parameters.")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_per_class", type=int, default=100)
    parser.add_argument("--num_labels", type=int, default=10)
    parser.add_argument("--normalized", action="store_true")  # remove punctuation, lowercase
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--base_model_name", type=str, default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--finetuned_model_name", type=str, default="model")
    parser.add_argument("--lang", type=str, default="de-DE")
    parser.add_argument(
        "--dataset",
        type=str,
        default="massive10",
        choices=["massive10", "massive60", "sib200", "sentiment"],
    )
    parser.add_argument("--train_data_path", type=str, default="data/de-massive/de-DE_train.csv")
    parser.add_argument("--test_data_path", type=str, default="data/de-massive/de-DE_test.csv")
    parser.add_argument("--val_data_path", type=str, default="")
    parser.add_argument(
        "--eval_results_file", type=str, default="results/massive10/de_summary.csv"
    )
    args = parser.parse_args()
    print(args)
    main(args)
