import argparse
import os
import random

import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score

device = "cuda" if torch.cuda.is_available() else "cpu"

import torch.nn as nn
from datasets import Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertModel

intent_labels = [
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
id2label = dict()
for idx, label in enumerate(intent_labels):
    id2label[idx] = label

# all original labels from the MASSIVE dataset
# intent_labels = ["alarm_query", "alarm_remove", "alarm_set", "audio_volume_down", "audio_volume_mute", "audio_volume_other", "audio_volume_up", "calendar_query", "calendar_remove", "calendar_set", "cooking_query", "cooking_recipe", "datetime_convert", "datetime_query", "email_addcontact", "email_query", "email_querycontact", "email_sendemail", "general_greet", "general_joke", "general_quirky", "iot_cleaning", "iot_coffee", "iot_hue_lightchange", "iot_hue_lightdim", "iot_hue_lightoff", "iot_hue_lighton", "iot_hue_lightup", "iot_wemo_off", "iot_wemo_on", "lists_createoradd", "lists_query", "lists_remove", "music_dislikeness", "music_likeness", "music_query", "music_settings", "news_query", "play_audiobook", "play_game", "play_music", "play_podcasts", "play_radio", "qa_currency", "qa_definition", "qa_factoid", "qa_maths", "qa_stock", "recommendation_events", "recommendation_locations", "recommendation_movies", "social_post", "social_query", "takeaway_order", "takeaway_query", "transport_query", "transport_taxi", "transport_ticket", "transport_traffic", "weather_query"]

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")


def get_text_and_labels(
    fname,
    collect_by_label=False,
    max_per_label=100,
    do_shuffle=False,
    skip_header=False,
    skip_index=False,
    separator=",",
):
    texts_with_labels = []
    df = pd.read_csv(fname, sep=separator, header=None)
    if skip_index and len(df.columns) > 2:
        texts = df[1].to_list()
        labels = df[2].to_list()
    else:
        texts = df[0].to_list()
        labels = df[1].to_list()
    if skip_header:
        labels = labels[1:]
        texts = texts[1:]
    for text, lbl in zip(texts, labels):
        text = str(text).strip()
        if lbl not in intent_labels or len(text) == 0:
            continue
        label = intent_labels.index(lbl)
        texts_with_labels.append((text, label))

    if collect_by_label:
        # texts_with_labels.sort(key = lambda x: x[1])
        label2texts = dict()
        for text, label in texts_with_labels:
            if label not in label2texts:
                label2texts[label] = []
            label2texts[label].append(text)
        for label in label2texts:
            all_texts = label2texts[label]
            random.shuffle(all_texts)
            label2texts[label] = all_texts[:max_per_label]
        texts_with_labels = [(text, label) for label in label2texts for text in label2texts[label]]
    if do_shuffle:
        random.shuffle(texts_with_labels)
    texts = []
    labels = []
    for i in range(len(texts_with_labels)):
        text, label = texts_with_labels[i]
        texts.append(text)
        labels.append(label)
    return texts, labels


def write_into_file(texts, labels, fname):
    with open(fname, "w") as f:
        f.write("text\tlabels\n")
        for text, label in zip(texts, labels):
            f.write(text + "\t" + str(label) + "\n")


def create_data(
    data_path,
    train_path,
    lang,
    val_proportion=0.25,
    num_labels=10,
    max_per_label=100,
):
    test_texts, test_labels = get_text_and_labels(
        os.path.join(data_path, lang + "_test.csv"),
        skip_header=True,
        skip_index=True,
        separator=",",
    )
    orig_train_texts, orig_train_labels = get_text_and_labels(
        train_path,
        collect_by_label=True,
        max_per_label=max_per_label,
        do_shuffle=True,
        skip_header=True,
        skip_index=True,
        separator=",",
    )

    val_limit = round(len(orig_train_texts) * val_proportion)
    train_texts = orig_train_texts[val_limit:]
    train_labels = orig_train_labels[val_limit:]
    val_texts = orig_train_texts[:val_limit]
    val_labels = orig_train_labels[:val_limit]
    if len(set(train_labels)) != num_labels:
        val_indices_to_remove = []
        for lbl_idx, lbl in enumerate(intent_labels):
            if lbl_idx not in train_labels:
                for j in range(len(val_texts)):
                    if val_labels[j] == lbl_idx:
                        train_texts.append(val_texts[j])
                        train_labels.append(val_labels[j])
                        val_indices_to_remove.append(j)
                        break
        new_val_texts = []
        new_val_labels = []
        for j in range(len(val_texts)):
            if j not in val_indices_to_remove:
                new_val_texts.append(val_texts[j])
                new_val_labels.append(val_labels[j])
        val_texts = new_val_texts
        val_labels = new_val_labels

    # write into files
    os.makedirs("data/prepared_data", exist_ok=True)

    df = pd.DataFrame.from_dict({"text": test_texts, "labels": test_labels})
    df.to_csv("data/prepared_data/test_" + lang + ".tsv", sep="\t")

    df = pd.DataFrame.from_dict({"text": train_texts, "labels": train_labels})
    df.to_csv("data/prepared_data/train_" + lang + ".tsv", sep="\t")

    df = pd.DataFrame.from_dict({"text": val_texts, "labels": val_labels})
    df.to_csv("data/prepared_data/dev_" + lang + ".tsv", sep="\t")


def tokenize_function(data):
    return tokenizer(
        [doc_tokens for doc_i, doc_tokens in enumerate(data["text"])],
        pad_to_max_length=True,
        padding="max_length",
        max_length=64,
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
            if no_improvement > patience:
                return


def evaluate(model, test_loader, test_set, criterion, setting, language):
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
    print(
        f"Test loss: {total_loss/len(test_loader)}\nTest acc: {round(total_acc/len(test_set)*100, 2)}%"
    )
    f1 = f1_score(expected_list, predictions_list, average="macro")
    print("F1 score:", round(f1 * 100, 2))


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    num_epochs = args.num_epochs  # 20
    batch_size = args.batch_size  # 32
    model_name = args.model_name  # "gold_baseline"
    lang = args.lang  # "de-DE"
    lang_prefix = lang.split("-")[0]
    num_labels = len(intent_labels)  # 10
    max_per_label = args.max_per_label  # 10

    train_path = args.train_path
    model_name = lang_prefix + "_" + model_name

    create_data(
        data_path="data/" + lang_prefix + "-massive",
        train_path=train_path,
        lang=lang,
        num_labels=num_labels,
        max_per_label=max_per_label,
    )
    train_set = Dataset.from_csv("data/prepared_data/train_" + lang + ".tsv", delimiter="\t")
    train_set = train_set.map(tokenize_function, batched=True, batch_size=batch_size)
    train_set.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    dev_set = Dataset.from_csv("data/prepared_data/dev_" + lang + ".tsv", delimiter="\t")
    dev_set = dev_set.map(tokenize_function, batched=True, batch_size=batch_size)
    dev_set.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    test_set = Dataset.from_csv("data/prepared_data/test_" + lang + ".tsv", delimiter="\t")
    test_set = test_set.map(tokenize_function, batched=True, batch_size=batch_size)
    test_set.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        "google-bert/bert-base-multilingual-cased", num_labels=num_labels
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5)  # 1e-5

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    train(model, model_name, optimizer, train_loader, dev_loader, criterion, num_epochs)

    model = AutoModelForSequenceClassification.from_pretrained(
        "google-bert/bert-base-multilingual-cased", num_labels=num_labels
    )

    model.load_state_dict(torch.load("saved_models/" + model_name + ".pt"))
    model.to(device)
    evaluate(model, test_loader, test_set, criterion, setting=train_path, language=lang_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training or evaluation parameters.")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_per_label", type=int, default=100)
    parser.add_argument("--model_name", type=str, default="model")
    parser.add_argument("--lang", type=str, default="de-DE")
    parser.add_argument("--train_path", type=str, default="???")
    args = parser.parse_args()
    main(args)
