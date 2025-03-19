import pandas as pd

# dev/test/train.csv files have label index and text, no headers!
# train_chat_bt.csv file has label index, original train text, backtranslated train text

all_class_names = [
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

label2id = dict()
for _id, class_name in enumerate(all_class_names):
    label2id[class_name] = _id


def save_filtered_data(input_fname, output_fname, generated_labels_fname=None):
    """Filter out wrong classes and assign ids to class names."""
    df = pd.read_csv(input_fname, sep="\t")
    texts = df["text"]
    labels = df["intent"]
    filtered_texts = []
    filtered_labels = []
    for txt, lbl in zip(texts, labels):
        if lbl in all_class_names:
            filtered_texts.append(txt)
            filtered_labels.append(lbl)
    if generated_labels_fname:
        with open(generated_labels_fname) as f:
            filtered_labels = [line.strip() for line in f.readlines()]
    assert len(filtered_labels) == len(filtered_texts)
    filtered_df = pd.DataFrame.from_dict({"text": filtered_texts, "intent": filtered_labels})
    filtered_df.to_csv(output_fname, sep="\t", index=False)


save_filtered_data("data/gold/ko-KR_train.csv", "data/gold/ko-KR_train_10-classes.csv")
