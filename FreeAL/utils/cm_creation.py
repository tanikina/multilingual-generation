from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score

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

# TODO: pass as parameter
gold = []
predicted = []

cm = confusion_matrix(
    y_true=[id2label[item] for item in gold],
    y_pred=[id2label[item] for item in predicted],
    labels=intent_labels,
)

shortened_labels = []
for lbl in intent_labels:
    shortened_labels.append("_".join([el[:5] for el in lbl.split("_")]))

cmd = ConfusionMatrixDisplay(cm, display_labels=shortened_labels)
cmd.plot(xticks_rotation="vertical")
fig = cmd.figure_
fig.tight_layout()
fig.savefig("conf_matrix.png")
