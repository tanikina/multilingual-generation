import argparse
import csv
import os

import tqdm
from transformers import pipeline

MAX_LENGTH = 512  # 768

# Note that batch_size depends on the model:
# "Helsinki-NLP/opus-mt-ko-en" can have 16
# "Helsinki-NLP/opus-mt-tc-big-en-ko" needs batch_size 16

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="massive", help="massive")
args = parser.parse_args()
pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-ko-en", device=0)
pipe_bt = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-ko", device=0)
# pipe_bt = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ko",device=0)
with open(os.path.join("data_in", args.task, "train_chat_vanilla.csv"), "r") as file:
    reader = csv.reader(file)
    data = list(reader)

ori_text = []
for row in data:
    ori_text.append(row[1])

results = []
results_bt = []

for i in range(len(ori_text) // 16 + 1):
    print(i)
    tmp_text = (
        ori_text[i * 16 : (i + 1) * 16] if (i + 1) * 16 < len(ori_text) else ori_text[i * 16 :]
    )
    batch_result = pipe(tmp_text, batch_size=16, max_length=MAX_LENGTH)
    mid_text = [tmp_result["translation_text"] for tmp_result in batch_result]
    results += mid_text


for i in range(len(results) // 16 + 1):
    print(i)
    tmp_text = results[i * 16 : (i + 1) * 16] if (i + 1) * 16 < len(results) else results[i * 16 :]
    batch_result = pipe_bt(tmp_text, batch_size=16, max_length=MAX_LENGTH)
    bt_text = [tmp_result["translation_text"] for tmp_result in batch_result]
    results_bt += bt_text


for i in range(len(data)):
    row = data[i]
    if len(row) == 3:
        continue
    else:
        bt = results_bt[i]
        row.append(bt)

with open(os.path.join("data_in", args.task, "train_chat_new.csv"), "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)
