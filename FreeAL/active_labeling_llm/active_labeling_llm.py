import argparse
import os
import pickle
import re
from typing import List

import numpy as np
import pandas as pd
import torch
from gd_logit_processor import GuidedDecodingLogitsProcessor, GuidedParser
from grammar import intent_grammar_10
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

# TODO
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

st_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
grammar = intent_grammar_10
gen_model_name = "TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ"

if "GPTQ" in gen_model_name:
    quantization_config = GPTQConfig(bits=8, disable_exllama=True)
    gen_model = AutoModelForCausalLM.from_pretrained(
        gen_model_name,
        low_cpu_mem_usage=True,
        device_map="auto",
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
else:
    gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(gen_model_name)

tokenizer.pad_token = tokenizer.eos_token
gd_parser = GuidedParser(grammar, tokenizer, model="gpt")
gen_model.config.pad_token_id = gen_model.config.eos_token_id


def askLLM(messages: List[str]):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(
        gen_model.device
    )
    guided_preprocessor = GuidedDecodingLogitsProcessor(gd_parser, input_ids.input_ids.shape[1])
    with torch.no_grad():
        generation = gen_model.generate(
            input_ids=input_ids.input_ids,
            logits_processor=[guided_preprocessor],
            eos_token_id=gd_parser.eos_token,
            pad_token_id=gen_model.config.pad_token_id,
            max_new_tokens=64,
            penalty_alpha=0.6,
            do_sample=True,
            top_k=5,
            top_p=0.95,
            temperature=0.1,
            repetition_penalty=1.2,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(input_ids.input_ids, generation)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.replace(" [e]", "").strip()


def get_embeddings(texts: List[str], st_model: SentenceTransformer, **kwargs) -> List[float]:
    # replace newlines, which can negatively affect performance.
    texts = [text.replace("\n", " ") for text in texts]
    embeddings = st_model.encode(texts).tolist()
    return embeddings


def cosine_similarity(
    st_model: SentenceTransformer, a_input_embedding: List[float], b_input_embedding: List[float]
):
    return st_model.similarity(
        np.array(a_input_embedding, dtype=float), np.array(b_input_embedding, dtype=float)
    ).item()


parser = argparse.ArgumentParser()
parser.add_argument("--refinery", action="store_true", default=False)
args = parser.parse_args()

# embedding of self-generated demonstrations
generated_embeddings_path = "embedding/embedding_massive_gen.csv"
generated_samples_path = (
    "data/llama8b/best/ko-llama8b_summarized_intent_description_10demos_self_check.csv"
)
# "data/llama8b/best/is-llama8b_summarized_intent_description_10demos_self_check.csv"
# "data/llama8b/best/de-llama8b_no_intent_description_10demos.csv"
if not os.path.isfile(generated_embeddings_path):
    df = pd.read_csv(generated_samples_path, header=None)
    gen_samples = df[0].to_list()
    gen_labels = df[1].to_list()
    gen_embeddings = get_embeddings(gen_samples, st_model)
    gen_encoded = {"text": gen_samples, "embeddings": gen_embeddings, "anno": gen_labels}
    df = pd.DataFrame.from_dict(gen_encoded)
    df.to_csv(generated_embeddings_path, index=False)

df = pd.read_csv(generated_embeddings_path)
df["embeddings"] = df.embeddings.apply(eval).apply(np.array)

# embedding of the unlabeled training dataset
train_embeddings_path = "embedding/embedding_massive_train.csv"
train_samples_path = "data/gold/ko-KR_train_10-classes.csv"
if not os.path.isfile(train_embeddings_path):
    df = pd.read_csv(train_samples_path, sep="\t")
    train_samples = df["text"].to_list()
    train_labels = df["intent"].to_list()
    train_embeddings = get_embeddings(train_samples, st_model)
    train_encoded = {"text": train_samples, "embeddings": train_embeddings}
    df = pd.DataFrame.from_dict(train_encoded)
    df.to_csv(train_embeddings_path, index=False)

df_train = pd.read_csv(train_embeddings_path)
df_train["embeddings"] = df_train.embeddings.apply(eval).apply(np.array)

if args.refinery:
    print("in refinery annotation")
    with open("self_training_slm/feedback/right_list_massive.pkl", "rb") as f:
        right_list_all = pickle.load(f)  # clean sample idx

    with open("self_training_slm/feedback/demo_index_massive.pkl", "rb") as f:
        top_indices = pickle.load(f)  # demonstration retrieval by SLM"s embeddings

    with open("self_training_slm/feedback/pred_label_massive.pkl", "rb") as f:
        pred_labels = pickle.load(f)  # pseudo-labels by SLM
else:
    print("in initial annotation")
start_idx = 0
test_num = len(df_train) - start_idx
sel_list = range(start_idx, start_idx + test_num)
df_train = df_train.iloc[sel_list].reset_index(drop=True)
all_text = df_train["text"]
all_embeddings = df_train["embeddings"]


def search_intext(embedding, n=3):
    query_embedding = embedding
    df["similarity"] = df.embeddings.apply(
        lambda x: cosine_similarity(st_model, x, query_embedding)
    )
    results = df.sort_values("similarity", ascending=False, ignore_index=True)
    intext_results = results.head(n)
    return intext_results


all_response = []
all_sentences = []
batch_messages = []
batch_sentences = []
batch_index = []
count = 0
initial_tag = True
for i in sel_list:
    n = 10
    sentence = all_text[i]
    embedding = all_embeddings[i]
    if not args.refinery:
        # initial round: retrieval by bert embeddings
        intext_results = search_intext(embedding, n)
    else:
        # refinery round: retrieval by SLM
        ds_examples = df_train.iloc[top_indices[i]].reset_index(drop=True)
        ds_annos = [pred_labels[idx] for idx in top_indices[i]]

    sentence = sentence.replace('"', '"').replace("\n", "")
    sentence_to_llm = sentence

    sentences_example = []
    annos_example = []
    all_classes_str = ", ".join(all_class_names)
    temp_messages = [
        {
            "role": "system",
            "content": f"You are a helpful assistant for the task of intent text classification. Your task is to classify the input as belonging to one of the following classes {all_classes_str}",
        },
    ]
    for j in range(n):
        if (
            args.refinery
        ):  # TODO: e.g. sometimes we do not have all the classes labeled: len(ds_examples) < num_classes!
            if j < len(ds_examples):
                temp_sentence = ds_examples.iloc[j]["text"].replace("\n", "").replace('"', '"')
                temp_anno = ds_annos[j]
            else:
                continue
        else:
            temp_sentence = intext_results.iloc[j]["text"].replace("\n", "").replace('"', '"')
            temp_anno = intext_results.iloc[j]["anno"]
        sentences_example.append(temp_sentence)
        annos_example.append(temp_anno)
        temp_messages.append({"role": "user", "content": temp_sentence})
        temp_messages.append({"role": "assistant", "content": temp_anno})

    temp_messages.append({"role": "user", "content": sentence_to_llm})

    batch_index.append(i)
    batch_messages.append(temp_messages)
    batch_sentences.append(sentence)
    count += 1
    batch_messages = batch_messages
    if count == 100:  # set a budget to reduce the negative impact of exceptions in annotations
        response = []
        for msg in batch_messages:
            response.append(askLLM(msg))
        all_response += response
        all_sentences += batch_sentences

        for i in range(len(batch_sentences)):
            final_anno = response[i]  # ["choices"][0]["message"]["content"]
            print(final_anno)
            final_label = str(label2id[final_anno])
            if args.refinery and batch_index[i] in right_list_all:
                final_label = str(pred_labels[batch_index[i]])
            if initial_tag:
                with open("results/output_massive_train.txt", "w") as f_out:
                    f_out.write(final_label + "\n")
                    initial_tag = False
            else:
                with open("results/output_massive_train.txt", "a") as f_out:
                    f_out.write(final_label + "\n")

        batch_messages = []
        batch_sentences = []
        batch_index = []
        count = 0
if len(batch_messages):
    response = []
    for msg in batch_messages:
        response.append(askLLM(msg))
    all_response += response
    all_sentences += batch_sentences
    for i in range(len(batch_sentences)):
        final_anno = response[i]  # ["choices"][0]["message"]["content"]
        final_label = str(label2id[final_anno])  # originally: 0, 1, -1
        if args.refinery and batch_index[i] in right_list_all:
            # clean samples do not require reannotation
            final_label = str(pred_labels[batch_index[i]])

        if initial_tag:
            with open("results/output_massive_train.txt", "w") as f_out:
                f_out.write(final_label + "\n")
                initial_tag = False
        else:
            with open("results/output_massive_train.txt", "a") as f_out:
                f_out.write(final_label + "\n")
