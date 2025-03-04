import random
from typing import Dict, List

import pandas as pd
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

random.seed(2024)

# set up the model and tokenizer
model_id = "TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ"
config = AutoConfig.from_pretrained(model_id)
config.quantization_config["disable_exllama"] = True
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", config=config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# read in intent2description.csv
df = pd.read_csv("data/massive-en/en-US_train.csv", sep="\t")
texts = df["text"].to_list()
intents = df["intent"].to_list()
intent2texts: Dict[str, List[str]] = dict()
for text, intent in zip(texts, intents):
    if intent not in intent2texts:
        intent2texts[intent] = []
    intent2texts[intent].append(text)

# ask LLM to summarize 5 randomly selected samples
threshold = 5
sorted_intents = sorted(intent2texts.keys())
out_intents = []
out_descriptions = []
for intent in sorted_intents:
    out_intents.append(intent)
    samples = intent2texts[intent]
    random.shuffle(samples)
    examples = "; ".join(samples[:threshold])
    messages = [
        {
            "role": "system",
            "content": "You are an excellent summarizer and generate concise descriptions for different intents given some examples.",
        },
        {
            "role": "user",
            "content": f"Your task is to generate a short description of the intent {intent} based on the following examples from the dataset that share the intent {intent}: {examples} Description:",
        },
    ]
    print(messages)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        # device="auto",
    )

    prompt = pipeline.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=64,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    model_output = outputs[0]["generated_text"][len(prompt) :]
    model_output = model_output.replace("\n", " ").replace('"', "")
    if "." in model_output:
        model_output = model_output[: model_output.rindex(".") + 1]
    if ":" in model_output:
        model_output = model_output[model_output.index(":") + 1 :]
    model_output = model_output.strip()
    print(intent, model_output)
    out_descriptions.append(model_output)

# write into file
df = pd.DataFrame.from_dict({"intent": out_intents, "description": out_descriptions})
df.to_csv("src/utils/intent2description_summarized.csv", index=False)
