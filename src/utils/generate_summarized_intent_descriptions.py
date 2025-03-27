import argparse
import random
from typing import Dict, List

import pandas as pd
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

random.seed(2024)


def generate_intent_descriptions(
    model_name: str, input_path: str, output_path: str, num_samples_per_intent: int, hf_token: str
):
    # set up the model and tokenizer
    config = AutoConfig.from_pretrained(model_name)
    config.quantization_config["disable_exllama"] = True
    if len(hf_token) > 0:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", config=config, token=hf_token
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", config=config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # read in intent2description.csv
    df = pd.read_csv(input_path, sep="\t")
    texts = df["text"].to_list()
    intents = df["intent"].to_list()
    intent2texts: Dict[str, List[str]] = dict()
    for text, intent in zip(texts, intents):
        if intent not in intent2texts:
            intent2texts[intent] = []
        intent2texts[intent].append(text)

    # ask LLM to summarize k randomly selected samples
    sorted_intents = sorted(intent2texts.keys())  # sorting class labels
    out_intents = []
    out_descriptions = []
    for intent in sorted_intents:
        out_intents.append(intent)
        samples = intent2texts[intent]
        random.shuffle(samples)
        examples = "; ".join(samples[:num_samples_per_intent])
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
        # print(messages)
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
        # clean up generated outputs
        model_output = outputs[0]["generated_text"][len(prompt) :]
        model_output = model_output.replace("\n", " ").replace('"', "")
        if "." in model_output:
            model_output = model_output[: model_output.rindex(".") + 1]
        if ":" in model_output:
            model_output = model_output[model_output.index(":") + 1 :]
        model_output = model_output.strip()
        print(f"{intent}: {model_output}")
        out_descriptions.append(model_output)

    # write into file
    df = pd.DataFrame.from_dict({"intent": out_intents, "description": out_descriptions})
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parameters for generating summarized intent descriptions."
    )
    parser.add_argument(
        "--model_name", type=str, default="TechxGenus/Meta-Llama-3-70B-Instruct-GPTQ"
    )
    parser.add_argument("--input_path", type=str, default="data/massive-en/en-US_train.csv")
    parser.add_argument(
        "--output_path", type=str, default="src/utils/intent2description_summarized.csv"
    )
    parser.add_argument("--num_samples_per_intent", type=int, default=10)
    parser.add_argument("--hf_token", type=str, default="")
    args = parser.parse_args()

    model_name = args.model_name
    input_path = args.input_path
    output_path = args.output_path
    num_samples_per_intent = args.num_samples_per_intent
    hf_token = args.hf_token

    generate_intent_descriptions(
        model_name, input_path, output_path, num_samples_per_intent, hf_token
    )
