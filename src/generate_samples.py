import argparse
import random
import sys
from os.path import abspath, basename, dirname, isfile

import pandas as pd
import torch

parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Gemma3ForCausalLM,
)

from class_labels import MASSIVE10_LABELS, MASSIVE60_LABELS, SIB200_LABELS

random.seed(2024)

HF_TOKEN = ""  # HuggingFace token to access the models
hf_token_path = "src/hf_token.txt"
if not (isfile(hf_token_path)):
    raise Exception(f"{hf_token_path} does not exist!")
with open(hf_token_path) as f:
    HF_TOKEN = f.readlines()[0].strip()
    if not (HF_TOKEN.startswith("hf_")):
        raise ValueError(f"Invalid HF_TOKEN: {HF_TOKEN}.")

device = "cuda" if torch.cuda.is_available() else "cpu"


def self_check(new_demo, language, class_name, class_description, pipeline, terminators):
    messages = [
        {
            "role": "system",
            "content": f"You are an excellent classifier and can reason whether a given sample in {language} belongs to the class {class_name} or not.",
        },
        {
            "role": "user",
            "content": f"Decide whether the following example belongs to the class {class_name} which means {class_description}. Answer yes if it belongs and represents a good sample (grammatically correct and complete) and no if it does not. Answer no if the example is not in {language}. Explain your answer in a concise way after generating yes or no. Input: {new_demo} Answer:",
        },
    ]
    max_new_tokens = 42
    if "deepseek" in pipeline.model.name_or_path:
        max_new_tokens = 2000
    #    system_content = messages[0]["content"]
    #    messages = messages[1:]
    #    messages[0]["content"] = system_content + " " + messages[0]["content"]

    prompt = pipeline.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    outputs = pipeline(
        prompt,
        max_new_tokens=max_new_tokens,  # 5
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    decoded = outputs[0]["generated_text"]
    if "deepseek" in pipeline.model.name_or_path:
        if "</think>" in decoded:
            decoded = decoded.split("</think>")[1].lower()
    else:
        decoded = decoded[len(prompt) :].lower()
    print(decoded)
    if "yes" in decoded:
        return True
    else:
        return False


def askLLM(message, tokenizer, model, parser, guided_preprocessor, labels):
    message += "\nSelect one of the following labels: " + ", ".join(labels)
    _input = tokenizer(message, return_tensors="pt")
    input_ids = _input.input_ids.to(device)
    response = ""

    with torch.no_grad():
        output = model.greedy_search(
            input_ids=input_ids,
            logits_processor=guided_preprocessor,
            eos_token_id=parser.eos_token,
            pad_token_id=model.config.pad_token_id,
        )
        try:
            response = tokenizer.decode(output[0]).split(message)[1]
        except Exception as e:
            response = labels[0]
            print(f"Failed! {e}", tokenizer.decode(output[0]))
    return response


def valid_sample(demo):
    # this is just a heuristic to filter out sentences with unusual length
    if len(demo) < 10 or len(demo) > 100:
        return False
    return True


def generate_demos(args):
    # prepare the parameters
    language = args.language
    if args.dataset == "massive10":
        labels = MASSIVE10_LABELS
    elif args.dataset == "massive60":
        labels = MASSIVE60_LABELS
    elif args.dataset == "sib-200":
        labels = SIB200_LABELS
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    input_path = args.input_path
    output_path = args.output_path
    df = pd.read_csv(input_path)
    demo_texts = list(df["text"])
    demo_labels = list(df["intent"])

    class2demos = dict()
    if args.use_translated_demos:
        # read the prepared translations
        df_translated_demos = pd.read_csv(
            input_path.replace("train.csv", "demo_centroids_chatgpt.csv"), header=None
        )  # "demo_centroids.csv" are based on GoogleTranslate, otherwise we use ChatGPT translations
        demo_texts = df_translated_demos[0].to_list()
        demo_labels = df_translated_demos[1].to_list()
    elif args.use_english_demos:
        # read English data
        lang_code = args.language.split("-")[0]
        df_english_demos = pd.read_csv(
            input_path.replace(basename(input_path), "en-US_train.csv").replace(
                f"/{lang_code}-{args.dataset}", "/en-{args.dataset}"
            )
        )
        demo_texts = df_english_demos["text"]
        demo_labels = df_english_demos["intent"]

    for txt, lbl in zip(demo_texts, demo_labels):
        if lbl in labels:
            if lbl not in class2demos:
                class2demos[lbl] = []
            class2demos[lbl].append(txt)

    num_samples_to_generate = args.num_samples_to_generate
    num_input_demos = args.num_input_demos

    do_self_check = args.do_self_check
    with_label_explanation = args.with_label_explanation
    use_simple_explanations = args.use_simple_explanations

    model_name = args.model_name
    verbose = args.verbose

    # Generation

    if "aya" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto", token=HF_TOKEN
        )
    elif "deepseek" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True, token=HF_TOKEN
        )
    elif "qwen" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=HF_TOKEN)
    elif "llama" in model_name.lower():
        config = AutoConfig.from_pretrained(model_name)
        config.quantization_config["disable_exllama"] = True
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", config=config, token=HF_TOKEN
        )
    elif "gemma" in model_name.lower():
        model = Gemma3ForCausalLM.from_pretrained(
            model_name, device_map="auto", token=HF_TOKEN
        ).eval()
    else:
        raise ValueError(
            "Unsupported model name {model_name}. Should be either Llama, Qwen or Aya model."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    self_demonstrations = []
    self_annotations = []

    label2explanation = dict()
    if use_simple_explanations:
        explanation_fname = "src/utils/intent2description.csv"
    else:
        explanation_fname = "src/utils/intent2description_summarized.csv"
    df_explanations = pd.read_csv(explanation_fname)
    labels = df_explanations["intent"].to_list()
    explanations = df_explanations["description"].to_list()
    for label, explanation in zip(labels, explanations):
        label2explanation[label] = explanation

    for class_name in labels:
        class_demos = class2demos[class_name]
        random.shuffle(class_demos)
        class_demos = class_demos[:num_input_demos]
        examples = class_demos

        if with_label_explanation:
            added_explanation = f"which has the following meaning: {label2explanation[class_name]}"
        else:
            added_explanation = ""
        if len(examples) > 0:
            self_generation_prompt = f"You are required to produce {num_samples_to_generate} examples in {language} that can have the label: {class_name} {added_explanation} Note that some examples from the dataset look as follows:\nExamples:\n{examples}\nNow generate {num_samples_to_generate} similar examples for the label {class_name}. Each example should be on a new line. Do not generate anything that cannot be classified as {class_name}.\nGenerated examples for label {class_name}:\n"
        else:
            self_generation_prompt = f"You are required to produce {num_samples_to_generate} examples in {language} that can have the label: {class_name} {added_explanation}. Generate {num_samples_to_generate} examples for the label {class_name}. Each example should be on a new line. Do not generate anything that cannot be classified as {class_name}.\nGenerated examples for label {class_name}:\n"

        messages = [
            {
                "role": "system",
                "content": f"You are an excellent text generator and can generate representative text samples for the given class in {language}.",
            },
            {"role": "user", "content": self_generation_prompt},
        ]

        self_demonstrations_per_class = []

        while len(self_demonstrations_per_class) < num_samples_to_generate:

            pipeline = transformers.pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                model_kwargs={"torch_dtype": torch.bfloat16},
            )

            max_new_tokens = 128
            if "deepseek" in model_name.lower():
                max_new_tokens = 2500
                # if "deepseek" in pipeline.model.name_or_path:
                #    system_content = messages[0]["content"]
                #    messages = messages[1:]
                #    messages[0]["content"] = system_content + " " + messages[0]["content"]

            prompt = pipeline.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            terminators = [pipeline.tokenizer.eos_token_id]

            if "aya" in model_name.lower():
                terminators.append(
                    pipeline.tokenizer.convert_tokens_to_ids("<|END_OF_TURN_TOKEN|>")
                )

            elif "llama" in model_name.lower():
                terminators.append(pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"))

            outputs = pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )

            decoded = outputs[0]["generated_text"][len(prompt) :]
            if verbose:
                print("DECODED before split:", decoded)

            try:
                if "</think>" in decoded:  # if using DeepSeek model
                    split = decoded.split("</think>")
                    if len(split) > 1:
                        decoded = split[1].strip()
                decoded = decoded.split("\n")[:num_samples_to_generate]
                decoded = [item for item in decoded if len(item) > 0]
                # skip the first one since it is typically "Here are x examples..."
                decoded = decoded[1:]
                if verbose:
                    print("DECODED after split:", decoded)
                demos_to_check = [
                    item[item.index(" ") + 1 :]
                    if item[0].replace(".", "").isdigit() and " " in item
                    else item
                    for item in decoded
                ]
                demos_to_check = [
                    demo.replace("*", "").replace('"', "")
                    for demo in demos_to_check
                    if valid_sample(demo)
                ]
                # the last entry is often truncated, thus we skip it
                demos_to_check = demos_to_check[:-1]

                if do_self_check:
                    new_demonstrations = []
                    for new_demo in demos_to_check:
                        if self_check(
                            new_demo,
                            language,
                            class_name,
                            label2explanation[class_name],
                            pipeline,
                            terminators,
                        ):
                            new_demonstrations.append(new_demo)
                            if verbose:
                                print("Good example (based on self-check):", new_demo, class_name)
                        else:
                            if verbose:
                                print("Bad example (based on self-check):", new_demo, class_name)
                else:
                    new_demonstrations = demos_to_check
                self_demonstrations_per_class.extend(new_demonstrations)
            except Exception as e:
                print("Failed decoding!", e)
                continue

        self_demonstrations_per_class = self_demonstrations_per_class[:num_samples_to_generate]

        self_demonstrations.extend(self_demonstrations_per_class)
        for i in range(len(self_demonstrations_per_class)):
            self_annotations.append(class_name)

        if len(self_annotations) != len(self_demonstrations):
            raise ValueError(
                f"Mismatch per class! {len(self_annotations)} annotations and {len(self_demonstrations)} demonstrations."
            )

    if verbose:
        print("****************************")
        print("Output:", self_demonstrations)

    # write into file
    df = pd.DataFrame(data={"text": self_demonstrations, "anno": self_annotations})
    df.to_csv(output_path, index=False, header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training parameters.")
    parser.add_argument("--language", type=str)
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)

    parser.add_argument("--dataset", type=str, default="massive10")
    parser.add_argument(
        "--model_name", type=str, default="TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ"
    )
    parser.add_argument("--num_samples_to_generate", type=int, default=100)
    parser.add_argument("--num_input_demos", type=int, default=10)
    parser.add_argument("--use_english_demos", type=bool, default=False)
    parser.add_argument("--use_translated_demos", type=bool, default=False)
    parser.add_argument("--with_label_explanation", type=bool, default=False)
    parser.add_argument("--use_simple_explanations", type=bool, default=False)
    parser.add_argument("--do_self_check", type=bool, default=False)
    parser.add_argument("--verbose", type=bool, default=False)

    args = parser.parse_args()
    print("Parameters:")
    for k, v in vars(args).items():
        print(k, v)
    generate_demos(args)
