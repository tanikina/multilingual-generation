import argparse
import ast
import os
import random
import sys
from os.path import abspath, basename, dirname, isfile

import pandas as pd
import torch
from vllm import LLM, SamplingParams

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

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

random.seed(2024)

lang_name_map = {
    "en-US": "English",
    "de-DE": "German",
    "th-TH": "Thai",
    "he-IL": "Hebrew",
    "id-ID": "Indonesian",
    "sw-KE": "Swahili",
    "ro-RO": "Romanian",
    "az-AZ": "Azerbaijani",
    "sl-SL": "Slovenian",
    "te-IN": "Telugu",
    "cy-GB": "Welsh",
}

HF_TOKEN = ""  # HuggingFace token to access the models
hf_token_path = "src/hf_token.txt"
if not (isfile(hf_token_path)):
    raise Exception(f"{hf_token_path} does not exist!")
with open(hf_token_path) as f:
    HF_TOKEN = f.readlines()[0].strip()
    if not (HF_TOKEN.startswith("hf_")):
        raise ValueError(f"Invalid HF_TOKEN: {HF_TOKEN}.")

device = "cuda" if torch.cuda.is_available() else "cpu"


def self_check(
    new_demo,
    lang_name,
    class_name,
    class_description,
    pipeline,
    terminators,
    use_vllm,
    vllm_model,
    vllm_sampling_params,
):
    messages = [
        {
            "role": "system",
            "content": f"You are an excellent classifier and can reason whether a given sample in {lang_name} belongs to the class {class_name} or not.",
        },
        {
            "role": "user",
            "content": f"Decide whether the following example belongs to the class {class_name} which means {class_description}. Answer yes if it belongs and represents a good sample (grammatically correct and complete) and no if it does not. Answer no if the example is not in {lang_name}. Explain your answer in a concise way after generating yes or no. Input: {new_demo} Answer:",
        },
    ]

    if use_vllm:
        decoded_outputs = vllm_model.generate(
            messages[0]["content"] + " " + messages[1]["content"], vllm_sampling_params
        )
        decoded = decoded_outputs[0].outputs[0].text  # use single prompt!
    else:
        max_new_tokens = 64

        prompt = pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        outputs = pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        decoded = outputs[0]["generated_text"]
        decoded = decoded[len(prompt) :].lower().replace("\n", " ").strip()

    if "yes" in decoded:
        return (True, decoded)
    else:
        return (False, decoded)


def valid_sample(demo):
    # this is just a heuristic to filter out sentences with unusual length
    if len(demo) < 10 or len(demo) > 100:
        return False
    return True


def generate_demos(args):
    # prepare the parameters
    language = args.language
    lang_name = lang_name_map[language]
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
                f"/{lang_code}-", "/en-"
            )
        )
        demo_texts = df_english_demos["text"]
        demo_labels = df_english_demos["intent"]

    for txt, lbl in zip(demo_texts, demo_labels):
        if lbl in labels:
            if lbl not in class2demos:
                class2demos[lbl] = []
            class2demos[lbl].append(txt)

    num_samples_to_generate = (
        args.num_samples_to_generate + 2
    )  # because we remove the first and the last ones
    num_input_demos = args.num_input_demos

    do_self_check = args.do_self_check
    use_vllm = args.use_vllm
    with_label_explanation = args.with_label_explanation
    use_simple_explanations = args.use_simple_explanations

    model_name = args.model_name
    verbose = args.verbose

    # Generation

    if use_vllm:
        vllm_model = LLM(model=model_name, tensor_parallel_size=1, max_model_len=2048)
        vllm_sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    else:
        if "aya" in model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype="auto", device_map="auto", token=HF_TOKEN
            )
        elif "qwen" in model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", token=HF_TOKEN
            )
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
                "Unsupported model name {model_name}. Should be either Llama, Qwen, Gemma or Aya model."
            )

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    self_demonstrations = []
    self_annotations = []

    # storing unfiltered responses when using self-check
    self_demonstrations_non_revised = []
    self_annotations_non_revised = []
    self_check_explanations = []
    self_check_annotations = []

    label2explanation = dict()
    if use_simple_explanations:
        explanation_fname = "src/utils/intent2description.csv"
    else:
        explanation_fname = "src/utils/intent2description_summarized.csv"
    df_explanations = pd.read_csv(explanation_fname)
    exp_labels = df_explanations["intent"].to_list()
    explanations = df_explanations["description"].to_list()
    for label, explanation in zip(exp_labels, explanations):
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
            self_generation_prompt = f"You are required to produce {num_samples_to_generate} examples in {lang_name} that can have the label: {class_name} {added_explanation} Note that some examples from the dataset look as follows:\nExamples:\n{examples}\nNow generate {num_samples_to_generate} similar examples for the label {class_name}. Each example should be on a new line. Do not generate anything that cannot be classified as {class_name} and do not repeat the instruction.\nGenerated examples for label {class_name}:\n"
        else:
            self_generation_prompt = f"You are required to produce {num_samples_to_generate} examples in {lang_name} that can have the label: {class_name} {added_explanation}. Generate {num_samples_to_generate} examples for the label {class_name}. Each example should be on a new line. Do not generate anything that cannot be classified as {class_name} and do not repeat the instruction.\nGenerated examples for label {class_name}:\n"

        messages = [
            {
                "role": "system",
                "content": f"You are an excellent text generator and can generate representative text samples for the given class in {lang_name}.",
            },
            {"role": "user", "content": self_generation_prompt},
        ]

        self_demonstrations_per_class = []
        self_demonstrations_per_class_non_revised = []
        self_check_explanations_per_class = []
        self_check_annotations_per_class = []

        if not use_vllm:
            pipeline = transformers.pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                model_kwargs={"torch_dtype": torch.bfloat16},
            )

            max_new_tokens = 128

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
        else:
            pipeline = None
            terminators = []

        while len(self_demonstrations_per_class) < num_samples_to_generate:
            if not use_vllm:
                outputs = pipeline(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                )

                decoded = outputs[0]["generated_text"][len(prompt) :]
            else:
                decoded = (
                    vllm_model.generate(
                        messages[0]["content"] + " " + messages[1]["content"], vllm_sampling_params
                    )[0]
                    .outputs[0]
                    .text
                )

            try:
                if "\n" in decoded:
                    decoded = decoded.split("\n")
                elif decoded.startswith("['"):
                    if not decoded.endswith("']"):
                        decoded = decoded + "']"
                    decoded = ast.literal_eval(decoded)
                elif decoded.startswith('["'):
                    if not decoded.endswith('"]'):
                        decoded = decoded + '"]'
                    decoded = ast.literal_eval(decoded)

                decoded = [item for item in decoded if len(item) > 0]
                # skip the first one since it is typically "Here are x examples..."
                if len(decoded) > 1:
                    decoded = decoded[1:]

                if verbose:
                    print("DECODED:", decoded)
                demos_to_check = [
                    item[item.index(" ") + 1 :]
                    if item[0].replace(".", "").isdigit() and " " in item
                    else item
                    for item in decoded
                ]

                demos_to_check = [
                    demo.replace("*", "").replace('"', "").replace("'", "")
                    for demo in demos_to_check
                    if valid_sample(demo)
                ]

                # the last entry is often truncated, thus we skip it
                if len(set(demos_to_check)) > 1:
                    demos_to_check = list(set(demos_to_check[:-1]))

                if do_self_check:
                    new_demonstrations = []
                    for new_demo in demos_to_check:
                        self_check_passed, self_check_verdict = self_check(
                            new_demo,
                            lang_name,
                            class_name,
                            label2explanation[class_name],
                            pipeline,
                            terminators,
                            use_vllm,
                            vllm_model,
                            vllm_sampling_params,
                        )
                        if self_check_passed:
                            new_demonstrations.append(new_demo)
                            self_check_annotations_per_class.append(1)
                        else:
                            self_check_annotations_per_class.append(0)
                        self_check_explanations_per_class.append(self_check_verdict)
                else:
                    new_demonstrations = demos_to_check
                self_demonstrations_per_class.extend(new_demonstrations)
                # add non-revised demonstrations to keep the same samples
                if do_self_check:
                    self_demonstrations_per_class_non_revised.extend(demos_to_check)
            except Exception as e:
                print("Failed decoding!", e)
                continue

        self_demonstrations_per_class = self_demonstrations_per_class[:num_samples_to_generate]
        self_demonstrations.extend(self_demonstrations_per_class)
        for i in range(len(self_demonstrations_per_class)):
            self_annotations.append(class_name)

        # with self-check we store both the "checked" and the originally generated samples with explanations
        if do_self_check:
            self_demonstrations_per_class_non_revised = self_demonstrations_per_class_non_revised[
                :num_samples_to_generate
            ]
            self_check_explanations_per_class = self_check_explanations_per_class[
                :num_samples_to_generate
            ]
            self_check_annotations_per_class = self_check_annotations_per_class[
                :num_samples_to_generate
            ]
            self_demonstrations_non_revised.extend(self_demonstrations_per_class_non_revised)
            self_check_explanations.extend(self_check_explanations_per_class)
            self_check_annotations.extend(self_check_annotations_per_class)
            for i in range(len(self_demonstrations_per_class_non_revised)):
                self_annotations_non_revised.append(class_name)

        if len(self_annotations) != len(self_demonstrations):
            raise ValueError(
                f"Mismatch per class! {len(self_annotations)} annotations and {len(self_demonstrations)} demonstrations."
            )

    if verbose:
        print("****************************")
        print("Output:", self_demonstrations)

    # write into file
    df = pd.DataFrame(data={"text": self_demonstrations, "intent": self_annotations})
    df.to_csv(output_path, index=True, header=True)

    if do_self_check:
        df_non_revised = pd.DataFrame(
            data={
                "text": self_demonstrations_non_revised,
                "intent": self_annotations_non_revised,
                "explanation": self_check_explanations,
                "verdict": self_check_annotations,
            }
        )
        df_non_revised.to_csv(
            output_path.replace(".csv", "_with-rejected.csv"), index=True, header=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training parameters.")
    parser.add_argument("--language", type=str, choices=list(lang_name_map.keys()))
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
    parser.add_argument("--use_vllm", type=bool, default=False)
    parser.add_argument("--verbose", type=bool, default=False)

    args = parser.parse_args()
    print("Parameters:")
    for k, v in vars(args).items():
        print(k, v)
    generate_demos(args)
