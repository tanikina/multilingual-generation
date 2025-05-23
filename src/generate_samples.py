import argparse
import os
import random
import re
import sys
from os.path import abspath, basename, dirname, isfile
from pathlib import Path

import pandas as pd
import torch
from vllm import LLM, SamplingParams

parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from class_labels import (
    MASSIVE10_LABELS,
    MASSIVE60_LABELS,
    SENTIMENT_LABELS,
    SIB200_LABELS,
)

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

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["HF_TOKEN"] = HF_TOKEN

device = "cuda" if torch.cuda.is_available() else "cpu"

FIRST_NUMBER_PATTERN = r"^(\d+)(?:\.|\))\s"  # to match "1. text" or "1) text" generated outputs
QUOTES = ["‘", "’", "”", "“", "„", '"', "'", "*"]


def self_check(
    new_demo,
    lang_name,
    class_name,
    class_description,
    pipeline,
    terminators,
    use_vllm,
    vllm_model,
    max_new_tokens=64,
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
        self_check_sampling_params = SamplingParams(
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.2,
            max_tokens=max_new_tokens,
        )
        decoded_outputs = vllm_model.chat(messages, self_check_sampling_params)
        decoded = decoded_outputs[0].outputs[0].text  # use single prompt!
    else:

        prompt = pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        outputs = pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
        )

        decoded = outputs[0]["generated_text"]
        decoded = decoded[len(prompt) :].replace("\n", " ").strip()

    if "yes" in decoded.lower():
        return (True, decoded)
    else:
        return (False, decoded)


def valid_sample(demo):
    # this is just a heuristic to filter out sentences with unusual length
    if len(demo) < 10 or len(demo) > 100:
        return False
    return True


def clean_up_quotes(text):
    for quote in QUOTES:
        if quote in text:
            text = text.replace(quote, "")
    return text


def remove_first_number(text):
    match = re.match(FIRST_NUMBER_PATTERN, text)
    if match:
        return text[match.end() :]
    else:
        return text


def generate_demos(args):
    # prepare the parameters
    language = args.language
    lang_name = lang_name_map[language]
    if args.dataset == "massive10":
        labels = MASSIVE10_LABELS
    elif args.dataset == "massive60":
        labels = MASSIVE60_LABELS
    elif args.dataset == "sib200":
        labels = SIB200_LABELS
    elif args.dataset == "sentiment":
        labels = SENTIMENT_LABELS
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
    threshold_per_class = args.num_samples_to_generate
    num_input_demos = args.num_input_demos

    do_self_check = args.do_self_check
    use_vllm = args.use_vllm
    with_label_explanation = args.with_label_explanation
    use_simple_explanations = args.use_simple_explanations

    model_name = args.model_name
    verbose = args.verbose

    # Generation

    # loading models and tokenizers
    if use_vllm:
        # setting up vllm generation
        vllm_model = LLM(
            model=model_name,
            tensor_parallel_size=1,
            max_model_len=4096,
        )  # use dtype="float16" if GPU has compute capacity < 8
        vllm_sampling_params = SamplingParams(
            temperature=0.4, top_p=0.9, repetition_penalty=1.2, max_tokens=512
        )
    else:
        if "aya" in model_name.lower() or "qwen" in model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
            ).eval()
        elif "llama" in model_name.lower():
            config = AutoConfig.from_pretrained(model_name)
            config.quantization_config["disable_exllama"] = True
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                config=config,
            ).eval()
        elif "gemma" in model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
            ).eval()
        else:
            raise ValueError(
                "Unsupported model name {model_name}. Should be either Llama, Qwen, Gemma or Aya model."
            )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        vllm_model = None
        vllm_sampling_params = None

    self_demonstrations = []
    self_annotations = []

    # storing unfiltered responses when using self-check
    self_demonstrations_non_revised = []
    self_annotations_non_revised = []
    self_check_explanations = []
    self_check_annotations = []

    # loading the mapping between the intent labels and their explanations
    label2explanation = dict()
    if use_simple_explanations:  # (non-summarized, human-written descriptions)
        explanation_fname = "src/utils/intent2description.csv"
    else:
        explanation_fname = args.summarized_explanation_fname
    df_explanations = pd.read_csv(explanation_fname)
    exp_labels = df_explanations["intent"].to_list()
    explanations = df_explanations["description"].to_list()
    for label, explanation in zip(exp_labels, explanations):
        label2explanation[label] = explanation

    # selecting random demonstrations per class
    # and constructing the prompt
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
            # truncate each example to max 128 tokens
            examples = [" ".join(example.split()[:128]) for example in examples]
            # check that the prompt length does not exceed max model length
            prompt_length_check_success = False
            while not prompt_length_check_success:
                self_generation_prompt = f"You are required to produce {num_samples_to_generate} examples in {lang_name} that can have the label: {class_name} {added_explanation} Note that some examples from the dataset look as follows:\nExamples:\n{examples}\nNow generate {num_samples_to_generate} similar examples for the label {class_name}. Each example should be on a new line. Do not generate anything that cannot be classified as {class_name} and do not repeat the instruction.\nGenerated examples for label {class_name}:\n"
                if use_vllm:
                    prompt_length_check_success = len(
                        vllm_model.get_tokenizer().encode(self_generation_prompt)
                    ) < (
                        vllm_model.llm_engine.vllm_config.model_config.max_model_len - 100
                    )  # 100 is the margin for the system prompt
                else:
                    prompt_length_check_success = len(tokenizer.encode(self_generation_prompt)) < (
                        model.config.max_position_embeddings - 100
                    )  # 100 is the margin for the system prompt, see https://stackoverflow.com/questions/76547541/huggingface-how-do-i-find-the-max-length-of-a-model#77286207
                if not prompt_length_check_success:
                    examples = examples[
                        :-1
                    ]  # reduce the number of examples until the valid length is reached
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

        # setting up pipeline generation
        if not use_vllm:
            pipeline = transformers.pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                model_kwargs={"torch_dtype": torch.bfloat16},
            )

            max_new_tokens = 512

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

        # collecting generated samples for each class
        while len(self_demonstrations_per_class) < threshold_per_class:
            if not use_vllm:
                outputs = pipeline(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.4,
                    top_p=0.9,
                )

                decoded = outputs[0]["generated_text"][len(prompt) :]
            else:
                decoded = vllm_model.chat(messages, vllm_sampling_params)[0].outputs[0].text

            try:
                # splitting generated text (vllm sometimes outputs a list instead of putting each sample on a new line)
                if "\n" in decoded:
                    decoded = decoded.split("\n")
                else:
                    if "', '" in decoded:
                        decoded = decoded.split("', '")
                    elif '", "' in decoded:
                        decoded = decoded.split('", "')

                # cleaning up generated samples, removing quotes
                decoded = [clean_up_quotes(item) for item in decoded if len(item) > 0]
                # skip the first one since it is typically "Here are x examples..."
                if len(decoded) > 1:
                    decoded = decoded[1:]

                if verbose:
                    print("DECODED:", decoded)
                # removing numbering (e.g. "1. text")
                demos_to_check = [remove_first_number(item) for item in decoded]

                demos_to_check = [demo.strip() for demo in demos_to_check if valid_sample(demo)]

                # the last entry is often truncated, thus we skip it
                if len(set(demos_to_check)) > 1:
                    demos_to_check = demos_to_check[:-1]

                # revising generated samples
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
                # adding non-revised demonstrations to a different list
                if do_self_check:
                    self_demonstrations_per_class_non_revised.extend(demos_to_check)
            except Exception as e:
                print("Failed decoding!", e)
                continue

        self_demonstrations_per_class = list(set(self_demonstrations_per_class))[
            :threshold_per_class
        ]
        self_demonstrations.extend(self_demonstrations_per_class)
        for i in range(len(self_demonstrations_per_class)):
            self_annotations.append(class_name)

        # with self-check we store both the "checked" and the originally generated samples with explanations
        if do_self_check:
            self_demonstrations_per_class_non_revised = self_demonstrations_per_class_non_revised[
                :threshold_per_class
            ]
            self_check_explanations_per_class = self_check_explanations_per_class[
                :threshold_per_class
            ]
            self_check_annotations_per_class = self_check_annotations_per_class[
                :threshold_per_class
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
    Path(output_path).parent.absolute().mkdir(parents=True, exist_ok=True)
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
    parser = argparse.ArgumentParser(description="Generation parameters.")
    parser.add_argument("--language", type=str, choices=list(lang_name_map.keys()))
    parser.add_argument("--input_path", type=str, default="data/de-massive/de-DE_train.csv")
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/generated/massive10/llama3_8b/de-DE_default_output.csv",
    )

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
    parser.add_argument(
        "--summarized_explanation_fname",
        type=str,
        default="src/utils/intent2description_summarized.csv",
    )
    parser.add_argument("--do_self_check", type=bool, default=False)
    parser.add_argument("--use_vllm", type=bool, default=False)
    parser.add_argument("--verbose", type=bool, default=False)

    args = parser.parse_args()
    print("Parameters:")
    for k, v in vars(args).items():
        print(k, v)
    generate_demos(args)
