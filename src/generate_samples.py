import argparse
import random
import sys
from os.path import abspath, dirname, isfile

import pandas as pd
import torch

parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from guided_decoding.gd_logit_processor import (
    GuidedDecodingLogitsProcessor,
    GuidedParser,
)
from guided_decoding.grammar import intent_grammar_10, intent_grammar_60

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
            "content": f"Decide whether the following example belongs to the class {class_name} which means {class_description}. Answer yes if it belongs and represents a good sample (grammatically correct and complete) and no if it does not. Explain your answer in a concise way after generating yes or no. Input: {new_demo} Answer:",
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


def askLLM(message, tokenizer, model, parser, guided_preprocessor, classes):
    message += "\nSelect one of the following labels: " + ", ".join(classes)
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
            response = classes[0]
            print(f"Failed! {e}", tokenizer.decode(output[0]))
    return response


def valid_sample(demo):
    # this is just a heuristic to filter out sentences with unusual length
    if len(demo) < 10 or len(demo) > 100:
        return False
    # we also filter out cases with invalid start (e.g. rephrasing prompt instructions etc)
    for wrong_start in ["I", "Let", "Here", "First"]:
        if demo.startswith(wrong_start):
            return False
    return True


def generate_demos(args):
    # prepare the parameters
    generate_per_label = args.generate_per_label
    language = args.language
    if args.num_classes == 10:
        classes = [
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
        intent_grammar = intent_grammar_10
    else:
        classes = [
            "alarm_query",
            "alarm_remove",
            "alarm_set",
            "audio_volume_down",
            "audio_volume_mute",
            "audio_volume_other",
            "audio_volume_up",
            "calendar_query",
            "calendar_remove",
            "calendar_set",
            "cooking_query",
            "cooking_recipe",
            "datetime_convert",
            "datetime_query",
            "email_addcontact",
            "email_query",
            "email_querycontact",
            "email_sendemail",
            "general_greet",
            "general_joke",
            "general_quirky",
            "iot_cleaning",
            "iot_coffee",
            "iot_hue_lightchange",
            "iot_hue_lightdim",
            "iot_hue_lightoff",
            "iot_hue_lighton",
            "iot_hue_lightup",
            "iot_wemo_off",
            "iot_wemo_on",
            "lists_createoradd",
            "lists_query",
            "lists_remove",
            "music_dislikeness",
            "music_likeness",
            "music_query",
            "music_settings",
            "news_query",
            "play_audiobook",
            "play_game",
            "play_music",
            "play_podcasts",
            "play_radio",
            "qa_currency",
            "qa_definition",
            "qa_factoid",
            "qa_maths",
            "qa_stock",
            "recommendation_events",
            "recommendation_locations",
            "recommendation_movies",
            "social_post",
            "social_query",
            "takeaway_order",
            "takeaway_query",
            "transport_query",
            "transport_taxi",
            "transport_ticket",
            "transport_traffic",
            "weather_query",
        ]
        intent_grammar = intent_grammar_60

    input_path = args.input_path
    output_path = args.output_path
    df = pd.read_csv(input_path, delimiter="\t")
    demo_texts = list(df["text"])
    demo_labels = list(df["intent"])
    selected_texts = []
    class2demos = dict()
    if args.use_translated_demos:
        # read the prepared translations
        df_translated_demos = pd.read_csv(
            input_path.replace("train.csv", "demo_centroids_chatgpt.csv"), header=None
        )  # "demo_centroids.csv" are based on GoogleTranslate, otherwise we use ChatGPT translations
        demo_texts = df_translated_demos[0].to_list()
        demo_labels = df_translated_demos[1].to_list()
    for txt, lbl in zip(demo_texts, demo_labels):
        if lbl in classes:
            selected_texts.append(txt)
            if lbl not in class2demos:
                class2demos[lbl] = []
            class2demos[lbl].append(txt)
    random.shuffle(selected_texts)

    threshold = args.unlabeled_samples_threshold
    num_self_demonstrations = args.num_self_demonstrations
    num_gold_demos = args.num_gold_demos

    do_self_check = args.do_self_check
    with_label_explanation = args.with_label_explanation
    use_simple_explanations = args.use_simple_explanations
    use_gold_demos = args.use_gold_demos
    unlabeled_examples = "\n".join(selected_texts[:threshold])

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

    if generate_per_label:
        for class_name in classes:
            if use_gold_demos:
                class_demos = class2demos[class_name]
                random.shuffle(class_demos)
                class_demos = class_demos[:num_gold_demos]
                examples = class_demos
                diff_labels = ""
                but = ""
            else:
                examples = unlabeled_examples
                diff_labels = " (with different labels)"
                but = " but"

            if with_label_explanation:
                added_explanation = (
                    f"which has the following meaning: {label2explanation[class_name]}"
                )
            else:
                added_explanation = ""
            if len(examples) > 0:
                self_generation_prompt = f"You are required to produce {num_self_demonstrations} examples in {language} that can have the label: {class_name} {added_explanation} Note that some examples from the dataset{diff_labels} look as follows:\nExamples:\n{examples}\nNow generate {num_self_demonstrations} similar examples{but} for the label {class_name}. Each example should be on a new line. Do not generate anything that cannot be classified as {class_name}.\nGenerated examples for label {class_name}:\n"
            else:
                self_generation_prompt = f"You are required to produce {num_self_demonstrations} examples in {language} that can have the label: {class_name} {added_explanation}. Generate {num_self_demonstrations} examples for the label {class_name}. Each example should be on a new line. Do not generate anything that cannot be classified as {class_name}.\nGenerated examples for label {class_name}:\n"

            messages = [
                {
                    "role": "system",
                    "content": f"You are an excellent text generator and can generate representative text samples for the given class in {language}.",
                },
                {"role": "user", "content": self_generation_prompt},
            ]

            self_demonstrations_per_class = []

            while len(self_demonstrations_per_class) < num_self_demonstrations:

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
                    decoded = decoded.split("\n")[:num_self_demonstrations]
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
                                    print(
                                        "Good example (based on self-check):", new_demo, class_name
                                    )
                            else:
                                if verbose:
                                    print(
                                        "Bad example (based on self-check):", new_demo, class_name
                                    )
                    else:
                        new_demonstrations = demos_to_check
                    self_demonstrations_per_class.extend(new_demonstrations)
                except Exception as e:
                    print("Failed decoding!", e)
                    continue

            self_demonstrations_per_class = self_demonstrations_per_class[:num_self_demonstrations]

            self_demonstrations.extend(self_demonstrations_per_class)
            for i in range(len(self_demonstrations_per_class)):
                self_annotations.append(class_name)

            if len(self_annotations) != len(self_demonstrations):
                raise ValueError(
                    f"Mismatch per class! {len(self_annotations)} annotations and {len(self_demonstrations)} demonstrations."
                )
    else:
        if use_gold_demos:
            demo_examples = ""
            for cls in class2demos:
                demo_examples += (
                    "Label: "
                    + cls
                    + " Samples: "
                    + ", ".join(class2demos[cls][:num_gold_demos])
                    + " "
                )
        else:
            demo_examples = "\n".join(selected_texts[:threshold])

        if with_label_explanation:
            classes_str = ""
            for cls in classes:
                classes_str += cls + ": " + label2explanation[cls] + ", "
            classes_str = classes_str[:-2]
        else:
            classes_str = ", ".join(classes)

        self_generation_prompt = f"You are required to produce {num_self_demonstrations} examples in {language} with labels for the task of intent classification. The task is to classify a sentence using one of the following classes: {classes_str}. Note that {threshold} of the labeled samples in the dataset are as follows:\nExamples:\n{demo_examples}\nNow generate {num_self_demonstrations} similar examples with each example on a new line.\nExamples:\n"
        input_ids = tokenizer(self_generation_prompt, return_tensors="pt").input_ids.to(device)

        max_new_tokens = 512
        if "deepseek" in model_name.lower():
            max_new_tokens = 2500

        while len(self_demonstrations) < num_self_demonstrations:
            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    top_k=40,
                    max_new_tokens=512,
                )

            decoded = tokenizer.decode(output[0])
            try:
                decoded = decoded[len(self_generation_prompt) :].split("\n")[
                    :num_self_demonstrations
                ]
                decoded = [item for item in decoded if len(item) > 0]
                decoded = [
                    item[item.index(" ") + 1 :]
                    if item[0].replace(".", "").isdigit() and " " in item
                    else item
                    for item in decoded
                ]
            except Exception as e:
                print("Failed decoding!", e)
            self_demonstrations.extend(decoded)
            self_demonstrations = list(set(self_demonstrations))
        self_demonstrations = self_demonstrations[:num_self_demonstrations]

        # self-annotation with GuidedDecoding
        parser = GuidedParser(
            intent_grammar, tokenizer, model="gpt", eos_token=tokenizer.encode(" [e]")[-1]
        )
        guided_preprocessor = GuidedDecodingLogitsProcessor(parser, input_ids.shape[1])

        for demo_text in self_demonstrations:
            message = (
                "Generate a label from the following list: "
                + " ".join(classes)
                + "\nInput text: "
                + demo_text
                + "\nLabel: "
            )
            not_labeled = True
            while not_labeled:
                label_tokens = askLLM(
                    message, tokenizer, model, parser, guided_preprocessor, classes
                ).split()
                for label_token in label_tokens:
                    if label_token in classes:
                        self_annotations.append(label_token)
                        if verbose:
                            print(demo_text, ">>>", label_token)
                        not_labeled = False
                        break

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

    parser.add_argument("--generate_per_label", type=bool, default=False)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument(
        "--model_name", type=str, default="TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ"
    )
    parser.add_argument("--unlabeled_samples_threshold", type=int, default=20)
    parser.add_argument("--num_self_demonstrations", type=int, default=20)
    parser.add_argument("--use_gold_demos", type=bool, default=False)
    parser.add_argument("--use_translated_demos", type=bool, default=False)
    parser.add_argument("--num_gold_demos", type=int, default=2)
    parser.add_argument("--with_label_explanation", type=bool, default=False)
    parser.add_argument("--use_simple_explanations", type=bool, default=False)
    parser.add_argument("--do_self_check", type=bool, default=False)
    parser.add_argument("--verbose", type=bool, default=False)

    args = parser.parse_args()
    print("Parameters:")
    for k, v in vars(args).items():
        print(k, v)
    generate_demos(args)
