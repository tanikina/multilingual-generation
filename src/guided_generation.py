"""Guided Decoding logits processor."""
import argparse
import copy
import re

import numpy as np
import pandas as pd
import torch
from lark import Lark
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"


class_grammar_10 = r"""
?start: action
action: operation done

done: " [e]"

operation: op1 |op2 |op3 |op4 |op5 |op6 |op7 |op8 |op9 |op10

op1: " alarm_query"
op2: " audio_volume_down"
op3: " calendar_remove"
op4: " cooking_recipe"
op5: " datetime_convert"
op6: " email_sendemail"
op7: " play_audiobook"
op8: " recommendation_movies"
op9: " transport_ticket"
op10: " weather_query"
"""

class_grammar_all = r"""
?start: action
action: operation done

done: " [e]"

operation: op1 |op2 |op3 |op4 |op5 |op6 |op7 |op8 |op9 |op10 |op11 |op12 |op13 |op14 |op15 |op16 |op17 |op18 |op19 |op20 |op21 |op22 |op23 |op24 |op25 |op26 |op27 |op28 |op29 |op30 |op31 |op32 |op33 |op34 |op35 |op36 |op37 |op38 |op39 |op40 |op41 |op42 |op43 |op44 |op45 |op46 |op47 |op48 |op49 |op50 |op51 |op52 |op53 |op54 |op55 |op56 |op57 |op58 |op59

op1: " alarm_query"
op2: " alarm_remove"
op3: " alarm_set"
op4: " audio_volume_down"
op5: " audio_volume_mute"
op6: " audio_volume_up"
op7: " calendar_query"
op8: " calendar_remove"
op9: " calendar_set"
op10: " cooking_query"
op11: " cooking_recipe"
op12: " datetime_convert"
op13: " datetime_query"
op14: " email_addcontact"
op15: " email_query"
op16: " email_querycontact"
op17: " email_sendemail"
op18: " general_greet"
op19: " general_joke"
op20: " general_quirky"
op21: " iot_cleaning"
op22: " iot_coffee"
op23: " iot_hue_lightchange"
op24: " iot_hue_lightdim"
op25: " iot_hue_lightoff"
op26: " iot_hue_lighton"
op27: " iot_hue_lightup"
op28: " iot_wemo_off"
op29: " iot_wemo_on"
op30: " lists_createoradd"
op31: " lists_query"
op32: " lists_remove"
op33: " music_dislikeness"
op34: " music_likeness"
op35: " music_query"
op36: " music_settings"
op37: " news_query"
op38: " play_audiobook"
op39: " play_game"
op40: " play_music"
op41: " play_podcasts"
op42: " play_radio"
op43: " qa_currency"
op44: " qa_definition"
op45: " qa_factoid"
op46: " qa_maths"
op47: " qa_stock"
op48: " recommendation_events"
op49: " recommendation_locations"
op50: " recommendation_movies"
op51: " social_post"
op52: " social_query"
op53: " takeaway_order"
op54: " takeaway_query"
op55: " transport_query"
op56: " transport_taxi"
op57: " transport_ticket"
op58: " transport_traffic"
op59: " weather_query"
"""


class GuidedDecodingLogitsProcessor(LogitsProcessor):
    def __init__(self, parser, prompt_length, filter_value=-float("Inf"), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser = parser
        self.prompt_length = prompt_length
        self.filter_value = filter_value

    def __call__(self, input_ids, scores):
        valid_tokens = torch.ones_like(scores) * self.filter_value

        # The tokens generated so far
        for b in range(scores.shape[0]):
            generated_tokens = input_ids[b, self.prompt_length :].cpu().tolist()
            next_tokens = self.parser.next_tokens(generated_tokens)
            int_next_tokens = np.array([int(t) for t in next_tokens])

            # Adjust the scores to allow only valid tokens
            valid_tokens[b, int_next_tokens] = scores[b, int_next_tokens]
        return valid_tokens


class GuidedParser:
    """A class defining the mapping between text grammar and tokenized grammar."""

    def __init__(self, init_grammar, tokenizer, model, eos_token=None):

        # The grammar with natural language text
        self.text_grammar = init_grammar

        # The natural language parser
        self.text_parser = Lark(self.text_grammar, parser="lalr")

        # The hugging face tokenizer
        self.tokenizer = tokenizer

        # Store the model being used. This influences some decoding settings
        self.model = model

        # The grammar compiled with tokens from the hugging face tokenizer
        self.token_grammar = self._compile_grammar(self.text_grammar, self.tokenizer)

        # The tokenized parser
        self.token_parser = Lark(self.token_grammar, parser="lalr")

        self.terminal_lookup = {}

        for terminal in self.token_parser.terminals:
            self.terminal_lookup[terminal.name] = terminal.pattern.value

        if eos_token is None:
            if model == "t5":
                self.eos_token = tokenizer.encode(" [e]")[-2]
            elif model == "gpt":
                self.eos_token = tokenizer.encode(" [e]")[-1]
            else:
                raise NameError(f"don't know model {model}")
        else:
            self.eos_token = eos_token

    def _compile_grammar(self, grammar, tokenizer):
        """Compiles a grammar into tokens."""

        # Create the tokenizer grammar
        tokenized_grammar = copy.deepcopy(grammar)

        # Find all the terminals
        terminals = re.findall('"([^"]*)"', grammar)

        # Store existing terminals
        existing_terms = {}

        # Records the update rules for the terminals
        indx = 0
        for term in tqdm(terminals):
            tokens = tokenizer.encode(term)

            replacement_rule = "("
            for tok in tokens:
                if tok == 1 and self.model == "t5":
                    continue
                # If it already exists, we don't want to add
                # the terminal again, just use the old one
                if tok in existing_terms:
                    name = existing_terms[tok]
                else:
                    name = f"ANON{indx} "
                    indx += 1
                    newrule = name + ": " + '"' + str(tok) + '"'
                    tokenized_grammar += f"\n{newrule}"
                    existing_terms[tok] = name
                replacement_rule += name

            # Close the list of terminals
            replacement_rule += ")"

            # Update the terminal with the tokens
            tokenized_grammar = tokenized_grammar.replace('"' + term + '"', replacement_rule)

        tokenized_grammar += '\n%ignore " "'
        return tokenized_grammar

    def next_tokens(self, tokens):
        """Get the next tokens."""
        string_tokens = " ".join([str(t) for t in tokens])
        interactive = self.token_parser.parse_interactive(string_tokens)
        interactive.exhaust_lexer()
        return [self.terminal_lookup[acc] for acc in interactive.accepts()]


def generate_label(instance, lang, model, tokenizer, parser, class_labels):
    class_labels_str = ", ".join(class_labels)
    messages = [
        {"role": "system", "content": "You are an assistant for intent recognition in dialogue."},
        {
            "role": "user",
            "content": f"Provide the label for the given instance in {lang} from the following set: {class_labels_str}. Instance: {instance} Label:",
        },
    ]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        print(e)
        text = messages[0]["content"] + " " + messages[1]["content"]

    input_ids = tokenizer([text], return_tensors="pt").to(model.device)
    input_ids = input_ids.input_ids

    # guided_preprocessor = GuidedDecodingLogitsProcessor(parser, input_ids.shape[1])
    res = None
    with torch.no_grad():
        generation = model.generate(
            input_ids=input_ids,  # logits_processor=[guided_preprocessor],
            eos_token_id=parser.eos_token,
            pad_token_id=model.config.pad_token_id,
            max_new_tokens=64,
            penalty_alpha=0.6,
            do_sample=True,
            top_k=5,
            top_p=0.95,
            temperature=0.1,
            repetition_penalty=1.2,
        )
        generated_ids = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(input_ids, generation)
        ]
        res = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        res = res.replace("[e]", "").strip()
    return res


def main(args):
    # Set up the model and tokenizer (in this case using a quantized version of Llama, for which you'll need auto-gptq installed)
    model_name = args.model_name
    test_data_path = args.test_data_path
    lang = args.lang
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    if args.class_grammar == "10":
        class_grammar = class_grammar_10
        class_labels = [
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
    elif args.class_grammar == "all":
        class_grammar = class_grammar_all
        all_labels = [
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

        if args.eval_on_10intents:
            # we just add more labels for generation while keeping the test data of the setting with 10 classes > assign new ids to the extra labels (not used in the test set)
            class_labels = [
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
            for lbl in all_labels:
                if lbl not in class_labels:
                    class_labels.append(lbl)
        else:
            class_labels = all_labels

    else:
        raise ValueError(
            "Incorrect class_grammar argument: {class_grammar}. Must be either 10 or all."
        )

    idx2label = dict()
    for idx, label in enumerate(class_labels):
        idx2label[idx] = label

    parser = GuidedParser(
        class_grammar, tokenizer, model="gpt", eos_token=tokenizer.encode(" [e]")[-1]
    )

    df = pd.read_csv(test_data_path, sep="\t")
    instances = df["text"]
    gold_labels = [idx2label[lbl] for lbl in df["labels"]]

    # generate labels with GD
    gen_labels = []
    for instance in instances:
        label = generate_label(instance, lang, model, tokenizer, parser, class_labels)
        gen_labels.append(label)
        if args.verbose:
            print(instance, label)

    # compute f1 score
    lbl2scores = dict()
    print("gold", gold_labels)
    print("generated", gen_labels)
    gold_class_labels = list(
        set(gold_labels)
    )  # needed to discard labels that LLM can generate but they are not in the test set
    for lbl in gold_class_labels:
        lbl2scores[lbl] = {"tp": 0, "fp": 0, "fn": 0, "prec": 0, "rec": 0, "f1": 0}
    total_correct = 0
    for i in range(len(gold_labels)):
        gold_lbl = gold_labels[i]
        gen_lbl = gen_labels[i]
        if gold_lbl == gen_lbl:
            lbl2scores[gold_lbl]["tp"] += 1
            total_correct += 1
        else:
            lbl2scores[gold_lbl]["fn"] += 1
            if gen_lbl in gold_class_labels:
                lbl2scores[gold_lbl]["fp"] += 1
    # compute precision and recall per label
    f1_scores_per_class = []
    for lbl in gold_class_labels:
        # precision
        denom = lbl2scores[lbl]["tp"] + lbl2scores[lbl]["fp"]
        if denom > 0:
            lbl2scores[lbl]["prec"] = lbl2scores[lbl]["tp"] / denom
        # recall
        denom = lbl2scores[lbl]["tp"] + lbl2scores[lbl]["fn"]
        if denom > 0:
            lbl2scores[lbl]["rec"] = lbl2scores[lbl]["tp"] / denom
        # f1
        denom = lbl2scores[lbl]["prec"] + lbl2scores[lbl]["rec"]
        if denom > 0:
            lbl2scores[lbl]["f1"] = (2 * lbl2scores[lbl]["prec"] * lbl2scores[lbl]["rec"]) / denom
        print(lbl, round(lbl2scores[lbl]["f1"], 3))
        f1_scores_per_class.append(lbl2scores[lbl]["f1"])
    print("Macro F1:", round(sum(f1_scores_per_class) / len(f1_scores_per_class), 3))
    print("Accuracy:", round(total_correct / len(gen_labels), 3))
    # save generated labels in a file
    res_dict = {"text": instances, "labels": gen_labels}
    df = pd.DataFrame.from_dict(res_dict)
    df.to_csv(args.output_fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training or evaluation parameters.")
    parser.add_argument(
        "--model_name", type=str, default="TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ"
    )
    parser.add_argument("--class_grammar", type=str, default="10")
    parser.add_argument("--lang", type=str, default="German")
    parser.add_argument("--test_data_path", type=str, default="???")
    parser.add_argument("--output_fname", type=str, default="???")
    parser.add_argument("--eval_on_10intents", type=bool, default=False)
    parser.add_argument("--verbose", type=bool, default=False)
    args = parser.parse_args()
    main(args)
