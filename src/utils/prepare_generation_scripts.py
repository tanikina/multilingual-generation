import argparse
from pathlib import Path

all_languages = [
    "en-US",
    "de-DE",
    "th-TH",
    "he-IL",
    "id-ID",
    "sw-KE",
    "ro-RO",
    "az-AZ",
    "sl-SL",
    "te-IN",
    "cy-GB",
]
ALL_DATASETS = ["massive10", "sib200", "sentiment"]
ALL_GEN_MODELS = ["gemma3_4b", "gemma3_27b", "llama3_8b", "llama3_70b"]


def prepare_scripts(datasets, gen_models):
    for dataset in datasets:
        # dataset = "massive10"
        if dataset.startswith("massive"):
            summarized_explanation_fname = "src/utils/intent2description_summarized.csv"
        elif dataset == "sib200":
            summarized_explanation_fname = "intent2description_summarized_sib200_llama70b.csv"
        elif dataset == "sentiment":
            summarized_explanation_fname = "intent2description_summarized_sentiment.csv"
        else:
            raise ValueError(
                f"Unknown dataset: {dataset}. Can be either massive10, sentiment or sib200."
            )

        for gen_model_name in gen_models:
            if gen_model_name == "gemma3_4b":
                full_gen_model_name = "google/gemma-3-4b-it"
            elif gen_model_name == "gemma3_27b":
                full_gen_model_name = "google/gemma-3-27b-it"
            elif gen_model_name == "llama3_8b":
                full_gen_model_name = "TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ"
            elif gen_model_name == "llama3_70b":
                full_gen_model_name = "TechxGenus/Meta-Llama-3-70B-Instruct-GPTQ"
            else:
                raise ValueError(f"Incorrect model name: {gen_model_name}")

            for lang in all_languages:
                # lang = "az-AZ"
                lang_prefix = lang.split("-")[0]

                out_path = f"scripts/{dataset}/{gen_model_name}/generate_{lang}.sh"
                out_str = "#!/bin/sh\n"

                # only_summarized_intent
                out_str += f"python src/generate_samples.py \\\n\
--language={lang} \\\n\
--input_path=data/{lang_prefix}-{dataset}/{lang}_train.csv \\\n\
--output_path=data/generated/{dataset}/{gen_model_name}/{lang}_only_summarized_intent.csv \\\n\
--dataset={dataset} \\\n\
--model_name={full_gen_model_name} \\\n\
--num_samples_to_generate=100 \\\n\
--num_input_demos=0 \\\n\
--with_label_explanation=True \\\n\
--use_vllm=True \\\n\
--summarized_explanation_fname=src/utils/{summarized_explanation_fname}\n"
                # summarized_intent_with_10_target_lang_demos
                out_str += f"python src/generate_samples.py \\\n\
--language={lang} \\\n\
--input_path=data/{lang_prefix}-{dataset}/{lang}_train.csv \\\n\
--output_path=data/generated/{dataset}/{gen_model_name}/{lang}_summarized_intent_with_10_target_lang_demos.csv \\\n\
--dataset={dataset} \\\n\
--model_name={full_gen_model_name} \\\n\
--num_samples_to_generate=100 \\\n\
--num_input_demos=10 \\\n\
--with_label_explanation=True \\\n\
--use_vllm=True \\\n\
--summarized_explanation_fname=src/utils/{summarized_explanation_fname}\n"
                # summarized_intent_with_10_english_demos
                out_str += f"python src/generate_samples.py \\\n\
--language={lang} \\\n\
--input_path=data/{lang_prefix}-{dataset}/{lang}_train.csv \\\n\
--output_path=data/generated/{dataset}/{gen_model_name}/{lang}_summarized_intent_with_10_english_demos.csv \\\n\
--dataset={dataset} \\\n\
--model_name={full_gen_model_name} \\\n\
--num_samples_to_generate=100 \\\n\
--num_input_demos=10 \\\n\
--with_label_explanation=True \\\n\
--use_english_demos=True \\\n\
--use_vllm=True \\\n\
--summarized_explanation_fname=src/utils/{summarized_explanation_fname}\n"
                # summarized_intent_with_10_target_lang_demos_and_revision
                out_str += f"python src/generate_samples.py \\\n\
--language={lang} \\\n\
--input_path=data/{lang_prefix}-{dataset}/{lang}_train.csv \\\n\
--output_path=data/generated/{dataset}/{gen_model_name}/{lang}_summarized_intent_with_10_target_lang_demos_and_revision.csv \\\n\
--dataset={dataset} \\\n\
--model_name={full_gen_model_name} \\\n\
--num_samples_to_generate=100 \\\n\
--num_input_demos=10 \\\n\
--with_label_explanation=True \\\n\
--do_self_check=True \\\n\
--use_vllm=True \\\n\
--summarized_explanation_fname=src/utils/{summarized_explanation_fname}\n"

                Path(out_path).parent.absolute().mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    f.write(out_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parameters for preparing the generation scripts."
    )
    parser.add_argument("--datasets", nargs="+", default=["sentiment"])
    parser.add_argument("--gen_models", nargs="+", default=["gemma3_4b"])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.all:
        datasets = ALL_DATASETS
        gen_models = ALL_GEN_MODELS
    else:
        datasets = args.datasets
        gen_models = args.gen_models

    for dataset in datasets:
        assert dataset in ALL_DATASETS
    for model in gen_models:
        assert model in ALL_GEN_MODELS

    print(f"Preparing the scripts for the following datasets: {datasets}")
    print(f"Preparing the scripts for the following models: {gen_models}")

    prepare_scripts(datasets, gen_models)
