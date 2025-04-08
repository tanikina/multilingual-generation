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
all_datasets = ["massive10", "sib200"]
all_gen_models = ["gemma3_4b", "gemma3_27b", "llama3_8b", "llama3_70b"]
all_settings = [
    "only_summarized_intent",
    "summarized_intent_with_10_target_lang_demos",
    "summarized_intent_with_10_english_demos",
    "summarized_intent_with_10_target_lang_demos_and_revision",
]


def main():
    for dataset in all_datasets:
        # dataset = "massive10"
        if dataset.startswith("massive"):
            dataset_base = "massive"
        else:
            dataset_base = dataset

        for gen_model_name in all_gen_models:
            # gen_model_name = "gemma3_4b"

            for lang in all_languages:
                # lang = "az-AZ"
                lang_prefix = lang.split("-")[0]

                out_path = (
                    f"scripts/downstream_evaluation/{dataset}/{gen_model_name}/evaluate_{lang}.sh"
                )
                out_str = "#!/bin/bash\nseeds=(1 2 3 4 5 6 7 8 9 10)\n"

                for setting in all_settings:
                    # setting = "only_summarized_intent"
                    fixed_template = """for seed in ${seeds[@]}; do
    python src/train_roberta.py \\
    --seed=$seed \\
    --num_epochs=50 \\
    --batch_size=16 \\
    --learning_rate=1e-5 \\
    --max_per_class=100 \\
    --num_labels=10 \\
    --base_model_name="FacebookAI/xlm-roberta-base" \\
    --balanced \\
    --normalized \\\n"""
                    out_str += fixed_template

                    finetuned_model_name = lang_prefix + "_" + setting + "_model_" + gen_model_name
                    train_data_path = (
                        "data/generated/"
                        + dataset
                        + "/"
                        + gen_model_name
                        + "/"
                        + lang
                        + "_"
                        + setting
                        + ".csv"
                    )
                    test_data_path = (
                        "data/" + lang_prefix + "-" + dataset_base + "/" + lang + "_test.csv"
                    )
                    eval_results_file = (
                        "results/"
                        + dataset
                        + "/"
                        + gen_model_name
                        + "/"
                        + lang_prefix
                        + "_results.csv"
                    )

                    flexible_template = (
                        f'    --finetuned_model_name="{finetuned_model_name}" \\\n'
                        f'    --lang="{lang}" \\\n'
                        f'    --dataset="{dataset}" \\\n'
                        f'    --train_data_path="{train_data_path}" \\\n'
                        f'    --test_data_path="{test_data_path}" \\\n'
                        f'    --eval_results_file="{eval_results_file}" \n'
                        "done\n"
                    )

                    out_str += flexible_template

                Path(out_path).parent.absolute().mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    f.write(out_str)


if __name__ == "__main__":
    main()
