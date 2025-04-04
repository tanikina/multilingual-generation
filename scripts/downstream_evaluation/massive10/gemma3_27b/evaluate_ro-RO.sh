#!/bin/bash
seeds=(1 2 3 4 5 6 7 8 9 10)
for seed in ${seeds[@]}; do
    python src/train_roberta.py \
    --seed=$seed \
    --num_epochs=50 \
    --batch_size=16 \
    --learning_rate=1e-5 \
    --max_per_class=100 \
    --num_labels=10 \
    --base_model_name="FacebookAI/xlm-roberta-base" \
    --finetuned_model_name="ro_only_summarized_intent_model" \
    --lang="ro-RO" \
    --dataset="massive10" \
    --balanced \
    --normalized \
    --train_data_path="data/generated/massive10/gemma3_27b/ro-RO_only_summarized_intent.csv" \
    --test_data_path="data/ro-massive/ro-RO_test.csv" \
    --eval_results_file="results/massive10/gemma3_27b/ro_results.csv"
done
for seed in ${seeds[@]}; do
    python src/train_roberta.py \
    --seed=$seed \
    --num_epochs=50 \
    --batch_size=16 \
    --learning_rate=1e-5 \
    --max_per_class=100 \
    --num_labels=10 \
    --base_model_name="FacebookAI/xlm-roberta-base" \
    --finetuned_model_name="ro_summarized_intent_with_10_target_lang_demos_model" \
    --lang="ro-RO" \
    --dataset="massive10" \
    --balanced \
    --normalized \
    --train_data_path="data/generated/massive10/gemma3_27b/ro-RO_summarized_intent_with_10_target_lang_demos.csv" \
    --test_data_path="data/ro-massive/ro-RO_test.csv" \
    --eval_results_file="results/massive10/gemma3_27b/ro_results.csv"
done
for seed in ${seeds[@]}; do
    python src/train_roberta.py \
    --seed=$seed \
    --num_epochs=50 \
    --batch_size=16 \
    --learning_rate=1e-5 \
    --max_per_class=100 \
    --num_labels=10 \
    --base_model_name="FacebookAI/xlm-roberta-base" \
    --finetuned_model_name="ro_summarized_intent_with_10_english_demos_model" \
    --lang="ro-RO" \
    --dataset="massive10" \
    --balanced \
    --normalized \
    --train_data_path="data/generated/massive10/gemma3_27b/ro-RO_summarized_intent_with_10_english_demos.csv" \
    --test_data_path="data/ro-massive/ro-RO_test.csv" \
    --eval_results_file="results/massive10/gemma3_27b/ro_results.csv"
done
for seed in ${seeds[@]}; do
    python src/train_roberta.py \
    --seed=$seed \
    --num_epochs=50 \
    --batch_size=16 \
    --learning_rate=1e-5 \
    --max_per_class=100 \
    --num_labels=10 \
    --base_model_name="FacebookAI/xlm-roberta-base" \
    --finetuned_model_name="ro_summarized_intent_with_10_target_lang_demos_and_revision_model" \
    --lang="ro-RO" \
    --dataset="massive10" \
    --balanced \
    --normalized \
    --train_data_path="data/generated/massive10/gemma3_27b/ro-RO_summarized_intent_with_10_target_lang_demos_and_revision.csv" \
    --test_data_path="data/ro-massive/ro-RO_test.csv" \
    --eval_results_file="results/massive10/gemma3_27b/ro_results.csv"
done
