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
    --finetuned_model_name="az_baseline_model" \
    --lang="az-AZ" \
    --dataset="massive10" \
    --balanced \
    --normalized \
    --train_data_path="data/az-massive/az-AZ_train.csv" \
    --test_data_path="data/az-massive/az-AZ_test.csv" \
    --eval_results_file="results/massive10/baseline/az_results.csv"
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
    --finetuned_model_name="cy_baseline_model" \
    --lang="cy-GB" \
    --dataset="massive10" \
    --balanced \
    --normalized \
    --train_data_path="data/cy-massive/cy-GB_train.csv" \
    --test_data_path="data/cy-massive/cy-GB_test.csv" \
    --eval_results_file="results/massive10/baseline/cy_results.csv"
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
    --finetuned_model_name="de_baseline_model" \
    --lang="de-DE" \
    --dataset="massive10" \
    --balanced \
    --normalized \
    --train_data_path="data/de-massive/de-DE_train.csv" \
    --test_data_path="data/de-massive/de-DE_test.csv" \
    --eval_results_file="results/massive10/baseline/de_results.csv"
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
    --finetuned_model_name="en_baseline_model" \
    --lang="en-US" \
    --dataset="massive10" \
    --balanced \
    --normalized \
    --train_data_path="data/en-massive/en-US_train.csv" \
    --test_data_path="data/en-massive/en-US_test.csv" \
    --eval_results_file="results/massive10/baseline/en_results.csv"
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
    --finetuned_model_name="he_baseline_model" \
    --lang="he-IL" \
    --dataset="massive10" \
    --balanced \
    --normalized \
    --train_data_path="data/he-massive/he-IL_train.csv" \
    --test_data_path="data/he-massive/he-IL_test.csv" \
    --eval_results_file="results/massive10/baseline/he_results.csv"
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
    --finetuned_model_name="id_baseline_model" \
    --lang="id-ID" \
    --dataset="massive10" \
    --balanced \
    --normalized \
    --train_data_path="data/id-massive/id-ID_train.csv" \
    --test_data_path="data/id-massive/id-ID_test.csv" \
    --eval_results_file="results/massive10/baseline/id_results.csv"
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
    --finetuned_model_name="ro_baseline_model" \
    --lang="ro-RO" \
    --dataset="massive10" \
    --balanced \
    --normalized \
    --train_data_path="data/ro-massive/ro-RO_train.csv" \
    --test_data_path="data/ro-massive/ro-RO_test.csv" \
    --eval_results_file="results/massive10/baseline/ro_results.csv"
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
    --finetuned_model_name="sl_baseline_model" \
    --lang="sl-SL" \
    --dataset="massive10" \
    --balanced \
    --normalized \
    --train_data_path="data/sl-massive/sl-SL_train.csv" \
    --test_data_path="data/sl-massive/sl-SL_test.csv" \
    --eval_results_file="results/massive10/baseline/sl_results.csv"
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
    --finetuned_model_name="sw_baseline_model" \
    --lang="sw-KE" \
    --dataset="massive10" \
    --balanced \
    --normalized \
    --train_data_path="data/sw-massive/sw-KE_train.csv" \
    --test_data_path="data/sw-massive/sw-KE_test.csv" \
    --eval_results_file="results/massive10/baseline/sw_results.csv"
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
    --finetuned_model_name="te_baseline_model" \
    --lang="te-IN" \
    --dataset="massive10" \
    --balanced \
    --normalized \
    --train_data_path="data/te-massive/te-IN_train.csv" \
    --test_data_path="data/te-massive/te-IN_test.csv" \
    --eval_results_file="results/massive10/baseline/te_results.csv"
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
    --finetuned_model_name="th_baseline_model" \
    --lang="th-TH" \
    --dataset="massive10" \
    --balanced \
    --normalized \
    --train_data_path="data/th-massive/th-TH_train.csv" \
    --test_data_path="data/th-massive/th-TH_test.csv" \
    --eval_results_file="results/massive10/baseline/th_results.csv"
done
