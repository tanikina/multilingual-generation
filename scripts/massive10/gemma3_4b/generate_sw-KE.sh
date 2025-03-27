#!/bin/sh
echo only_summarized_intent
python src/generate_samples.py \
--language=sw-KE \
--input_path=data/de-massive/sw-KE_train.csv \
--output_path=data/generated/massive10/gemma3_4b/sw-KE_only_summarized_intent.csv \
--dataset=massive10 \
--model_name=google/gemma-3-4b-it \
--num_samples_to_generate=100 \
--num_input_demos=0 \
--with_label_explanation=True \
--use_vllm=True
echo summarized_intent_with_10_target_lang_demos
python src/generate_samples.py \
--language=sw-KE \
--input_path=data/de-massive/sw-KE_train.csv \
--output_path=data/generated/massive10/gemma3_4b/sw-KE_summarized_intent_with_10_target_lang_demos.csv \
--dataset=massive10 \
--model_name=google/gemma-3-4b-it \
--num_samples_to_generate=100 \
--num_input_demos=10 \
--with_label_explanation=True \
--use_vllm=True
echo summarized_intent_with_10_english_demos
python src/generate_samples.py \
--language=sw-KE \
--input_path=data/de-massive/sw-KE_train.csv \
--output_path=data/generated/massive10/gemma3_4b/sw-KE_summarized_intent_with_10_english_demos.csv \
--dataset=massive10 \
--model_name=google/gemma-3-4b-it \
--num_samples_to_generate=100 \
--num_input_demos=10 \
--with_label_explanation=True \
--use_english_demos=True \
--use_vllm=True
echo summarized_intent_with_10_target_lang_demos_and_revision
python src/generate_samples.py \
--language=sw-KE \
--input_path=data/de-massive/sw-KE_train.csv \
--output_path=data/generated/massive10/gemma3_4b/sw-KE_summarized_intent_with_10_target_lang_demos_and_revision.csv \
--dataset=massive10 \
--model_name=google/gemma-3-4b-it \
--num_samples_to_generate=100 \
--num_input_demos=10 \
--with_label_explanation=True \
--do_self_check=True \
--use_vllm=True
