# A Rigorous Evaluation of Data Generation Strategies for Low-Resource Languages

![low_resources_synthetic_methodology2](https://github.com/user-attachments/assets/8f2420a3-a10a-4b0b-9d31-61abc051d965)

## Motivation

Large Language Models (LLMs) are increasingly used to generate synthetic textual data for training smaller specialized models. However, a comparison of various generation strategies for low-resource language settings is lacking.  While various prompting strategies have been proposed—such as demonstrations, label-based summaries, and self-revision—their comparative effectiveness remains unclear, especially for low-resource languages.

In this project, we systematically evaluate the performance of these generation strategies and their combinations across 11 typologically diverse languages, including several extremely low-resource ones. Using three NLP tasks and four open-source LLMs, we assess downstream model performance on generated versus gold-standard data.

Our results show that strategic combinations of generation methods—particularly target-language demonstrations with LLM-based revisions—yield strong performance, **narrowing the gap with real data to as little as 5% in some settings**. We also find that smart prompting techniques can reduce the **advantage of larger LLMs**, highlighting efficient generation strategies for synthetic data generation in low-resource scenarios with smaller models.

### Contributions:

1. We provide an exhaustive evaluation of different common strategies for **generating synthetic data for low-resource languages** and formulate suggestions for the most effective combination of strategies for extreme low-resource settings.
2. We confirm that while models with a higher number of parameters outperform their smaller counterparts on the generation task in the low-resource setting, **the gap in performance is small** when a right generation technique is used.
3. We show that - using the right combination of techniques, and for some configurations - **the drop of performance for a model trained on LLM-generated data is as small as up to only 5%** absolute when compared to the same number of "real" data.
4. We show that a **combination of demonstrations in the target language with LLM-based revisions** generally leads to the best performance across most languages, especially in extremely low-resource settings.

## Generation setup

1. **Generation Strategies:**

   - Summarized Label (SL)
   - EnglishDemos + SL
   - EnglishDemos + Revision
   - TargetDemos
   - TargetDemos + SL
   - TargetDemos + Revision
   - TargetDemos + SL + Revision

2. **Models:**

   [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it)

   [`google/gemma-3-27b-it`](https://huggingface.co/google/gemma-3-27b-it)

   [`TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ`](https://huggingface.co/TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ)

   [`TechxGenus/Meta-Llama-3-70B-Instruct-GPTQ`](https://huggingface.co/TechxGenus/Meta-Llama-3-70B-Instruct-GPTQ)

3. **Languages and Datasets:**

   Datasets:
   MASSIVE [(FitzGerald et al., 2023)](https://aclanthology.org/2023.acl-long.235/) and SIB-200 [(Adelani et al., 2024)](https://aclanthology.org/2024.eacl-long.14.pdf). See [`scripts/prepare_data.sh`](https://github.com/tanikina/multilingual-generation/blob/main/scripts/prepare_data.sh) for the script that extracts and prepares the data. For the sentiment task we use the data from [(Gurgurov et al., 2024)](https://aclanthology.org/2024.kallm-1.7/), [(Gurgurov et al., 2025)](https://aclanthology.org/2025.findings-naacl.67/), and [(Mollanorozy et al., 2023)](https://aclanthology.org/2023.sigtyp-1.9/).

   Languages:

   ```
   mid-to-high-resourced: Thai, Hebrew, Indonesian, Swahili, German, English

   low-resourced: Romanian, Azerbaijan, Slovenian, Telugu, Welsh
   ```

4. **Generation scripts for MASSIVE:**

   Note: if you want to modify `intent2description`, e.g., using a different amount of examples, or a different model (by default the descriptions are generated with Llama-70b), you can run the following code:

   ```
       python src/utils/generate_summarized_intent_descriptions.py \
       --model_name="TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ" \
       --output_path="src/utils/intent2description_summarized_llama8b.csv" \
       --num_samples_per_intent=10
   ```

   [`scripts/massive10/gemma3_4b`](https://github.com/tanikina/multilingual-generation/tree/main/scripts/massive10/gemma3_4b)

   [`scripts/massive10/gemma3_27b`](https://github.com/tanikina/multilingual-generation/tree/main/scripts/massive10/gemma3_27b)

   [`scripts/massive10/llama3_8b`](https://github.com/tanikina/multilingual-generation/tree/main/scripts/massive10/llama3_8b)

5. **Estimated running time** for vllm vs pipeline approach:

   tested the same generation strategy with `gemma3-4b-it` (L40S with 48GB)

   **pipeline** approach (self-revision generation setting for German, 100 per class): 44:08 min

   **vllm** (self-revision generation settings for German, 100 per class): 23:30 min

## Evaluation setup

The scripts to run the downstream evaluation with `FacebookAI/xlm-roberta-base` can be generated with `python src/utils/prepare_evaluation_scripts.py`. The resulting scripts will be stored in `scripts/downstream_evaluation`.

For each setting we fine-tune the model with 10 different seeds, computing F1 and accuracy scores. The results are stored in the `results/{dataset_name}/{model_name}` directory. E.g., the results for gemma3-4b on the MASSIVE data for German will be in `results/massive10/gemma3_4b/de_results.csv`.

We also store one confusion matrix per setting in the `figures` directory (the matrix is generated for each run, but since all the runs in the same setting have the same path, the matrix will be overwritten by the subsequent runs, so in practice we always have the confusion matrix corresponding to the last seed).

## Important

We use `pre-commit` to make sure that the code is formatted properly. Before pushing the changes to this repository, please run: `pre-commit run --all` to make sure that all checks pass. If changes are minor, you can commit them to the main branch, otherwise, please create a separate pull request.
