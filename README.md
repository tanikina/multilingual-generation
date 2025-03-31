# Multilingual generation

## Motivation

[Xiao et al. (2023)](https://aclanthology.org/2023.emnlp-main.896/) proposed an approach to leverage large (LLM) and small (SLM) language models to iteratively annotate the data from the unlabeled pool without any involvement of human annotators. However, the first step requires the creation of demonstrations "from scratch" which is feasible with large pre-trained models like ChatGPT given a small amount of distinct labels for simple categories (e.g., positive vs negative movie reviews). In this work, we are investigating whether this approach can scale to different languages, multiple labels and whether we can leverage open-sourced language models for the generation tasks.

### Research Questions 1 (Data Generation):

1. What are the best ways to generate samples "from scratch" **in a multilingual setting**? Which prompt strategies work the best and to what extent we need gold demonstrations and unlabeled data in the target language?
2. Which **filtering and selection methods** work well with the generated data? E.g., coreset-based selection vs. cartography etc.
3. Can we improve downstream performance by combining generated data with paraphrased demonstrations? Is it beneficial to **combine traditional data augmentation methods with generation**?
4. How can we **measure the quality of generated data** and what metrics are most beneficial for finding "good quality" samples.

### Contributions 1 (Data Generation):

1. A comprehensive analysis of different sample generation and selection strategies for multiple languages.
2. Benchmarking several SOTA models of different sizes on the dataset generation task.
3. Exploring different way of assessing data quality, finding outliers and noise in the data.

### Research Questions 2 (FreeAL):

1. Does FreeAL setting generalize to **different languages and multiple labels**?
2. To what extent can we **replace the available gold and silver data** with generated samples?
3. How can we mitigate the **effects of unbalanced distribution** that is crucial for good sample selection with FreeAL?

### Contributions 2 (FreeAL):

1. Benchmarking FreeAL approach in the multilingual setting with multiple labels and open-sourced models, identifying problems and proposing solutions for imbalanced class distribution and overlapping labels.
2. Assessing the impact of sample selection strategies in the FreeAL setting, and experimenting with different ways of consistency regularization (e.g., using LLM-paraphrasing instead of backtranslation).

## Generation setup

1. **Settings:**

   - only intent description with a summarized intent
   - intent description with 10 demos per class in English (random selection)
   - \[optional\] if revision helps, also try 10 demos per class in English (random selection) + revision
   - intent description with 10 demos per class in the target language (random selection)
   - intent description with 10 demos per class (random selection) + revision

2. **Models:**

   `google/gemma-3-4b-it`

   `google/gemma-3-27b-it`

   `TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ`

   `TechxGenus/Meta-Llama-3-70B-Instruct-GPTQ` \[optional\]

3. **Languages and Datasets:**

   Datasets:
   MASSIVE [(FitzGerald et al., 2023)](https://aclanthology.org/2023.acl-long.235/) and SIB-200 [(Adelani et al., 2024)](https://aclanthology.org/2024.eacl-long.14.pdf). See [`scripts/prepare_data.sh`](https://github.com/tanikina/multilingual-generation/blob/main/scripts/prepare_data.sh) for the script that extracts and prepares the data.

   Languages:

   ```
   mid-to-high-resourced: German, Thai, Hebrew, Indonesian, Swahili

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

   tested the same generation strategies with `gemma3-4b-it` on RTX3090 (24GB)

   **pipeline** approach (4 generation settings for German, 100 per class): 2h 11m

   **vllm** (4 generation settings for German, 100 per class): 1h 18m

**Important:** we use `pre-commit` to make sure that the code is formatted properly. Before pushing the changes to this repository, please run: `pre-commit run --all` to make sure that all checks pass. If changes are , please create a separate pull request.
