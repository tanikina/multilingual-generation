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
1. Benchmarking FreeAL approach in the multilingual setting with multiple labels and open-sourced models, identifying problems and proposing solutions for inbalanced class distribution and overlapping labels.
2. Assessing the impact of sample selection strategies in the FreeAL setting, and experimenting with different ways of consistency regularization (e.g., using LLM-paraphrasing instead of backtranslation).
