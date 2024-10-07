# Mirror-Consistency

## Overview

This project evaluates the performance of different large language models (LLMs) on the Mirror-Consistency metric using various datasets. The experiments are conducted using four major LLMs and can be customized by altering the model or dataset configurations.

**This work has been accepted to EMNLP 2024 as Short Findings.**

## Models

We have utilized four LLMs for our experiments:

1. **gpt3.5-turbo-0613** - For using this model, provide the corresponding API by replacing the `_gpt35_api` function in `model.py`.

2. **qwen-turbo** - This model also requires the corresponding API which should be replaced in the `_qwen_turbo_api` function in `model.py`.

3. **Llama3-8B-Instruct** - To use this model, download the [Hugging Face version of the model parameters](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and set the `model_path` parameter in `run.py` to the path of the downloaded model weights.

4. **Llama3-70B-Instruct** - Similar to the Llama3-8B model, ensure to download and correctly reference the model parameters in `run.py`.

To switch between these models, modify the `config.model_name` in `run.py`.

## Datasets

Our experiments utilize the following datasets:

- **[GSM8K](https://github.com/openai/grade-school-math)**: A dataset of grade-school math word problems to test arithmetic reasoning.

- **[SVAMP](https://github.com/arkilpatel/SVAMP)**: A dataset designed to test the robustness of mathematical problem-solving models.

- **[Date Understanding](https://github.com/google/BIG-bench/)**: A dataset focusing on the comprehension of date and time expressions in natural language.

- **[StrategyQA](https://github.com/eladsegal/strategyqa)**: A question-answering dataset that requires multi-hop reasoning and strategy.

To use a different dataset, update the `config.dataset_name` in `run.py`.

## Running Experiments

For Mirror-Consistency experiments:

1. **Set Up the Model**: Follow the instructions in the [Models](#models) section to configure the desired model.

2. **Modify Parameters**: Adjust `run.py` with the desired model and dataset parameters.

3. **Execute Script**: Run `python run.py` directly to start the experiments.
We also provide tools for detailed performance analysis in `complete_evaluate.py`.

## Additional Resources

- **`check_pipeline.ipynb`**: A Jupyter notebook that serves as a simple example of the generation process using the configured models.







