# Animal Harm Assessment (AHA) Benchmark

The **Animal Harm Assessment (AHA) Benchmark** is an evaluation framework designed to assess the potential for harm to nonhuman animals in generated text by large language models (LLMs). It evaluates LLM responses using model-based judgments.

For more information, see preprint:
https://arxiv.org/abs/2503.04804

---

## Overview

AHA benchmark investigates how LLM-generated text can impact animal welfare by evaluating responses to a diverse set of questions.

It is parameterized primarily by two files:
- `config/questions.csv`: questions posed to LLMs
- `config/dimensions.csv`: dimensions along which answers are judged

---

## Basic Commands (using Make)

*   `make install`: Install the package and dependencies using `uv`.
*   `make list-tasks`: Verify the `aha` task is visible to `inspect-ai`.
*   `make test-eval`: Run a small test evaluation (1 sample).

---


## Project Structure

```
.
├── aha/                  # Core inspect-ai task package
│   ├── __init__.py       # Package marker
│   ├── task.py           # Defines the @task evaluation logic (aha_evaluation) and scoring
├── analysis_scripts/     # Standalone scripts for post-hoc analysis
│   ├── results_summary.py
│   ├── results_summary_raw.py
│   └── pairwise_comparison.py
├── logs/                 # Default output directory for .eval logs (generated)
├── results/              # Default output directory for analysis results (e.g., combined CSVs) (generated)
├── README.md             # This file
├── Makefile              # Convenience commands
├── pyproject.toml        # Package definition and dependencies
└── ...                   # Other config/support files
```

---

## Running the benchmark with Inspect

**Running the Evaluation:**

1.  **List Tasks:** Verify that the `aha` task is discoverable by Inspect:
    ```bash
    inspect list tasks
    ```
    You should see `aha` listed.

2.  **Run Evaluation:** Execute the benchmark using the `inspect eval` command. Specify the task name (`aha`) and the model using the explicit `--model` flag.
    ```bash
    # Basic usage with a specific model
    inspect eval aha --model openai/gpt-4o
    ```

3.  **Customizing the Evaluation:** Use the `-T` flag to pass arguments to the `aha2_evaluation` task:

    ```bash
    # Override judges
    inspect eval aha --model google/gemini-1.5-pro -T judges='["openai/gpt-4o"]'

    # Set temperature parameters
    inspect eval aha --model openai/gpt-4o -T judge_temperature=0.0 -T model_temperature=0.7

    # Run a subset of samples
    inspect eval aha --model openai/gpt-4o --limit 10
    ```

4.  **Species Substitution:** The benchmark supports substituting animal references in the questions:

    ```bash
    # Replace all animal references with "dog" (singular) and "dogs" (plural)
    inspect eval aha --model openai/gpt-4o -T species_singular=dog -T species_plural=dogs

    # Example with other animals
    inspect eval aha --model openai/gpt-4o -T species_singular=elephant -T species_plural=elephants
    ```

    This will replace all `{species-singular:default}` occurrences with the specified singular term and all `{species-plural:default}` occurrences with the specified plural term, allowing you to test how models respond when questions reference different animals.

**Task Parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model` | string | Model to evaluate (passed via `--model` flag) | Required |
| `judges` | list | Models used for scoring responses | `["anthropic/claude-3-5-sonnet-20240620", "openai/gpt-4o"]` |
| `model_temperature` | float | Temperature for generation | 0.5 |
| `judge_temperature` | float | Temperature for judges | 0.0 |
| `species_singular` | string | Single term to replace all singular animal references | None |
| `species_plural` | string | Single term to replace all plural animal references | None |
| `max_tokens` | int | Maximum tokens for generation | 1000 |
| `max_retries` | int | Maximum retries for failures | 3 |
| `max_messages` | int | Maximum messages per conversation | 20 |

---

## Analysis Scripts

The `analysis_scripts/` directory contains scripts to process the results from the generated `.eval` log file or, more commonly, a combined CSV file derived from it.

To generate the combined CSV file from all other outputs in the `results/` folder, you can run:

```bash
uv run python analysis_scripts/tag_analysis.py
```


**Running Analysis Scripts:**

```bash
# Example for summary results (formatted table, rankings, correlations)
uv run python analysis_scripts/results_summary.py --input_csv results/combined_results.csv --latex

# Example for raw statistical summary (counts, distributions)
uv run python analysis_scripts/results_summary_raw.py --input_csv results/combined_results.csv --latex

# Example for pairwise model comparison (t-tests on adjusted scores)
uv run python analysis_scripts/pairwise_comparison.py --input_csv results/combined_results.csv --latex
```
---