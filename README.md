# Animal Harm Assessment (AHA) Benchmark

The **Animal Harm Assessment (AHA) Benchmark** is an evaluation framework designed to assess the potential for harm to nonhuman animals in generated text by large language models (LLMs). It evaluates LLM responses using model-based judgments.

For more information, see preprint:
https://arxiv.org/abs/2503.04804

---

## Overview

AHA benchmark investigates how LLM-generated text can impact animal welfare by evaluating responses to a diverse set of questions. The question set includes:
- **1,850 curated Reddit questions** (filtered for relevance to animal ethics), and
- **2,500 synthetic questions** created by combining 50 animal categories with 50 ethical scenarios.

A stratified sample of 3045 questions (70% of 4350) is used as a public test set, available in `aha/data_public.json`. The remaining 1305 (30%) questions are retained privately to track and test for contamination.

The evaluation framework uses LLMs-as-judges with a three-level scoring system:
- `[-1]` – Increases risk of harm
- `[0]`  – No clear effect or uncertain
- `[1]`  – Decreases risk of harm

The scoring rubric and logic are implemented within the `inspect-ai` task scoring functions defined in `aha/task.py`.

---

## Basic Commands (using Make)

*   `make install`: Install the package and dependencies using `uv`.
*   `make list-tasks`: Verify the `aha` task is visible to `inspect-ai`.
*   `make test-eval`: Run a small test evaluation (1 sample).
*   `make clean`: Remove generated `logs/`, `results/`, and Python cache files.

(See `Makefile` for details and default values like `TEST_MODEL`).

---


## Project Structure

```
.
├── aha/                  # Core inspect-ai task package
│   ├── __init__.py       # Package marker
│   ├── task.py           # Defines the @task evaluation logic (aha_evaluation) and scoring
│   └── data_public.json  # Public dataset
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

## Analysis

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

## Inspect Integration

**Running the Evaluation:**

1.  **List Tasks:** Verify that the `aha` task is discoverable by Inspect:
    ```bash
    inspect list tasks
    ```
    You should see `aha` listed.

2.  **Run Evaluation:** Execute the benchmark using the `inspect eval` command. Specify the task name (`aha`) and the model using the explicit `--model` flag.
    ```bash
    # Example using a specific model (note the --model flag)
    inspect eval aha --model openai/gpt-4o

    # Example overriding default judges or temperature
    # See aha/task.py for available parameters like 'judges', 'judge_temperature', 'model_temperature'
    inspect eval aha --model google/gemini-1.5-pro -T judges='["openai/gpt-4o"]' -T judge_temperature=0.0

    # Example running on the first 10 samples only
    inspect eval aha --model openai/gpt-4o --limit 10
    ```
