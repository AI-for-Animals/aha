# AHA Benchmark 2.0

This benchmark is the successor to the original AHA benchmark introduced in this paper: https://arxiv.org/abs/2503.04804.

![AHA 2.0 Benchmark](img.png)


## Basic commands

*   `make install`: Install the package and dependencies using `uv`.
*   `make list-tasks`: Confirm the task is visible to `inspect-ai`.
*   `make test-eval`: Run a small test evaluation (1 sample).


## Data

The benchmark is parameterized primarily by two files:
- `data/questions.csv`: questions posed to LLMs
- `data/dimensions.csv`: dimensions along which answers are judged

Alternatively, you can provide a huggingface dataset id (see below).

## Running the benchmark with Inspect

Execute the benchmark:

    ```bash
    # Run the eval
    uv run inspect eval src --model openai/gpt-4o

    # Draw from our huggingface dataset instead of local csvs
    uv run inspect eval src -T dataset_repo_id="nishubuilder/aha-2"

    uv run inspect eval src \
        # Specify a scorer model and its temperature
        --model openai/gpt-4o -S temperature=0.0 \
        # Run multiple graders with different models
        -T grader_models='["anthropic/claude-3-7-sonnet-20250219", "google/gemini-2.0-flash"]'
    ```

Use the `-T` flag to pass task parameters:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| dataset_repo_id | string | optional huggingface dataset id to pull samples from instead of using data/*.csv | nishubuilder/aha-2 |
| `grader_models` | list | Models used for scoring responses | `["anthropic/claude-3-5-sonnet-20240620", "openai/gpt-4o"]` |
| `grader_temperature` | float | Temperature for graders | 0.0 |


Use the `-S` flag to pass solver (test model) parameters:
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `temperature` | float | Temperature for generation | 0.0 |


See `uv run inspect eval --help` for all available options.

## Viewing results

Inspect generates `.eval` files that are, by default, placed in a `logs/` folder. These can be viewed through a local web server:


    uv run inspect view

You can also generate radar charts from a given .eval file with:


    uv run python src/radar_plot.py --lookback=<num evals (one per model) to plot together>
