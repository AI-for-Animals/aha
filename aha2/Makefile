.PHONY: help install list-tasks test-eval

DEFAULT_SOLVER_MODEL ?= openai/gpt-4o-mini

help:
	@echo "Available targets:"
	@echo "  install        : Install the package and dependencies using uv pip"
	@echo "  list-tasks     : List available Inspect tasks"
	@echo "  test-eval      : Run a small test evaluation (limit 1). Uses DEFAULT_TEST_MODEL (${DEFAULT_SOLVER_MODEL})."


install:
	@echo "Installing package using uv..."
	uv pip install -e .

list-tasks:
	@echo "Listing Inspect tasks..."
	uv run inspect list tasks | cat

test-eval:
	@echo "Running test evaluation with model: ${DEFAULT_SOLVER_MODEL}, limit 1..."
	uv run inspect eval src --model ${DEFAULT_SOLVER_MODEL} --limit 1 -T judges='["${DEFAULT_SOLVER_MODEL}"]' | cat

lint:
	@echo "Linting code..."
	uv run black src
	uv run autopep8 --in-place --recursive src
