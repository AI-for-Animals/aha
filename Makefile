# Makefile for AHA Inspect Task

.PHONY: help install list-tasks test-eval analyze clean install-dev test

# Default model for test-eval - can be overridden on the command line
# Example: make test-eval DEFAULT_TEST_MODEL=openai/gpt-4o
DEFAULT_TEST_MODEL ?= openai/gpt-4o

# Variables for analyze target - can be overridden on the command line
# Example: make analyze ANALYZE_LOG_FILE=logs/my_log.eval ANALYZE_SOLVER_NAME=google/gemini-pro
ANALYZE_LOG_FILE ?=
ANALYZE_SOLVER_NAME ?=

# Default solver name for analysis if ANALYZE_SOLVER_NAME is not provided
DEFAULT_SOLVER_NAME = openai/gpt-4.1-mini

help:
	@echo "Available targets:"
	@echo "  install        : Install the package and dependencies using uv pip"
	@echo "  list-tasks     : List available Inspect tasks"
	@echo "  test-eval      : Run a small test evaluation (limit 1). Uses DEFAULT_TEST_MODEL (${DEFAULT_TEST_MODEL})."
	@echo "                   Override model with DEFAULT_TEST_MODEL=<model_name>"
	@echo "  analyze        : Run analysis script on a log file."
	@echo "                   Uses latest log in logs/ if ANALYZE_LOG_FILE is not set."
	@echo "                   Uses ${DEFAULT_SOLVER_NAME} if ANALYZE_SOLVER_NAME is not set."
	@echo "                   Override with ANALYZE_LOG_FILE=<path> ANALYZE_SOLVER_NAME=<model_name>"
	@echo "  clean          : Remove generated files (logs, results, __pycache__, *.egg-info)"
	@echo "  install-dev    : Install package with development dependencies (e.g., pytest)"
	@echo "  test           : Run the pytest test suite"

install:
	@echo "Installing package using uv..."
	uv pip install -e .

install-dev:
	@echo "Installing package with dev dependencies using uv..."
	uv pip install -e '.[dev]'

list-tasks:
	@echo "Listing Inspect tasks..."
	inspect list tasks | cat

test-eval:
	@echo "Running test evaluation with model: ${DEFAULT_TEST_MODEL}, limit 1..."
	# Override judges using -T key=value syntax
	# Explicitly set INSPECT_EVAL_MODEL for the scorer's get_model call
	# Use --model flag for clarity and to avoid parsing issues
	INSPECT_EVAL_MODEL=${DEFAULT_TEST_MODEL} inspect eval aha --model ${DEFAULT_TEST_MODEL} --limit 1 -T judges='["${DEFAULT_TEST_MODEL}"]' | cat

analyze:
	@echo "Running analysis script..."
	@LOG_FILE_TO_USE="${ANALYZE_LOG_FILE}"; \
	 ACTUAL_SOLVER_NAME="${ANALYZE_SOLVER_NAME}"; \
	 if [ -z "$${LOG_FILE_TO_USE}" ]; then \
	   echo "ANALYZE_LOG_FILE not specified, finding most recent log in logs/..."; \
	   LOG_FILE_TO_USE=$$(ls -t logs/*.eval 2>/dev/null | head -n 1); \
	 fi; \
	 if [ -z "$${LOG_FILE_TO_USE}" ] || [ ! -f "$${LOG_FILE_TO_USE}" ]; then \
	   echo "Error: No log file found or specified."; \
	   echo "Please run 'make test-eval' first or specify ANALYZE_LOG_FILE="; \
	   exit 1; \
	 fi; \
	 if [ -z "$${ACTUAL_SOLVER_NAME}" ]; then \
	   ACTUAL_SOLVER_NAME="${DEFAULT_SOLVER_NAME}"; \
	   echo "ANALYZE_SOLVER_NAME not specified, using default: $${ACTUAL_SOLVER_NAME}"; \
	 fi; \
	 echo "Using Log File: $${LOG_FILE_TO_USE}"; \
	 echo "Using Solver Name: $${ACTUAL_SOLVER_NAME}"; \
	 python analysis.py --log-file "$${LOG_FILE_TO_USE}" --output-dir results --solver-name "$${ACTUAL_SOLVER_NAME}"

test:
	@echo "Running pytest..."
	pytest tests/

clean:
	@echo "Cleaning generated files..."
	rm -rf logs/
	rm -rf results/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} + 