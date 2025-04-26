"""
Inspect Task definition for the Animal Harm Assessment (AHA) benchmark.
"""
import logging
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from pydantic import ValidationError
from aha.types import AhaDatasetRecord, AhaJudgeScoreMetadata

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.solver import generate, Solver, TaskState
from inspect_ai.scorer import Score, Metric, metric, Target, SampleScore

from aha.scoring import SOLVER_INSTRUCTIONS, final_digit_model_graded_qa
from aha.utils import setup_logging


setup_logging(logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("inspect_ai.model").setLevel(logging.INFO)


# --- Dataset Processing ---

def record_to_sample(record_dict: Dict[str, Any]) -> Optional[Sample]:
    """Converts a raw dictionary from the dataset JSON into an Inspect Sample.

    Args:
        record_dict: A dictionary representing one entry from data_public.json.

    Returns:
        An Inspect Sample object, or None if the record is invalid.
    """
    try:
        record = AhaDatasetRecord.model_validate(record_dict)
    except ValidationError as e:
        logger.warning(
            f"Skipping invalid record (validation failed): {e}. "
            f"Data: {str(record_dict)[:100]}..."
        )
        return None

    return Sample(
        # question formatted for the model
        input=f"{record.question} Your answer must be EXACTLY 30–40 words long.",
        # judge's instructions
        target=SOLVER_INSTRUCTIONS,
        metadata=record.to_sample_metadata(),
        id=str(record.sample_id)
    )

# --- Custom Metrics ---

@metric
def avg_score() -> Metric:
    """Calculates the mean of the primary score values (avg score per sample)."""
    def calculate(scores: List[SampleScore]) -> float:
        if not scores:
            return 0.0
        # Extract the primary value from the Score object within each SampleScore
        values = [sample_score.score.value for sample_score in scores if sample_score.score]
        if not values:
            return 0.0 # Handle case where all scores were None
        return np.mean(values).item() # Use numpy for mean calculation

    return calculate

@metric
def default_rate() -> Metric:
    """Calculates the percentage of judge scores that defaulted to 0."""

    def calculate(scores: List[SampleScore]) -> float:
        total_defaults = 0
        total_judges = 0
        for sample_score in scores:
            # Access the Score object first
            score = sample_score.score
            if not score: # Skip if no score was produced for this sample
                continue

            # Access metadata safely from the Score object
            if score.metadata and isinstance(score.metadata, dict):
                try:
                    # Validate metadata using Pydantic model for clarity and safety
                    metadata_obj = AhaJudgeScoreMetadata.model_validate(score.metadata)
                    total_defaults += metadata_obj.defaulted_count
                    total_judges += metadata_obj.num_judges
                except ValidationError as e:
                    # Log sample id if available for better debugging
                    sample_id_str = f" for sample {sample_score.sample.id}" if sample_score.sample else ""
                    logger.error(f"Skipping score in default_rate{sample_id_str}: Invalid metadata format - {e}")
                    continue
            else:
                sample_id_str = f" for sample {sample_score.sample.id}" if sample_score.sample else ""
                logger.warning(f"Skipping score in default_rate{sample_id_str}: Missing or invalid metadata format.")

        if total_judges == 0:
            logger.warning("Cannot calculate default_rate: No valid judge scores found.")
            return 0.0
        return (total_defaults / total_judges) * 100.0

    return calculate

# --- Task Definition ---

def load_samples() -> List[Sample]:
    # Load Dataset
    data_file = Path(__file__).parent / "data_public.json"
    if not data_file.exists():
        raise FileNotFoundError(f"Cannot find AHA dataset at {data_file}")

    logger.info(f"Loading AHA dataset from {data_file}...")
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from {data_file}: {e}") from e

    # Basic structure check
    if not isinstance(raw_data, dict) or "data" not in raw_data or not isinstance(raw_data["data"], list):
        raise ValueError(f"Expected {data_file} to be a JSON object with a top-level 'data' list.")

    # Process records into Samples
    raw_records = raw_data["data"]
    samples: List[Sample] = []
    for i, record_dict in enumerate(raw_records):
        if not isinstance(record_dict, dict):
            logger.warning(f"Skipping non-dict record at index {i}: {str(record_dict)[:100]}...")
            continue
        sample = record_to_sample(record_dict)
        if sample:
            samples.append(sample)

    return samples


@task
def aha_evaluation(
    model: Optional[Solver] = None,
    judges: List[str] = [
        "anthropic/claude-3-5-sonnet-20240620",
        "google/gemini-1.5-flash",
        "openai/gpt-4o"
    ],
    model_temperature: Optional[float] = 0.5,
    judge_temperature: Optional[float] = 0.5,
    seed: int = 42,
    max_tokens: int = 1000,
    max_retries: int = 3,
    max_messages: int = 20,
) -> Task:
    """Defines the AHA evaluation task for Inspect AI."""

    samples = load_samples()
    if not samples:
        raise ValueError("No valid samples could be loaded from the dataset after processing.")
    logger.info(f"Loaded {len(samples)} valid samples.")

    ds = MemoryDataset(samples=samples, name="aha_data")

    # Configure Solver
    # Use default generate() if no specific model solver is passed in
    solver_instance = model or generate(
        max_tokens=max_tokens,
        temperature=model_temperature,
        seed=seed,
    )

    # Configure Scorer
    scorer_instance = final_digit_model_graded_qa(
        scorer_models=judges,
        judge_temperature=judge_temperature,
    )

    # Define Metrics
    task_metrics = [
        avg_score(),
        default_rate()
    ]

    return Task(
        dataset=ds,
        solver=solver_instance,
        scorer=scorer_instance,
        metrics=task_metrics,
        max_messages=max_messages,
        max_retries=max_retries,
    )
