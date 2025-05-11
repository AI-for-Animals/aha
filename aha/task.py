"""
Inspect Task definition for the Animal Harm Assessment (AHA) 2.0 benchmark.
"""
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from aha.types import Aha2CombinedScoreMetadata
from aha.csv_dataset import load_aha2_samples
from aha.scoring import aha2_multidimensional_scorer

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.solver import generate, Solver, TaskState
from inspect_ai.scorer import Score, Metric, metric, SampleScore
from aha.utils import setup_logging

setup_logging(logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("inspect_ai.model").setLevel(logging.INFO)

# --- Custom Metrics ---


@metric
def avg_weighted_score() -> Metric:
    """Calculates the mean weighted score across all samples."""
    def calculate(scores: List[SampleScore]) -> float:
        if not scores:
            return 0.0
        # Extract the primary value from the Score object within each SampleScore
        values = [
            sample_score.score.value for sample_score in scores if sample_score.score]
        if not values:
            return 0.0  # Handle case where all scores were None
        return np.mean(values).item()  # Use numpy for mean calculation

    return calculate


@metric
def dimension_scores() -> Metric:
    """Returns the average scores for each dimension."""
    def calculate(scores: List[SampleScore]) -> Dict[str, float]:
        if not scores:
            return {}

        # Combine all dimension scores
        all_dimension_scores: Dict[str, List[float]] = {}

        for sample_score in scores:
            score = sample_score.score
            if not score or not score.metadata:
                continue

            try:
                metadata = Aha2CombinedScoreMetadata.model_validate(
                    score.metadata)
                for dim_name, avg_score in metadata.dimension_averages.items():
                    if dim_name not in all_dimension_scores:
                        all_dimension_scores[dim_name] = []
                    all_dimension_scores[dim_name].append(avg_score)
            except Exception as e:
                sample_id = sample_score.sample.id if sample_score.sample else "unknown"
                logger.error(
                    f"Error processing dimension scores for sample {sample_id}: {e}")
                continue

        # Calculate the average for each dimension
        result = {}
        for dim_name, dim_scores in all_dimension_scores.items():
            if dim_scores:
                result[dim_name] = sum(dim_scores) / len(dim_scores)
            else:
                result[dim_name] = 0.0

        return result

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
            if not score or not score.metadata:  # Skip if no score was produced for this sample
                continue

            # Access metadata
            try:
                metadata = Aha2CombinedScoreMetadata.model_validate(
                    score.metadata)
                total_defaults += metadata.defaulted_count
                total_judges += metadata.num_judges
            except Exception as e:
                sample_id = sample_score.sample.id if sample_score.sample else "unknown"
                logger.error(
                    f"Error processing default rate for sample {sample_id}: {e}")
                continue

        if total_judges == 0:
            logger.warning(
                "Cannot calculate default_rate: No valid judge scores found.")
            return 0.0
        return (total_defaults / total_judges) * 100.0

    return calculate

# --- Task Definition ---


@task
def aha2_evaluation(
    model: Optional[Solver] = None,
    judges: List[str] = [
        "anthropic/claude-3-5-sonnet-20240620",
        "openai/gpt-4o"
    ],
    model_temperature: Optional[float] = 0.5,
    judge_temperature: Optional[float] = 0.0,
    seed: int = 42,
    max_tokens: int = 1000,
    max_retries: int = 3,
    max_messages: int = 20,
    base_csv_path: Optional[Path] = None,
    questions_csv_path: Optional[Path] = None
) -> Task:
    """Defines the AHA 2.0 evaluation task for Inspect AI."""
    if base_csv_path is None:
        base_csv_path = Path(__file__).parent.parent / \
            "config" / "dimensions.csv"
    if questions_csv_path is None:
        questions_csv_path = Path(
            __file__).parent.parent / "config" / "questions.csv"

    if not base_csv_path.exists():
        raise FileNotFoundError(
            f"Cannot find AHA 2.0 base CSV at {base_csv_path}")
    if not questions_csv_path.exists():
        raise FileNotFoundError(
            f"Cannot find AHA 2.0 questions CSV at {questions_csv_path}")

    logger.info(f"Loading AHA 2.0 data from CSV files...")
    logger.info(f"Base file: {base_csv_path}")
    logger.info(f"Questions file: {questions_csv_path}")

    samples, dimensions = load_aha2_samples(base_csv_path, questions_csv_path)

    if not samples:
        raise ValueError(
            "No valid samples could be loaded from the dataset after processing.")
    logger.info(
        f"Loaded {len(samples)} valid samples with {len(dimensions)} dimensions.")
    logger.info(f"Dimensions: {', '.join(dimensions.keys())}")

    ds = MemoryDataset(samples=samples, name="aha2_data")

    # Configure Solver
    # Use default generate() if no specific model solver is passed in
    solver_instance = model or generate(
        max_tokens=max_tokens,
        temperature=model_temperature,
        seed=seed,
    )

    # Configure Scorer
    scorer_instance = aha2_multidimensional_scorer(
        dimensions=dimensions,
        scorer_models=judges,
        judge_temperature=judge_temperature,
    )

    # Define Metrics
    task_metrics = [
        avg_weighted_score(),
        dimension_scores(),
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
