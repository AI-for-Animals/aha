"""
Inspect Task definition for the Animal Harm Assessment (AHA) 2.0 benchmark.
"""
import logging
from pathlib import Path
from typing import List, Optional

from aha.types import Aha2CombinedScoreMetadata
from aha.csv_dataset import load_aha2_samples, load_dimensions
from aha.scoring import aha_multi_scorer

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset
from inspect_ai.solver import generate, Solver
from inspect_ai.scorer import Metric, SampleScore
from aha.utils import setup_logging

setup_logging(logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("inspect_ai.model").setLevel(logging.INFO)


# @metric
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
                metadata = Aha2CombinedScoreMetadata.model_validate(
                    score.metadata)
                total_defaults += metadata.defaulted_count
                total_judges += metadata.num_judges

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
    questions_csv_path: Optional[Path] = None,
    species_singular: Optional[str] = None,
    species_plural: Optional[str] = None
) -> Task:
    """
    Defines the AHA 2.0 evaluation task for Inspect AI.
    
    Args:
        model: The model to use for solving the task
        judges: List of models to use for scoring
        model_temperature: Temperature for the model
        judge_temperature: Temperature for the judges
        seed: Random seed
        max_tokens: Maximum tokens to generate
        max_retries: Maximum retries for failures
        max_messages: Maximum messages per conversation
        base_csv_path: Path to dimensions CSV file
        questions_csv_path: Path to questions CSV file
        species_singular: Single replacement value for all {species-singular:*} variables
        species_plural: Single replacement value for all {species-plural:*} variables
    """
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
    
    if species_singular or species_plural:
        logger.info(f"Using species replacements - singular: '{species_singular}', plural: '{species_plural}'")

    dimensions = load_dimensions(base_csv_path)
    dimensions_by_name = {dim.name: dim for dim in dimensions}
    samples = load_aha2_samples(dimensions=dimensions_by_name, 
                                questions_csv_path=questions_csv_path, 
                                species_singular=species_singular,
                                species_plural=species_plural)

    if not samples:
        raise ValueError(
            "No valid samples could be loaded from the dataset after processing.")
    ds = MemoryDataset(samples=samples, name="aha2_data")
    solver_instance = model or generate(
        max_tokens=max_tokens,
        temperature=model_temperature,
        seed=seed,
    )
    scorer_instances = aha_multi_scorer(
        dimensions=dimensions_by_name,
        models=judges,
    )
    task_metrics = [
        # default_rate()
    ]

    return Task(
        dataset=ds,
        solver=solver_instance,
        scorer=scorer_instances,
        # metrics=task_metrics,
        max_messages=max_messages,
        max_retries=max_retries,
    )
