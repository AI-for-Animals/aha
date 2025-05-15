"""
Inspect Task definition for the Animal Harm Assessment (AHA) 2.0 benchmark.
"""
import logging
import os
from pathlib import Path
from typing import List, Optional

from inspect_ai.model import GenerateConfig, GenerateConfigArgs
from aha.csv_dataset import load_dimensions, load_samples
from aha.scoring import aha_multi_scorer

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset
from inspect_ai.solver import generate, Solver
from aha.utils import remove_none_values, setup_logging

setup_logging(logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("inspect_ai.model").setLevel(os.getenv("INSPECT_MODEL_LOG_LEVEL", logging.INFO))


@task
def aha2_evaluation(
    model: Optional[Solver] = None,
    judges: List[str] = [
        "openai/gpt-4o-mini"
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

    dimensions = load_dimensions(base_csv_path)
    dimensions_by_name = {dim.name: dim for dim in dimensions}
    samples = load_samples(dimensions=dimensions_by_name, 
                           questions_csv_path=questions_csv_path,)

    if not samples:
        raise ValueError(
            "No valid samples could be loaded from the dataset after processing.")
    ds = MemoryDataset(samples=samples, name="aha2_data")

    generate_config_args: GenerateConfigArgs = remove_none_values(dict(
        max_tokens=max_tokens,
        seed=seed,
        temperature=model_temperature,
    ))
    solver_instance = model or generate(**generate_config_args)

    scorer_instances = aha_multi_scorer(
        dimensions=dimensions_by_name,
        judge_models=judges,
        generate_config=GenerateConfig(**remove_none_values({
            **generate_config_args,
            'temperature': judge_temperature,
        }))
    )

    return Task(
        dataset=ds,
        solver=solver_instance,
        scorer=scorer_instances,
        max_messages=max_messages,
        max_retries=max_retries,
    )
