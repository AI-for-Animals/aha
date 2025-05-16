import logging
from pathlib import Path
from typing import List, Optional

from inspect_ai.model import GenerateConfig, GenerateConfigArgs
from src.samples import load_dimensions, load_samples
from src.scorer import aha2_scorer

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset
from inspect_ai.solver import generate, Solver
from src.utils import remove_nones

logger = logging.getLogger(__name__)

@task
def aha2(
    model: Optional[Solver] = None,
    judges: List[str] = [
        "openai/gpt-4o-mini"
    ],
    model_temperature: Optional[float] = 0.5,
    judge_temperature: Optional[float] = 0.0,
    dimensions_csv_path: Optional[Path] = None,
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
        dimensions_csv_path: Path to dimensions CSV file
        questions_csv_path: Path to questions CSV file
    """
    default_base_path = Path(__file__).parent.parent / "data"
    dimensions_csv_path = dimensions_csv_path or default_base_path / "dimensions.csv"
    questions_csv_path = questions_csv_path or default_base_path / "questions.csv"

    dimensions = load_dimensions(dimensions_csv_path)
    dimensions_by_name = {dim.name: dim for dim in dimensions}
    samples = load_samples(dimensions=dimensions_by_name, 
                           questions_csv_path=questions_csv_path)

    if not samples:
        raise ValueError(
            "No valid samples could be loaded from the dataset after processing.")
    ds = MemoryDataset(samples=samples, name="aha2_data")

    generate_config_args: GenerateConfigArgs = remove_nones(dict(
        temperature=model_temperature,
    ))
    solver_instance = model or generate(**generate_config_args)

    scorer_instances = aha2_scorer(
        dimensions=dimensions_by_name,
        judge_models=judges,
        generate_config=GenerateConfig(**remove_nones({
            **generate_config_args,
            'temperature': judge_temperature,
        }))
    )

    return Task(
        dataset=ds,
        solver=solver_instance,
        scorer=scorer_instances,
    )
