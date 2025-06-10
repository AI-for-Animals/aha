import logging
from pathlib import Path
from typing import Optional

from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import mean

from src.metrics import (
    avg_by_dimension,
    dimension_normalized_avg,
    remove_nones,
)
from src.samples import load_dimensions, load_samples
from src.samples_hf import load_dataset_from_hf, load_dimensions_from_hf
from src.scorer import aha2_scorer

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset
from inspect_ai.solver import generate

logger = logging.getLogger(__name__)


@task
def aha2(
    dataset_repo_id: str = None,
    grader_models: Optional[list[str]] = None,
    judge_max_connections: int = 10,
    judge_temperature: float = 0.0,
    judge_max_tokens: int = 1024,
    judge_max_retries: int = 0,
    questions_csv_path: Optional[str] = None,
) -> Task:
    if dataset_repo_id:
        dims = load_dimensions_from_hf(dataset_repo_id)
        dims_by_name = {d.name: d for d in dims}
        dataset = load_dataset_from_hf(dataset_repo_id, dims_by_name.keys())
    else:
        default_base_path = Path(__file__).parent.parent / "data"
        dimensions_csv_path = default_base_path / "dimensions.csv"
        questions_csv_path = (
            default_base_path / (questions_csv_path or "questions.csv")
        )

        dimensions = load_dimensions(dimensions_csv_path)
        dims_by_name = {dim.name: dim for dim in dimensions}
        samples = load_samples(
            dimensions=dims_by_name, questions_csv_path=questions_csv_path
        )
        dataset = MemoryDataset(samples=samples, name="aha2_data")
    generate_judge_config_args = remove_nones(
        dict(
            max_tokens=judge_max_tokens,
            temperature=judge_temperature,
            max_retries=judge_max_retries,
            max_connections=judge_max_connections,
        )
    )
    return Task(
        dataset=dataset,
        solver=generate(),
        metrics=[
            mean(),
            dimension_normalized_avg(dims_by_name),
            avg_by_dimension(),
        ],
        scorer=aha2_scorer(
            dimensions=dims_by_name,
            grader_models=grader_models,
            grader_config=GenerateConfig(**generate_judge_config_args),
        ),
    )
