import logging
from pathlib import Path
from typing import Optional

from src.metrics import avg_by_dimension, dimension_normalized_avg, question_normalized_avg
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
    grader_temperature: Optional[float] = None,
) -> Task:
    if dataset_repo_id:
        dims = load_dimensions_from_hf(dataset_repo_id)
        dims_by_name = {d.name: d for d in dims}
        dataset = load_dataset_from_hf(dataset_repo_id, dims_by_name.keys())
    else:
        default_base_path = Path(__file__).parent.parent / "data"
        dimensions_csv_path = default_base_path / "dimensions.csv"
        questions_csv_path = default_base_path / "questions.csv"

        dimensions = load_dimensions(dimensions_csv_path)
        dims_by_name = {dim.name: dim for dim in dimensions}
        samples = load_samples(dimensions=dims_by_name,
                               questions_csv_path=questions_csv_path)
        dataset = MemoryDataset(samples=samples, name="aha2_data")

    return Task(dataset=dataset,
                solver=generate(),
                metrics=[
                    question_normalized_avg(dims_by_name),
                    dimension_normalized_avg(dims_by_name),
                    avg_by_dimension(),
                ],
                scorer=aha2_scorer(dimensions=dims_by_name,
                                   grader_models=grader_models,
                                   grader_temperature=grader_temperature))
