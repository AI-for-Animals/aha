import json
import logging
from typing import Any, Callable, Iterable

from datasets import load_dataset
from inspect_ai.dataset import Dataset, Sample, hf_dataset

from src.types import AhaDimension

logger = logging.getLogger(__name__)


def load_dimensions_from_hf(repo_id: str) -> list[AhaDimension]:
    ds_dims = load_dataset(repo_id, "dimensions", split="train", token=False)
    dimensions = []
    for row in ds_dims:
        dimension_data = {
            "name": row["dimension"],
            "weight": row.get("default_weighting", 1.0),
            "scoring_description": row["scoring"],
            "guiding_question": row["guiding_question"],
            "indicators": row.get("observable_indicators", ""),
        }
        dimensions.append(AhaDimension.model_validate(dimension_data))
    return dimensions


def record_to_sample(
    valid_dims: Iterable[str],
) -> Callable[[dict[str, Any]], Sample | None]:
    valid = set(valid_dims)

    def _convert(record: dict[str, Any]) -> Sample | None:
        tags, variables, question = (
            record["tags"],
            record["variables"],
            record["question"],
        )
        if not tags or any(t not in valid for t in tags) or not question:
            return None

        # variable name -> [value1, value2]
        parsed_vars = {}
        if variables:
            variable_sets = [v.strip() for v in variables.split("\n")]
            for variable_set in variable_sets:
                split_set = [v.strip() for v in variable_set.split(":")]
                if len(split_set) != 2:
                    logger.warning(f"Invalid variable set: {variable_set}")
                    continue
                variable, values = split_set
                parsed_vars.setdefault(variable, []).extend(values.split(","))
        return Sample(
            input=question, target=json.dumps(
                {"tags": tags, "variables": parsed_vars})
        )

    return _convert


def load_dataset_from_hf(repo_id: str, valid_dims: Iterable[str]) -> Dataset:
    """Load questions from Hugging Face as an Inspect dataset."""
    return hf_dataset(
        path=repo_id,
        data_dir="questions",
        split="train",
        sample_fields=record_to_sample(valid_dims),
        trust=True,
        shuffle=True,
        auto_id=True,
        token=False,
    )
