from datasets import load_dataset
from src.types import AhaDimension
from inspect_ai.dataset import Sample


def load_from_hf(
    repo_id: str,
    questions_config: str = "questions",
    dimensions_config: str = "dimensions",
) -> tuple[list[AhaDimension], list[Sample]]:
    """Load dimensions and question samples from a Hugging-Face dataset repo that
    holds two configs (``questions`` and ``dimensions``).

    Returns
    -------
    tuple
        ``(dimensions, samples)`` where
        * dimensions is a list[AhaDimension]
        * samples is a list[inspect_ai.dataset.Sample]
    """
    ds_qs = load_dataset(repo_id, questions_config, split="train")
    ds_dims = load_dataset(repo_id, dimensions_config, split="train")

    # Build dimension objects
    dimensions_list: list[AhaDimension] = []
    dims_by_name: dict[str, AhaDimension] = {}
    for row in ds_dims:  # each row is a dict
        dim = AhaDimension(
            name=row["dimension"],
            weight=row.get("default_weighting", 1.0),
            scoring_description=row.get("scoring", ""),
            guiding_question=row.get("guiding_question", ""),
            indicators=row.get("observable_indicators", ""),
        )
        dimensions_list.append(dim)
        dims_by_name[dim.name] = dim

    # Build Sample objects from questions, filtering any tag not present in dimensions
    samples: list[Sample] = []
    for i, row in enumerate(ds_qs):
        tags: list[str] = row.get("tags", [])
        if not tags or any(t not in dims_by_name for t in tags):
            continue
        samples.append(
            Sample(
                input=row["question"],
                target=tags,
                id=str(i + 1),
            )
        )

    return dimensions_list, samples
