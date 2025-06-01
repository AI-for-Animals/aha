from collections import defaultdict
import logging

import numpy as np
from inspect_ai.scorer import Metric, SampleScore, metric

from typing import TypeVar

from src.types import AhaDimension
from src.utils import get_dimension_weighted_avg

logger = logging.getLogger(__name__)


K = TypeVar("K")
V = TypeVar("V")


def remove_nones(d: dict[K, V]) -> dict[K, V]:
    return {k: v for k, v in d.items() if v is not None}


@metric
def dimension_normalized_avg(dims_by_name: dict[str, AhaDimension]) -> Metric:
    def metric(scores: list[SampleScore]) -> float:
        return get_dimension_weighted_avg(_get_avg_by_dimension(scores), dims_by_name)

    return metric


def _get_avg_by_dimension(scores: list[SampleScore]) -> dict[str, float]:
    scores_per_dimension: dict[str, list[float]] = defaultdict(list)
    for per_dim_scores in [s.score.metadata["dimension_scores"] for s in scores]:
        for dim, dim_score in remove_nones(per_dim_scores).items():
            scores_per_dimension[dim].append(dim_score)
    return {dim: np.mean(values).item() for dim, values in scores_per_dimension.items()}


@metric
def avg_by_dimension() -> Metric:
    return _get_avg_by_dimension
