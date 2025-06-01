from collections import defaultdict
import logging

import numpy as np
from inspect_ai.scorer import Metric, SampleScore, metric

from typing import TypeVar

from src.types import AhaDimension

logger = logging.getLogger(__name__)


K = TypeVar("K")
V = TypeVar("V")


def remove_nones(d: dict[K, V]) -> dict[K, V]:
    return {k: v for k, v in d.items() if v is not None}


@metric
def question_normalized_avg(dims_by_name: dict[str, AhaDimension]) -> Metric:
    def metric(scores: list[SampleScore]) -> float:
        avg_score_per_question = []
        for sample_score in scores:
            valid_scores = remove_nones(sample_score.score.value)
            if valid_scores:
                weighted_scores = [
                    score * dims_by_name[dim].weight
                    for dim, score in valid_scores.items()
                ]
                avg_score_per_question.append(np.mean(weighted_scores).item())
        if not avg_score_per_question:
            return 0.0
        return np.mean(avg_score_per_question).item()

    return metric


@metric
def dimension_normalized_avg(dims_by_name: dict[str, AhaDimension]) -> Metric:
    def metric(scores: list[SampleScore]) -> float:
        logger.info(f"scores in dimension_normalized_avg: {scores}")
        dimension_avgs = _get_avg_by_dimension(scores)
        if not dimension_avgs:
            return None
        weighted_avgs = [
            score * dims_by_name[dim].weight for dim, score in dimension_avgs.items()
        ]
        return np.mean(weighted_avgs).item()

    return metric


def _get_avg_by_dimension(scores: list[SampleScore]) -> dict[str, float]:
    scores_per_dimension: dict[str, list[float]] = defaultdict(list)
    for sample_score in scores:
        for dim, dim_score in remove_nones(sample_score.score.value).items():
            scores_per_dimension[dim].append(dim_score)
    return {dim: np.mean(values).item() for dim, values in scores_per_dimension.items()}


@metric
def avg_by_dimension() -> Metric:
    return _get_avg_by_dimension
