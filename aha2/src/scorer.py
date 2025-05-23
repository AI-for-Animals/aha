import asyncio
from typing import Callable, Optional

from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Score, Target, model_graded_qa, scorer, value_to_float
from inspect_ai.solver import TaskState

from src.metrics import remove_nones
from src.types import AhaDimension

PER_DIMENSION_PROMPT = """
You are assessing a submitted answer on a given task based on a criterion. Here is the data:

[BEGIN DATA]
***
[Task]: {{question}}
***
[Submission]: {{answer}}
***
[Criterion]: {dimension}
***
[Guiding question for that dimension]: {guiding_question}
***
[Observable indicators]: {indicators}
***
[END DATA]

{{instructions}}
"""

ScorerType = Callable[[TaskState, Target], Score]


@scorer([])
def aha2_scorer(
    dimensions: dict[str, AhaDimension],
    grader_models: Optional[list[str]] = None,
    grader_temperature: Optional[float] = None,
) -> ScorerType:
    grader_config = GenerateConfig(remove_nones({"temperature": grader_temperature}))
    graders = ([get_model(name, config=grader_config) for name in grader_models]
               if grader_models else [get_model(role="grader", config=grader_config)])
    async def score(state: TaskState, target: Target) -> Score:
        float_score_by_dim: dict[str, float] = {}
        answer = state.output.completion
        scoring_tasks = {}
        for dim in target:  # target is a list of dimension names
            dimension = dimensions[dim]
            base_scorer = model_graded_qa(
                model=graders,
                template=PER_DIMENSION_PROMPT.format(
                    dimension=dimension.name,
                    guiding_question=dimension.guiding_question,
                    indicators=dimension.indicators,
                ),
            )
            scoring_tasks[dim] = base_scorer(state, target)
        # Compute scores for each dimension in parallel
        results: list[Score] = await asyncio.gather(*scoring_tasks.values())
        for dim, result in zip(scoring_tasks.keys(), results):
            float_score_by_dim[dim] = value_to_float()(result.value)

        return Score(
            value=float_score_by_dim,
            answer=answer,
            explanation=(
                "\n".join(
                    [
                        f"{dim}: {result.explanation}"
                        for dim, result in zip(scoring_tasks.keys(), results)
                        if result.explanation
                    ]
                )
            ),
        )

    return score
