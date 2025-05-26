import asyncio
import json
import numpy as np
from typing import Callable, Optional
import re

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
    grader_config: GenerateConfig = GenerateConfig(),
) -> ScorerType:
    graders = ([get_model(name, config=grader_config) for name in grader_models]
               if grader_models else [get_model(role="grader", config=grader_config)])
    async def score(state: TaskState, target: Target) -> Score:
        float_score_by_dim_i: dict[str, float] = {}
        float_score_by_dim: dict[str, float] = {}
        answer = state.output.completion
        scoring_tasks = {}
        parsed_target = json.loads(str(target.text))
        target_dimensions = parsed_target['tags']
        variables = parsed_target['variables']  # Dimension name -> variable name -> [value1, value2]
        for dim in target_dimensions:
            variable_sets = [{}]
            for var_name, var_values in variables.get(dim, {}).items():  # variable name -> [value1, value2]
                updated_variable_sets = []
                for v in var_values:
                    updated_variable_sets.extend([
                        {**v_set, var_name: v}
                        for v_set in variable_sets
                    ])
                variable_sets = updated_variable_sets
            dimension = dimensions[dim]
            for i, v_set in enumerate(variable_sets):
                guiding_q = dimension.guiding_question
                for k, v in v_set.items():
                    guiding_q = guiding_q.replace('{{' + k + '}}', v)

                # Replace any remaining template variables with their literal key names
                pattern = r"\{\{([^}]+)\}\}"
                matches = re.findall(pattern, guiding_q)
                for match in matches:
                    guiding_q = guiding_q.replace("{{" + match + "}}", match)

                base_scorer = model_graded_qa(
                    model=graders,
                    template=PER_DIMENSION_PROMPT.format(
                        dimension=dimension.name,
                        guiding_question=guiding_q,
                        indicators=dimension.indicators,
                    ),
                )
                scoring_tasks[f'{dim}_{i}'] = base_scorer(state, target)
        # Compute scores for each dimension in parallel
        results: list[Score] = await asyncio.gather(*scoring_tasks.values())
        for dim_i, result in zip(scoring_tasks.keys(), results):
            dim, i = dim_i.split('_')
            float_score_by_dim_i.setdefault(dim, []).append(value_to_float()(result.value))

        for dim, scores in float_score_by_dim_i.items():
            float_score_by_dim[dim] = np.mean(scores).item()

        return Score(
            value=float_score_by_dim,
            answer=answer,
        )

    return score
