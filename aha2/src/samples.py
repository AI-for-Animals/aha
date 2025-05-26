import csv
import json
import re
import logging
from pathlib import Path

from src.types import AhaDimension
from inspect_ai.dataset import Sample

logger = logging.getLogger(__name__)

def load_dimensions(dimensions_csv_path: Path) -> list[AhaDimension]:
    dimensions = []
    if not dimensions_csv_path.exists():
        raise FileNotFoundError(
            f"Cannot find AHA 2.0 dimensions CSV at {dimensions_csv_path}")
    with open(dimensions_csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            variables = list(re.findall(r'\{\{(.*?)\}\}', row.get('Guiding Question', '')))
            dimensions.append(AhaDimension(
                name=row['Dimension'],
                weight=float(row.get('Default weighting', 1.0)),
                scoring_description=row.get('Scoring', ''),
                guiding_question=row.get('Guiding Question', ''),
                indicators=row.get(
                    'Observable Indicators (pattern-match)', ''),
                variables=variables,
            ))
    return dimensions


def load_samples(
    questions_csv_path: Path, 
    dimensions: dict[str, AhaDimension]
) -> list[Sample]:
    samples = []
    if not questions_csv_path.exists():
        raise FileNotFoundError(
            f"Cannot find AHA 2.0 questions CSV at {questions_csv_path}")
    with open(questions_csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if not row.get('Question') or not row.get('Validated tags'):
                logger.warning(
                    f"Question {i+1} is missing a question or validated tags, skipping")
                continue

            tags = [tag.strip() for tag in row['Validated tags'].split(',')]
            if bad_tag := next((tag for tag in tags if tag not in dimensions), None):
                logger.warning(
                    f"Question {i+1} has a tag that does not match any dimension, skipping: {bad_tag}")
                continue
            if not tags:
                logger.warning(
                    f"Question {i+1} has no valid tags matching dimensions, skipping")
                continue
            variables = row.get('Variables', None)

            # Dimension name -> variable name -> [value1, value2]
            parsed_vars = {}
            if variables:
                variable_sets = [v.strip() for v in variables.split('\n')]
                for j, variable_set in enumerate(variable_sets):
                    split_set = [v.strip() for v in variable_set.split(':')]
                    if len(split_set) != 3:
                        logger.warning(
                            f"Question {i+1} has an invalid {j+1}th variable set, skipping")
                        continue
                    dimension_name, variable, values = split_set
                    if dimension_name not in dimensions:
                        logger.warning(
                            f"Question {i+1} has an invalid {j+1}th variable set, skipping")
                        continue
                    if variable not in dimensions[dimension_name].variables:
                        logger.warning(
                            f"Question {i+1} has an invalid {j+1}th variable set, skipping")
                        continue
                    parsed_vars.setdefault(dimension_name, {}).setdefault(variable, []).extend(values.split(','))
            samples.append(Sample(
                input=row['Question'],
                target=json.dumps({'tags': tags, 'variables': parsed_vars}),
                id=str(i + 1),
            ))
    return samples
