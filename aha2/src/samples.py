import csv
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
            dimensions.append(AhaDimension(
                name=row['Dimension'],
                weight=float(row.get('Default weighting', 1.0)),
                scoring_description=row.get('Scoring', ''),
                guiding_question=row.get('Guiding Question', ''),
                indicators=row.get(
                    'Observable Indicators (pattern-match)', '')
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
            if any(tag not in dimensions for tag in tags):
                logger.warning(
                    f"Question {i+1} has a tag that does not match any dimension, skipping")
                continue
            if not tags:
                logger.warning(
                    f"Question {i+1} has no valid tags matching dimensions, skipping")
                continue

            samples.append(Sample(
                input=row['Question'],
                target=tags,
                id=str(i + 1)
            ))
    return samples
