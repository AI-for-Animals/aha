import csv
import logging
from pathlib import Path
from typing import List, Dict

from aha.types import AhaDimension, Aha2DatasetRecord
from inspect_ai.dataset import Sample

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_dimensions(base_csv_path: Path) -> Dict[str, AhaDimension]:
    """
    Loads the scoring dimensions from the base CSV file.

    Args:
        base_csv_path: Path to the AHA Bench 2.0 - Base.csv file

    Returns:
        Dictionary of dimension names to AhaDimension objects
    """
    dimensions = {}
    try:
        with open(base_csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Skip empty rows or header rows
                if not row.get('Dimension') or row['Dimension'] == 'Dimension':
                    continue

                dimension = AhaDimension(
                    name=row['Dimension'],
                    weight=float(row.get('Default weighting', 1.0)),
                    scoring_description=row.get('Scoring', ''),
                    guiding_question=row.get('Guiding Question', ''),
                    indicators=row.get(
                        'Observable Indicators (pattern-match)', '')
                )
                dimensions[dimension.name] = dimension

        logger.info(f"Loaded {len(dimensions)} scoring dimensions")
        return dimensions
    except Exception as e:
        logger.error(f"Error loading dimensions from {base_csv_path}: {e}")
        raise


def load_questions(questions_csv_path: Path, dimensions: Dict[str, AhaDimension]) -> List[Aha2DatasetRecord]:
    """
    Loads the questions from the questions CSV file.

    Args:
        questions_csv_path: Path to the AHA Bench 2.0 - Questions list.csv file
        dimensions: Dictionary of dimension names to AhaDimension objects

    Returns:
        List of Aha2DatasetRecord objects
    """
    records = []

    # Create a case-insensitive lookup dictionary for dimensions
    dimension_lookup = {
        dim_name.lower(): dim_name for dim_name in dimensions.keys()}

    # Track all unmatched tags to raise a comprehensive error at the end
    unmatched_tags = set()

    try:
        with open(questions_csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            # Check if validated tags column exists
            has_validated_column = 'Validated tags' in reader.fieldnames
            if has_validated_column:
                logger.info(
                    "Using 'Validated tags' column for dimension mapping")
            else:
                logger.info(
                    "'Validated tags' column not found, using 'Tags' column with case-insensitive matching")

            for i, row in enumerate(reader):
                # Skip empty rows or header rows
                if not row.get('Question'):
                    continue

                # Parse tags, using "Validated tags" if present, otherwise use "Tags"
                tags = []

                if has_validated_column and row.get('Validated tags'):
                    # Use validated tags directly - they should already match dimension names
                    validated_tags = [tag.strip()
                                      for tag in row['Validated tags'].split(',')]
                    for tag in validated_tags:
                        if not tag:  # Skip empty tags
                            continue

                        # Even with validated tags, verify they match dimensions
                        if tag in dimensions:
                            tags.append(tag)
                        else:
                            # Still track any unmatched validated tags
                            unmatched_tags.add(tag)

                elif row.get('Tags'):
                    # Fall back to original tags with case-insensitive matching
                    raw_tags = [tag.strip() for tag in row['Tags'].split(',')]
                    for tag in raw_tags:
                        if not tag:  # Skip empty tags
                            continue

                        # Case-insensitive match
                        tag_lower = tag.lower()
                        if tag_lower in dimension_lookup:
                            # Use the correctly-cased dimension name
                            tags.append(dimension_lookup[tag_lower])
                        else:
                            # Track the unmatched tag
                            unmatched_tags.add(tag)

                # Ensure there's at least one valid tag/dimension
                if not tags:
                    logger.warning(
                        f"Question {i+1} has no valid tags matching dimensions, skipping")
                    continue

                record = Aha2DatasetRecord(
                    sample_id=i + 1,  # Use row index + 1 as sample_id
                    question=row['Question'],
                    tags=tags,  # Only include validated tags
                    correct_answer=row.get('Correct answer'),
                    comments=row.get('Comments/Discussion')
                )
                records.append(record)

        # If there were any unmatched tags, raise an error
        if unmatched_tags:
            sorted_unmatched = sorted(unmatched_tags)
            error_msg = f"The following tags in the questions CSV do not match any dimension: {', '.join(sorted_unmatched)}"
            raise ValueError(error_msg)

        logger.info(f"Loaded {len(records)} questions")
        return records
    except Exception as e:
        logger.error(f"Error loading questions from {questions_csv_path}: {e}")
        raise


def record_to_aha2_sample(record: Aha2DatasetRecord, dimensions: Dict[str, AhaDimension]) -> Sample:
    """
    Converts an Aha2DatasetRecord to an Inspect Sample.

    Args:
        record: The dataset record
        dimensions: Dictionary of dimension names to AhaDimension objects

    Returns:
        An Inspect Sample object
    """
    # Create input with no special formatting - just the question
    sample_input = f"{record.question}"

    # Only include dimensions that match the record's tags
    applicable_dimensions = {
        tag: dimensions[tag] for tag in record.tags if tag in dimensions
    }

    # For the target, include only the relevant dimensions
    dimension_info = "\n".join([
        f"- {d.name}: {d.guiding_question}" for d in applicable_dimensions.values()
    ])

    # This target will be used for formatting the judge prompt later
    target = f"Evaluate the answer for the following dimensions:\n{dimension_info}"

    # Add the applicable dimension names to the metadata
    metadata = record.to_sample_metadata()
    metadata["applicable_dimensions"] = list(applicable_dimensions.keys())

    return Sample(
        input=sample_input,
        target=target,
        metadata=metadata,
        id=str(record.sample_id)
    )


def load_aha2_samples(base_csv_path: Path, questions_csv_path: Path) -> tuple[List[Sample], Dict[str, AhaDimension]]:
    """
    Loads both dimensions and questions and converts them to Inspect Samples.

    Args:
        base_csv_path: Path to the AHA Bench 2.0 - Base.csv file
        questions_csv_path: Path to the AHA Bench 2.0 - Questions list.csv file

    Returns:
        Tuple of (list of Samples, dictionary of dimensions)
    """
    dimensions = load_dimensions(base_csv_path)
    records = load_questions(questions_csv_path, dimensions)

    samples = []
    for record in records:
        samples.append(record_to_aha2_sample(record, dimensions))

    return samples, dimensions
