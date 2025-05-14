import csv
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional

from aha.types import AhaDimension, Aha2DatasetRecord
from inspect_ai.dataset import Sample

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_dimensions(base_csv_path: Path) -> list[AhaDimension]:
    dimensions = []
    with open(base_csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Skip empty rows or header rows
            if not row.get('Dimension') or row['Dimension'] == 'Dimension':
                continue
            dimensions.append(AhaDimension(
                name=row['Dimension'],
                weight=float(row.get('Default weighting', 1.0)),
                scoring_description=row.get('Scoring', ''),
                guiding_question=row.get('Guiding Question', ''),
                indicators=row.get(
                    'Observable Indicators (pattern-match)', '')
            ))
    return dimensions


def process_species_variables(text: str, species_singular: Optional[str] = None, species_plural: Optional[str] = None) -> str:
    # Replace all instances of {species-singular:*} with the provided singular species
    if species_singular is not None:
        singular_pattern = r'\{species-singular:([^}]*)\}'
        text = re.sub(singular_pattern, species_singular, text)
    
    # Replace all instances of {species-plural:*} with the provided plural species
    if species_plural is not None:
        plural_pattern = r'\{species-plural:([^}]*)\}'
        text = re.sub(plural_pattern, species_plural, text)
    
    return text


def load_questions(
    questions_csv_path: Path, 
    dimensions: dict[str, AhaDimension],
    species_singular: Optional[str] = None,
    species_plural: Optional[str] = None
) -> List[Aha2DatasetRecord]:
    """
    Loads the questions from the questions CSV file.

    Args:
        questions_csv_path: Path to the AHA Bench 2.0 - Questions list.csv file
        dimensions: Dictionary of dimension names to AhaDimension objects
        species_singular: Single replacement value for all {species-singular:*} variables
        species_plural: Single replacement value for all {species-plural:*} variables

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
                
                # Process species variables in the question
                question = process_species_variables(
                    row['Question'],
                    species_singular,
                    species_plural
                )
                
                # Process species variables in the correct answer if it exists
                correct_answer = row.get('Correct answer', '')
                if correct_answer:
                    correct_answer = process_species_variables(
                        correct_answer,
                        species_singular,
                        species_plural
                    )
                
                # Process species variables in comments if they exist
                comments = row.get('Comments/Discussion', '')
                if comments:
                    comments = process_species_variables(
                        comments,
                        species_singular,
                        species_plural
                    )

                record = Aha2DatasetRecord(
                    sample_id=i + 1,  # Use row index + 1 as sample_id
                    question=question,
                    tags=tags,  # Only include validated tags
                    correct_answer=correct_answer,
                    comments=comments
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


def load_aha2_samples(
    dimensions: dict[str, AhaDimension],
    questions_csv_path: Path,
    species_singular: Optional[str] = None,
    species_plural: Optional[str] = None
) -> list[Sample]:
    records = load_questions(
        questions_csv_path, 
        dimensions,
        species_singular=species_singular,
        species_plural=species_plural
    )
    return [
        Sample(
            input=record.question,
            # The target will get used as the `criterion` in the judge prompt
            target=record.tags,
            metadata=record.to_sample_metadata(),
            id=str(record.sample_id)
        )
        for record in records
    ]
