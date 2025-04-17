"""
Inspect Task definition for the Animal Harm Assessment (AHA) benchmark.
"""
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.solver import generate, Solver
from aha.scoring import SOLVER_INSTRUCTIONS, final_digit_model_graded_qa
from aha.utils import setup_logging

setup_logging(logging.INFO)
logging.getLogger("inspect_ai.model").setLevel(logging.INFO)

def record_to_sample(record: Dict[str, Any]) -> Optional[Sample]:
    # Check for required fields first (only question and sample_id needed now)
    required_fields = ["sample_id", "question"]
    missing_fields = [field for field in required_fields if field not in record]
    if missing_fields:
        logging.warning(
            f"Record missing required fields ({', '.join(missing_fields)}): "
            f"sample_id={record.get('sample_id', '?')}, "
            f"question='{record.get('question', '?')[:50]}...'"
        )
        return None # Skip this record

    meta = record.get("generation_tags", {}).copy()
    meta["sample_id"] = record["sample_id"]
    meta["tags"] = record.get("tags", [])

    # Add reference answers to metadata if they exist
    ans_match = record.get("answer_matching_behavior", None)
    ans_not_match = record.get("answer_not_matching_behavior", None)
    if ans_match:
        meta["answer_matching_behavior"] = ans_match
    if ans_not_match:
        meta["answer_not_matching_behavior"] = ans_not_match

    # Target should be the solver instructions for the judge model
    return Sample(
        input=f"{record['question']} Your answer must be EXACTLY 30–40 words long.",
        target=SOLVER_INSTRUCTIONS,
        metadata=meta
    )

@task
def aha_evaluation(
    model: Optional[Solver] = None,
    judges: List[str] = ["anthropic/claude-3-5-sonnet-20240620", "google/gemini-1.5-flash", "openai/gpt-4o"],
    model_temperature: Optional[float] = 0.5,
    judge_temperature: Optional[float] = 0.5,
    seed: int = 42,
    max_tokens: int = 1000,
    max_retries: int = 3,
    max_messages: int = 20,
) -> Task:
    data_file = Path(__file__).parent / "data_public.json"
    if not data_file.exists():
        raise FileNotFoundError(f"Cannot find data_public.json at {data_file}")

    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from {data_file}: {e}") from e

    if not isinstance(raw_data, dict) or "data" not in raw_data or not isinstance(raw_data["data"], list):
        raise ValueError(f"Expected {data_file} to be a JSON object with a top-level 'data' list.")

    records = raw_data["data"]

    # Process records into Samples, handling potential errors and None returns
    samples = []
    for i, record in enumerate(records):
        try:
            # Ensure record is a dictionary before processing
            if not isinstance(record, dict):
                logging.warning(f"Skipping non-dict record at index {i}: {str(record)[:100]}...")
                continue

            sample = record_to_sample(record)
            if sample: # Only append if record_to_sample didn't return None
                 samples.append(sample)
        except Exception as e:
            # Catch any unexpected errors during processing of a single record
            logging.error(f"Unexpected error processing record {record.get('sample_id', '?')} at index {i}: {e}", exc_info=True)
            continue # Skip this record

    if not samples:
        # Raise error only if NO valid samples were loaded after processing ALL records
        raise ValueError("No valid samples could be loaded from the dataset after processing all records.")

    ds = MemoryDataset(samples=samples, name=data_file.stem)

    solver_instance = model or generate(
        max_tokens=max_tokens,
        temperature=model_temperature if model_temperature is not None else None,
        seed=seed
    )

    scorer_instance = final_digit_model_graded_qa(
        scorer_models=judges,
        judge_temperature=judge_temperature if judge_temperature is not None else None
    )

    return Task(
        dataset=ds,
        solver=solver_instance,
        scorer=scorer_instance,
        max_retries=max_retries,
        metrics=[]
    )
