import argparse
import csv
from pathlib import Path
from typing import List, Dict

from datasets import Dataset, DatasetDict, Features, Value, Sequence


QUESTIONS_CSV = "questions.csv"
DIMENSIONS_CSV = "dimensions.csv"


def _load_questions(path: Path) -> List[Dict]:
    """Load question rows from the questions.csv file."""
    questions_path = path / QUESTIONS_CSV
    if not questions_path.exists():
        raise FileNotFoundError(f"{questions_path} not found")

    records = []
    with questions_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            tags = [t.strip() for t in row["Validated tags"].split(",") if t.strip()]
            question = row.get("Question", "").strip()
            if not question or not tags:
                continue
            records.append({
                "id": idx,
                "question": row.get("Question", "").strip(),
                "tags": tags,
            })
    return records


def _load_dimensions(path: Path) -> List[Dict]:
    """Load dimension rows from the dimensions.csv file."""
    dims_path = path / DIMENSIONS_CSV
    if not dims_path.exists():
        raise FileNotFoundError(f"{dims_path} not found")

    records = []
    with dims_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            records.append({
                "id": idx,
                "dimension": row.get("Dimension", "").strip(),
                "default_weighting": float(row.get("Default weighting", 1.0)),
                "scoring": row.get("Scoring", "").strip(),
                "guiding_question": row.get("Guiding Question", "").strip(),
                "observable_indicators": row.get("Observable Indicators (pattern-match)", "").strip(),
            })
    return records


def build_dataset(data_dir: Path) -> DatasetDict:
    """Create a DatasetDict with two splits: questions and dimensions."""
    questions = _load_questions(data_dir)
    dimensions = _load_dimensions(data_dir)

    question_features = Features({
        "id": Value("int32"),
        "question": Value("string"),
        "tags": Sequence(Value("string")),
    })

    dimension_features = Features({
        "id": Value("int32"),
        "dimension": Value("string"),
        "default_weighting": Value("float32"),
        "scoring": Value("string"),
        "guiding_question": Value("string"),
        "observable_indicators": Value("string"),
    })

    ds_questions = Dataset.from_list(questions, features=question_features)
    ds_dimensions = Dataset.from_list(dimensions, features=dimension_features)

    return ds_questions, ds_dimensions


def main():
    parser = argparse.ArgumentParser(description="Convert AHA 2.0 CSVs to a Hugging Face dataset.")
    parser.add_argument("--data-dir", type=str, default="../data", help="Directory containing CSV files.")
    parser.add_argument("--push", action="store_true", help="Push the dataset to the Hugging Face Hub.")
    parser.add_argument("--repo-id", type=str, help="Target Hub repo (e.g. ai-for-animals/aha2). Required if --push is set.")
    parser.add_argument("--private", action="store_true", help="Push the dataset as a private repo.")

    args = parser.parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()

    ds_questions, ds_dimensions = build_dataset(data_dir)

    if args.push:
        if not args.repo_id:
            raise ValueError("--repo-id must be provided when using --push")
        ds_questions.push_to_hub(
            args.repo_id,
            config_name="questions",
            private=args.private,
        )
        ds_dimensions.push_to_hub(
            args.repo_id,
            config_name="dimensions",
            private=args.private,
        )
        print(
            "Questions + dimensions pushed to https://huggingface.co/datasets/"
            f"{args.repo_id} (configs: 'questions', 'dimensions')"
        )
    else:
        out_path = data_dir / "hf_dataset"
        (DatasetDict({"train": ds_questions})).save_to_disk(out_path / "questions")
        ds_dimensions.save_to_disk(out_path / "dimensions")
        print(f"Dataset saved locally to {out_path}")


if __name__ == "__main__":
    main()
