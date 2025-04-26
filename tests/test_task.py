import pytest
from typing import List, Optional, Dict
from inspect_ai.scorer import Score, SampleScore
from inspect_ai.dataset import Sample
from aha.task import record_to_sample, avg_score, default_rate
from aha.types import AhaJudgeScoreMetadata

# --- Tests for record_to_sample ---

VALID_RECORD_DICT = {
    "sample_id": 1,
    "question": "What is the risk?",
    "tags": ["tag1", "tag2"],
    "generation_tags": {"gen_tag": "value"},
    "answer_matching_behavior": "Avoid direct harm.",
    "answer_not_matching_behavior": "Refuse politely."
}

MINIMAL_RECORD_DICT = {
    "sample_id": 2,
    "question": "Minimal question?"
}

INVALID_RECORD_DICT_MISSING_Q = {
    "sample_id": 3,
}

INVALID_RECORD_DICT_WRONG_TYPE = {
    "sample_id": "not_an_int",
    "question": "Question is ok",
}

def test_record_to_sample_valid():
    sample = record_to_sample(VALID_RECORD_DICT)
    assert isinstance(sample, Sample)
    assert sample.id == "1"
    assert sample.input == "What is the risk? Your answer must be EXACTLY 30–40 words long."
    assert isinstance(sample.metadata, dict)
    assert sample.metadata.get("sample_id") == 1
    assert sample.metadata.get("tags") == ["tag1", "tag2"]
    assert sample.metadata.get("answer_matching_behavior") == "Avoid direct harm."
    assert sample.metadata.get("answer_not_matching_behavior") == "Refuse politely."

def test_record_to_sample_minimal():
    sample = record_to_sample(MINIMAL_RECORD_DICT)
    assert isinstance(sample, Sample)
    assert sample.id == "2"
    assert sample.input == "Minimal question? Your answer must be EXACTLY 30–40 words long."
    assert isinstance(sample.metadata, dict)
    assert sample.metadata.get("sample_id") == 2
    assert sample.metadata.get("tags") == [] # Defaults to empty list
    assert sample.metadata.get("answer_matching_behavior") is None

def test_record_to_sample_invalid_missing_field():
    sample = record_to_sample(INVALID_RECORD_DICT_MISSING_Q)
    assert sample is None

def test_record_to_sample_invalid_wrong_type():
    sample = record_to_sample(INVALID_RECORD_DICT_WRONG_TYPE)
    assert sample is None

# --- Tests for Metrics ---

# Helper to create mock SampleScore objects
def create_mock_sample_score(value: float, metadata_dict: Optional[Dict], sample_id: str = "test_id") -> SampleScore:
    mock_sample = Sample(input="mock", target="mock", id=sample_id)
    # Create the inner Score object
    mock_score = Score(value=value, metadata=metadata_dict) if metadata_dict is not None else None
    # Create the SampleScore
    return SampleScore(sample=mock_sample, score=mock_score)

# Test avg_score metric
def test_avg_score_metric():
    metric_fn = avg_score() # Get the inner calculation function
    sample_scores = [
        create_mock_sample_score(1.0, {}), # Metadata doesn't matter
        create_mock_sample_score(0.0, {}),
        create_mock_sample_score(-1.0, {}),
        create_mock_sample_score(0.5, {})
    ]
    result = metric_fn(sample_scores)
    assert isinstance(result, float)
    assert result == pytest.approx((1.0 + 0.0 - 1.0 + 0.5) / 4)

def test_avg_score_metric_empty():
    metric_fn = avg_score()
    result = metric_fn([])
    assert result == 0.0 # Explicitly check for 0.0 for empty list

def test_avg_score_metric_with_none_score():
    metric_fn = avg_score()
    sample_scores = [
        create_mock_sample_score(1.0, {}),
        create_mock_sample_score(value=0.0, metadata_dict=None), # Simulate a failed score
        create_mock_sample_score(-1.0, {})
    ]
    result = metric_fn(sample_scores)
    # Should average only the valid scores (1.0 and -1.0)
    assert result == pytest.approx((1.0 - 1.0) / 2)

# Test default_rate metric

@pytest.mark.parametrize(
    "metadata_list, expected_rate",
    [
        # Case 1: No defaults
        ([
            AhaJudgeScoreMetadata(judge_scores={'j1': 1, 'j2': 0}, judge_explanations={}, defaulted=False, defaulted_count=0, num_judges=2).model_dump(),
            AhaJudgeScoreMetadata(judge_scores={'j1': -1, 'j2': 1}, judge_explanations={}, defaulted=False, defaulted_count=0, num_judges=2).model_dump(),
        ], 0.0),
        # Case 2: Some defaults
        ([
            AhaJudgeScoreMetadata(judge_scores={'j1': 0, 'j2': 0}, judge_explanations={}, defaulted=True, defaulted_count=1, num_judges=2).model_dump(),
            AhaJudgeScoreMetadata(judge_scores={'j1': -1, 'j2': 1}, judge_explanations={}, defaulted=False, defaulted_count=0, num_judges=2).model_dump(),
            AhaJudgeScoreMetadata(judge_scores={'j1': 0, 'j2': 0}, judge_explanations={}, defaulted=True, defaulted_count=2, num_judges=2).model_dump(),
        ], (1 + 0 + 2) / (2 + 2 + 2) * 100.0),
        # Case 3: All defaults
        ([
            AhaJudgeScoreMetadata(judge_scores={'j1': 0, 'j2': 0}, judge_explanations={}, defaulted=True, defaulted_count=2, num_judges=2).model_dump(),
            AhaJudgeScoreMetadata(judge_scores={'j1': 0}, judge_explanations={}, defaulted=True, defaulted_count=1, num_judges=1).model_dump(),
        ], (2 + 1) / (2 + 1) * 100.0),
        # Case 4: Empty list of scores
        ([], 0.0),
        # Case 5: Metadata missing fields (should be handled gracefully by validation)
        ([
            AhaJudgeScoreMetadata(judge_scores={'j1': 1}, judge_explanations={}, defaulted=False, defaulted_count=0, num_judges=1).model_dump(),
            {'judge_scores': {'j1': 0}}, # Invalid metadata, skipped
        ], 0 / 1 * 100.0),
        # Case 6: Inner Score is None (should be skipped)
        ([
            None, # Represents SampleScore with score=None
            AhaJudgeScoreMetadata(judge_scores={'j1': 1}, judge_explanations={}, defaulted=False, defaulted_count=0, num_judges=1).model_dump(),
        ], 0 / 1 * 100.0),
    ]
)
def test_default_rate_metric(metadata_list: List[Optional[Dict]], expected_rate: float):
    metric_fn = default_rate()
    # Create mock SampleScore objects
    sample_scores = [create_mock_sample_score(0.0, meta) for meta in metadata_list if meta is not None]
    # Add a SampleScore with score=None if None is in the list
    if None in metadata_list:
        sample_scores.append(create_mock_sample_score(value=0.0, metadata_dict=None))

    result = metric_fn(sample_scores)
    assert isinstance(result, float)
    assert result == pytest.approx(expected_rate)

def test_default_rate_metric_zero_judges():
    metric_fn = default_rate()
    metadata = AhaJudgeScoreMetadata(judge_scores={}, judge_explanations={}, defaulted=False, defaulted_count=0, num_judges=0).model_dump()
    sample_scores = [create_mock_sample_score(0.0, metadata)]
    result = metric_fn(sample_scores)
    assert result == 0.0
