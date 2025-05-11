from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union


class AhaDatasetRecord(BaseModel):
    """Structure of a single record in the AHA dataset JSON."""
    sample_id: int
    question: str
    tags: List[str] = Field(default_factory=list)
    generation_tags: Dict[str, Any] = Field(default_factory=dict)
    answer_matching_behavior: Optional[str] = None
    answer_not_matching_behavior: Optional[str] = None

    def to_sample_metadata(self) -> dict:
        return {
            k: v for k, v in self.model_dump().items()
            if k in ["sample_id", "tags", "answer_matching_behavior",
                     "answer_not_matching_behavior"]
        }


class AhaJudgeScoreMetadata(BaseModel):
    """Metadata stored within the Score object by the AHA scorer."""
    judge_scores: Dict[str, int]
    judge_explanations: Dict[str, str]
    defaulted: bool
    defaulted_count: int
    num_judges: int

# NEW CLASSES FOR AHA 2.0


class AhaDimension(BaseModel):
    """Represents a single scoring dimension in AHA 2.0."""
    name: str
    weight: float = 1.0
    scoring_description: str
    guiding_question: str
    indicators: str = ""


class AhaDimensionScore(BaseModel):
    """Score for a single dimension in AHA 2.0."""
    dimension: str
    score: int  # 1 for correct, 0 for incorrect
    explanation: str


class Aha2DatasetRecord(BaseModel):
    """Structure of a single record in the AHA 2.0 dataset CSV."""
    sample_id: int
    question: str
    tags: List[str] = Field(default_factory=list)
    correct_answer: Optional[str] = None
    comments: Optional[str] = None

    def to_sample_metadata(self) -> dict:
        return {
            k: v for k, v in self.model_dump().items()
            if k in ["sample_id", "tags", "correct_answer", "comments"]
        }


class Aha2JudgeScoreMetadata(BaseModel):
    """Metadata stored within the Score object by the AHA 2.0 scorer."""
    dimension_scores: Dict[str, AhaDimensionScore]
    weighted_total_score: float
    max_possible_score: float
    judge_model: str
    defaulted: bool = False


class Aha2CombinedScoreMetadata(BaseModel):
    """Combined metadata from multiple judges for AHA 2.0 scoring."""
    judge_results: Dict[str, Aha2JudgeScoreMetadata]
    average_weighted_score: float
    dimension_averages: Dict[str, float]
    defaulted: bool = False
    defaulted_count: int = 0
    num_judges: int = 0
