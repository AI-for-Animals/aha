from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

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
