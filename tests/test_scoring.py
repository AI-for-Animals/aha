import pytest
from aha.scoring import _validate_judge_response, _normalize_response


def test_normalize_response_string():
    assert _normalize_response("hello world") == "hello world"

def test_normalize_response_list_of_strings():
    assert _normalize_response(["hello", "world"]) == "hello world"

class MockContentBlock:
    def __init__(self, text: str):
        self.text = text

def test_normalize_response_list_of_content_blocks():
    blocks = [MockContentBlock(text="hello"), MockContentBlock(text="world")]
    assert _normalize_response(blocks) == "hello"

def test_normalize_response_single_content_block():
    block = MockContentBlock(text="single block")
    assert _normalize_response(block) == "single block"

def test_normalize_response_other_types():
    assert _normalize_response(123) == "123"
    assert _normalize_response(None) == "None"

# --- Tests for _validate_judge_response ---

@pytest.mark.parametrize(
    "response_text, expected_valid, expected_score",
    [
        # Valid cases
        ("[A]\nAssessment text thirty to forty words long exactly ok.\n[-1]", True, -1),
        ("[B]\nAnother assessment text, thirty to forty words. Padding.\n[0]", True, 0),
        ("[E]\nMore assessment text, needs to be long enough for words. thirty to forty.\n[1]", True, 1),
        # Valid with extra whitespace
        ("  [C]  \n  Assessment text thirty to forty words long exactly ok. \n  [0]  ", True, 0),
        # Invalid category
        ("[F]\nAssessment text thirty to forty words long exactly ok.\n[-1]", False, 0),
        # Invalid score
        ("[A]\nAssessment text thirty to forty words long exactly ok.\n[2]", False, 0),
        ("[A]\nAssessment text thirty to forty words long exactly ok.\n[-2]", False, 0),
        # Missing category
        ("Assessment text thirty to forty words long exactly ok.\n[-1]", False, 0),
        # Missing score
        ("[A]\nAssessment text thirty to forty words long exactly ok.", False, 0),
        # Malformed brackets
        ("A\nAssessment text thirty to forty words long exactly ok.\n[-1]", False, 0),
        ("[A]\nAssessment text thirty to forty words long exactly ok.\n-1", False, 0),
        ("A]\nAssessment text thirty to forty words long exactly ok.\n[-1]", False, 0),
        ("[A\nAssessment text thirty to forty words long exactly ok.\n[-1]", False, 0),
        # Empty responses
        ("", False, 0),
        ("\n", False, 0),
    ]
)
def test_validate_response(response_text, expected_valid, expected_score):
    is_valid, score = _validate_judge_response(response_text)
    assert is_valid == expected_valid
    assert score == expected_score
