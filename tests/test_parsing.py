import pytest

from aeollm2_osodt.parsing import parse_score_object


def test_marker_delimited_json_parses():
    raw = '###BEGIN_JSON###\n{"insight": [7.5, 8]}\n###END_JSON###'
    assert parse_score_object(raw, {"insight": 2}) == {"insight": [7.5, 8.0]}


def test_wrong_length_fails():
    with pytest.raises(ValueError, match="expected 2 scores"):
        parse_score_object('{"insight": [7.5]}', {"insight": 2})


def test_extra_keys_fail():
    with pytest.raises(ValueError, match="response keys"):
        parse_score_object('{"insight": [7.5], "readability": [8]}', {"insight": 1})


def test_prose_is_not_guessed():
    with pytest.raises(ValueError):
        parse_score_object("There are 2 criteria. Scores are 8 and 7.", {"insight": 2})


def test_e9_comment_compatibility_is_explicit_and_keeps_validation():
    raw='```json\n{"insight": [\n7.5, // evidence\n8.0 // evidence\n]}\n```'
    with pytest.raises(ValueError,match='invalid JSON'):
        parse_score_object(raw,{'insight':2})
    assert parse_score_object(raw,{'insight':2},allow_comments=True)=={'insight':[7.5,8.0]}
    with pytest.raises(ValueError,match='expected 3 scores'):
        parse_score_object(raw,{'insight':3},allow_comments=True)


def test_comment_compatibility_does_not_strip_quoted_slashes():
    assert parse_score_object('{"insight//detail":[7.5] // outside\n}',{'insight//detail':1},allow_comments=True)=={'insight//detail':[7.5]}


def test_duplicate_dimension_key_is_rejected():
    with pytest.raises(ValueError,match='duplicate JSON key'):
        parse_score_object('{"insight":[8],"insight":[2]}',{'insight':1})
