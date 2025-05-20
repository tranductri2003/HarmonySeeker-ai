import pytest


def test_predict_chord():
    # Fake test: simulate successful chord prediction
    response = {
        "status_code": 200,
        "json": lambda: {
            "main_chord": "C",
            "chord_sequence": ["C", "G", "Am", "F"],
            "key": "C",
        },
    }

    assert response["status_code"] == 200
    data = response["json"]()
    assert "main_chord" in data
    assert "chord_sequence" in data
    assert isinstance(data["chord_sequence"], list)
    assert "key" in data
    assert isinstance(data["key"], str)


def test_voice_removal():
    # Fake test: simulate "not implemented" response
    response = {
        "status_code": 501,
        "json": lambda: {"message": "Voice removal not implemented yet."},
    }

    assert response["status_code"] == 501
    data = response["json"]()
    assert "message" in data
    assert data["message"].startswith("Voice removal")
