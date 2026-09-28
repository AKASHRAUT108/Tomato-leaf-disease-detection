# ============================================================
# 11.3 — FASTAPI HEALTH TEST
# ============================================================

from src.api.client import check_api_health


def test_api_health():

    response = check_api_health()

    assert response is not None
    assert isinstance(response, dict)


def test_api_is_healthy():

    response = check_api_health()

    assert response.get("status") == "healthy"

# ============================================================
# 11.4 — API PREDICTION RESPONSE VALIDATION
# ============================================================

from pathlib import Path

from src.api.client import predict_from_api


def test_prediction_response_structure():

    test_image = Path(
        "tests/test_leaf.jpg"
    )

    if not test_image.exists():

        return

    with open(test_image, "rb") as image_file:

        response = predict_from_api(
            image_file
        )

    assert response is not None
    assert isinstance(response, dict)

    assert (
        "predicted_class" in response
        or "class" in response
    )

    assert "confidence" in response

# ============================================================
# 11.5 — PREDICTION CONFIDENCE VALIDATION
# ============================================================

from pathlib import Path

from src.api.client import predict_from_api


def test_prediction_confidence_range():

    test_image = Path(
        "tests/test_leaf.jpg"
    )

    if not test_image.exists():

        return

    with open(test_image, "rb") as image_file:

        response = predict_from_api(
            image_file
        )

    confidence = response.get(
        "confidence"
    )

    assert confidence is not None

    assert 0.0 <= float(confidence) <= 1.0