import os
import requests


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)

def check_api_health():
    response = requests.get(
        f"{API_URL}/health",
        timeout=10
    )

    response.raise_for_status()

    return response.json()


def predict_from_api(image_file):
    if hasattr(image_file, "getvalue"):
        image_data = image_file.getvalue()
    else:
        image_data = image_file.read()

    content_type = getattr(
        image_file,
        "type",
        "image/jpeg"
    )

    files = {
        "file": (
            getattr(image_file, "name", "image.jpg"),
            image_data,
            content_type
        )
    }

    response = requests.post(
        f"{API_URL}/predict",
        files=files,
        timeout=30
    )

    response.raise_for_status()

    return response.json()