from pathlib import Path
from PIL import Image

from src.inference.predict import predict_image


# Change this to any real tomato image for testing
TEST_IMAGE = Path(
    "data/processed/tomato/test/healthy227"
)


image_files = list(TEST_IMAGE.glob("*.jpg"))

if not image_files:
    image_files = list(TEST_IMAGE.glob("*.jpeg"))

if not image_files:
    image_files = list(TEST_IMAGE.glob("*.png"))


if not image_files:
    raise FileNotFoundError(
        "No test image found."
    )


image_path = image_files[0]

print(f"Testing image: {image_path}")

image = Image.open(image_path)

result = predict_image(image)

print("\nPrediction Result")
print("=================")
print(f"Class: {result['class']}")
print(f"Confidence: {result['confidence'] * 100:.2f}%")