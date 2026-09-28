from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "mobilenetv2_finetuned_best.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"

IMAGE_SIZE = (224, 224)


# ============================================================
# Load model
# ============================================================

def load_model():
    """Load the fine-tuned MobileNetV2 model."""

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "preprocess_input": preprocess_input
        },
        safe_mode=False
    )

    return model


# ============================================================
# Load class names
# ============================================================

def load_class_names():
    """Load class names from JSON file."""

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# Image preprocessing
# ============================================================

def preprocess_image(image):
    """Prepare image for MobileNetV2 prediction."""

    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image).astype("float32")

    # MobileNetV2 preprocessing
    image_array = preprocess_input(image_array)

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ============================================================
# Prediction
# ============================================================

def predict_image(image):
    """Predict tomato leaf disease from an image."""

    model = load_model()
    class_names = load_class_names()

    processed_image = preprocess_image(image)

    predictions = model.predict(
        processed_image,
        verbose=0
    )

    predicted_index = int(np.argmax(predictions[0]))

    confidence = float(
        predictions[0][predicted_index]
    )

    predicted_class = class_names[predicted_index]

    return {
        "class": predicted_class,
        "confidence": confidence,
        "probabilities": predictions[0].tolist()
    }