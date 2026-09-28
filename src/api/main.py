# ============================================================
# STAGE 9 — FASTAPI BACKEND
# ============================================================

from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

from src.inference.predict import predict_image


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Tomato Leaf Disease Detection API",
    description="API for tomato leaf disease classification",
    version="1.0.0"
)


# ============================================================
# 9.1 — HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Tomato Leaf Disease Detection API",
        "status": "running"
    }


@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }


# ============================================================
# 9.2 — IMAGE PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # --------------------------------------------------------
    # Validate file type
    # --------------------------------------------------------

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/jpg"
    }

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, JPEG or PNG image."
        )


    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    contents = await file.read()

    try:

        image = Image.open(
            BytesIO(contents)
        )

        image = image.convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )


    # --------------------------------------------------------
    # Image size validation
    # --------------------------------------------------------

    width, height = image.size

    if width < 100 or height < 100:

        raise HTTPException(
            status_code=400,
            detail="Image is too small."
        )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        result = predict_image(image)

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )


    # --------------------------------------------------------
    # API response
    # --------------------------------------------------------

    return {
        "filename": file.filename,
        "predicted_class": result["class"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"]
    }