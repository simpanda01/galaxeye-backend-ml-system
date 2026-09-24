from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from PIL import Image
from io import BytesIO

from app.model import SatelliteClassifier
from app.database import (
    init_database,
    save_prediction,
    get_predictions
)


app = FastAPI(
    title="GalaxEye Satellite Classification API",
    version="1.0.0"
)

classifier = SatelliteClassifier()

# Predictions below this confidence are flagged as uncertain.
CONFIDENCE_THRESHOLD = 0.60

# Create database/table when the application starts.
init_database()


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "GalaxEye Satellite Classification API"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )

    try:
        contents = await file.read()

        image = Image.open(BytesIO(contents))

        result = classifier.predict(image)

        confidence = result["confidence"]

        # Flag low-confidence predictions instead of treating
        # every prediction as equally reliable.
        status = (
            "accepted"
            if confidence >= CONFIDENCE_THRESHOLD
            else "uncertain"
        )

        save_prediction(
            filename=file.filename,
            prediction=result["prediction"],
            confidence=confidence,
            status=status
        )

        return {
            "filename": file.filename,
            "prediction": result["prediction"],
            "confidence": confidence,
            "status": status,
            "model_version": "resnet18-v1",
            "stored": True
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process image: {str(e)}"
        )


@app.get("/predictions")
def get_all_predictions(
    prediction: str | None = Query(
        default=None,
        description="Filter by predicted class"
    ),
    min_confidence: float | None = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Return predictions above this confidence"
    )
):
    return get_predictions(
        prediction=prediction,
        min_confidence=min_confidence
    )