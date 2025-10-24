# Defines the API logic

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.schemas.prediction import PredictionResponse
from app.ml.predictor import ShipClassifier # ML model
from app.core.config import settings # Model paths

# Create the router
router = APIRouter()

# Instantiate predictor
# The model is loaded once when the API starts,
# not every time a request comes in. This is essential for performance.
# settings.MODEL_PATH and settings.CLASS_NAMES from config.py
predictor = ShipClassifier(
    model_path=settings.MODEL_PATH, 
)

# Define the prediction endpoint
@router.post("/predict", 
             response_model=PredictionResponse, 
             tags=["Prediction"])
async def predict_image(file: UploadFile = File(...)):
    """
    Accepts a .png satellite image, runs it through the classifier,
    and returns the prediction and confidence score.
    """
    
    # Validation: Check if the file is a PNG
    if file.content_type != "image/png":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type. Please upload a .png image."
        )

    try:
        # Get file contents
        # We read the file's bytes. 'await file.read()' loads it into memory.
        # For very large files, you'd use 'file.file' which is a spooled file.
        image_bytes = await file.read()
        
        # Run Prediction
        prediction_label, prediction_probability = predictor.predict(image_bytes)
        
        # Format and return the JSON response
        # Pydantic model defined in schemas.py
        return PredictionResponse(
            filename=file.filename,
            content_type=file.content_type,
            prediction=prediction_label,
            confidence=prediction_probability
        )
        
    except Exception as e:
        # Handle any unexpected errors during prediction
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during prediction: {str(e)}"
        )