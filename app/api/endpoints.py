# Defines the API logic

from fastapi import APIRouter, Form, UploadFile, File, HTTPException, status
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

# Define what inputs the endpoint accepts
async def predict_image(
    file: UploadFile = File(..., description="The satellite image in PNG format."),
    resolution_m_per_pixel: float = Form(..., description="Image resolution in meters per pixel (e.g., 3.0)")
):
    """
    Accepts a .png satellite image, runs it through the classifier,
    and returns the prediction and confidence score.
    """
    
    # Validation
    if file.content_type != "image/png":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type. Please upload a .png image."
        )
    
    if resolution_m_per_pixel <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Resolution must be a positive value."
        )

    try:
        # Check file size and read contents
        image_bytes = await file.read()
        file_size_mb = len(image_bytes) / (1024 * 1024)  # Convert bytes to megabytes
        if file_size_mb > settings.MAX_IMAGE_SIZE_MB:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds the maximum limit of {settings.MAX_IMAGE_SIZE_MB} MB."
            )
        
        # Run Prediction
        # Returns ship_count in the image and positions
        ship_count, positions = predictor.predict(image_bytes, resolution_m_per_pixel)
        
        # Format and return the JSON response
        # Pydantic model defined in schemas.py
        return PredictionResponse(
            filename=file.filename,
            ship_count=ship_count,
            positions=positions
        )
        
    except Exception as e:
        # Handle any unexpected errors during prediction
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during prediction: {str(e)}"
        )