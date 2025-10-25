# Defines the API logic

from fastapi import APIRouter, Form, UploadFile, File, HTTPException, status
from app.schemas.prediction import PredictionResponse
from fastapi.responses import StreamingResponse
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

# Define the analyze json return format endpoint
@router.post("/analyze/json", 
             response_model=PredictionResponse, 
             tags=["Analysis"])

# Define what inputs the endpoint accepts
async def analyze_image(
    file: UploadFile = File(..., description="The satellite image in PNG format."),
    resolution_m_per_pixel: float = Form(..., description="Image resolution in meters per pixel (e.g., 3.0)")
):
    """
    Accepts an image, runs the ship classification analysis, and returns
    the total ship count and their unique coordinates in JSON format.
    """
    
    # Validation
    if file.content_type != "image/png":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Unsupported file type. Please upload a .png image."
        )
    
    if resolution_m_per_pixel <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Resolution must be a positive value."
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
    

# Define the marked .png image return format endpoint
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw
import io

@router.post("/analyze/image", 
             # No response_model, as we are returning raw image data
             tags=["Analysis"])
async def analyze_image_png(
    file: UploadFile = File(..., description="The satellite image in PNG format."),
    resolution_m_per_pixel: float = Form(..., description="Image resolution in meters per pixel (e.g., 3.0)")
):
    """
    Accepts an image, marks detected ships with red circles, and returns 
    the annotated image as a PNG file stream.
    """
    
    # 1. Validation (File/Resolution - same as above)
    # ... (Include the same validation checks as the JSON endpoint) ...

    try:
        image_bytes = await file.read() 
        
        _, positions = predictor.predict(image_bytes, resolution_m_per_pixel)
        
        # Visualization: Draw on the original image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(image)
        circle_radius = 50  # pixels
        width, height = image.size
        for x, y in positions:
            x_pos, y_pos = x * width, y * height
            bbox = [x_pos - circle_radius, y_pos - circle_radius, x_pos + circle_radius, y_pos + circle_radius]
            draw.ellipse(bbox, outline="red", width=3)

        # Save the image to an in-memory buffer
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Return the image stream
        return StreamingResponse(
            content=img_byte_arr, 
            media_type="image/png", 
            headers={"Content-Disposition": "inline; filename=annotated_ship_image.png"}
        )

    except Exception as e:
        print(f"Visualization Error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Visualization failed: {str(e)}")