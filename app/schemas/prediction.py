# Defines what the JSON response should look like after a prediction is made

from pydantic import BaseModel

class PredictionResponse(BaseModel):
    """
    Defines the JSON response structure for a prediction.

    Tells FastAPI that a PredictionResponse object must have:
    filename (string), 
    content_type (string), 
    prediction (int), and 
    confidence (float).
    """
    filename: str
    content_type: str
    prediction: int
    confidence: float

    class Config:
        # This allows FastAPI to generate a sample response for its docs
        schema_extra = {
            "example": {
                "filename": "image_01.png",
                "content_type": "image/png",
                "prediction": 1,
                "confidence": 0.92,
            }
        }