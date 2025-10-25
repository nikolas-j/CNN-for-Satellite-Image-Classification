# Defines what the JSON response should look like after a prediction is made

from pydantic import BaseModel
from typing import List, Tuple, Union

Coordinate = Tuple[float, float]
class PredictionResponse(BaseModel):
    """
    Defines the JSON response structure for a prediction.
    Tells FastAPI that a PredictionResponse object must have:
    filename (string), 
    ship_count (integer), and 
    positions (list of tuples).
    """

    filename: str
    ship_count: int
    positions: List[Coordinate]

    class Config:
        # This allows FastAPI to generate a sample response for its docs
        schema_extra = {
            "example": {
                "filename": "image_01.png",
                "ship_count": 4,
                "positions": [[0.4,0.5],[0.9,0.1]]
            }
        }