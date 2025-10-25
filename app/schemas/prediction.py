# app/schemas/prediction.py

from pydantic import BaseModel
from typing import List

# Coordinate definition to explicitly be a list/array structure
Coordinate = List[float] 

class PredictionResponse(BaseModel):
    """
    Defines the JSON response structure for a prediction.
    """
    filename: str
    ship_count: int
    positions: List[Coordinate] 
    class Config:
        schema_extra = {
            "example": {
                "filename": "image_01.png",
                "ship_count": 2,
                "positions": [[0.5, 0.4233], [0.122, 0.299]] 
            }
        }