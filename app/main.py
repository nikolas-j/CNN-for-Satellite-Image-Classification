# Entry point for the application

from fastapi import FastAPI
from app.api import endpoints  # Import the router from our endpoints file
from app.core.config import settings # Import configuration settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0"
)

# Tells the main 'app' to include all the routes (like /predict)
# that are defined in the 'endpoints.py' file.
# endpoints will have prefix /api/v1/predict
app.include_router(endpoints.router, prefix="/api/v1")

@app.get("/", tags=["Root"])
def read_root():
    """
    A simple 'hello world' endpoint to check if the API is running.
    """
    return {"status": "Satellite Classifier API is running."}