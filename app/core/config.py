# Configuration settings for the application
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Satellite Ship Classifier API"
    MODEL_PATH: str = "app/ml/models/model_weights_v1.pth"
    MAX_IMAGE_SIZE_MB: int = 100

settings = Settings()
