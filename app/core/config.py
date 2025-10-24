# Configuration settings for the application


from pydantic import BaseSettings

# Placeholder for configuration settings
class Settings(BaseSettings):
    PROJECT_NAME: str = "Satellite Ship Classifier API"

    # Configure the model to be used
    MODEL_PATH: str = "app/ml/models/model_weights_v1.pth"

settings = Settings()