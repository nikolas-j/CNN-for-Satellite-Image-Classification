import pytest
from fastapi.testclient import TestClient
from app.main import app  # Import your main FastAPI app
import io

# 1. Create a "client" for your tests
# This 'client' acts like a browser or a script calling your API.
client = TestClient(app)

# 2. Write a test for a successful prediction
def test_predict_success():
    # Make sure you have a sample image in your 'tests/' folder
    file_path = "tests/test_images/image4.png" 
    
    with open(file_path, "rb") as f:
        # 'files' is how you upload a file with TestClient
        response = client.post(
            "/api/v1/predict",
            files={"file": ("sample_forest.png", f, "image/png")}
        )
    
    # Check 1: Did we get a 200 OK status?
    assert response.status_code == 200
    
    # Check 2: Is the response valid JSON?
    data = response.json()
    
    # Check 3: Does the JSON contain the keys we expect?
    assert "filename" in data
    assert "prediction" in data
    assert "confidence" in data
    
    # Check correct prediction (optional, based on your test image)
    # assert data["prediction"] == "1"

# Test for handling of unsupported file types
def test_predict_wrong_file_type():
    # Create a dummy text file in memory
    not_a_png = io.BytesIO(b"this is a text file, not a png")
    
    response = client.post(
        "/api/v1/predict",
        files={"file": ("test.txt", not_a_png, "text/plain")}
    )
    
    # Check 1: Got 415 error?
    assert response.status_code == 415
    
    # Check 2: Does the error detail match?
    assert "Unsupported file type" in response.json()["detail"]