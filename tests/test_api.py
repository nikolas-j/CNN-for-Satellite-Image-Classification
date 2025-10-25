import pytest
from fastapi.testclient import TestClient
from app.main import app
import io


# 'client' acts like a browser calling your API.
client = TestClient(app)

def test_predict_success():

    file_path = "tests/test_images/image4.png" 
    required_resolution = "3.0"

    # Send a POST request to /api/v1/predict with the test image
    with open(file_path, "rb") as f:

        files_payload = {
            "file": (file_path, f, "image/png")
        }

        data_payload = {
            "resolution_m_per_pixel": required_resolution
        }

        response = client.post(
            "/api/v1/predict",
            files=files_payload,
            data=data_payload
        )
    
    # 🚨 TEMPORARY DIAGNOSTIC CODE 🚨
    if response.status_code != 200:
        print("\n--- API Traceback (500 Error) ---")
        # Try to print the JSON detail if it exists
        try:
            print(response.json())
        except:
            # If no JSON, print the raw text content
            print(response.text)
        print("---------------------------------")
    # 🚨 END DIAGNOSTIC CODE 🚨
    
    assert response.status_code == 200
    data = response.json()
    assert "ship_count" in data
    assert isinstance(data["positions"], list)


def test_predict_wrong_file_type():

    not_a_png = io.BytesIO(b"this is a text file, not a png")
    
    response = client.post(
        "/api/v1/predict",
        files={"file": ("test.txt", not_a_png, "text/plain")},
        data={"resolution_m_per_pixel": "3.0"}
    )
    
    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["detail"]


def test_predict_missing_resolution():
    file_path = "tests/test_images/image4.png" 
    
    with open(file_path, "rb") as f:
        response = client.post(
            "/api/v1/predict",
            files={"file": ("sample_forest.png", f, "image/png")}
            # 'data' parameter is omitted entirely
        )
    
    assert response.status_code == 422
    assert "resolution_m_per_pixel" in response.json()["detail"][0]["loc"]