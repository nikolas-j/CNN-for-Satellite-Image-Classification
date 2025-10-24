# Satellite Image Ship Classification ML Pipeline

This project implements a machine learning satellite image classification pipeline for counting and locating container ships in open waters. The service is implemented using an API which rececives an satellite image from the client and returns the count and positions the ships. The classification is done by first parsing the image into suitable chuncks and using a convolutional neural network to classify the images as either 1 or 0 for containing a ship or not, respectively. The dataset used to train the ML from Planet's openly licenced image set of images from the San Francisco Bay area.


### ML Service / API Structure:

1. Request: 

A user sends an HTTP POST request to /api/v1/predict with a .png file payload.

2. Routing:

- Server gets the request.

- It passes it to the app object in main.py.

- The app sees the /api/v1/predict URL and matches it to the router from endpoints.py.

3. Endpoint Logic:

- FastAPI, using the UploadFile type, efficiently handles the file upload.

- Validation: The if file.content_type... line checks the file's "MIME type." If it's not image/png, it immediately stops and sends a 415 error back to the user.

- Inference: If validation passes, the file's bytes are read and passed to predictor.predict(...).

- The CNN proudces a prediction and a prediction confidence (Softmax of the higher value)

- Response: The function returns a PredictionResponse object. FastAPI automatically converts this Pydantic object into a clean JSON string containing the results and sends it back to the user with a 200 OK status.


