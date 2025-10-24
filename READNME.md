# Satellite Image Ship Classification ML Pipeline






### How It All Works:

1. Request: 

A user sends an HTTP POST request to http://your-server.com/api/v1/predict and attaches a .png file.

2. Routing:

- The uvicorn server gets the request.

- It passes it to the app object in main.py.

- The app sees the /api/v1/predict URL and matches it to the router from endpoints.py.

3. Endpoint Logic:

- FastAPI, using the UploadFile type, efficiently handles the file upload.

- Validation: The if file.content_type... line checks the file's "MIME type." If it's not image/png, it immediately stops and sends a 415 error back to the user.

- Inference: If validation passes, the file's bytes are read and passed to predictor.predict(...).

- Response: The function returns a PredictionResponse object. FastAPI automatically converts this Pydantic object into a clean JSON string and sends it back to the user with a 200 OK status.