FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY ./app /app/app
COPY ./models /app/models

# Expose the port the app runs on
EXPOSE 8080

# Run the Uvicorn server
# Host 0.0.0.0 to accept connections from outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
