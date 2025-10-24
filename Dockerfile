# 1. Use an official lightweight Python image
FROM python:3.10-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the application code
COPY ./app /app/app
COPY ./models /app/models

# 5. Expose the port the app runs on
EXPOSE 8080

# 6. Run the Uvicorn server
#    Host 0.0.0.0 is needed to accept connections from outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]