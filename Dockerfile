# Use slim Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency list and install
COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data files
COPY . /app

# Run the app with uvicorn (main.py contains `app = FastAPI(...)`)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]

