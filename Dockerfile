FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Create necessary directories
RUN mkdir -p /app/data

# Copy CSV file to the correct location
COPY resultados_60_ofc.csv /app/resultados_60_ofc.csv

# Copy all model and scaler files
COPY src/model_*.keras /app/
COPY src/scaler_*.pkl /app/

# Command to run the application
CMD ["python", "src/main.py"] 