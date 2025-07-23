# Use a base image with Python
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy source code
COPY src/ ./src/
COPY models/ ./models/

# Install required packages
RUN pip install scikit-learn joblib

# Default command
CMD ["python", "src/predict.py"]

