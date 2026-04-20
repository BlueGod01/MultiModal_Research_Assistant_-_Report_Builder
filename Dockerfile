FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UPLOAD_DIR=/app/uploads

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app

# Create uploads directory (ensure it exists)
RUN mkdir -p /app/uploads

# Expose port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
