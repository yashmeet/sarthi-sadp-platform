FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for pandas and data processing
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY test_import.py ./

# Create directory for temporary dataset storage
RUN mkdir -p /tmp/kaggle_datasets

# Download NLTK data that may be needed
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)" || echo "NLTK download failed, continuing..."

# Test that the application imports correctly
RUN python test_import.py

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV KAGGLE_USERNAME=""
ENV KAGGLE_KEY=""

# Expose port
EXPOSE 8080

# Start the service with explicit startup timeout
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "120"]