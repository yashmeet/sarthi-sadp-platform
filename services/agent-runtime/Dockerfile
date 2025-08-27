FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install essential packages including database support
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    google-generativeai==0.3.0 \
    pydantic==2.5.0 \
    psycopg2-binary==2.9.9 \
    structlog==23.2.0 \
    google-cloud-secret-manager==2.16.2

# Copy the simple production files
COPY src/main_simple_prod.py ./
COPY src/poml_simple_api.py ./

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "main_simple_prod:app", "--host", "0.0.0.0", "--port", "8080"]