# Use a slim Python 3.11 environment optimized for container deployment
FROM python:3.11-slim

# Expose port required by Cloud Run
EXPOSE 8080

# Environment variables to optimize Streamlit
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_HEADLESS=true

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Strategically copy the sanitized source application and needed models
# (Excluding raw data via .dockerignore)
COPY app/ app/
COPY src/ src/
COPY data/processed/ data/processed/

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health || exit 1

# Start Streamlit application
ENTRYPOINT ["python", "-m", "streamlit", "run", "app/app.py"]
