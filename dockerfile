# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (if needed for chromadb, etc.)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Create directory for ChromaDB (persistent volume)
RUN mkdir -p /app/data/chromadb

# Expose port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]