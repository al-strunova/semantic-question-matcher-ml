FROM python:3.13-slim

WORKDIR /app

# Try without build tools first - add back if pip install fails
# RUN apt-get update && apt-get install -y \
#     gcc \
#     g++ \
#     && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application
COPY . .

# These env vars help with Flask and Python imports
ENV FLASK_APP=src.app
ENV PYTHONPATH=/app

# Expose Flask port
EXPOSE 5000

# Run Flask
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]