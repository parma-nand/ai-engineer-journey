FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Upgrade pip and install torch + other dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.2.0+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]