FROM python:3.10-slim
 
WORKDIR /app
 
# Copy requirements and install (CPU-only torch)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
 
# Copy app files
COPY main.py .
COPY model.pkl .
COPY scaler.pkl .
 
# Expose port
EXPOSE 8000
 
# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]