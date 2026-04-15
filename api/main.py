"""
FastAPI Batch Processing for AMOLED Defect Detection
"""

import io
import base64
import uuid
import asyncio
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import cv2
from PIL import Image

# Import your existing model
import sys
sys.path.append('/home/amrmuhammad/dev/amoled-defect-detection')
from src.model_transfer import TransferDefectDetector

# Initialize
app = FastAPI(title="AMOLED Defect Detection API", version="1.0.0")

# Load model once at startup
detector = TransferDefectDetector(input_shape=(128, 128, 3))
detector.load_model('models/defect_detector_v2.keras')
print("✅ Model loaded")

# Simple in-memory job storage (replace with Redis for production)
job_storage = {}

class PredictionRequest(BaseModel):
    image_base64: str
    image_id: Optional[str] = None

class BatchRequest(BaseModel):
    images: List[PredictionRequest]

class PredictionResponse(BaseModel):
    image_id: str
    defect: bool
    confidence: float
    defect_type: Optional[str]
    inference_time_ms: float

class BatchResponse(BaseModel):
    job_id: str
    status: str
    total_images: int
    results: Optional[List[PredictionResponse]] = None

def process_single_image(image_bytes: bytes) -> dict:
    """Run inference on a single image"""
    import time
    start = time.time()
    
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Predict
    confidence, label = detector.predict(img)
    
    inference_time = (time.time() - start) * 1000  # ms
    
    # Map confidence to defect type (simple heuristic)
    defect_type = None
    if confidence > 0.8:
        # In a full implementation, add a classifier for defect types
        defect_type = "defect_detected"
    
    return {
        "defect": label == "DEFECTIVE",
        "confidence": float(confidence),
        "defect_type": defect_type,
        "inference_time_ms": round(inference_time, 2)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(file: UploadFile = File(...)):
    """Analyze a single image file"""
    contents = await file.read()
    result = process_single_image(contents)
    return PredictionResponse(
        image_id=file.filename,
        **result
    )

@app.post("/batch", response_model=BatchResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    """Analyze multiple images in a batch"""
    job_id = str(uuid.uuid4())
    job_storage[job_id] = {"status": "processing", "results": []}
    
    results = []
    for file in files:
        contents = await file.read()
        result = process_single_image(contents)
        results.append(PredictionResponse(
            image_id=file.filename,
            **result
        ))
    
    job_storage[job_id] = {"status": "completed", "results": results}
    
    return BatchResponse(
        job_id=job_id,
        status="completed",
        total_images=len(files),
        results=results
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_storage[job_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
