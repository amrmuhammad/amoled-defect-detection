"""
FastAPI Batch Processing for AMOLED Defect Detection - Multi-Class Version
Returns defect type (Clean, Dead Pixel, Stuck Pixel, Mura, Scratch, Dust)
"""

import io
import uuid
import time
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import cv2
from PIL import Image

# Import your multi-class model
import sys
sys.path.append('/home/amrmuhammad/dev/amoled-defect-detection')
from src.model_multi_class import MultiClassDefectDetector

# Initialize
app = FastAPI(title="AMOLED Defect Detection API", version="2.0.0")

# Load multi-class model once at startup
detector = MultiClassDefectDetector(input_shape=(128, 128, 3), num_classes=6)
detector.load_model('models/multi_class_defect_detector.keras')
print("✅ Multi-class model loaded (98.67% accuracy)")

# Simple in-memory job storage (replace with Redis for production)
job_storage = {}

class PredictionResponse(BaseModel):
    image_id: str
    defect: bool
    defect_type: str
    confidence: float
    inference_time_ms: float

class BatchResponse(BaseModel):
    job_id: str
    status: str
    total_images: int
    results: Optional[List[PredictionResponse]] = None

def process_single_image(image_bytes: bytes, image_name: str) -> dict:
    """Run multi-class inference on a single image"""
    start_time = time.time()
    
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Predict (returns class_index, class_name, confidence)
    class_idx, class_name, confidence = detector.predict(img)
    
    inference_time = (time.time() - start_time) * 1000  # ms
    
    return {
        "image_id": image_name,
        "defect": class_idx != 0,
        "defect_type": class_name,
        "confidence": round(confidence, 4),
        "inference_time_ms": round(inference_time, 2)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(file: UploadFile = File(...)):
    """Analyze a single image file"""
    contents = await file.read()
    result = process_single_image(contents, file.filename)
    return PredictionResponse(**result)

@app.post("/batch", response_model=BatchResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    """Analyze multiple images in a batch"""
    job_id = str(uuid.uuid4())
    job_storage[job_id] = {"status": "processing", "results": []}
    
    results = []
    for file in files:
        try:
            contents = await file.read()
            result = process_single_image(contents, file.filename)
            results.append(PredictionResponse(**result))
        except Exception as e:
            results.append(PredictionResponse(
                image_id=file.filename,
                defect=False,
                defect_type="Error",
                confidence=0.0,
                inference_time_ms=0.0
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
    return {"status": "healthy", "model": "multi-class", "accuracy": "98.67%"}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_storage[job_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
