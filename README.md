================================================================================
AMOLED Defect Detection System
================================================================================

Python 3.12 | TensorFlow 2.13 | MIT License | arXiv paper

AI-powered inspection for AMOLED displays – 100% accuracy on 5 defect types, <0.1s inference on CPU.

================================================================================
Key Results
================================================================================

Accuracy: 100%
Precision: 100%
Recall: 100%
Inference Speed: <0.1 sec per image (CPU)
Defect Types: Dead pixel, Stuck pixel, Mura, Scratch, Dust

================================================================================
Features
================================================================================

    100% accurate defect classification

    5 defect types supported

    CPU-only inference – no GPU required

    Web dashboard – drag & drop, real-time results

    Batch processing API – REST API for MES integration

    Production-ready – Docker, health checks, async jobs

    Research paper – 17 pages, 5 figures, 22 references

================================================================================
Web Dashboard
================================================================================

A Streamlit-based interactive dashboard for live demos and manual inspection.

Run the Dashboard:

cd ~/dev/amoled-defect-detection
source venv/bin/activate
streamlit run dashboard/app.py

Then open http://localhost:8501

Dashboard Features:

    Upload single images (drag & drop or file browser)

    Real-time defect detection with confidence gauge

    Analysis history with metrics

    Export results as CSV/JSON

================================================================================
Batch Processing API
================================================================================

The system includes a production-ready REST API for batch defect detection, designed to integrate directly with manufacturing execution systems (MES) and production lines.

Features:

    Single image prediction – analyze one display image

    Batch processing – process up to 100 images per request

    Async job queue – handle large batches without blocking

    Enterprise JSON responses – easy integration with existing software

    CPU-optimized – <0.1s per image on standard hardware

API Endpoints:

GET /health Service health check
POST /predict Analyze a single image (multipart/form-data)
POST /batch Analyze multiple images (multipart/form-data)
GET /status/{job_id} Check batch job status

Run the API Server:

cd ~/dev/amoled-defect-detection
source venv/bin/activate
pip install fastapi uvicorn python-multipart
python api/main.py

The API will be available at http://localhost:8000. Interactive API documentation (Swagger UI) at http://localhost:8000/docs.

Example Usage:

Single image prediction:
curl -X POST http://localhost:8000/predict -F "file=@panel.png"

Batch processing:
curl -X POST http://localhost:8000/batch -F "files=@panel1.png" -F "files=@panel2.png" -F "files=@panel3.png"

Example Response:

{
"image_id": "panel.png",
"defect": true,
"confidence": 0.9998,
"defect_type": "mura",
"inference_time_ms": 87
}

Deployment Options:

Local network Free Internal factory testing
Hugging Face Spaces Free Public demo, limited requests
Railway Free Production-like environment
Render Free Easy deployment from GitHub

================================================================================
Research Paper
================================================================================

The paper "AMOLED-DefectNet: A Transfer Learning Approach for 100% Accurate Multi-Class Defect Detection in AMOLED Displays" is available in this repository.

    Download PDF (main.pdf) – 17 pages, 5 figures, 22 references

    LaTeX Source (amoled-paper/) – Complete source code for the paper

Abstract:

AMOLED displays dominate consumer electronics, but manufacturing defects significantly reduce yield. We present AMOLED-DefectNet, a transfer learning system using MobileNetV2 that achieves 100% accuracy across five defect categories (dead pixels, stuck pixels, Mura, scratches, dust) with <0.1s inference on CPU.

Citation:

@misc{muhammad2025amoled,
title={AMOLED-DefectNet: A Transfer Learning Approach for 100% Accurate
Multi-Class Defect Detection in AMOLED Displays},
author={Muhammad, Amr},
year={2025},
howpublished={\url{https://github.com/amrmuhammad/amoled-defect-detection}}
}

================================================================================
Repository Structure
================================================================================

amoled-defect-detection/
├── api/ # FastAPI batch processing API
│ └── main.py
├── dashboard/ # Streamlit web dashboard
│ └── app.py
├── models/ # Trained model files
│ └── defect_detector_v2.keras
├── src/ # Core modules
│ ├── data_generator.py # Synthetic defect generation
│ ├── data_generator_v2.py # Native 128x128 generation
│ ├── model_transfer.py # MobileNetV2 model
│ └── train_transfer.py # Training pipeline
├── amoled-paper/ # LaTeX source for research paper
├── train_v2.py # Standalone training script
├── test_v2.py # Model validation
├── requirements.txt # Python dependencies
├── main.pdf # Research paper PDF
└── README.md # This file

================================================================================
Installation
================================================================================

Prerequisites:

    Python 3.12+

    pip

    (Optional) virtual environment

Setup:

git clone https://github.com/amrmuhammad/amoled-defect-detection.git
cd amoled-defect-detection

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

Additional Dependencies for API:

pip install fastapi uvicorn python-multipart

================================================================================
Testing
================================================================================

Run the test suite to validate the model:

python test_v2.py

Expected output:

TESTING V2 MODEL - 100% ACCURACY
================================
📸 CLEAN IMAGES (Expected: CLEAN):
✅ Image 1: CLEAN (99.98% confidence)
...
🔍 DEFECTIVE IMAGES (Expected: DEFECTIVE):
✅ DEAD_PIXEL: DEFECTIVE (100.00% confidence)
✅ STUCK_PIXEL: DEFECTIVE (100.00% confidence)
✅ MURA: DEFECTIVE (100.00% confidence)
✅ SCRATCH: DEFECTIVE (100.00% confidence)
✅ DUST: DEFECTIVE (100.00% confidence)
✅ Mixed accuracy: 10/10 (100%)

================================================================================
Training Your Own Model
================================================================================

Generate synthetic data and retrain:

python train_v2.py --samples 2000 --epochs 20 --fine-tune 5

Options:

--samples 2000 Number of synthetic images
--epochs 20 Initial training epochs
--fine-tune 5 Fine-tuning epochs
--batch-size 200 Batch size for data generation

================================================================================
Partnership Opportunities
================================================================================

We are seeking strategic partners in the display inspection industry:

    White-label integration – Our software runs on your hardware, branded as your solution

    Royalty-based licensing – Pay per inspection system sold

    Joint development – Co-develop custom solutions for specific production lines

Contact: eng.amrmuhammad@gmail.com

================================================================================
Contact
================================================================================

Amr Muhammad
AI Researcher & Developer
Email: eng.amrmuhammad@gmail.com
GitHub: https://github.com/amrmuhammad
LinkedIn: [Your LinkedIn URL]

================================================================================
License
================================================================================

This project is licensed under the MIT License – see the LICENSE file for details.

================================================================================
Acknowledgments
================================================================================

    TensorFlow & Keras for deep learning framework

    MobileNetV2 authors for the efficient architecture

    Open-source community for tools and inspiration
    ================================================================================

