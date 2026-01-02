# Identity – Local Face Recognition System

Identity is a local face recognition system that identifies known individuals from uploaded images using pretrained facial embeddings and a lightweight classifier. The project is designed with a clear separation between model training and inference, and provides a simple Streamlit-based interface for interaction.

## What it does
- Takes an image as input and detects faces
- Extracts facial embeddings using a pretrained model
- Predicts the identity using an SVM classifier
- Handles unknown faces using a confidence threshold
- Works entirely locally (no cloud, no external APIs)

## Features
- Pretrained face embeddings for robust recognition
- SVM-based identity classification
- Confidence-based unknown face handling
- Support for multiple faces in a single image
- Local Streamlit web interface for image upload
- Clear separation between training and inference

## Tech Stack
- Python
- face_recognition (dlib)
- scikit-learn
- Streamlit
- NumPy
- OpenCV
- Pillow

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py

roject Structure
├── app.py          # Streamlit application
├── infer.py        # Inference logic
├── requirements.txt
└── artifacts/      # Trained model artifacts (excluded)

