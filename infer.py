import pickle
import face_recognition
import numpy as np
import sys
import os

ARTIFACTS_PATH = "artifacts"

# Load trained classifier
with open(os.path.join(ARTIFACTS_PATH, "classifier.pkl"), "rb") as f:
    clf = pickle.load(f)

with open(os.path.join(ARTIFACTS_PATH, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

# Check input
if len(sys.argv) < 2:
    print("Usage: python infer.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

# Load image
image = face_recognition.load_image_file(image_path)
encodings = face_recognition.face_encodings(image)

if len(encodings) == 0:
    print("No face detected.")
    sys.exit(0)

# Predict
for i, encoding in enumerate(encodings):
    probs = clf.predict_proba([encoding])[0]
    best_idx = np.argmax(probs)
    confidence = probs[best_idx]

    name = label_encoder.inverse_transform([best_idx])[0]

    threshold = 0.6

    if confidence < threshold:
        print(f"Face {i+1}: Unknown ({confidence:.2f})")
    else:
        print(f"Face {i+1}: {name} ({confidence:.2f})")
