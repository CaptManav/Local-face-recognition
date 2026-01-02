import streamlit as st
import pickle
import numpy as np
import face_recognition
from PIL import Image
import os

st.set_page_config(page_title="Face Recognition", layout="centered")
st.title("🧠 Face Recognition App")

ARTIFACTS_PATH = "artifacts"

@st.cache_resource
def load_model():
    with open(os.path.join(ARTIFACTS_PATH, "classifier.pkl"), "rb") as f:
        clf = pickle.load(f)
    with open(os.path.join(ARTIFACTS_PATH, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    return clf, le

clf, label_encoder = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    image_np = np.array(image)
    encodings = face_recognition.face_encodings(image_np)

    if len(encodings) == 0:
        st.error("No face detected in the image.")
    else:
        st.subheader("Results")
        threshold = 0.6

        for i, encoding in enumerate(encodings):
            probs = clf.predict_proba([encoding])[0]
            best_idx = np.argmax(probs)
            confidence = probs[best_idx]
            name = label_encoder.inverse_transform([best_idx])[0]

            if confidence < threshold:
                st.write(f"Face {i+1}: ❓ Unknown ({confidence:.2f})")
            else:
                st.write(f"Face {i+1}: ✅ {name} ({confidence:.2f})")
