"""
AMOLED Defect Detection Dashboard - Multi-Class Version
Detects: Clean, Dead Pixel, Stuck Pixel, Mura, Scratch, Dust
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="AMOLED Defect Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        text-align: center;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        background: rgba(0,0,0,0.3);
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .defect-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    .defect-detected {
        background-color: #ff4444;
        color: white;
    }
    .defect-clean {
        background-color: #00c851;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header"><h1>🔍 AMOLED Defect Detection System</h1><p>Multi-Class Defect Classification (6 Types)</p></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/monitor--v1.png", width=80)
    st.markdown("### About")
    st.info("""
    **Model Performance:**
    - 🎯 Accuracy: **98.67%**
    - 📊 Classes: **6** (Clean + 5 defects)
    
    **Supported Defects:**
    - 💀 Dead Pixel
    - 🎨 Stuck Pixel
    - 🌊 Mura
    - ➖ Scratch
    - 🌫️ Dust
    """)
    
    st.markdown("---")
    st.markdown("### How to Use")
    st.markdown("""
    1. Upload a display image
    2. Click 'Analyze'
    3. View defect type and confidence
    """)
    
    st.markdown("---")
    st.markdown("### Technical Details")
    st.markdown("""
    - Model: MobileNetV2 (fine‑tuned)
    - Input: 128×128 RGB
    - Inference: <0.1s on CPU
    """)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'model' not in st.session_state:
    st.session_state.model = None

@st.cache_resource
def load_model():
    """Load the multi-class model"""
    model_path = 'models/multi_class_defect_detector.keras'
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        return model
    else:
        st.error(f"Model not found at {model_path}")
        return None

# Load model
if st.session_state.model is None:
    with st.spinner("Loading AI model (multi-class)..."):
        st.session_state.model = load_model()
        if st.session_state.model:
            st.success("✅ Model loaded (98.67% accuracy)")

def preprocess_image(image):
    """Preprocess image for model inference"""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    image_resized = cv2.resize(image, (128, 128))
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)
    return image_batch, image_resized

def predict_image(model, image):
    """Return (class_index, class_name, confidence)"""
    processed, _ = preprocess_image(image)
    probs = model.predict(processed, verbose=0)[0]
    class_idx = np.argmax(probs)
    confidence = probs[class_idx]
    class_names = ['Clean', 'Dead Pixel', 'Stuck Pixel', 'Mura', 'Scratch', 'Dust']
    return class_idx, class_names[class_idx], float(confidence)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Display Image")
    
    upload_method = st.radio("Choose input method:", ["Upload File", "Use Sample"])
    
    uploaded_file = None
    if upload_method == "Upload File":
        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload",
            type=['jpg', 'jpeg', 'png', 'bmp']
        )
    else:
        st.markdown("#### Sample Images")
        os.makedirs("demo/sample_images", exist_ok=True)
        sample_files = [f for f in os.listdir("demo/sample_images") if f.endswith(('.png', '.jpg', '.jpeg'))]
        if sample_files:
            selected_sample = st.selectbox("Select a sample image:", sample_files)
            if selected_sample:
                sample_path = os.path.join("demo/sample_images", selected_sample)
                uploaded_file = open(sample_path, 'rb')
                st.info(f"Using sample: {selected_sample}")
        else:
            st.warning("No sample images found. Please upload an image first.")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    st.markdown("### 🔬 Analysis Results")
    
    if uploaded_file is not None and st.session_state.model is not None:
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                image_np = np.array(image)
                class_idx, class_name, confidence = predict_image(st.session_state.model, image_np)
                
                # Display results
                if class_idx == 0:
                    st.success(f"### ✅ CLEAN DISPLAY")
                    st.metric("Confidence", f"{confidence:.2%}")
                else:
                    st.error(f"### ⚠️ {class_name.upper()} DETECTED")
                    st.metric("Confidence", f"{confidence:.2%}")
                
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    title={'text': "Confidence"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred" if class_idx != 0 else "darkgreen"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
                    }
                ))
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
                
                # Add to history
                st.session_state.history.append({
                    'timestamp': datetime.now(),
                    'defect_type': class_name,
                    'confidence': confidence,
                    'result': 'Defective' if class_idx != 0 else 'Clean'
                })
    
    elif st.session_state.model is None:
        st.warning("⚠️ Model not loaded.")
    else:
        st.info("👈 Upload an image and click 'Analyze Image'")

# History and Analytics
st.markdown("---")
st.markdown("### 📊 Detection History")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    
    col1, col2, col3, col4 = st.columns(4)
    total = len(df)
    defective = len(df[df['result'] == 'Defective'])
    clean = total - defective
    avg_conf = df['confidence'].mean()
    
    with col1:
        st.metric("Total Analyses", total)
    with col2:
        st.metric("Defective Found", defective, delta=f"{(defective/total*100):.0f}%")
    with col3:
        st.metric("Clean Panels", clean)
    with col4:
        st.metric("Avg Confidence", f"{avg_conf:.1%}")
    
    # Show recent history
    st.markdown("#### Recent Results")
    display_df = df.tail(10).copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%H:%M:%S')
    display_df = display_df.rename(columns={'timestamp': 'Time', 'defect_type': 'Defect Type', 'confidence': 'Confidence'})
    display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.1%}")
    st.dataframe(display_df[['Time', 'Defect Type', 'Confidence']], use_container_width=True)
    
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("No analysis history yet.")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.8rem;'>"
    "AMOLED Defect Detection System | Multi-Class Model (98.67% Accuracy)"
    "</div>",
    unsafe_allow_html=True
)
