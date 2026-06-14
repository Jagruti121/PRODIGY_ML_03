

import os
import cv2
import numpy as np
import matplotlib
# Force Matplotlib to use a non-interactive backend to prevent freezing
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =====================================================================
# 1. PAGE CONFIGURATION & THEME SETUP
# =====================================================================
st.set_page_config(
    page_title="SVM Image Classifier",
    page_icon="🐱🐶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set clean style for confusion matrix plot
sns.set_theme(style="white")

# =====================================================================
# 2. DATA PIPELINE (IMAGE LOADING & FEATURE EXTRACTION)
# =====================================================================
IMAGE_SIZE = 64  # Resize to 64x64 to balance model accuracy and training speed

@st.cache_data
def load_and_preprocess_images(data_dir, max_samples=1000):
    """
    Loads images from directory, extracts labels from filenames,
    resizes, flattens, and normalizes pixel data.
    """
    if not os.path.exists(data_dir):
        return None, None

    features = []
    labels = []
    
    # List and shuffle images to get a balanced representation of cats and dogs
    all_images = os.listdir(data_dir)
    np.random.seed(42)
    np.random.shuffle(all_images)
    
    cat_count = 0
    dog_count = 0
    half_samples = max_samples // 2

    for img_name in all_images:
        # Determine class label from Kaggle filename format (e.g., "cat.0.jpg")
        if img_name.startswith('cat'):
            if cat_count >= half_samples:
                continue
            label = 0  # 0 for Cat
            cat_count += 1
        elif img_name.startswith('dog'):
            if dog_count >= half_samples:
                continue
            label = 1  # 1 for Dog
            dog_count += 1
        else:
            continue

        img_path = os.path.join(data_dir, img_name)
        
        # Read image in grayscale to simplify vector space
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        # Resize to standard size
        img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Flatten the 2D image matrix into a 1D feature array (64 * 64 = 4096 features)
        img_flattened = img_resized.flatten()
        
        features.append(img_flattened)
        labels.append(label)
        
        # Break early if total sample limit is reached
        if len(features) >= max_samples:
            break

    X = np.array(features, dtype=np.float32)
    y = np.array(labels)
    
    # Normalize pixel values from range [0, 255] to [0, 1]
    X /= 255.0
    
    return X, y

# =====================================================================
# DATA INITIALIZATION (With Visual Progress Feedback)
# =====================================================================
DATA_DIRECTORY = 'train'

# Wrap the loading process in a spinner to prevent blank page rendering
with st.spinner("🤖 Processing image dataset and extracting pixel feature vectors... Please wait..."):
    # Set to 200 samples for immediate confirmation. You can bump this to 1000 later!
    X, y = load_and_preprocess_images(DATA_DIRECTORY, max_samples=200)

if X is None or len(X) == 0:
    st.error(f"🚨 Missing Dataset: Could not find the '{DATA_DIRECTORY}' directory or it is empty. "
             f"Please extract Kaggle's 'train.zip' into a folder named '{DATA_DIRECTORY}' in this project root.")
    st.stop()

# Split into train and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================================================================
# 3. MODEL TRAINING ENGINE
# =====================================================================
@st.cache_resource
def train_svm_classifier(X_t, y_t):
    """Trains an SVM model using an RBF kernel."""
    # Using probability=True allows us to generate certainty scores later
    svm = SVC(kernel='rbf', C=5.0, gamma='scale', probability=True, random_state=42)
    svm.fit(X_t, y_t)
    return svm

with st.spinner("🏋️‍♂️ Training Support Vector Machine Classifier..."):
    svm_model = train_svm_classifier(X_train, y_train)

# Generate test predictions for evaluation metrics
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# =====================================================================
# 4. SIDEBAR PANEL SYSTEM DOCUMENTATION
# =====================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluent/100/000000/dog.png", width=80)
    st.title("SVM Vision Classifier")
    st.markdown("""
    This app utilizes a **Support Vector Machine (SVM)** configured with a non-linear **RBF Kernel** to draw hyperplanes in high-dimensional pixel space.
    
    **Pipeline Architecture:**
    * **Input Format:** Raw Binary JPGs
    * **Downsampling:** 64×64 Grayscale
    * **Feature Vector Size:** 4,096 dimensions
    * **Kernel Vector Space:** Radial Basis Function
    """)
    st.divider()
    st.caption("v1.1.0 • Stable Startup Build")

# =====================================================================
# 5. MAIN USER INTERFACE HEADER
# =====================================================================
st.title("🐱🐶 Support Vector Machine Image Classification Platform")
st.markdown("Classify canine and feline profiles using high-dimensional support vector margins.")

tab_inference, tab_metrics = st.tabs(["🔮 Real-Time Image Inference", "📈 Model Diagnostic Analytics"])

# =====================================================================
# TAB 1: REAL-TIME IMAGE INFERENCE
# =====================================================================
with tab_inference:
    st.subheader("Upload a Custom Image")
    st.write("Upload a JPG or PNG image of a cat or dog to query the trained SVM backend model:")
    
    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convert file stream to OpenCV matrix format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        col_ui_img, col_ui_pred = st.columns([1, 1])
        
        with col_ui_img:
            st.image(opencv_img, channels="BGR", caption="Uploaded Image File", use_container_width=True)
            
        with col_ui_pred:
            st.markdown("### Backend Decision Output")
            
            # Replicate pipeline transformations on the custom uploaded image
            gray_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
            resized_img = cv2.resize(gray_img, (IMAGE_SIZE, IMAGE_SIZE))
            flattened_input = resized_img.flatten().reshape(1, -1)
            normalized_input = flattened_input / 255.0
            
            # Predict class and calculate probability values
            prediction_class = svm_model.predict(normalized_input)[0]
            prediction_probs = svm_model.predict_proba(normalized_input)[0]
            
            label_mapping = {0: "Cat 🐱", 1: "Dog 🐶"}
            final_label = label_mapping[prediction_class]
            confidence_score = prediction_probs[prediction_class]
            
            st.markdown(f"""
            <div style="background-color:#f8f9fa; padding:24px; border-radius:12px; border-left: 6px solid #0066cc; box-shadow: 0px 4px 6px rgba(0,0,0,0.05);">
                <h5 style="margin-top:0; color:#495057;">Classification Result:</h5>
                <h2 style="color:#0066cc; margin:10px 0;">{final_label}</h2>
                <p style="font-size:1.1em; color:#6c757d; margin-bottom:0;">Model Confidence: <b>{confidence_score:.2%}</b></p>
            </div>
            """, unsafe_allow_html=True)

# =====================================================================
# TAB 2: MODEL DIAGNOSTIC ANALYTICS
# =====================================================================
with tab_metrics:
    st.subheader("📊 Classifier Validation Performance Dashboard")
    
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    with kpi_col1:
        st.metric(label="Validation Target Accuracy", value=f"{accuracy:.2%}")
    with kpi_col2:
        st.metric(label="Training Sample Size Evaluated", value=f"{len(X_train)} Images")
    with kpi_col3:
        st.metric(label="Validation Holdout Test Size", value=f"{len(X_test)} Images")
        
    st.write("---")
    
    chart_col, text_col = st.columns([3, 2])
    
    with chart_col:
        st.markdown("**Confusion Matrix Heatmap**")
        
        # Compute confusion matrix array
        cm = confusion_matrix(y_test, y_pred)
        
        fig_cm = plt.figure(figsize=(5, 3.8))
        ax_cm = fig_cm.add_subplot(111)
        
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Cat (Predicted)', 'Dog (Predicted)'], 
            yticklabels=['Cat (Actual)', 'Dog (Actual)'], 
            cbar=False, ax=ax_cm
        )
        plt.title("Classification Distribution Heatmap", fontsize=11, fontweight='bold', pad=10)
        
        st.pyplot(fig_cm)
        plt.close(fig_cm)
        
    with text_col:
        st.markdown("**Classification Precision & Recall Metrics Report**")
        
        # Extract classification report as dictionary layout
        report_dict = classification_report(y_test, y_pred, target_names=['Cat', 'Dog'], output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        
        st.dataframe(report_df.iloc[:-2, :3], use_container_width=True)
        st.write(
            "An SVM model processing raw downscaled dimensions will comfortably hit an accuracy threshold around **60% to 68%**. "
            "Because an SVM uses linear or radial distance equations rather than structural context, spatial variations (like different body poses, rotations, and changing lighting) introduce significant feature noise. "
            "To hit high-end accuracies (>90%), these features are typically fed into convolutional extraction neural networks (CNNs) before passing to a classification head."
        )


