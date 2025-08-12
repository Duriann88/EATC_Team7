import streamlit as st
from PIL import Image
import pandas as pd
import joblib

# ----- Custom Cyberpunk CSS -----
cyberpunk_css = """
<style>
    body {
        background-color: #0d0d0d;
        color: #00ff99;
    }
    .stApp {
        background-color: #0d0d0d;
        color: #00ff99;
    }
    h1, h2, h3, h4 {
        color: #00ff99 !important;
        text-shadow: 0 0 10px #00ff99;
    }
    .top-nav {
        display: flex;
        justify-content: center;
        background-color: #1a1a1a;
        padding: 10px;
        border-bottom: 2px solid #00ff99;
    }
    .nav-button {
        color: #00ff99;
        padding: 10px 20px;
        cursor: pointer;
        font-weight: bold;
        margin: 0 10px;
        background: transparent;
        border: 2px solid #00ff99;
        border-radius: 5px;
        transition: 0.3s;
    }
    .nav-button:hover {
        background-color: #00ff99;
        color: black;
        box-shadow: 0 0 10px #00ff99;
    }
</style>
"""
st.markdown(cyberpunk_css, unsafe_allow_html=True)

# ----- Top Navigation Bar -----
pages = ["Home", "EDA Insights", "Model Info", "Real-Time Prediction"]
selected_tab = st.session_state.get("selected_tab", "Home")

def set_tab(tab):
    st.session_state.selected_tab = tab

st.markdown('<div class="top-nav">', unsafe_allow_html=True)
for page in pages:
    if st.button(page, key=page, help=f"Go to {page}"):
        set_tab(page)
st.markdown('</div>', unsafe_allow_html=True)

# --- HOME PAGE ---
if selected_tab == "Home":
    st.title("Malware Detection System")

    st.markdown("""
    Welcome!  
    This web app helps detect whether a Windows executable file (like a `.exe`) is **benign or a type of malware** — using a trained Machine Learning model.

    ---

    ### Why can’t I just upload a `.exe` file?

    To keep your files **safe** and your device **secure**, we don't allow `.exe` uploads directly.  
    But no worries — we provide a **simple Python script** to help you extract important features from your `.exe` on your own computer. That way, you're always in control of your own data. 🔐

    You just:
    - Run the script locally
    - It creates a `.csv` file from your `.exe`
    - Upload that `.csv` here for prediction 

    ---

    ###  What’s going on behind the scenes?

    Our model looks at **different technical features** inside the executable file, like:

    - Which **DLLs** the file imports
    - What’s inside its **PE Header** and **Sections**
    - Which **Windows API functions** it tries to use

    All of this tells us how suspicious (or safe) the file might be.

    ---

    ###  What kind of model are we using?

    We trained a **Random Forest Classifier** on a dataset of ~30,000 executable files — both clean and malicious (from different malware families like Spyware, RAT, Banking Trojans, and more).

    **Model Accuracy**: ~89%  
    **Balanced Detection**: It handles imbalanced classes well using internal weighting.

    ---

    ###  What you can do here:

    -  Download sample `.csv` files (already extracted) to try the prediction
    -  Download the feature extractor tool to use with your own `.exe` files
    -  Upload a feature file and get a real-time prediction
    -  Explore how the model sees the data (EDA Insights tab)

    This is designed to be beginner-friendly and safe for experimenting!

    ---
    """)


elif selected_tab == "EDA Insights":
    st.title("EDA Insights")
    st.markdown("Below are key visualizations from the Exploratory Data Analysis (EDA) performed on the dataset.")

    st.subheader("1. Class Distribution")
    st.image("class_distribution.png", caption="Class Distribution of Malware Types", use_container_width=True)

    st.subheader("2. Outlier Detection (Boxplots by Class)")
    st.image("outlier_boxplots.png", caption="Boxplot showing outliers per class", use_container_width=True)

    st.subheader("3. Feature Correlation Heatmap")
    st.image("corelation_heatmap.png", caption="Correlation heatmap on sample of 3000 rows", use_container_width=True)

    st.subheader("4. Distribution of Key Features by Class")
    col1, col2 = st.columns(2)
    with col1:
        st.image("AddressOfEntryPoint_distribution_by_class.png", use_container_width=True, caption="AddressOfEntryPoint")
        st.image("rsrc_Misc_VirtualSize_distribution_by_class.png", use_container_width=True, caption="rsrc_Misc_VirtualSize")
        st.image("rsrc_PointerToRawData_distribution_by_class.png", use_container_width=True, caption="rsrc_PointerToRawData")
    with col2:
        st.image("text_Misc_VirtualSize_distribution_by_class.png", use_container_width=True, caption="text_Misc_VirtualSize")
        st.image("TimeDateStamp_distribution_by_class.png", use_container_width=True, caption="TimeDateStamp")

elif selected_tab == "Model Info":
    st.title("Model Information")
    st.markdown("""
    ### Final Model: Optuna-Tuned XGBoost
    - **Algorithm**: XGBoost (multi:softmax)
    - **Optimizer**: Optuna (30 trials, Stratified K-Fold CV)
    - **Class imbalance handling**: Class weights via `compute_class_weight`
    - **Objective**: Multi-class malware classification
    - **Metrics used**: mlogloss, merror
    """)

elif selected_tab == "Real-Time Prediction":
    st.title(" Real-Time Malware Prediction")

    # --- File download section ---
    st.subheader(" Download Support Files")

    col1, col2 = st.columns([1, 2])
    with col1:
        with open("tools/sample_input.csv", "rb") as f:
            st.download_button(
                label="Sample Input CSV",
                data=f,
                file_name="sample_input.csv",
                mime="text/csv"
            )
    with col2:
        st.markdown("Use this sample to understand how your feature input file should be formatted before uploading.")

    col3, col4 = st.columns([1, 2])
    with col3:
        with open("tools/extract_features.py", "rb") as f:
            st.download_button(
                label="extract_features.py",
                data=f,
                file_name="extract_features.py",
                mime="text/x-python"
            )
    with col4:
        st.markdown("This script helps extract features from raw executables. Include this in your own data pipeline if needed.")

    # --- Upload CSV for prediction ---
    st.subheader("Upload CSV for Prediction")
    uploaded_file = st.file_uploader("Upload your feature CSV", type=["csv"])

    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview")
        st.dataframe(user_df.head())

        # Load model and feature order
        model = joblib.load("models/final_xgb_model.pkl")
        feature_order = pd.read_csv("new_feature_order.csv")["Feature"].tolist()

        # Match columns
        try:
            X_user = user_df[feature_order]
        except KeyError as e:
            st.error(f"Missing required features: {e}")
            st.stop()

        # Predict
        predictions = model.predict(X_user)
        results_df = user_df.copy()
        results_df["Predicted_Type"] = predictions

        st.write("### Prediction Results")
        st.dataframe(results_df)

        # Download predictions
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
