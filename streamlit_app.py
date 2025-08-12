import streamlit as st
import pandas as pd
import pickle
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Malware Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DARK THEME STYLING ---
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: #ffffff;
        }
        .reportview-container {
            background-color: #121212;
        }
        .sidebar .sidebar-content {
            background-color: #1e1e1e;
        }
        .css-1d391kg, .css-hxt7ib, .css-1q8dd3e, .css-1v0mbdj {
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
selected_tab = st.sidebar.selectbox("Go to", ["Home", "EDA Insights","Model Performance", "Real-Time Prediction"])


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

# --- REAL-TIME PREDICTION TAB ---
elif selected_tab == "Real-Time Prediction":
    st.title(" Real-Time Malware Prediction")

    # --- File download section ---
    st.subheader(" Download Support Files")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Button: Sample input CSV
        with open("tools/sample_input.csv", "rb") as f:
            st.download_button(
                label="Sample Input CSV",
                data=f,
                file_name="sample_input.csv",
                mime="text/csv"
            )
    with col2:
        st.markdown(
            "Use this sample to understand how your feature input file should be formatted before uploading."
        )

    col3, col4 = st.columns([1, 2])

    with col3:
        # Button: extract_features.py
        with open("tools/extract_features.py", "rb") as f:
            st.download_button(
                label="💻 extract_features.py",
                data=f,
                file_name="extract_features.py",
                mime="text/x-python"
            )
    with col4:
        st.markdown(
            "This script helps extract features from raw executables. Include this in your own data pipeline if needed."
        )

    # --- Upload for prediction ---
    st.subheader(" Upload Your Feature CSV for Prediction")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            # Load uploaded file
            input_df = pd.read_csv(uploaded_file)

            # Drop SHA256 if present
            if "SHA256" in input_df.columns:
                input_df = input_df.drop(columns=["SHA256"])

            # Load trained model
            model = joblib.load("xgb_model.joblib")

            # Load correct feature order
            feature_list = pd.read_csv("new_feature_order.csv")["Feature"].tolist()

            # Align features
            input_df = input_df.reindex(columns=feature_list, fill_value=0)

            # Predict
            prediction = model.predict(input_df)[0]

            # Map label
            label_map = {
                0: "Benign",
                1: "RedLineStealer",
                2: "Downloader",
                3: "RAT",
                4: "BankingTrojan",
                5: "SnakeKeyLogger",
                6: "Spyware"
            }

            result = label_map.get(prediction, "Unknown")
            st.success(f" Prediction: **{result}**")

        except Exception as e:
            st.error(f" Prediction failed: {e}")



elif selected_tab == "EDA Insights":
    st.title("EDA Insights")
    st.markdown("Upload a CSV file to generate EDA visualizations dynamically.")

    uploaded_csv = st.file_uploader("Upload your CSV for EDA", type=["csv"])

    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)

        st.subheader("1. Class Distribution")
        if 'Type' in df.columns:
            fig, ax = plt.subplots()
            sns.countplot(x='Type', data=df, palette='viridis', ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning("Column 'Type' not found in CSV.")

        st.subheader("2. Outlier Detection (Boxplots by Class)")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if 'Type' in df.columns and len(numeric_cols) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='Type', y=numeric_cols[0], data=df, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        st.subheader("3. Feature Correlation Heatmap")
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", annot=False, ax=ax)
            st.pyplot(fig)

        st.subheader("4. Top Correlated Features (Line Plot by Class)")
        if 'Type' in df.columns and len(numeric_cols) > 1:
            top_corr_features = df[numeric_cols].corr()['Type'].abs().sort_values(ascending=False).head(5).index
            fig, ax = plt.subplots()
            for feature in top_corr_features:
                sns.lineplot(x='Type', y=feature, data=df, ax=ax, label=feature)
            plt.xticks(rotation=45)
            st.pyplot(fig)

    else:
        st.info("Please upload a CSV file to view EDA results.")


# --- Model performance tab ---
elif selected_tab == "Model Performance":
    st.title("Model Performance")
    st.markdown("Here’s how our trained model performed on the malware classification task.")

    # --- Classification Report ---
    st.subheader("Classification Report")
    try:
        with open("classification_report.txt", "r") as f:
            report = f.read()
        st.code(report, language='text')
    except Exception as e:
        st.error(f"Error loading classification_report.txt: {e}")

    # --- Confusion Matrix ---
    st.subheader("Confusion Matrix")
    try:
        st.image("confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)
    except Exception as e:
        st.error(f"Error loading confusion_matrix.png: {e}")

    # --- Feature Importance Plot ---
    st.subheader("Top Feature Importances")
    try:
        feature_df = pd.read_csv("feature_importances.csv").head(15)
        st.bar_chart(data=feature_df.set_index("Feature"))
    except Exception as e:
        st.error(f"Error loading feature_importances.csv: {e}")

    # --- Model Summary ---
    st.subheader(" Model Summary")
    st.markdown("""
    - **Model Type:** Random Forest Classifier  
    - **Training Samples:** ~30,000 executables  
    - **Number of Features Used:** Top 200 (selected using ExtraTreesClassifier)  
    - **Class Imbalance Handling:** `class_weight='balanced'`  
    - **Best Accuracy Achieved:** ~89.1%  
    - **Used Libraries:** Scikit-learn, Pandas, Matplotlib, Seaborn
    """)
