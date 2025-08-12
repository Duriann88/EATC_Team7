import streamlit as st
import pandas as pd
import joblib

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Malware Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN DARK THEME STYLING ---
st.markdown("""
<style>
    /* Overall background and text */
    body, .reportview-container, .css-18e3th9 {
        background-color: #121212 !important;
        color: #E0E0E0 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #1F1F1F;
        color: #00B8D9;
        font-weight: 600;
    }

    /* Sidebar selectbox text */
    .css-1v0mbdj, .css-1d391kg, .css-hxt7ib, .css-1q8dd3e {
        color: #E0E0E0;
    }

    /* Buttons */
    button[data-baseweb="button"] {
        background-color: #00B8D9 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.6em 1.5em !important;
        font-weight: 600 !important;
        transition: background-color 0.3s ease;
    }
    button[data-baseweb="button"]:hover {
        background-color: #007A99 !important;
        cursor: pointer;
    }

    /* Headers */
    h1, h2, h3 {
        color: #00B8D9;
    }

    /* Links */
    a {
        color: #00B8D9;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }

    /* Markdown */
    .markdown-text-container p {
        font-size: 1.1em;
        line-height: 1.6em;
    }

    /* File uploader */
    .stFileUploader>div>div>input {
        border-radius: 8px;
    }

    /* Image captions */
    figcaption {
        font-style: italic;
        font-size: 0.85em;
        color: #AAAAAA;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
selected_tab = st.sidebar.selectbox("Go to", ["Home", "EDA Insights", "Model Performance", "Real-Time Prediction"])

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

    ### What’s going on behind the scenes?

    Our model looks at **different technical features** inside the executable file, like:

    - Which **DLLs** the file imports
    - What’s inside its **PE Header** and **Sections**
    - Which **Windows API functions** it tries to use

    All of this tells us how suspicious (or safe) the file might be.

    ---

    ### What kind of model are we using?

    We trained an **XGBoost Classifier** tuned with Optuna on a dataset of ~30,000 executable files — both clean and malicious (from different malware families like Spyware, RAT, Banking Trojans, and more).

    **Model Accuracy**: ~89%  
    **Balanced Detection**: It handles imbalanced classes well using internal weighting.

    ---

    ### What you can do here:

    - Download sample `.csv` files (already extracted) to try the prediction
    - Download the feature extractor tool to use with your own `.exe` files
    - Upload a feature file and get a real-time prediction
    - Explore how the model sees the data (EDA Insights tab)

    This is designed to be beginner-friendly and safe for experimenting!

    ---
    """)

# --- EDA INSIGHTS TAB ---
elif selected_tab == "EDA Insights":
    st.title("EDA Insights")
    st.markdown("Below are key visualizations from the Exploratory Data Analysis (EDA) performed on the dataset.")

    with st.expander("1. Class Distribution"):
        st.image("class_distribution.png", caption="Class Distribution of Malware Types", use_container_width=True)

    with st.expander("2. Outlier Detection (Boxplots by Class)"):
        st.image("outlier_boxplots.png", caption="Boxplot showing outliers per class", use_container_width=True)

    with st.expander("3. Feature Correlation Heatmap"):
        st.image("corelation_heatmap.png", caption="Correlation heatmap on sample of 3000 rows", use_container_width=True)

    with st.expander("4. Top Correlated Features (Line Plot by Class)"):
        st.image("line_plot.png", caption="Class-wise trends for top correlated features", use_container_width=True)

    with st.expander("5. Distribution of Key Features by Class"):
        st.markdown("These plots show how the distribution of key features varies across malware families.")

        col1, col2 = st.columns(2)
        with col1:
            st.image("AddressOfEntryPoint_distribution_by_class.png", use_container_width=True, caption="AddressOfEntryPoint")
            st.image("rsrc_Misc_VirtualSize_distribution_by_class.png", use_container_width=True, caption="rsrc_Misc_VirtualSize")
            st.image("rsrc_PointerToRawData_distribution_by_class.png", use_container_width=True, caption="rsrc_PointerToRawData")
        with col2:
            st.image("text_Misc_VirtualSize_distribution_by_class.png", use_container_width=True, caption="text_Misc_VirtualSize")
            st.image("TimeDateStamp_distribution_by_class.png", use_container_width=True, caption="TimeDateStamp")

# --- MODEL PERFORMANCE TAB ---
elif selected_tab == "Model Performance":
    st.title("Model Performance")
    st.markdown("Here’s how our trained model performed on the malware classification task.")

    st.subheader("Classification Report")
    try:
        with open("classification_report.txt", "r") as f:
            report = f.read()
        st.code(report, language='text')
    except Exception as e:
        st.error(f"Error loading classification_report.txt: {e}")

    st.subheader("Confusion Matrix")
    try:
        st.image("confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)
    except Exception as e:
        st.error(f"Error loading confusion_matrix.png: {e}")

    st.subheader("Top Feature Importances")
    try:
        feature_df = pd.read_csv("feature_importances.csv").head(15)
        st.bar_chart(data=feature_df.set_index("Feature"))
    except Exception as e:
        st.error(f"Error loading feature_importances.csv: {e}")

    st.subheader("Model Summary")
    st.markdown("""
    - **Model Type:** XGBoost Classifier (Optuna-tuned)
    - **Training Samples:** ~30,000 executables
    - **Number of Features Used:** Top 200 (selected using ExtraTreesClassifier)
    - **Class Imbalance Handling:** `scale_pos_weight` or balanced objective tuning in Optuna
    - **Best Accuracy Achieved:** ~89%
    - **Used Libraries:** XGBoost, Scikit-learn, Pandas, Matplotlib, Seaborn
    """)

# --- REAL-TIME PREDICTION TAB ---
elif selected_tab == "Real-Time Prediction":
    st.title("Real-Time Malware Prediction")

    st.subheader("Download Support Files")

    col1, col2 = st.columns([1, 2])
    with col1:
        with open("tools/sample_input.csv", "rb") as f:
            st.download_button(
                label="📥 Sample Input CSV",
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
                label="💻 extract_features.py",
                data=f,
                file_name="extract_features.py",
                mime="text/x-python"
            )
    with col4:
        st.markdown("This script helps extract features from raw executables. Include this in your own data pipeline if needed.")

    st.subheader("Upload Your Feature CSV for Prediction")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        with st.spinner('Analyzing your file...'):
            try:
                input_df = pd.read_csv(uploaded_file)

                if "SHA256" in input_df.columns:
                    input_df = input_df.drop(columns=["SHA256"])

                # Load your XGBoost model (update path if needed)
                model = joblib.load("xgb_optuna_model.joblib")

                # Load feature order
                feature_list = pd.read_csv("new_feature_order.csv")["Feature"].tolist()

                # Reindex features to model expected order, fill missing with 0
                input_df = input_df.reindex(columns=feature_list, fill_value=0)

                # Predict
                prediction = model.predict(input_df)[0]

                # Map label - update if your classes changed
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

                st.success(f"Prediction: **{result}**")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
