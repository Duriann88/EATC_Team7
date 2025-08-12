import streamlit as st
import pandas as pd
import joblib

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Malware Detection System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- MODERN GRADIENT THEME STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main background with gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Content container */
    .block-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem auto;
        max-width: 1200px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    }
    
    /* Hide default elements */
    #MainMenu, footer, header, .stDeployButton {
        visibility: hidden;
    }
    
    /* Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Navigation Pills */
    .nav-pills {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        margin: 2rem 0;
        padding: 0.5rem;
        background: rgba(255, 255, 255, 0.8);
        border-radius: 50px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    .nav-pill {
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        cursor: pointer;
        border: none;
        background: transparent;
    }
    
    .nav-pill.active {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }
    
    .nav-pill:not(.active) {
        color: #6b7280;
        background: rgba(255, 255, 255, 0.5);
    }
    
    .nav-pill:not(.active):hover {
        background: rgba(102, 126, 234, 0.1);
        color: #667eea;
        transform: translateY(-1px);
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        font-size: 2.5rem;
        text-align: center;
        margin: 2rem 0;
    }
    
    h2, h3 {
        color: #374151;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Cards and sections */
    .feature-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.7));
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981, #059669) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 3px 10px rgba(16, 185, 129, 0.3) !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5) !important;
    }
    
    /* File uploader */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 12px;
        padding: 1rem;
        border: 2px dashed #667eea;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #764ba2;
        background: rgba(255, 255, 255, 0.9);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.1)) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        border-radius: 12px !important;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(220, 38, 38, 0.1)) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        border-radius: 12px !important;
    }
    
    /* Expandable sections */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Charts and metrics */
    .metric-container {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.7));
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Links */
    a {
        color: #667eea !important;
        text-decoration: none !important;
        font-weight: 500;
    }
    
    a:hover {
        color: #764ba2 !important;
        text-decoration: underline !important;
    }
    
    /* Radio button hide */
    .stRadio {
        display: none;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- ENHANCED NAVIGATION ---
def render_navigation():
    tabs = ["Home", "EDA Insights", "Model Performance", "Real-Time Prediction"]
    icons = ["🏠", "📊", "🤖", "🔍"]
    
    # Get current tab from session state, default to 0
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = 0
    
    st.markdown('<div class="nav-pills">', unsafe_allow_html=True)
    
    cols = st.columns(4)
    for i, (tab, icon) in enumerate(zip(tabs, icons)):
        with cols[i]:
            active_class = "active" if i == st.session_state.current_tab else ""
            if st.button(f"{icon} {tab}", key=f"nav_{i}", help=f"Navigate to {tab}"):
                st.session_state.current_tab = i
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return tabs[st.session_state.current_tab]

# --- PAGE CONTENT FUNCTIONS ---
def home_page():
    st.markdown('<h1>🛡️ Malware Detection System</h1>', unsafe_allow_html=True)
    
    # Hero section
    st.markdown("""
    <div class="feature-card">
        <h2 style="text-align: center; margin-top: 0;">🎯 Advanced AI-Powered Malware Detection</h2>
        <p style="text-align: center; font-size: 1.1rem; color: #6b7280;">
            Detect malicious Windows executables using state-of-the-art machine learning techniques
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🔒 Secure Analysis</h3>
            <p>Upload feature CSV files extracted from your executables. No direct .exe uploads for maximum security.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 89% Accuracy</h3>
            <p>XGBoost model trained on 30,000+ samples with Optuna hyperparameter optimization.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>⚡ Real-time</h3>
            <p>Get instant predictions with detailed malware family classification.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How it works
    st.markdown("""
    <div class="feature-card">
        <h2>🔬 How It Works</h2>
        <p>Our system analyzes technical features extracted from Windows executable files:</p>
        <ul style="font-size: 1.1rem; line-height: 1.8;">
            <li><strong>PE Header Analysis:</strong> Examines portable executable structure and metadata</li>
            <li><strong>Import Tables:</strong> Analyzes DLL imports and Windows API function calls</li>
            <li><strong>Section Analysis:</strong> Studies code and data sections for suspicious patterns</li>
            <li><strong>Entry Point Detection:</strong> Identifies unusual entry point characteristics</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Model info
    st.markdown("""
    <div class="feature-card">
        <h2>🤖 Model Details</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;">
            <div class="metric-container">
                <h3 style="margin: 0; color: #667eea;">Algorithm</h3>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">XGBoost Classifier</p>
            </div>
            <div class="metric-container">
                <h3 style="margin: 0; color: #667eea;">Training Data</h3>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">~30,000 samples</p>
            </div>
            <div class="metric-container">
                <h3 style="margin: 0; color: #667eea;">Accuracy</h3>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">~89%</p>
            </div>
            <div class="metric-container">
                <h3 style="margin: 0; color: #667eea;">Features</h3>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">Top 200</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Detectable malware types
    st.markdown("""
    <div class="feature-card">
        <h2>🦠 Detectable Malware Types</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;">
            <div style="text-align: center; padding: 1rem; background: rgba(239, 68, 68, 0.1); border-radius: 10px;">
                <strong>🔴 RedLineStealer</strong>
            </div>
            <div style="text-align: center; padding: 1rem; background: rgba(245, 158, 11, 0.1); border-radius: 10px;">
                <strong>📥 Downloader</strong>
            </div>
            <div style="text-align: center; padding: 1rem; background: rgba(168, 85, 247, 0.1); border-radius: 10px;">
                <strong>🐭 RAT</strong>
            </div>
            <div style="text-align: center; padding: 1rem; background: rgba(34, 197, 94, 0.1); border-radius: 10px;">
                <strong>🏦 Banking Trojan</strong>
            </div>
            <div style="text-align: center; padding: 1rem; background: rgba(59, 130, 246, 0.1); border-radius: 10px;">
                <strong>🐍 SnakeKeyLogger</strong>
            </div>
            <div style="text-align: center; padding: 1rem; background: rgba(236, 72, 153, 0.1); border-radius: 10px;">
                <strong>🕵️ Spyware</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def eda_page():
    st.markdown('<h1>📊 EDA Insights</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <p style="font-size: 1.1rem; text-align: center;">
            Explore comprehensive data analysis and visualizations from our malware dataset
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📈 1. Class Distribution", expanded=True):
        st.markdown("Distribution of different malware families in our training dataset:")
        try:
            st.image("assets/class_distribution.png", caption="Class Distribution of Malware Types", use_container_width=True)
        except:
            st.info("📷 Image file not found: class_distribution.png")

    with st.expander("📦 2. Outlier Detection Analysis"):
        st.markdown("Boxplot analysis showing statistical outliers across different malware classes:")
        try:
            st.image("assets/outlier_boxplots.png", caption="Statistical outliers per malware class", use_container_width=True)
        except:
            st.info("📷 Image file not found: outlier_boxplots.png")

    with st.expander("🔗 3. Feature Correlation Matrix"):
        st.markdown("Correlation heatmap revealing relationships between different features:")
        try:
            st.image("assets/correlation_heatmap.png", caption="Feature correlation analysis (sample of 3000 rows)", use_container_width=True)
        except:
            st.info("📷 Image file not found: corelation_heatmap.png")

    with st.expander("📊 4. Feature Trends by Class"):
        st.markdown("Line plots showing how top correlated features vary across malware families:")
        try:
            st.image("assets/line_plot.png", caption="Class-wise trends for highly correlated features", use_container_width=True)
        except:
            st.info("📷 Image file not found: line_plot.png")
        
        for i, (img_path, title) in enumerate(feature_images):
            with col1 if i % 2 == 0 else col2:
                try:
                    st.image(img_path, use_container_width=True, caption=title)
                except:
                    st.info(f"📷 Image file not found: {img_path}")

def model_perf_page():
    st.markdown('<h1>🤖 Model Performance</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <p style="font-size: 1.1rem; text-align: center;">
            Comprehensive evaluation metrics and performance analysis of our malware detection model
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Performance metrics overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h3 style="margin: 0; color: #10b981;">Overall Accuracy</h3>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.5rem; font-weight: 700;">~89%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <h3 style="margin: 0; color: #667eea;">Model Type</h3>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">XGBoost</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <h3 style="margin: 0; color: #f59e0b;">Training Size</h3>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">30K+ samples</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-container">
            <h3 style="margin: 0; color: #ef4444;">Features Used</h3>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">Top 200</p>
        </div>
        """, unsafe_allow_html=True)

    # Classification Report
    st.markdown("""
    <div class="feature-card">
        <h2>📈 Classification Report</h2>
    """, unsafe_allow_html=True)
    
    try:
        with open("classification_report.txt", "r") as f:
            report = f.read()
        st.code(report, language='text')
    except Exception as e:
        st.error(f"❌ Error loading classification_report.txt: {e}")
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Confusion Matrix
    st.markdown("""
    <div class="feature-card">
        <h2>🔍 Confusion Matrix</h2>
        <p>Visual representation of model predictions vs actual classifications:</p>
    """, unsafe_allow_html=True)
    
    try:
        st.image("confusion_matrix.png", caption="Confusion Matrix - Model Prediction Accuracy", use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error loading confusion_matrix.png: {e}")
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Feature Importance
    st.markdown("""
    <div class="feature-card">
        <h2>⭐ Top Feature Importances</h2>
        <p>The most influential features in malware classification:</p>
    """, unsafe_allow_html=True)
    
    try:
        feature_df = pd.read_csv("feature_importances.csv").head(15)
        st.bar_chart(data=feature_df.set_index("Feature"))
    except Exception as e:
        st.error(f"❌ Error loading feature_importances.csv: {e}")
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Model Technical Details
    st.markdown("""
    <div class="feature-card">
        <h2>🔧 Technical Implementation</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem;">
            <div>
                <h3>Optimization</h3>
                <ul>
                    <li>Optuna hyperparameter tuning</li>
                    <li>Cross-validation based selection</li>
                    <li>Class imbalance handling</li>
                </ul>
            </div>
            <div>
                <h3>Feature Engineering</h3>
                <ul>
                    <li>ExtraTreesClassifier feature selection</li>
                    <li>Top 200 most important features</li>
                    <li>PE header and section analysis</li>
                </ul>
            </div>
            <div>
                <h3>Technology Stack</h3>
                <ul>
                    <li>XGBoost, Scikit-learn</li>
                    <li>Pandas, NumPy</li>
                    <li>Matplotlib, Seaborn</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def prediction_page():
    st.markdown('<h1>🔍 Real-Time Prediction</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <p style="font-size: 1.1rem; text-align: center;">
            Upload your feature CSV file to get instant malware detection results
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Download section
    st.markdown("""
    <div class="feature-card">
        <h2>📥 Download Support Files</h2>
        <p>Get the tools and samples you need to analyze your executable files:</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Sample Input CSV**")
        st.markdown("Use this sample to understand the required file format")
        try:
            with open("tools/sample_input.csv", "rb") as f:
                st.download_button(
                    label="📥 Download Sample CSV",
                    data=f,
                    file_name="sample_input.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        except:
            st.error("❌ Sample file not found")
    
    with col2:
        st.markdown("**Feature Extractor Tool**")
        st.markdown("Python script to extract features from your .exe files")
        try:
            with open("tools/extract_features.py", "rb") as f:
                st.download_button(
                    label="🐍 Download Extractor",
                    data=f,
                    file_name="extract_features.py",
                    mime="text/x-python",
                    use_container_width=True
                )
        except:
            st.error("❌ Extractor tool not found")
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Prediction section
    st.markdown("""
    <div class="feature-card">
        <h2>🎯 Upload & Analyze</h2>
        <p>Upload your feature CSV file extracted from an executable:</p>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose your feature CSV file", 
        type=["csv"],
        help="Upload a CSV file containing extracted features from your executable"
    )

    if uploaded_file is not None:
        with st.spinner('🔄 Analyzing your file...'):
            try:
                # Load and preprocess data
                input_df = pd.read_csv(uploaded_file)
                st.success("✅ File uploaded successfully!")
                
                # Show file info
                st.markdown(f"**📊 File Info:** {input_df.shape[0]} samples, {input_df.shape[1]} features")
                
                # Remove SHA256 if present
                if "SHA256" in input_df.columns:
                    input_df = input_df.drop(columns=["SHA256"])

                # Load model and make prediction
                model = joblib.load("xgb_model.joblib")
                feature_list = pd.read_csv("new_feature_order.csv")["Feature"].tolist()
                input_df = input_df.reindex(columns=feature_list, fill_value=0)

                prediction = model.predict(input_df)[0]
                prediction_proba = model.predict_proba(input_df)[0]

                # Label mapping
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
                confidence = prediction_proba[prediction] * 100

                # Display results
                if result == "Benign":
                    st.success(f"🟢 **SAFE**: {result}")
                    st.markdown(f"**Confidence:** {confidence:.1f}%")
                else:
                    st.error(f"🔴 **MALWARE DETECTED**: {result}")
                    st.markdown(f"**Confidence:** {confidence:.1f}%")
                    st.warning("⚠️ This file appears to be malicious. Handle with extreme caution!")

                # Show top probabilities
                st.markdown("### 📊 Detection Probabilities")
                prob_df = pd.DataFrame({
                    'Malware Type': [label_map[i] for i in range(len(prediction_proba))],
                    'Probability': prediction_proba * 100
                }).sort_values('Probability', ascending=False)
                
                st.bar_chart(prob_df.set_index('Malware Type'))

            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.markdown("**Possible issues:**")
                st.markdown("- File format doesn't match expected structure")
                st.markdown("- Missing required features")
                st.markdown("- Model files not found")
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- MAIN APP ---
def main():
    # Render navigation and get current tab
    current_tab = render_navigation()
    
    # Render content based on selected tab
    if current_tab == "Home":
        home_page()
    elif current_tab == "EDA Insights":
        eda_page()
    elif current_tab == "Model Performance":
        model_perf_page()
    elif current_tab == "Real-Time Prediction":
        prediction_page()

if __name__ == "__main__":
    main()
