# 🛡️ Windows Malware Detection

A machine learning-powered malware detection system that analyzes Windows PE (Portable Executable) files using static analysis techniques. This project leverages XGBoost to identify and classify malware families based on extracted static features.


## 🎯 Project Overview

This project focuses on detecting and classifying Windows malware using machine learning techniques. The system analyzes static features extracted from PE files, including:

- **DLL Imports** - Dynamic Link Library dependencies
- **PE Headers** - Portable Executable file headers and metadata
- **Windows API Calls** - System function call patterns


## 📁 Project Structure

```
EATC_Team7/
├── app.py                        # Streamlit web application
├── EATC_ASG2 (Final).ipynb      # Model training & analysis notebook
├── random_forest_model.joblib   # Trained Random Forest model
├── xgb_model.joblib             # Trained XGBoost model
├── new_feature_order.csv        # Feature order used during training
├── requirements.txt             # Python dependencies
├── tools/                       # Helper scripts and feature extractors
├── assets/                      # Images and visual content
├── Datasets/                 # Demo CSV files for testing
├── API_Functions.zip            # Windows API functions dataset
├── DLLs_Imported.zip           # DLL imports dataset
├── PE_Header.zip               # PE header features dataset
└── PE_Section.zip              # PE section information dataset
```

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/EATC_Team7.git
   cd EATC_Team7
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

4. **Alternatively, Access the Web Interface**
  visit our live application: [https://eatcteam7.streamlit.app/](https://eatcteam7.streamlit.app/)

## 💻 Usage

### Web Application Features

- **📤 File Upload**: Upload CSV files containing PE features
- **🔮 Malware Prediction**: Get instant malware family classification
- **📊 Data Visualization**: Explore dataset through interactive charts
- **📈 Model Insights**: View model performance metrics and feature importance
- **📥 Sample Data**: Download sample datasets for testing

### Using the Jupyter Notebook

Open `EATC_ASG2 (Final).ipynb` to:
- Explore the complete data analysis pipeline
- Understand feature engineering techniques
- Review model training and evaluation process
- Experiment with hyperparameter tuning

## 🧠 Model Architecture

### Algorithms Used

- **🚀 XGBoost Classifier**
  - Gradient boosting framework
  - High performance and accuracy
  - Efficient handling of missing values

### Performance Metrics

XGBoost Accuracy ~89%

### Feature Engineering

The model uses over 2000+ static features including:
- Import address table analysis
- Section entropy calculations
- PE header characteristics
- API call frequency patterns

### Source
- **Primary Dataset**: [Kaggle - Windows Malwares](https://www.kaggle.com/datasets/joebeachcapital/windows-malwares/data)
