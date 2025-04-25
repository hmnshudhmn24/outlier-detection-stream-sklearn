# 🛡️ Real-Time Outlier Detection System with Scikit-learn

This project builds an unsupervised real-time outlier detection system using **Scikit-learn** and **Streamlit**. It's useful for fraud detection, anomaly spotting, or system health monitoring.

## ⚙️ Models Used

- **Isolation Forest**
- **One-Class SVM**
- **Local Outlier Factor (LOF)**

## 📦 Features

- Dummy streaming simulation with sleep delay
- Real-time labeling as "Outlier" or "Normal"
- Streamlit dashboard interface
- Supports multiple detection algorithms

## 📁 Dataset Format

Use a tabular dataset (numeric values only). A sample is included:

```csv
amount,time,transaction_type
120,1,0
135,2,0
900,3,1
...
```

## 🚀 How to Run

1. Install dependencies:

```bash
pip install pandas numpy scikit-learn streamlit
```

2. Run the app:

```bash
streamlit run src/main.py
```

3. View the dashboard and watch real-time predictions update live.

## 🎛️ Streamlit Interface

- Sidebar to select detection model
- Real-time display of prediction with simulated stream
