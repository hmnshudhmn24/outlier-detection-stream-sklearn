import pandas as pd
import numpy as np
import time
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Load dataset
df = pd.read_csv('data/data.csv')
features = df.columns.tolist()
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Select model
model_option = st.sidebar.selectbox("Choose Outlier Detection Model", ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"])
st.title("🚨 Real-Time Outlier Detection Dashboard")

if model_option == "Isolation Forest":
    model = IsolationForest(contamination=0.1)
elif model_option == "One-Class SVM":
    model = OneClassSVM(nu=0.1)
else:
    model = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)

# Fit the model
if model_option == "Local Outlier Factor":
    model.fit(df_scaled)
else:
    model.fit(df_scaled)

# Simulate real-time detection
stream_data = df.sample(frac=1).reset_index(drop=True)

for i in range(len(stream_data)):
    x = scaler.transform([stream_data.iloc[i]])
    if model_option == "Local Outlier Factor":
        prediction = model.predict(x)
    else:
        prediction = model.predict(x)
    label = "❌ Outlier" if prediction[0] == -1 else "✅ Normal"
    st.write(f"**Incoming Data:** {stream_data.iloc[i].to_dict()} --> **{label}**")
    time.sleep(0.3)
