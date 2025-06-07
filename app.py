import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

model_nb = joblib.load("model_nb.pkl")
model_lr = joblib.load("model_logres.pkl")

scaler_nb = joblib.load("scaler_nb.pkl")
scaler_lr = joblib.load("scaler_logres.pkl")

# Dummy accuracy scores
accuracy_scores = {
    "Naive Bayes": 35.4,
    "Logistic Regression": 37.9
}

# =================== Streamlit UI ===================
st.set_page_config(page_title="EV User Type Predictor", layout="centered")
st.title("🚗 EV User Type Predictor")
st.markdown("Masukkan data sesi pengisian daya di bawah ini untuk memprediksi **User Type** dan **karakteristik kluster** pengguna.")

# Model selection
model_choice = st.selectbox("Pilih Model", ["Naive Bayes", "Logistic Regression"])
st.markdown(f"**Akurasi model:** {accuracy_scores[model_choice]}%")

# Input form
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        energy = st.number_input("Energy Consumed (kWh)", min_value=0.0, max_value=100.0, step=0.1)
        duration = st.number_input("Charging Duration (hours)", min_value=0.0, max_value=24.0, step=0.1)
        rate = st.number_input("Charging Rate (kW)", min_value=0.0, max_value=300.0, step=0.5)
        soc_start = st.number_input("State of Charge (Start %)", min_value=0, max_value=100)

    with col2:
        soc_end = st.number_input("State of Charge (End %)", min_value=0, max_value=100)
        distance = st.number_input("Distance Driven (since last charge) (km)", min_value=0.0, max_value=1000.0, step=1.0)
        temp = st.number_input("Temperature (\u00b0C)", min_value=-10.0, max_value=73.0, step=0.5)
        charger_type = st.selectbox("Charger Type", options=["Level 1", "Level 2", "DC Fast"], index=0)

    submitted = st.form_submit_button("🔍 Submit")

# Charger type mapping
charger_map = {"Level 1": 0, "Level 2": 1, "DC Fast": 2}

if submitted:
    try:
        input_data = np.array([[energy, duration, rate, soc_start, soc_end, distance, temp, charger_map[charger_type]]])

        # Validasi manual
        if soc_end < soc_start:
            st.error("❌ End SOC tidak boleh lebih kecil dari Start SOC.")
        else:
            # Standarisasi
            if model_choice == "Naive Bayes":
                scaler = scaler_nb
            else:
                scaler = scaler_lr
            input_scaled = scaler.transform(input_data)

            # Prediksi
            if model_choice == "Naive Bayes":
                prediction = model_nb.predict(input_scaled)[0]
            else:
                prediction = model_lr.predict(input_scaled)[0]

            # Karakteristik kluster 
            cluster_info = {
                0: "🔋 Commuter - penggunaan rutin, jarak sedang, pengisian stabil",
                1: "🚀 Long Distance - jarak jauh, biaya besar, SOC tinggi",
                2: "🌿 Casual Driver - jarang isi, hemat energi"
            }

            # Output prediksi dan karakteristik
            st.success(f"✅ Prediksi: **User Type {prediction}**")
            st.info(cluster_info.get(prediction, "Karakteristik tidak diketahui."))

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {e}")
