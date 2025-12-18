import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import gdown
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="📈 Stock Price Prediction",
    page_icon="📊",
    layout="wide"
)

# =========================
# CUSTOM UI STYLE
# =========================
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #141e30, #243b55);
    color: white;
}
h1, h2, h3 {
    color: #00e5ff;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("📈 Stock Price Prediction Using LSTM")
st.markdown(
    "Predict future stock prices using a deep learning LSTM model."
)

# =========================
# LOAD LSTM MODEL FROM GOOGLE DRIVE
# =========================
@st.cache_resource
def load_lstm_model():
    # 🔑 REPLACE WITH YOUR REAL GOOGLE DRIVE FILE ID
    file_id = "1fY6Zp1CqN5ZXScdxwGKt_Q1WAxjvQnWc"

    # ✅ CORRECT GOOGLE DRIVE URL
    url = f"https://drive.google.com/file/d/{file_id}/view"

    temp_model = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")

    # ✅ FUZZY DOWNLOAD HANDLES DRIVE CONFIRMATION
    gdown.download(
        url,
        temp_model.name,
        quiet=False,
        fuzzy=True
    )

    model = load_model(temp_model.name, compile=False)
    return model

# =========================
# MODEL LOADING
# =========================
with st.spinner("🔄 Loading LSTM model..."):
    model = load_lstm_model()

st.success("✅ Model loaded successfully")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "📂 Upload Stock CSV file (must contain 'Close' column)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "Close" not in df.columns:
        st.error("❌ CSV must contain a 'Close' column")
        st.stop()

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # =========================
    # DATA PREPROCESSING
    # =========================
    close_prices = df["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    sequence_length = 60
    X = []

    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])

    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # =========================
    # PREDICTION
    # =========================
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)

    actual = close_prices[sequence_length:]

    # =========================
    # VISUALIZATION
    # =========================
    st.subheader("📊 Actual vs Predicted Stock Prices")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(actual, label="Actual Price")
    ax.plot(predictions, label="Predicted Price")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()

    st.pyplot(fig)

    # =========================
    # NEXT DAY PREDICTION
    # =========================
    last_sequence = scaled_data[-sequence_length:]
    last_sequence = last_sequence.reshape(1, sequence_length, 1)

    next_day_price = model.predict(last_sequence)
    next_day_price = scaler.inverse_transform(next_day_price)

    st.markdown("## 📌 Next Day Predicted Price")
    st.success(f"₹ {next_day_price[0][0]:.2f}")

else:
    st.info("⬆️ Upload a CSV file to start prediction")

# =========================
# FOOTER
# =========================
st.markdown("""
---
Developed using **LSTM & Streamlit**  
""")
