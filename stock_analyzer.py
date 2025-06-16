# stock_analyzer.py

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import time
from io import BytesIO
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

from telegram_utils import send_message, send_photo
from config import TELEGRAM_CHAT_ID, LQ45_TICKERS

# Set seed agar model konsisten
np.random.seed(42)
tf.random.set_seed(42)

def get_trading_days(start_date, count):
    """Mengembalikan N hari kerja dari start_date"""
    days = []
    while len(days) < count:
        if start_date.weekday() < 5:
            days.append(start_date)
        start_date += timedelta(days=1)
    return days

def analyze_stock(ticker):
    """Prediksi harga saham dengan LSTM dan kirim grafik ke Telegram"""
    try:
        data = yf.Ticker(ticker)
        df = data.history(period="6mo", interval="1d")

        if df.empty:
            send_message(f"⚠ Tidak ada data historis untuk *{ticker}*.\nCek ulang apakah kode saham valid di Yahoo Finance.", TELEGRAM_CHAT_ID)
            return None, None

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

        X, y = [], []
        for i in range(60, len(scaled) - 7):
            X.append(scaled[i - 60:i])
            y.append(scaled[i:i + 7])
        if not X:
            send_message(f"⚠ Data tidak cukup untuk menganalisis *{ticker}*", TELEGRAM_CHAT_ID)
            return None, None

        X = np.array(X).reshape(-1, 60, 1)
        y = np.array(y)

        model_path = f"models/{ticker}_model.h5"
        os.makedirs("models", exist_ok=True)
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(60, 1)),
                LSTM(50),
                Dense(7)
            ])
            model.compile(optimizer="adam", loss="mse")
            model.fit(X, y, epochs=10, batch_size=32, verbose=0)
            model.save(model_path)

        prediction = model.predict(scaled[-60:].reshape(1, 60, 1)).flatten()
        prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()

        harga_now = df["Close"].iloc[-1]
        harga_beli = min(prediction[1:3])
        harga_jual = max(prediction)
        kenaikan = ((harga_jual - harga_now) / harga_now) * 100

        result = {
            "Saham": ticker,
            "Harga Sekarang": harga_now,
            "Prediksi Beli": harga_beli,
            "Prediksi Jual": harga_jual,
            "Kenaikan (%)": kenaikan
        }

        dates = [d.strftime("%d-%m-%Y") for d in get_trading_days(datetime.today(), 7)]
        plt.figure(figsize=(8, 4))
        plt.plot(dates, prediction, marker="o", label="Prediksi")
        plt.axhline(harga_now, linestyle="--", color="red", label="Harga Sekarang")
        plt.title(f"Prediksi {ticker.replace('.JK','')}")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        return result, buf

    except Exception as e:
        send_message(f"⚠ Gagal analisis *{ticker}*: {e}", TELEGRAM_CHAT_ID)
        return None, None

def analyze_stocks():
    """Analisa massal saham LQ45"""
    results = []
    for ticker in LQ45_TICKERS:
        time.sleep(3)  # Perpanjang jeda untuk hindari pemblokiran
        result, _ = analyze_stock(ticker)
        if result and result["Kenaikan (%)"] > 7:
            results.append(result)
    return results

def send_stock_chart(stock_code):
    """Analisa 1 saham, kirim grafik & rekomendasi ke Telegram"""
    code = stock_code.upper()
    if not code.endswith(".JK"):
        code += ".JK"

    result, chart = analyze_stock(code)
    if not result:
        return

    send_photo(chart, TELEGRAM_CHAT_ID)
    pesan = (
        f"📈 *{code.replace('.JK','')}*\n"
        f"Harga Sekarang: Rp{result['Harga Sekarang']:.2f}\n"
        f"Prediksi Beli: Rp{result['Prediksi Beli']:.2f}\n"
        f"Prediksi Jual: Rp{result['Prediksi Jual']:.2f}\n"
        f"Kenaikan: {result['Kenaikan (%)']:.2f}%"
    )
    send_message(pesan, TELEGRAM_CHAT_ID)
