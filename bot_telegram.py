import requests
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from io import BytesIO
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime, timedelta

# ---------------------------
# Konfigurasi Telegram Bot
# ---------------------------
TELEGRAM_API_TOKEN = "7126717961:AAFz9fwffUPXlbAg3LqK9-zBF11hmI95KDw"
TELEGRAM_CHAT_ID = "6778588870"

def send_telegram_message(message):
    """Mengirim pesan teks ke Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_API_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    response = requests.post(url, data=data)
    return response.status_code

def send_telegram_photo(image):
    """Mengirim gambar ke Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_API_TOKEN}/sendPhoto"
    files = {"photo": image}
    data = {"chat_id": TELEGRAM_CHAT_ID}
    response = requests.post(url, files=files, data=data)
    return response.status_code

def get_updates(offset=None):
    """Mengambil update baru dari Telegram dengan mekanisme offset."""
    url = f"https://api.telegram.org/bot{TELEGRAM_API_TOKEN}/getUpdates"
    params = {}
    if offset:
        params["offset"] = offset
    response = requests.get(url, params=params)
    data = response.json()
    if "result" in data:
        return data["result"]
    return []

# ---------------------------
# Daftar Saham LQ45 (Statis)
# ---------------------------
lq45_tickers = [
    "BBCA.JK", "TLKM.JK", "BBRI.JK", "BMRI.JK", "ASII.JK", "UNVR.JK", "PGAS.JK", "TINS.JK", 
    "MDKA.JK", "ANTM.JK", "ICBP.JK", "INDF.JK", "ADRO.JK", "BRPT.JK", "CPIN.JK", "EXCL.JK", 
    "GGRM.JK", "HMSP.JK", "KLBF.JK", "MEDC.JK", "MIKA.JK", "SMGR.JK", "TKIM.JK", "WIKA.JK", 
    "WSKT.JK", "BBTN.JK", "BFIN.JK", "BUKA.JK", "ELSA.JK", "ERAA.JK", "INDY.JK", "JPFA.JK", 
    "MNCN.JK", "PTPP.JK", "SCMA.JK", "SIDO.JK", "SMRA.JK", "TBIG.JK", "TOWR.JK", "UNTR.JK", 
    "WIKA.JK", "WIIM.JK", "ZINC.JK"
]

print(f"✅ Menggunakan daftar saham LQ45 statis ({len(lq45_tickers)} saham)")

# ---------------------------
# Fungsi Analisis Saham
# ---------------------------
def analyze_stocks():
    """Melakukan analisis saham dan prediksi menggunakan model LSTM.
       Hasil analisis disimpan dalam variabel global 'predicted_stocks'."""
    global predicted_stocks
    predicted_stocks = []  # Kosongkan data analisis sebelumnya
    today = datetime.today()
    
    for ticker in lq45_tickers:
        # Tambahkan jeda agar tidak membebani akses ke Yahoo Finance
        time.sleep(2)
        
        data = yf.Ticker(ticker)
        df = data.history(period="6mo")  # Gunakan data 6 bulan
        
        if df.empty:
            continue
        
        # Preprocessing Data untuk LSTM
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_scaled = scaler.fit_transform(df["Close"].values.reshape(-1, 1))
        
        # Membuat dataset untuk LSTM
        X_train, y_train = [], []
        for i in range(60, len(df_scaled) - 7):
            X_train.append(df_scaled[i-60:i, 0])
            y_train.append(df_scaled[i:i+7, 0])  # Prediksi 7 hari ke depan
        
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        # Gunakan model LSTM yang sudah dilatih sebelumnya
        model_path = f"{ticker}_lstm_model.h5"
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                LSTM(units=50, return_sequences=False),
                Dense(units=7)  # Output: prediksi 7 hari ke depan
            ])
            model.compile(optimizer="adam", loss="mean_squared_error")
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
            model.save(model_path)
        
        # Prediksi harga 7 hari ke depan
        last_60_days = df_scaled[-60:].reshape(1, 60, 1)
        predicted_prices = model.predict(last_60_days)
        predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))
        
        if len(predicted_prices.flatten()) < 7:
            continue  # Lewati jika prediksi kurang dari 7 hari
        
        # Hitung level beli dan jual otomatis
        harga_sekarang = df["Close"].iloc[-1]
        harga_prediksi_tertinggi = max(predicted_prices.flatten())
        harga_prediksi_beli = min(predicted_prices.flatten()[1:3])
        
        # Hitung persentase kenaikan harga
        kenaikan_persen = ((harga_prediksi_tertinggi - harga_sekarang) / harga_sekarang) * 100
        
        # Filter hanya saham dengan kenaikan > 7%
        if kenaikan_persen >= 7:
            predicted_stocks.append({
                "Saham": ticker,  # Simpan ticker dengan ".JK" untuk proses internal
                "Harga Sekarang": harga_sekarang,
                "Prediksi Beli": harga_prediksi_beli,
                "Prediksi Jual": harga_prediksi_tertinggi,
                "Kenaikan (%)": kenaikan_persen
            })
        
        # Buat grafik prediksi harga dengan tanggal
        tanggal_prediksi = [(today + timedelta(days=i)).strftime("%d-%m-%Y") for i in range(1, len(predicted_prices.flatten()) + 1)]
        
        plt.figure(figsize=(8, 4))
        plt.plot(tanggal_prediksi, predicted_prices.flatten(), marker="o", linestyle="-", label="Prediksi Harga")
        plt.axhline(y=harga_sekarang, color="r", linestyle="--", label="Harga Sekarang")
        plt.title(f"Prediksi Harga {ticker.replace('.JK','')}")
        plt.xlabel("Tanggal")
        plt.ylabel("Harga (Rp)")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()
        
        # Pastikan grafik tidak terpotong
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        send_telegram_photo(buf)
        
    # Kirim daftar saham terbaik (maksimal 10 saham)
    if predicted_stocks:
        pesan = "📊 *10 Saham LQ45 dengan Kenaikan Harga >7%* 📊\n\n"
        # Saat mengirim pesan, hilangkan ekstensi ".JK"
        for stock in sorted(predicted_stocks, key=lambda x: x["Kenaikan (%)"], reverse=True)[:10]:
            saham_clean = stock["Saham"].replace(".JK", "")
            pesan += (f"🔹 {saham_clean} | Harga Sekarang: Rp{stock['Harga Sekarang']:.2f} | "
                      f"Prediksi Jual: Rp{stock['Prediksi Jual']:.2f} | Kenaikan: {stock['Kenaikan (%)']:.2f}%\n")
        send_telegram_message(pesan)
    else:
        send_telegram_message("⚠ Tidak ada saham dengan kenaikan harga di atas 7%.")

# ---------------------------
# Analisis awal saat bot dijalankan
# ---------------------------
analyze_stocks()

# ---------------------------
# Memproses update dari Telegram dengan offset
# ---------------------------
last_update_id = None

while True:
    updates = get_updates(offset=last_update_id)
    if updates:
        for update in updates:
            last_update_id = update["update_id"] + 1  # update offset agar pesan tidak diproses ulang
            if "message" in update and "text" in update["message"]:
                text = update["message"]["text"].lower()
                if text == "cek":
                    send_telegram_message("🔄 Mengirim ulang hasil analisis terakhir...")
                    if predicted_stocks:
                        for stock in sorted(predicted_stocks, key=lambda x: x["Kenaikan (%)"], reverse=True)[:10]:
                            saham_clean = stock["Saham"].replace(".JK", "")
                            send_telegram_message(f"🔹 {saham_clean} | Harga Sekarang: Rp{stock['Harga Sekarang']:.2f} | "
                                                  f"Prediksi Jual: Rp{stock['Prediksi Jual']:.2f} | Kenaikan: {stock['Kenaikan (%)']:.2f}%")
                    else:
                        send_telegram_message("⚠ Tidak ada data analisis terakhir.")
                elif text == "ulang":
                    send_telegram_message("🔄 Melakukan analisis ulang...")
                    analyze_stocks()
                    send_telegram_message("✅ Analisis ulang selesai! Gunakan perintah 'Cek' untuk melihat hasil terbaru.")
    time.sleep(5)
