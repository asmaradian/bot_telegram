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
# KONFIGURASI TELEGRAM BOT
# ---------------------------
TELEGRAM_API_TOKEN = "7126717961:AAFz9fwffUPXlbAg3LqK9-zBF11hmI95KDw"
TELEGRAM_CHAT_ID = "6778588870"

def send_telegram_message(message):
    """Kirim pesan teks ke Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_API_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    response = requests.post(url, data=data)
    return response.status_code

def send_telegram_photo(image):
    """Kirim gambar ke Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_API_TOKEN}/sendPhoto"
    files = {"photo": image}
    data = {"chat_id": TELEGRAM_CHAT_ID}
    response = requests.post(url, files=files, data=data)
    return response.status_code

def get_updates(offset=None):
    """Ambil update baru dari Telegram dengan mekanisme offset."""
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
# Fungsi BANTUAN UNTUK MENGAMBIL HARI PERDAGANGAN (TRADING DAYS)
# ---------------------------
def get_trading_days(start_date, count):
    """
    Mengembalikan list 'count' trading days mulai dari start_date.
    Jika start_date adalah hari kerja, maka akan disertakan,
    jika tidak, akan dilewati sampai hari kerja berikutnya.
    """
    trading_days = []
    current_date = start_date
    while len(trading_days) < count:
        if current_date.weekday() < 5:  # 0=Senin s/d 4=Jumat
            trading_days.append(current_date)
        current_date += timedelta(days=1)
    return trading_days

# ---------------------------
# DAFTAR SAHAM LQ45 (Statis)
# ---------------------------
# Ticker internal memakai ekstensi ".JK" untuk penggunaan API; pada output ekstensi akan dihapus.
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
# VARIABEL GLOBAL UNTUK MENYIMPAN HASIL ANALISIS
# ---------------------------
predicted_stocks = []  # Menyimpan saham dengan kenaikan >7%
last_message_text = ""  # Untuk menghindari pengolahan duplikasi pesan

# ---------------------------
# FUNGSI ANALISIS INDIVIDUAL
# ---------------------------
def analyze_stock(ticker):
    """
    Melakukan analisis untuk satu ticker.
    Mengembalikan tuple (result, buf) jika sukses, atau (None, None) jika gagal.
    """
    today = datetime.today()  # Titik awal untuk label tanggal
    try:
        data = yf.Ticker(ticker)
        df = data.history(period="6mo")
        if df.empty:
            return None, None

        # Preprocessing data
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_scaled = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

        # Buat dataset untuk LSTM
        X_train, y_train = [], []
        for i in range(60, len(df_scaled) - 7):
            X_train.append(df_scaled[i-60:i, 0])
            y_train.append(df_scaled[i:i+7, 0])
        if not X_train:
            return None, None
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Muat atau latih model LSTM
        model_path = f"{ticker}_lstm_model.h5"
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                LSTM(units=50, return_sequences=False),
                Dense(units=7)
            ])
            model.compile(optimizer="adam", loss="mean_squared_error")
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
            model.save(model_path)

        # Prediksi 7 hari mendatang
        last_60_days = df_scaled[-60:].reshape(1, 60, 1)
        predicted_prices = model.predict(last_60_days)
        predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))
        if len(predicted_prices.flatten()) < 7:
            return None, None

        harga_sekarang = df["Close"].iloc[-1]
        harga_prediksi_tertinggi = max(predicted_prices.flatten())
        harga_prediksi_beli = min(predicted_prices.flatten()[1:3])
        kenaikan_persen = ((harga_prediksi_tertinggi - harga_sekarang) / harga_sekarang) * 100

        result = {
            "Saham": ticker,
            "Harga Sekarang": harga_sekarang,
            "Prediksi Beli": harga_prediksi_beli,
            "Prediksi Jual": harga_prediksi_tertinggi,
            "Kenaikan (%)": kenaikan_persen
        }

        # Buat label tanggal prediksi hanya untuk hari perdagangan dari HARI INI
        trading_dates = get_trading_days(datetime.today(), len(predicted_prices.flatten()))
        tanggal_prediksi = [dt.strftime("%d-%m-%Y") for dt in trading_dates]

        # Buat grafik prediksi
        plt.figure(figsize=(8, 4))
        plt.plot(tanggal_prediksi, predicted_prices.flatten(), marker="o", linestyle="-", label="Prediksi Harga")
        plt.axhline(y=harga_sekarang, color="r", linestyle="--", label="Harga Sekarang")
        plt.title(f"Prediksi Harga {ticker.replace('.JK','')}")
        plt.xlabel("Tanggal")
        plt.ylabel("Harga (Rp)")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()
        return result, buf
    except Exception as e:
        print(f"Error saat analisis {ticker}: {e}")
        return None, None

# ---------------------------
# FUNGSI ANALISIS MASSAL
# ---------------------------
def analyze_stocks():
    """
    Melakukan analisis untuk seluruh saham dalam lq45_tickers.
    Menyimpan saham dengan kenaikan >7% ke variable global predicted_stocks.
    Grafik tidak dikirim untuk analisis massal.
    """
    global predicted_stocks
    predicted_stocks = []  # Hapus hasil analisis sebelumnya
    for ticker in lq45_tickers:
        time.sleep(2)  # Jeda agar tidak membanjiri request ke Yahoo Finance
        result, _ = analyze_stock(ticker)
        if result and result["Kenaikan (%)"] >= 7:
            predicted_stocks.append(result)
    if predicted_stocks:
        pesan = "✅ *Analisis selesai!*\nGunakan perintah `cek` untuk melihat ringkasan 10 saham terbaik."
    else:
        pesan = "⚠ Analisis selesai, tetapi tidak ditemukan saham dengan kenaikan >7%."
    send_telegram_message(pesan)

# ---------------------------
# FUNGSI MENGIRIM GRAFIK DAN KETERANGAN SAHAM INDIVIDUAL
# ---------------------------
def send_stock_chart(stock_code):
    """
    Menerima kode saham (contoh: 'BBCA' atau 'BBCA.JK').
    Jika tidak ada '.JK', maka akan ditambahkan.
    Mengirim grafik dan keterangan analisis untuk ticker tersebut.
    """
    kode = stock_code.upper()
    if not kode.endswith(".JK"):
        kode_full = kode + ".JK"
    else:
        kode_full = kode
    if kode_full not in lq45_tickers:
        send_telegram_message(f"⚠ Kode saham {kode} tidak ditemukan dalam daftar.")
        return
    result, buf = analyze_stock(kode_full)
    if result is None:
        send_telegram_message(f"⚠ Gagal melakukan analisis untuk {kode}.")
    else:
        send_telegram_photo(buf)
        saham_clean = result["Saham"].replace(".JK", "")
        pesan = (f"📈 *{saham_clean}*\n"
                 f"Harga Sekarang: Rp{result['Harga Sekarang']:.2f}\n"
                 f"Prediksi Beli: Rp{result['Prediksi Beli']:.2f}\n"
                 f"Prediksi Jual: Rp{result['Prediksi Jual']:.2f}\n"
                 f"Kenaikan: {result['Kenaikan (%)']:.2f}%")
        send_telegram_message(pesan)

# ---------------------------
# PROSES UPDATE DARI TELEGRAM DENGAN OFFSET
# ---------------------------
last_update_id = None
processed_ids = set()  # Melacak update_id yang telah diproses

# ---------------------------
# LOOP UTAMA BOT
# ---------------------------
while True:
    updates = get_updates(offset=last_update_id)
    if updates:
        for update in updates:
            update_id = update["update_id"]
            # Jika update sudah diproses, lewati
            if update_id in processed_ids:
                continue
            processed_ids.add(update_id)
            last_update_id = update_id + 1  # Perbarui offset
            if "message" in update and "text" in update["message"]:
                text = update["message"]["text"].lower().strip()
                # Cegah pengolahan duplikasi pesan jika pesan sama dengan pesan terakhir
                if text == last_message_text:
                    print(f"Duplikasi pesan terdeteksi: '{text}'")
                    continue
                last_message_text = text
                # Logging update untuk debug
                print(f"Memproses update_id {update_id} dengan pesan: '{text}'")
                if text == "analisa":
                    send_telegram_message("🔄 Mulai analisis saham...")
                    analyze_stocks()
                elif text == "cek":
                    send_telegram_message("🔄 Mengirim ringkasan analisis...")
                    if predicted_stocks:
                        summary = "📊 *10 Saham LQ45 dengan Kenaikan >7%:*\n\n"
                        top10 = sorted(predicted_stocks, key=lambda x: x["Kenaikan (%)"], reverse=True)[:10]
                        for stock in top10:
                            saham_clean = stock["Saham"].replace(".JK", "")
                            summary += (f"🔹 {saham_clean} | Harga: Rp{stock['Harga Sekarang']:.2f} | "
                                        f"Prediksi Jual: Rp{stock['Prediksi Jual']:.2f} | "
                                        f"Kenaikan: {stock['Kenaikan (%)']:.2f}%\n")
                        send_telegram_message(summary)
                    else:
                        send_telegram_message("⚠ Data analisis belum tersedia. Gunakan perintah `analisa` terlebih dahulu.")
                elif text == "ulang":
                    send_telegram_message("🔄 Melakukan analisis ulang... (hasil sebelumnya dihapus)")
                    analyze_stocks()
                else:
                    # Anggap pesan merupakan kode saham
                    send_stock_chart(text)
    time.sleep(5)
