# bot.py

from telegram_utils import send_message, get_updates, send_photo
from stock_analyzer import analyze_stocks, send_stock_chart
from config import TELEGRAM_CHAT_ID

import time

last_update_id = None
last_message_text = ""
predicted_stocks = []

while True:
    updates = get_updates(offset=last_update_id)
    for update in updates:
        uid = update["update_id"]
        msg = update.get("message", {}).get("text", "").lower().strip()

        if uid == last_update_id or msg == last_message_text:
            continue  # Hindari duplikasi

        last_update_id = uid + 1
        last_message_text = msg

        if msg == "analisa":
            send_message("📊 Memulai analisis saham...", TELEGRAM_CHAT_ID)
            predicted_stocks = analyze_stocks()
        elif msg == "cek":
            if not predicted_stocks:
                send_message("⚠ Belum ada data. Ketik `analisa` dulu.", TELEGRAM_CHAT_ID)
            else:
                summary = "*📈 Ringkasan Saham:* \n\n"
                for stock in predicted_stocks[:10]:
                    kode = stock["Saham"].replace(".JK", "")
                    summary += (
                        f"🔹 {kode} | Harga: Rp{stock['Harga Sekarang']:.2f} | "
                        f"Prediksi Jual: Rp{stock['Prediksi Jual']:.2f} | "
                        f"Kenaikan: {stock['Kenaikan (%)']:.2f}%\n"
                    )
                send_message(summary, TELEGRAM_CHAT_ID)
        else:
            send_stock_chart(msg)

    time.sleep(5)

# trigger deploy
