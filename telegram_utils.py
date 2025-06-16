# telegram_utils.py

import requests

# Kamu bisa memindahkan nilai token dan chat ID ke config.py agar lebih aman
from config import TELEGRAM_API_TOKEN

def send_message(text, chat_id):
    """Mengirim pesan teks ke chat Telegram tertentu."""
    url = f"https://api.telegram.org/bot{TELEGRAM_API_TOKEN}/sendMessage"
    data = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    return requests.post(url, data=data)

def send_photo(image_file, chat_id):
    """Mengirim gambar (grafik saham) ke Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_API_TOKEN}/sendPhoto"
    files = {"photo": image_file}
    data = {"chat_id": chat_id}
    return requests.post(url, files=files, data=data)

def get_updates(offset=None):
    """Mengambil pesan terbaru dari Telegram dengan offset (untuk menghindari duplikasi)."""
    url = f"https://api.telegram.org/bot{TELEGRAM_API_TOKEN}/getUpdates"
    params = {"offset": offset} if offset else {}
    response = requests.get(url, params=params).json()
    return response.get("result", [])
