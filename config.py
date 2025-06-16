# config.py

import os

TELEGRAM_API_TOKEN = os.getenv("TELEGRAM_API_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Daftar saham LQ45 statis dari Yahoo Finance (.JK = Bursa Efek Indonesia)
LQ45_TICKERS = [
    "BBCA.JK", "TLKM.JK", "BBRI.JK", "BMRI.JK", "ASII.JK", "UNVR.JK", "PGAS.JK", "TINS.JK",
    "MDKA.JK", "ANTM.JK", "ICBP.JK", "INDF.JK", "ADRO.JK", "BRPT.JK", "CPIN.JK", "EXCL.JK",
    "GGRM.JK", "HMSP.JK", "KLBF.JK", "MEDC.JK", "MIKA.JK", "SMGR.JK", "TKIM.JK", "WIKA.JK",
    "WSKT.JK", "BBTN.JK", "BFIN.JK", "BUKA.JK", "ELSA.JK", "ERAA.JK", "INDY.JK", "JPFA.JK",
    "MNCN.JK", "PTPP.JK", "SCMA.JK", "SIDO.JK", "SMRA.JK", "TBIG.JK", "TOWR.JK", "UNTR.JK",
    "WIIM.JK", "ZINC.JK"
]
