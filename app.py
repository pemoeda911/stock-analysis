import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
import google.generativeai as genai

# --- KONFIGURASI HALAMAN WEB ---
st.set_page_config(page_title="Screener Saham BEI", page_icon="📈", layout="wide")

# --- FUNGSI PENGAMBILAN DATA (DENGAN CACHE) ---
@st.cache_data(ttl=3600, show_spinner=False)
def dapatkan_kurs_usd_idr():
    try:
        ticker = yf.Ticker("IDR=X")
        hist = ticker.history(period="1d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1]), "Yahoo Finance"
    except Exception:
        pass
    
    try:
        response = requests.get("https://open.er-api.com/v6/latest/USD", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'rates' in data and 'IDR' in data['rates']:
                return float(data['rates']['IDR']), "ExchangeRate-API"
    except Exception:
        pass

    return 16200.0, "Sistem Cadangan (Statis)"

@st.cache_data(ttl=3600, show_spinner=False)
def dapatkan_data_saham(ticker_symbol, periode="6mo", kurs_usd_aktif=16200.0):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = {}
        for attempt in range(3):
            try:
                info = ticker.info
                if info and 'symbol' in info: break
            except Exception:
                time.sleep(1) 
        
        hist = ticker.history(period=periode)
        if hist.empty or len(hist) < 30: return None 
        
        harga_terakhir = hist['Close'].iloc[-1]
        mata_uang = info.get('financialCurrency', 'IDR')
        kurs_usd = kurs_usd_aktif if mata_uang == 'USD' else 1.0
        
        eps = info.get('trailingEps')
        bvps = info.get('bookValue')
        
        if eps is not None: eps = eps * kurs_usd
        if bvps is not None: bvps = bvps * kurs_usd
        
        pe_ratio = (harga_terakhir / eps) if (eps and eps > 0) else np.nan
        pbv = (harga_terakhir / bvps) if (bvps and bvps > 0) else np.nan
        
        roe = info.get('returnOnEquity')
        der = info.get('debtToEquity') 
        div_yield = info.get('dividendYield') or info.get('trailingAnnualDividendYield')
        
        if der is not None: der = der / 100.0
            
        harga_wajar_graham = np.nan
        if eps and bvps and eps > 0 and bvps > 0:
            harga_wajar_graham = np.sqrt(22.5 * eps * bvps)
            
        hist['MA20'] = hist['Close'].rolling(window=20).mean()
        ma20_terakhir = hist['MA20'].iloc[-1]
        
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
        exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd_bullish = (exp1 - exp2).iloc[-1] > (exp1 - exp2).ewm(span=9, adjust=False).mean().iloc[-1]
        
        low_14 = hist['Low'].rolling(window=14).min()
        high_14 = hist['High'].rolling(window=14).max()
        stoch_k = 100 * ((hist['Close'] - low_14) / (high_14 - low_14 + 1e-9))
        
        return {
            'Ticker': ticker_symbol.replace('.JK', ''),
            'Harga': harga_terakhir,
            'Harga_Wajar': harga_wajar_graham,
            'PE_Ratio': pe_ratio,
            'PBV': pbv,
            'ROE': roe,
            'RSI_14': hist['RSI'].iloc[-1],
            'MACD_Bullish': macd_bullish,
            'Stoch_K': stoch_k.iloc[-1]
        }
    except Exception:
        return None

# --- FUNGSI FORMATTING MATA UANG ---
def format_mata_uang(val):
    """Memformat angka menjadi string dengan simbol Rp di kiri dan nilai rata kanan menggunakan spasi non-breaking."""
    if pd.isnull(val) or np.isnan(val):
        return "-"
    # Menambahkan spasi agar terlihat rata kanan
    return f"Rp &nbsp;&nbsp;&nbsp; {int(val):,}"

# --- ANTARMUKA PENGGUNA ---
# ... (kode lainnya tetap sama)
# Saat memproses dataframe 'tampil':

# tampil['Harga'] = tampil['Harga'].apply(format_mata_uang)
# tampil['Harga_Wajar'] = tampil['Harga_Wajar'].apply(format_mata_uang)

# --- (Pastikan untuk menggunakan st.markdown(tampil.to_html(escape=False), unsafe_allow_html=True) 
# jika ingin perataan benar-benar tampil di browser, karena st.dataframe 
# akan me-render tag HTML tersebut sebagai teks mentah) ---
