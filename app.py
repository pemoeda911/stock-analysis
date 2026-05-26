import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
import google.generativeai as genai

# --- KONFIGURASI HALAMAN WEB ---
st.set_page_config(page_title="Screener Saham BEI", page_icon="📈", layout="wide")

# --- CSS UNTUK STYLING TABEL ---
st.markdown("""
<style>
    .custom-table { width: 100%; border-collapse: collapse; font-size: 0.9em; margin-bottom: 2rem; }
    .custom-table th { background-color: #f0f2f6; padding: 12px; border: 1px solid #ddd; text-align: center; font-weight: bold; }
    .custom-table td { padding: 10px; border: 1px solid #ddd; vertical-align: middle; }
    .currency-cell { display: flex; justify-content: space-between; font-family: 'Courier New', Courier, monospace; }
    .currency-symbol { text-align: left; color: #555; }
    .currency-value { text-align: right; font-weight: bold; }
    .text-center { text-align: center; }
</style>
""", unsafe_allow_html=True)

# --- FUNGSI FORMATTING HTML ---
def render_currency_html(val):
    if pd.isnull(val) or isinstance(val, str) or np.isnan(val):
        return '<div class="text-center">-</div>'
    return f'<div class="currency-cell"><span class="currency-symbol">Rp</span><span class="currency-value">{int(val):,}</span></div>'

# --- FUNGSI PENGAMBILAN DATA ---
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
                if info and 'symbol' in info: 
                    break
            except Exception:
                time.sleep(1) 
        
        if not info or 'symbol' not in info:
            session = requests.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            })
            ticker_custom = yf.Ticker(ticker_symbol, session=session)
            try:
                info = ticker_custom.info
                if info is None:
                    info = {}
            except Exception:
                info = {} 
        
        hist = ticker.history(period=periode)
        if hist.empty or len(hist) < 30:
            return None 
        
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
        
        rata_volume_20hari = hist['Volume'].rolling(window=20).mean().iloc[-1]
        rata_volume_5hari = hist['Volume'].iloc[-5:].mean()
        
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        hist['RSI'] = 100 - (100 / (1 + rs))
        rsi_terakhir = hist['RSI'].iloc[-1]
        
        exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
        exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_bullish = macd.iloc[-1] > signal.iloc[-1]
        
        bb_std = hist['Close'].rolling(window=20).std()
        bb_lower = hist['MA20'] - (2 * bb_std)
        bb_upper = hist['MA20'] + (2 * bb_std)
        dekat_bb_bawah = harga_terakhir <= (bb_lower.iloc[-1] * 1.05)
        
        low_20 = hist['Low'].rolling(window=20).min().iloc[-1]
        high_20 = hist['High'].rolling(window=20).max().iloc[-1]
        
        target_beli_teknikal = (bb_lower.iloc[-1] + low_20) / 2
        target_jual_teknikal = (bb_upper.iloc[-1] + high_20) / 2
        
        low_14 = hist['Low'].rolling(window=14).min()
        high_14 = hist['High'].rolling(window=14).max()
        stoch_k = 100 * ((hist['Close'] - low_14) / (high_14 - low_14 + 1e-9))
        stoch_k_terakhir = stoch_k.iloc[-1]
        
        return {
            'Ticker': ticker_symbol.replace('.JK', ''),
            'Harga': harga_terakhir,
            'Harga_Wajar': harga_wajar_graham,
            'Target_Beli': target_beli_teknikal,
            'Target_Jual': target_jual_teknikal,
            'PE_Ratio': pe_ratio,
            'PBV': pbv,
            'ROE': roe,
            'RSI_14': rsi_terakhir if pd.notnull(rsi_terakhir) else 50.0,
            'MACD_Bullish': macd_bullish,
            'Stoch_K': stoch_k_terakhir,
            'Di_Atas_MA20': harga_terakhir > ma20_terakhir if pd.notnull(ma20_terakhir) else False,
            'Dekat_BB_Bawah': dekat_bb_bawah
        }
    except Exception as e:
        print(f"Error fetching {ticker_symbol}: {str(e)}")
        return None

def saring_saham_pilihan(daftar_saham, periode, strategi_pilihan, kurs_usd_aktif):
    hasil_analisis = []
    error_tickers = []
    
    progress_text = "Memindai saham..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, tkr in enumerate(daftar_saham):
        data = dapatkan_data_saham(tkr, periode, kurs_usd_aktif)
        if data:
            hasil_analisis.append(data)
        else:
            error_tickers.append(tkr.replace('.JK', ''))
            
        my_bar.progress((i + 1) / len(daftar_saham), text=f"Memindai {tkr}...")
        time.sleep(0.5) 
        
    my_bar.empty() 
    
    if error_tickers:
        st.warning(f"⚠️ Gagal menarik data historis untuk: **{', '.join(error_tickers)}**. Cek apakah kode saham valid.")
            
    if not hasil_analisis:
        return pd.DataFrame()
        
    df = pd.DataFrame(hasil_analisis)
    df['Sinyal_Investasi'] = 'Pantau'
    df['Sinyal_Trading_Pendek'] = 'Pantau/Tahan' 
    
    df.loc[(df['RSI_14'] > 75) | (df['Stoch_K'] > 80), 'Sinyal_Trading_Pendek'] = 'JUAL (Overbought)'
    df.loc[(df['RSI_14'] < 35) | (df['Stoch_K'] < 20), 'Sinyal_Trading_Pendek'] = 'BELI (Oversold)'
    df.loc[(df['MACD_Bullish'] == True) & (df['Di_Atas_MA20'] == True) & (df['RSI_14'] >= 35) & (df['RSI_14'] <= 70), 'Sinyal_Trading_Pendek'] = 'BELI (Momentum)'
    
    if strategi_pilihan == 1:
        df.loc[(df['PBV'] < 1.5) & (df['PE_Ratio'] < 15) & (df['ROE'] > 0.08), 'Sinyal_Investasi'] = 'BELI (Value Saham Murah)'
    elif strategi_pilihan == 2:
        df.loc[(df['PBV'] < 1.5) & (df['ROE'] > 0.1), 'Sinyal_Investasi'] = 'BELI (Fundamental Kuat)'
    elif strategi_pilihan == 3:
        df.loc[(df['Di_Atas_MA20'] == True) & (df['MACD_Bullish'] == True), 'Sinyal_Investasi'] = 'BELI (Momentum Kuat)'
    elif strategi_pilihan == 4:
        df.loc[((df['RSI_14'] < 40) | (df['Stoch_K'] < 20)) & (df['Dekat_BB_Bawah'] == True), 'Sinyal_Investasi'] = 'BELI (Oversold / Diskon)'
    elif strategi_pilihan == 5:
        df.loc[(df['PBV'] < 1.5) & (df['ROE'] > 0.05) & (df['Di_Atas_MA20'] == True) & (df['MACD_Bullish'] == True), 'Sinyal_Investasi'] = 'BELI (Super Kombinasi)'

    return df


# --- PROMPT AI STANDAR ---
def buat_prompt_ai(df, strategi):
    kolom_penting = ['Ticker', 'Harga', 'Harga_Wajar', 'Target_Beli', 'Target_Jual', 'PBV', 'PE_Ratio', 'ROE', 'RSI_14', 'MACD_Bullish', 'Sinyal_Investasi', 'Sinyal_Trading_Pendek']
    df_ai = df[kolom_penting].copy()
    
    df_ai['MACD_Bullish'] = df_ai['MACD_Bullish'].apply(lambda x: "Uptrend" if x else "Downtrend")
    df_ai['ROE'] = df_ai['ROE'].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "-")
    df_ai['PBV'] = df_ai['PBV'].apply(lambda x: f"{x:.2f}x" if pd.notnull(x) else "-")
    df_ai['PE_Ratio'] = df_ai['PE_Ratio'].apply(lambda x: f"{x:.2f}x" if pd.notnull(x) else "-")
    
    data_str = df_ai.to_string(index=False)
    
    prompt = f"""
    Anda adalah seorang Analis Saham Senior dari Wall Street.
    Berikan analisa komprehensif berdasarkan hasil screening data saham Bursa Efek Indonesia (BEI) hari ini.
    
    Strategi Screening: "{strategi}"
    
    Data Saham Terkini:
    {data_str}
    
    INSTRUKSI (Gunakan Markdown rapi):
    1. **Tinjauan Pasar:** Ringkasan singkat kondisi kumpulan saham ini.
    2. **Analisa Fundamental:** Sebutkan 1-2 saham dengan valuasi menarik (murah/ROE tinggi).
    3. **Analisa Teknikal:** Sebutkan saham di area support optimal (dekat Target_Beli/RSI Oversold) atau momentum uptrend kuat.
    4. **Rekomendasi Taktis:** Rekomendasi Beli/Tahan untuk top pick. Wajib sebutkan angka Target Beli & Target Jual dari tabel (Jangan buat harga sendiri).
    5. **Manajemen Risiko:** Peringatan objektif singkat.
    
    Gunakan Bahasa Indonesia yang tajam dan profesional.
    """
    return prompt

# --- FUNGSI AI: 1. GEMINI (UTAMA) ---
def analisa_gemini_ai(df, strategi, api_key):
    try:
        genai.configure(api_key=api_key)
        prompt = buat_prompt_ai(df, strategi)
        
        valid_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        if not valid_models: return False, "API Key valid, tetapi model tidak tersedia."
            
        target_model = valid_models[0]
        for pref in ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-1.0-pro', 'models/gemini-pro']:
            if pref in valid_models:
                target_model = pref
                break
                
        clean_model_name = target_model.replace('models/', '')
        model = genai.GenerativeModel(clean_model_name)
        response = model.generate_content(prompt)
        return True, response.text
    except Exception as e:
        return False, str(e)

# --- FUNGSI AI: 2. GROQ / LLAMA 3 (CADANGAN) ---
def analisa_groq_ai(df, strategi, api_key):
    """Fungsi ini memanggil API Groq (Llama 3) menggunakan requests standar (tanpa perlu install library groq)."""
    try:
        prompt = buat_prompt_ai(df, strategi)
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama3-70b-8192", # Menggunakan model Llama 3 70B yang cerdas dan gratis
            "messages": [
                {"role": "system", "content": "Anda adalah asisten AI keuangan berbahasa Indonesia."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        
        if response.status_code == 200:
            hasil_teks = response.json()['choices'][0]['message']['content']
            return True, hasil_teks
        else:
            return False, f"Error Groq API ({response.status_code}): {response.text}"
            
    except Exception as e:
        return False, str(e)

# --- FUNGSI AI: 3. OPENAI / CHATGPT (ALTERNATIF BARU) ---
def analisa_openai_ai(df, strategi, api_key):
    """Fungsi ini memanggil API OpenAI (ChatGPT) menggunakan requests."""
    try:
        prompt = buat_prompt_ai(df, strategi)
        
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini", # Model ringan, cerdas, dan cepat dari OpenAI
            "messages": [
                {"role": "system", "content": "Anda adalah Analis Saham Profesional berbahasa Indonesia."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        
        if response.status_code == 200:
            hasil_teks = response.json()['choices'][0]['message']['content']
            return True, hasil_teks
        else:
            return False, f"Error OpenAI API ({response.status_code}): {response.text}"
            
    except Exception as e:
        return False, str(e)


# --- FUNGSI AI: 4. TRADISIONAL (DARURAT) ---
def buat_narasi_analisa(df, nama_strategi):
    if df.empty: return ""
    total = len(df)
    saham_beli = list(set(df[df['Sinyal_Investasi'].str.contains('BELI', na=False)]['Ticker'].tolist() + df[df['Sinyal_Trading_Pendek'].str.contains('BELI', na=False)]['Ticker'].tolist()))
    
    narasi = f"### 💡 Ringkasan Eksekutif (Berasaskan Peraturan)\n"
    narasi += f"Dari **{total} saham** yang dipindai menggunakan strategi **{nama_strategi}**, berikut adalah intisari pasarnya:\n\n"
    if saham_beli: narasi += f"- 🎯 **Fokus Utama:** Saham **{', '.join(saham_beli)}** masuk ke dalam zona **BELI**.\n"
    else: narasi += f"- ⏳ **Fokus Utama:** Belum ada saham yang memenuhi kriteria **BELI** yang kuat. Mayoritas fase *Pantau*.\n"
    
    saham_murah = df[(df['PBV'] < 1.5) & (df['ROE'] > 0.10)]['Ticker'].tolist()
    saham_uptrend = df[(df['MACD_Bullish'] == True) & (df['Di_Atas_MA20'] == True)]['Ticker'].tolist()
    saham_oversold = df[(df['RSI_14'] < 35) | (df['Stoch_K'] < 20)]['Ticker'].tolist()
    
    if saham_murah: narasi += f"- 💰 **Valuasi & Fundamental:** Saham **{', '.join(saham_murah)}** terdeteksi sedang salah harga (Murah: PBV < 1.5x) namun sehat (ROE > 10%).\n"
    if saham_uptrend: narasi += f"- 📈 **Momentum Teknikal:** Harga **{', '.join(saham_uptrend)}** sedang dalam tren naik (Uptrend) yang kuat.\n"
    if saham_oversold: narasi += f"- 🛒 **Peluang Rebound:** Saham **{', '.join(saham_oversold)}** sudah jenuh jual (Oversold), berpeluang melantun naik.\n"
    
    return narasi

def buat_ringkasan_aksi(df):
    saham_rekomendasi = df[
        (df['Sinyal_Investasi'].str.contains('BELI', na=False)) |
        (df['Sinyal_Trading_Pendek'].str.contains('BELI', na=False))
    ]
    if saham_rekomendasi.empty: return ""
        
    ringkasan = "### 🎯 Rekomendasi Aksi & Target Harga\n"
    ringkasan += "Daftar saham dengan sinyal **BELI** beserta area harga ideal untuk dieksekusi:\n\n"
    
    ringkasan += "| Kode Saham | Kategori Sinyal Beli | Area Target Beli (Support) | Area Target Jual (Resisten) |\n"
    ringkasan += "| :--- | :--- | :--- | :--- |\n"
    
    for _, row in saham_rekomendasi.iterrows():
        ticker = row['Ticker']
        sinyal_list = []
        if 'BELI' in str(row['Sinyal_Investasi']): sinyal_list.append("Investasi")
        if 'BELI' in str(row['Sinyal_Trading_Pendek']): 
            jenis_trading = str(row['Sinyal_Trading_Pendek']).split('(')[-1].replace(')', '')
            sinyal_list.append(f"Trading ({jenis_trading})")
            
        kategori = " & ".join(sinyal_list)
        t_beli = f"Rp {int(row['Target_Beli']):,}" if pd.notnull(row['Target_Beli']) and not np.isnan(row['Target_Beli']) else "-"
        t_jual = f"Rp {int(row['Target_Jual']):,}" if pd.notnull(row['Target_Jual']) and not np.isnan(row['Target_Jual']) else "-"
        ringkasan += f"| **{ticker}** | {kategori} | **{t_beli}** | **{t_jual}** |\n"
        
    return ringkasan

# --- ANTARMUKA PENGGUNA (UI) ---
st.title("📈 Web App Screener Saham BEI")
st.markdown("Aplikasi web ini menyaring saham berdasarkan analisis fundamental dan sentimen teknikal secara _real-time_, diperkuat dengan **Sistem AI Ganda (Gemini & Llama 3)**.")

# Sidebar Pengaturan
st.sidebar.header("⚙️ Pengaturan Analisis")

st.sidebar.subheader("📋 Kelompok Emiten")
kategori_saham = {
    "🌟 Rekomendasi (Campuran)": "PKPK, BSDE, PWON, CTRA, INDF, ICBP, PGAS, ADRO, EXCL, TLKM, ITMG, BBCA",
    "🏦 Perbankan (Big Bank)": "BBCA, BBRI, BMRI, BBNI, BRIS, ARTO",
    "⚡ Energi & Tambang": "ADRO, ITMG, PTBA, UNTR, PGAS, MEDC, HRUM",
    "🏢 Properti & Konstruksi": "BSDE, PWON, CTRA, SMRA, PTPP, WIKA, WSKT",
    "🍜 Barang Konsumsi": "INDF, ICBP, MYOR, UNVR, KLBF, AMRT",
    "📱 Telekomunikasi & Tech": "TLKM, EXCL, ISAT, GOTO, BUKA, MTEL"
}
pilihan_kategori = st.sidebar.radio("Pilih Daftar Saham:", list(kategori_saham.keys()))
daftar_ticker = st.sidebar.text_area("Daftar Kode Saham (Bisa diedit manual):", kategori_saham[pilihan_kategori])
ticker_pantauan = [t.strip() + ".JK" for t in daftar_ticker.split(",") if t.strip()]

st.sidebar.markdown("---")
opsi_periode = {"3 Bulan": "3mo", "6 Bulan": "6mo", "1 Tahun": "1y"}
pilihan_periode_label = st.sidebar.selectbox("Periode Historis:", list(opsi_periode.keys()), index=1)
periode = opsi_periode[pilihan_periode_label]

opsi_strategi = {
    "1. [Investasi] Value Saham Murah": 1,
    "2. [Investasi] Fundamental Sehat & Kuat": 2,
    "3. [Trading] Momentum & Volume": 3,
    "4. [Trading] Buy The Dip (Diskon)": 4,
    "5. [Swing] Fundamental Murah + Tren Naik": 5
}
pilihan_strategi_label = st.sidebar.radio("Strategi Screening:", list(opsi_strategi.keys()), index=4)
strategi = opsi_strategi[pilihan_strategi_label]

st.sidebar.markdown("---")
st.sidebar.subheader("💵 Pengaturan Kurs (USD/IDR)")
opsi_kurs = st.sidebar.radio("Sumber Kurs:", ["Otomatis (Live API)", "Manual"])
if opsi_kurs == "Otomatis (Live API)":
    kurs_val, kurs_sumber = dapatkan_kurs_usd_idr()
else:
    kurs_val = st.sidebar.number_input("Masukkan Nilai Kurs (Rp):", min_value=10000.0, max_value=25000.0, value=16200.0, step=100.0)
    kurs_sumber = "Manual (Input User)"
st.sidebar.info(f"**Kurs Aktif:** Rp {kurs_val:,.0f}\n\n**Sumber:** {kurs_sumber}")

# --- PENGATURAN API KEY (SECRETS) ---
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Analisis AI Mendalam")
st.sidebar.markdown("Aktifkan fitur ini agar AI menganalisa data saham layaknya analis Wall Street.")
gunakan_ai = st.sidebar.checkbox("Gunakan AI untuk Analisa")

if gunakan_ai:
    pilihan_ai = st.sidebar.radio("Pilih Engine AI:", [
        "Otomatis (Sistem Fallback)", 
        "Google Gemini", 
        "Meta Llama 3 (Groq)", 
        "OpenAI (ChatGPT)"
    ], help="Pilih 'Otomatis' untuk menggunakan Gemini sebagai prioritas utama, dengan opsi lain sebagai cadangan jika limit.")
    
    # Mengambil API Key dari Secrets Streamlit (Jika Ada)
    try: gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
    except: gemini_api_key = ""

    try: groq_api_key = st.secrets.get("GROQ_API_KEY", "")
    except: groq_api_key = ""
    
    try: openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
    except: openai_api_key = ""

    # Memungkinkan input manual di UI jika secrets kosong sesuai pilihan AI
    if pilihan_ai in ["Otomatis (Sistem Fallback)", "Google Gemini"] and not gemini_api_key:
        gemini_api_key = st.sidebar.text_input("Gemini API Key:", type="password")
        
    if pilihan_ai in ["Otomatis (Sistem Fallback)", "Meta Llama 3 (Groq)"] and not groq_api_key:
        groq_api_key = st.sidebar.text_input("Groq API Key:", type="password", help="Dapatkan di console.groq.com")
        
    if pilihan_ai in ["Otomatis (Sistem Fallback)", "OpenAI (ChatGPT)"] and not openai_api_key:
        openai_api_key = st.sidebar.text_input("OpenAI API Key:", type="password", help="Dapatkan di platform.openai.com")


# Tombol Eksekusi
if st.sidebar.button("Jalankan Pemindaian 🚀", type="primary"):
    df_hasil = saring_saham_pilihan(ticker_pantauan, periode, strategi, kurs_val)
    
    if not df_hasil.empty:
        df_hasil = df_hasil.sort_values(by=['Sinyal_Investasi', 'Sinyal_Trading_Pendek'], ascending=[False, False])
        
        # --- LOGIKA SISTEM AI BERLAPIS (FALLBACK SYSTEM) ---
        if gunakan_ai:
            with st.spinner(f"🤖 AI sedang membedah data menggunakan {pilihan_ai if pilihan_ai != 'Otomatis (Sistem Fallback)' else 'Sistem Otomatis'}..."):
                narasi_komprehensif = ""
                
                # Fungsi Helper Eksekutor AI
                def jalankan_gemini():
                    if gemini_api_key:
                        sukses, hasil = analisa_gemini_ai(df_hasil, pilihan_strategi_label, gemini_api_key)
                        if sukses: return f"### 🤖 Analisa Eksekutif AI (Sumber: Google Gemini)\n\n{hasil}"
                        st.warning(f"⚠️ **Gangguan Gemini:** {hasil}")
                    return ""
                    
                def jalankan_groq():
                    if groq_api_key:
                        sukses, hasil = analisa_groq_ai(df_hasil, pilihan_strategi_label, groq_api_key)
                        if sukses: return f"### 🤖 Analisa Eksekutif AI (Sumber: Meta Llama 3 via Groq)\n\n{hasil}"
                        st.warning(f"⚠️ **Gangguan Groq AI:** {hasil}")
                    return ""
                    
                def jalankan_openai():
                    if openai_api_key:
                        sukses, hasil = analisa_openai_ai(df_hasil, pilihan_strategi_label, openai_api_key)
                        if sukses: return f"### 🤖 Analisa Eksekutif AI (Sumber: OpenAI ChatGPT)\n\n{hasil}"
                        st.warning(f"⚠️ **Gangguan OpenAI:** {hasil}")
                    return ""

                # Eksekusi berdasarkan pilihan Radio Button
                if pilihan_ai == "Google Gemini":
                    narasi_komprehensif = jalankan_gemini()
                elif pilihan_ai == "Meta Llama 3 (Groq)":
                    narasi_komprehensif = jalankan_groq()
                elif pilihan_ai == "OpenAI (ChatGPT)":
                    narasi_komprehensif = jalankan_openai()
                elif pilihan_ai == "Otomatis (Sistem Fallback)":
                    # Lapis 1: Gemini
                    narasi_komprehensif = jalankan_gemini()
                    # Lapis 2: Groq
                    if not narasi_komprehensif:
                        st.warning("🔄 Beralih ke Llama 3 (Groq) sebagai cadangan pertama...")
                        narasi_komprehensif = jalankan_groq()
                    # Lapis 3: OpenAI
                    if not narasi_komprehensif:
                        st.warning("🔄 Beralih ke ChatGPT (OpenAI) sebagai cadangan kedua...")
                        narasi_komprehensif = jalankan_openai()

                # Lapis Terakhir: Analisa Tradisional (Jika semua AI gagal atau API Key tidak ada)
                if not narasi_komprehensif:
                    st.warning("⚠️ Analisa AI gagal atau kunci API tidak lengkap. Menjalankan Analisa Tradisional.")
                    narasi_komprehensif = buat_narasi_analisa(df_hasil, pilihan_strategi_label)
        else:
            narasi_komprehensif = buat_narasi_analisa(df_hasil, pilihan_strategi_label)
            
        ringkasan_aksi = buat_ringkasan_aksi(df_hasil)
        
        # Formatting Tampilan Utama Tabel HTML
        tampil = df_hasil.copy()
        for col in ['Harga', 'Harga_Wajar', 'Target_Beli', 'Target_Jual']:
            tampil[col] = tampil[col].apply(render_currency_html)
            
        tampil['PBV'] = tampil['PBV'].apply(lambda x: f"{x:.2f}x" if pd.notnull(x) else '<div class="text-center">-</div>')
        tampil['ROE'] = tampil['ROE'].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else '<div class="text-center">-</div>')
        tampil['RSI_14'] = tampil['RSI_14'].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else '<div class="text-center">-</div>')
        tampil['MACD'] = tampil['MACD_Bullish'].apply(lambda x: "Bull 🟢" if x else "Bear 🔴")
        
        kolom = ['Ticker', 'Harga', 'Harga_Wajar', 'Target_Beli', 'Target_Jual', 'PBV', 'ROE', 'RSI_14', 'MACD', 'Sinyal_Investasi', 'Sinyal_Trading_Pendek']
        
        st.success("✅ Pemindaian Selesai!")
        
        # Output UI
        st.info(narasi_komprehensif)
        
        if ringkasan_aksi:
            st.success(ringkasan_aksi)
            
        st.divider()
        st.markdown(tampil[kolom].to_html(classes="custom-table", escape=False, index=False), unsafe_allow_html=True)
        
        st.info("""
        **Catatan Target Harga & Integritas Data:**
        * **Harga Wajar:** Menggunakan rumus *Graham Number*. Jika kosong (`-`), berarti EPS perusahaan negatif (rugi) atau datanya belum diperbarui.
        * **Target Beli & Jual:** Dihitung dari area Support & Resistance teknikal dinamis.
        * **Sistem Koreksi Kurs Aktif:** Aplikasi telah mendeteksi dan mengoreksi laporan keuangan emiten tambang (berbasis Dolar AS) ke Rupiah secara otomatis.
        """)
    else:
        st.error("Gagal mengambil data atau tidak ada saham yang valid.")
