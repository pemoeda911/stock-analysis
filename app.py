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

# --- FUNGSI ANALISA MENDALAM DENGAN GEMINI AI ---
def analisa_gemini_ai(df, strategi, api_key):
    try:
        genai.configure(api_key=api_key)
        
        # 1. Siapkan DataFrame khusus AI (bersihkan format agar AI mudah membaca)
        kolom_penting = ['Ticker', 'Harga', 'Harga_Wajar', 'Target_Beli', 'Target_Jual', 'PBV', 'PE_Ratio', 'ROE', 'RSI_14', 'MACD_Bullish', 'Sinyal_Investasi', 'Sinyal_Trading_Pendek']
        df_ai = df[kolom_penting].copy()
        
        # Format angka menjadi string yang jelas agar AI tidak salah menerjemahkan desimal
        df_ai['MACD_Bullish'] = df_ai['MACD_Bullish'].apply(lambda x: "Uptrend (Bullish)" if x else "Downtrend (Bearish)")
        df_ai['ROE'] = df_ai['ROE'].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "Data Kosong")
        df_ai['PBV'] = df_ai['PBV'].apply(lambda x: f"{x:.2f}x" if pd.notnull(x) else "Data Kosong")
        df_ai['PE_Ratio'] = df_ai['PE_Ratio'].apply(lambda x: f"{x:.2f}x" if pd.notnull(x) else "Data Kosong")
        
        data_str = df_ai.to_string(index=False)
        
        prompt = f"""
        Anda adalah seorang Analis Saham Senior dan Fund Manager Institusional dari Wall Street.
        Tugas Anda adalah memberikan analisa komprehensif, teliti, dan akurat berdasarkan hasil screening data pasar saham Bursa Efek Indonesia (BEI) hari ini.
        
        Strategi Screening yang digunakan pengguna: "{strategi}"
        
        Data Saham Terkini (Sudah disaring):
        {data_str}
        
        INSTRUKSI ANALISA (Jawab menggunakan format Markdown yang rapi):
        1. **Tinjauan Pasar (Overview):** Berikan pandangan singkat tentang kondisi kumpulan saham ini berdasarkan metrik yang ada (Apakah mayoritas mahal, murah, uptrend, atau oversold?).
        2. **Analisa Fundamental (Valuasi & Kualitas):** Identifikasi 1 atau 2 saham dengan valuasi paling menarik (misal: diskon dalam terhadap Harga_Wajar, PBV rendah, PE rasional) namun memiliki profitabilitas (ROE) yang baik.
        3. **Analisa Teknikal (Momentum & Timing):** Identifikasi saham yang berada di area pantulan optimal (dekat Target_Beli, RSI Oversold) atau yang memiliki dorongan tren kuat (MACD Bullish).
        4. **Rekomendasi Taktis & Eksekusi:** Berikan rekomendasi spesifik (Beli/Tahan/Jual) untuk saham-saham top pick. Wajib sebutkan area Target Beli (Support) dan Target Jual (Resisten) sesuai data angka di tabel (JANGAN mengarang harga target sendiri).
        5. **Manajemen Risiko:** Berikan satu paragraf peringatan objektif tentang risiko dari pilihan saham tersebut (misal: "meskipun PBV rendah, tren masih bearish").
        
        Gunakan gaya bahasa Indonesia yang tajam, meyakinkan, namun tetap objektif. Jangan sekadar membaca ulang tabel, tapi berikan *insight* korelasi antar datanya.
        """
        
        # 2. Sistem Fallback Otomatis (Mengatasi Error 404 Model Not Found)
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
        except Exception as e_model:
            # Jika gemini-1.5-flash tidak ditemukan di versi SDK server, turunkan ke gemini-pro
            if "404" in str(e_model) or "not found" in str(e_model).lower():
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(prompt)
            else:
                raise e_model # Lempar error jika masalahnya bukan karena versi model

        return response.text
    except Exception as e:
        return f"⚠️ **Gagal menghasilkan analisa AI:** {str(e)}\n\nPastikan API Key valid, dan tidak ada masalah koneksi pada server Streamlit."

# --- FUNGSI PEMBUAT NARASI OTOMATIS (FALLBACK TRADISIONAL) ---
def buat_narasi_analisa(df, nama_strategi):
    if df.empty: return ""
    total = len(df)
    saham_beli = list(set(df[df['Sinyal_Investasi'].str.contains('BELI', na=False)]['Ticker'].tolist() + df[df['Sinyal_Trading_Pendek'].str.contains('BELI', na=False)]['Ticker'].tolist()))
    
    narasi = f"### 💡 Ringkasan Eksekutif (Berbasis Aturan)\n"
    narasi += f"Dari **{total} saham** yang dipindai menggunakan strategi **{nama_strategi}**, berikut adalah intisari pasarnya:\n\n"
    if saham_beli: narasi += f"- 🎯 **Fokus Utama:** Saham **{', '.join(saham_beli)}** masuk ke dalam zona **BELI**.\n"
    else: narasi += f"- ⏳ **Fokus Utama:** Belum ada saham yang memenuhi kriteria **BELI** yang kuat. Mayoritas fase *Pantau*.\n"
    return narasi

def buat_ringkasan_aksi(df):
    saham_rekomendasi = df[
        (df['Sinyal_Investasi'].str.contains('BELI', na=False)) |
        (df['Sinyal_Trading_Pendek'].str.contains('BELI', na=False))
    ]
    if saham_rekomendasi.empty: return ""
        
    ringkasan = "### 🎯 Rekomendasi Aksi & Target Harga\n"
    ringkasan += "Daftar saham dengan sinyal **BELI** beserta area harga ideal untuk dieksekusi:\n\n"
    
    # Render tabel markdown biasa khusus untuk ringkasan aksi agar mudah dibaca cepat
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
st.markdown("Aplikasi web ini menyaring saham berdasarkan analisis fundamental dan sentimen teknikal secara _real-time_, diperkuat dengan **AI Analis Profesional**.")

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

# --- INTEGRASI GOOGLE GEMINI AI (AMAN MENGGUNAKAN SECRETS) ---
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Analisis AI Mendalam")
st.sidebar.markdown("Aktifkan fitur ini agar AI menganalisa data saham layaknya analis Wall Street.")
gunakan_ai = st.sidebar.checkbox("Gunakan Google Gemini AI")

# Mengambil API Key secara aman dari sistem Secrets Streamlit
try:
    api_key_input = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    api_key_input = ""

# Tombol Eksekusi
if st.sidebar.button("Jalankan Pemindaian 🚀", type="primary"):
    df_hasil = saring_saham_pilihan(ticker_pantauan, periode, strategi, kurs_val)
    
    if not df_hasil.empty:
        df_hasil = df_hasil.sort_values(by=['Sinyal_Investasi', 'Sinyal_Trading_Pendek'], ascending=[False, False])
        
        # Eksekusi Analisa
        if gunakan_ai and api_key_input:
            with st.spinner("🤖 AI sedang membedah data dan menyusun analisa komprehensif..."):
                narasi_komprehensif = analisa_gemini_ai(df_hasil, pilihan_strategi_label, api_key_input)
                narasi_komprehensif = f"### 🤖 Analisa Eksekutif AI (Gemini)\n\n{narasi_komprehensif}"
        else:
            if gunakan_ai and not api_key_input:
                st.warning("⚠️ Fitur AI tidak dapat berjalan karena API Key tidak ditemukan di dalam rahasia (Secrets) server.")
            narasi_komprehensif = buat_narasi_analisa(df_hasil, pilihan_strategi_label)
            
        ringkasan_aksi = buat_ringkasan_aksi(df_hasil)
        
        # Formatting Tampilan Utama Tabel HTML
        tampil = df_hasil.copy()
        
        # Kolom Mata Uang
        for col in ['Harga', 'Harga_Wajar', 'Target_Beli', 'Target_Jual']:
            tampil[col] = tampil[col].apply(render_currency_html)
            
        # Kolom Desimal & Kategori
        tampil['PBV'] = tampil['PBV'].apply(lambda x: f"{x:.2f}x" if pd.notnull(x) else '<div class="text-center">-</div>')
        tampil['ROE'] = tampil['ROE'].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else '<div class="text-center">-</div>')
        tampil['RSI_14'] = tampil['RSI_14'].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else '<div class="text-center">-</div>')
        tampil['MACD'] = tampil['MACD_Bullish'].apply(lambda x: "Bull 🟢" if x else "Bear 🔴")
        
        # Kolom yang akan dirender (hanya menampilkan data utama)
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
