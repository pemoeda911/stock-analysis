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
    """Mengambil kurs USD ke IDR terkini secara berlapis (Yahoo -> Public API -> Statis).
       Mengembalikan tuple: (nilai_kurs, sumber_data)
    """
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
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive"
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
        ev_ebitda = info.get('enterpriseToEbitda')
        eps_growth = info.get('earningsQuarterlyGrowth') or info.get('earningsGrowth')
        
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
            'DER': der,
            'Div_Yield': div_yield,
            'EPS_Growth': eps_growth,
            'RSI_14': rsi_terakhir if pd.notnull(rsi_terakhir) else 50.0,
            'Di_Atas_MA20': harga_terakhir > ma20_terakhir if pd.notnull(ma20_terakhir) else False,
            'Lonjakan_Volume': rata_volume_5hari > (rata_volume_20hari * 1.1) if pd.notnull(rata_volume_20hari) else False,
            'MACD_Bullish': macd_bullish,
            'Dekat_BB_Bawah': dekat_bb_bawah,
            'Stoch_K': stoch_k_terakhir
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
        df.loc[(df['Div_Yield'] > 0.05) & ((df['DER'].isna()) | (df['DER'] < 2.0)), 'Sinyal_Investasi'] = 'BELI (Dividen Tinggi)'
    elif strategi_pilihan == 3:
        df.loc[(df['Di_Atas_MA20'] == True) & (df['Lonjakan_Volume'] == True) & (df['MACD_Bullish'] == True), 'Sinyal_Investasi'] = 'BELI (Momentum Kuat)'
    elif strategi_pilihan == 4:
        df.loc[((df['RSI_14'] < 40) | (df['Stoch_K'] < 20)) & (df['Dekat_BB_Bawah'] == True), 'Sinyal_Investasi'] = 'BELI (Oversold / Diskon)'
    elif strategi_pilihan == 5:
        df.loc[(df['PBV'] < 1.5) & (df['ROE'] > 0.05) & (df['Di_Atas_MA20'] == True) & (df['MACD_Bullish'] == True), 'Sinyal_Investasi'] = 'BELI (Super Kombinasi)'

    return df

# --- FUNGSI ANALISA MENDALAM DENGAN GEMINI AI ---
def analisa_gemini_ai(df, strategi, api_key):
    try:
        genai.configure(api_key=api_key)
        # Menggunakan model Gemini 1.5 Flash untuk analisis cepat dan cerdas
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Konversi dataframe penting ke format string untuk dibaca AI
        kolom_penting = ['Ticker', 'Harga', 'PBV', 'PE_Ratio', 'ROE', 'RSI_14', 'MACD_Bullish', 'Sinyal_Investasi', 'Sinyal_Trading_Pendek']
        df_ai = df[kolom_penting].copy()
        
        # Mengganti to_markdown() dengan to_string() untuk menghindari error import 'tabulate'
        data_str = df_ai.to_string(index=False)
        
        prompt = f"""
        Anda adalah seorang Analis Saham Profesional dan Fund Manager berpengalaman dari Wall Street.
        Tugas Anda adalah menganalisa hasil screening saham di Bursa Efek Indonesia (BEI) berikut ini.
        
        Strategi Screening yang dipilih user: "{strategi}"
        
        Tabel Data Saham Terkini:
        {data_str}
        
        Berikan ringkasan eksekutif dan analisa mendalam yang mencakup:
        1. **Highlight Fundamental:** Dari data di atas, saham mana yang memiliki valuasi paling menarik (undervalued berdasarkan PBV/PE) atau fundamental terkuat (berdasarkan ROE)?
        2. **Kondisi Teknikal:** Saham mana yang sedang memiliki momentum teknikal yang kuat (MACD Bullish, Sinyal Trading Beli), atau sedang berada di area pantulan murah (Oversold berdasarkan RSI)?
        3. **Rekomendasi Taktis:** Kesimpulan aksi apa yang sebaiknya diambil. Fokus pada 1-3 saham terbaik untuk dieksekusi berdasarkan kombinasi fundamental dan teknikal.
        
        Gunakan gaya bahasa Indonesia yang profesional, tajam, dan mudah dipahami. Jangan sekadar mengulang isi tabel, berikan *insight* dan interpretasi data yang bernilai bagi investor. Format menggunakan Markdown.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ **Gagal menghasilkan analisa AI:** {str(e)}\n\nPastikan API Key valid dan koneksi internet stabil."

# --- FUNGSI PEMBUAT NARASI OTOMATIS (FALLBACK TRADISIONAL) ---
def buat_narasi_analisa(df, nama_strategi):
    if df.empty:
        return ""
    
    total = len(df)
    saham_beli_invest = df[df['Sinyal_Investasi'].str.contains('BELI', na=False)]['Ticker'].tolist()
    saham_beli_trading = df[df['Sinyal_Trading_Pendek'].str.contains('BELI', na=False)]['Ticker'].tolist()
    
    rekomendasi_beli = list(set(saham_beli_invest + saham_beli_trading))
    saham_murah = df[(df['PBV'] < 1.5) & (df['ROE'] > 0.10)]['Ticker'].tolist()
    saham_uptrend = df[(df['MACD_Bullish'] == True) & (df['Di_Atas_MA20'] == True)]['Ticker'].tolist()
    saham_oversold = df[(df['RSI_14'] < 35) | (df['Stoch_K'] < 20)]['Ticker'].tolist()
    saham_overbought = df[(df['RSI_14'] > 75) | (df['Stoch_K'] > 80)]['Ticker'].tolist()
    
    narasi = f"### 💡 Ringkasan Eksekutif (Berbasis Aturan)\n"
    narasi += f"Dari **{total} saham** yang dipindai menggunakan strategi **{nama_strategi}**, berikut adalah intisari pasarnya:\n\n"
    
    if rekomendasi_beli:
        narasi += f"- 🎯 **Fokus Utama:** Saham **{', '.join(rekomendasi_beli)}** masuk ke dalam zona **BELI**.\n"
    else:
        narasi += f"- ⏳ **Fokus Utama:** Belum ada saham yang memenuhi kriteria **BELI** yang kuat. Mayoritas fase *Pantau*.\n"
        
    narasi += "- 💰 **Valuasi & Fundamental:** "
    if saham_murah:
        narasi += f"Saham **{', '.join(saham_murah)}** terdeteksi sedang 'salah harga' (Sangat Murah: PBV < 1.5x) namun sehat (ROE > 10%).\n"
    else:
        narasi += f"Belum ditemukan saham dengan valuasi sangat terdiskon sekaligus profitabilitas kuat.\n"
        
    narasi += "- 📈 **Momentum Teknikal:** "
    if saham_uptrend:
        narasi += f"Harga **{', '.join(saham_uptrend)}** sedang dalam tren naik (*Uptrend*) yang solid.\n"
    else:
        narasi += f"Mayoritas saham sedang lesu atau berkonsolidasi.\n"
        
    if saham_oversold:
        narasi += f"- 🛒 **Peluang Rebound:** Saham **{', '.join(saham_oversold)}** sudah jenuh jual (*Oversold*), berpeluang mantul naik.\n"
    if saham_overbought:
        narasi += f"- ⚠️ **Rawan Koreksi:** Saham **{', '.join(saham_overbought)}** sudah jenuh beli (*Overbought*). Hati-hati koreksi.\n"
        
    return narasi

def buat_ringkasan_aksi(df):
    saham_rekomendasi = df[
        (df['Sinyal_Investasi'].str.contains('BELI', na=False)) |
        (df['Sinyal_Trading_Pendek'].str.contains('BELI', na=False))
    ]
    
    if saham_rekomendasi.empty:
        return ""
        
    ringkasan = "### 🎯 Rekomendasi Aksi & Target Harga\n"
    ringkasan += "Daftar saham dengan sinyal **BELI** beserta area harga ideal untuk dieksekusi:\n\n"
    
    ringkasan += "| Kode Saham | Kategori Sinyal Beli | Area Target Beli (Support) | Area Target Jual (Resisten) |\n"
    ringkasan += "| :--- | :--- | :--- | :--- |\n"
    
    for _, row in saham_rekomendasi.iterrows():
        ticker = row['Ticker']
        sinyal_list = []
        if 'BELI' in str(row['Sinyal_Investasi']):
            sinyal_list.append("Investasi")
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

# Sidebar untuk Input
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

daftar_ticker = st.sidebar.text_area(
    "Daftar Kode Saham (Bisa diedit manual):", 
    kategori_saham[pilihan_kategori]
)
ticker_pantauan = [t.strip() + ".JK" for t in daftar_ticker.split(",") if t.strip()]

st.sidebar.markdown("---")

opsi_periode = {"3 Bulan": "3mo", "6 Bulan": "6mo", "1 Tahun": "1y"}
pilihan_periode_label = st.sidebar.selectbox("Periode Historis:", list(opsi_periode.keys()), index=1)
periode = opsi_periode[pilihan_periode_label]

opsi_strategi = {
    "1. [Investasi] Value Saham Murah": 1,
    "2. [Investasi] High Dividend Yield": 2,
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

# --- INTEGRASI GOOGLE GEMINI AI (BARU) ---
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Analisis AI Mendalam")
st.sidebar.markdown("Aktifkan fitur ini agar AI menganalisa data saham layaknya analis Wall Street.")
gunakan_ai = st.sidebar.checkbox("Gunakan Google Gemini AI")

# HARDCODE API KEY DI SINI
# PENTING: Jangan unggah script ini ke GitHub Publik jika API Key terisi!
api_key_input = "MASUKKAN_API_KEY_ANDA_DI_SINI"

# Tombol Eksekusi
if st.sidebar.button("Jalankan Pemindaian 🚀", type="primary"):
    
    df_hasil = saring_saham_pilihan(ticker_pantauan, periode, strategi, kurs_val)
    
    if not df_hasil.empty:
        df_hasil = df_hasil.sort_values(by=['Sinyal_Investasi', 'Sinyal_Trading_Pendek'], ascending=[False, False])
        
        # Eksekusi Analisa AI vs Tradisional
        if gunakan_ai and api_key_input and api_key_input != "MASUKKAN_API_KEY_ANDA_DI_SINI":
            with st.spinner("🤖 AI sedang membedah data dan menyusun analisa mendalam..."):
                narasi_komprehensif = analisa_gemini_ai(df_hasil, pilihan_strategi_label, api_key_input)
                narasi_komprehensif = f"### 🤖 Analisa Eksekutif AI (Gemini)\n\n{narasi_komprehensif}"
        else:
            if gunakan_ai and (not api_key_input or api_key_input == "MASUKKAN_API_KEY_ANDA_DI_SINI"):
                st.warning("Silakan ganti placeholder 'MASUKKAN_API_KEY_ANDA_DI_SINI' di dalam script dengan API Key yang valid.")
            narasi_komprehensif = buat_narasi_analisa(df_hasil, pilihan_strategi_label)
            
        ringkasan_aksi = buat_ringkasan_aksi(df_hasil)
        
        tampil = df_hasil.copy()
        
        tampil['PBV'] = tampil['PBV'].apply(lambda x: f"{x:.2f}x" if pd.notnull(x) else "-")
        tampil['ROE'] = tampil['ROE'].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "-")
        tampil['RSI_14'] = tampil['RSI_14'].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else "-")
        tampil['MACD'] = tampil['MACD_Bullish'].apply(lambda x: "Bull 🟢" if x else "Bear 🔴")
        tampil['Harga'] = tampil['Harga'].apply(lambda x: f"Rp{int(x):,}")
        tampil['Harga_Wajar'] = tampil['Harga_Wajar'].apply(lambda x: f"Rp{int(x):,}" if pd.notnull(x) and not np.isnan(x) else "-")
        tampil['Target_Beli'] = tampil['Target_Beli'].apply(lambda x: f"Rp{int(x):,}" if pd.notnull(x) and not np.isnan(x) else "-")
        tampil['Target_Jual'] = tampil['Target_Jual'].apply(lambda x: f"Rp{int(x):,}" if pd.notnull(x) and not np.isnan(x) else "-")
        
        kolom = ['Ticker', 'Harga', 'Harga_Wajar', 'Target_Beli', 'Target_Jual', 'PBV', 'ROE', 'RSI_14', 'MACD', 'Sinyal_Investasi', 'Sinyal_Trading_Pendek']
        
        st.success("✅ Pemindaian Selesai!")
        
        st.info(narasi_komprehensif)
        
        if ringkasan_aksi:
            st.success(ringkasan_aksi)
            
        st.divider()
        st.dataframe(tampil[kolom], use_container_width=True, hide_index=True)
        
        st.info("""
        **Catatan Target Harga & Integritas Data:**
        * **Harga Wajar:** Menggunakan rumus *Graham Number*. Jika kosong (`-`), berarti EPS perusahaan negatif (rugi) atau datanya belum diperbarui oleh Yahoo Finance.
        * **Target Beli & Jual:** Dihitung dari area Support & Resistance teknikal.
        * **Sel Kosong (`-`):** Terjadi karena keterbatasan database gratis Yahoo Finance untuk beberapa saham lapis menengah/kecil di BEI (IDX).
        * **Sistem Koreksi Kurs Aktif:** Aplikasi telah mendeteksi dan mengoreksi laporan keuangan emiten tambang (berbasis Dolar AS) ke Rupiah secara otomatis.
        """)
    else:
        st.error("Gagal mengambil data atau tidak ada saham yang valid.")
