import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time

# --- KONFIGURASI HALAMAN WEB ---
st.set_page_config(page_title="Screener Saham BEI", page_icon="📈", layout="wide")

# --- FUNGSI PENGAMBILAN DATA (DENGAN CACHE) ---
@st.cache_data(ttl=3600, show_spinner=False)
def dapatkan_kurs_usd_idr():
    """Mengambil kurs USD ke IDR terkini secara berlapis (Yahoo -> Public API -> Statis).
       Mengembalikan tuple: (nilai_kurs, sumber_data)
    """
    # 1. Coba via Yahoo Finance (Sistem Utama)
    try:
        ticker = yf.Ticker("IDR=X")
        hist = ticker.history(period="1d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1]), "Yahoo Finance"
    except Exception:
        pass
    
    # 2. Cadangan Dinamis: via Public Exchange Rate API (Akurat & Terpercaya)
    try:
        response = requests.get("https://open.er-api.com/v6/latest/USD", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'rates' in data and 'IDR' in data['rates']:
                return float(data['rates']['IDR']), "ExchangeRate-API"
    except Exception:
        pass

    # 3. Jaring Pengaman Terakhir (Hanya dipakai jika server benar-benar tidak bisa akses internet luar)
    return 16200.0, "Sistem Cadangan (Statis)"

# Menggunakan cache agar web tidak lemot saat pengguna mengganti filter
@st.cache_data(ttl=3600, show_spinner=False)
def dapatkan_data_saham(ticker_symbol, periode="6mo", kurs_usd_aktif=16200.0):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = {}
        
        # 1. Coba ambil fundamental data dengan retry (3 kali percobaan)
        for attempt in range(3):
            try:
                info = ticker.info
                # Cek apakah info valid dan bukan sekadar dictionary kosong
                if info and 'symbol' in info: 
                    break
            except Exception:
                time.sleep(1) # Jeda sebelum mencoba lagi
        
        # 2. Jika cara bawaan gagal (sering terjadi di Streamlit Cloud), gunakan Custom Session
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
                info = {} # Menyerah dan gunakan data kosong jika kedua cara tetap diblokir
        
        # 3. Ekstraksi Data Historis Harga (Dipindahkan ke atas agar bisa dipakai menghitung rasio)
        hist = ticker.history(period=periode)
        if hist.empty or len(hist) < 30:
            return None # Skip jika data harga kurang untuk dihitung MA/RSI
        
        harga_terakhir = hist['Close'].iloc[-1]
        
        # --- 4. SISTEM KOREKSI KURS INDEPENDEN (FIX ANOMALI YAHOO) ---
        # Cek mata uang fundamental. Jika USD, ubah EPS dan BVPS ke Rupiah.
        mata_uang = info.get('financialCurrency', 'IDR')
        kurs_usd = kurs_usd_aktif if mata_uang == 'USD' else 1.0
        
        eps = info.get('trailingEps')
        bvps = info.get('bookValue')
        
        if eps is not None: eps = eps * kurs_usd
        if bvps is not None: bvps = bvps * kurs_usd
        
        # Hitung Ulang Valuasi (Mengabaikan PBV & PE bawaan Yahoo yang sering kacau pada emiten Dolar)
        pe_ratio = (harga_terakhir / eps) if (eps and eps > 0) else np.nan
        pbv = (harga_terakhir / bvps) if (bvps and bvps > 0) else np.nan
        
        # Fundamental lainnya (Rasio persentase biasanya aman dari anomali kurs)
        roe = info.get('returnOnEquity')
        der = info.get('debtToEquity') 
        div_yield = info.get('dividendYield') or info.get('trailingAnnualDividendYield')
        ev_ebitda = info.get('enterpriseToEbitda')
        eps_growth = info.get('earningsQuarterlyGrowth') or info.get('earningsGrowth')
        
        if der is not None: der = der / 100.0
            
        harga_wajar_graham = np.nan
        # Rumus Graham (Sekarang sudah kebal terhadap masalah mata uang)
        if eps and bvps and eps > 0 and bvps > 0:
            harga_wajar_graham = np.sqrt(22.5 * eps * bvps)
            
        # Ekstraksi data teknikal lanjutan
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
        # Jika terjadi error fatal di luar dugaan
        print(f"Error fetching {ticker_symbol}: {str(e)}")
        return None

def saring_saham_pilihan(daftar_saham, periode, strategi_pilihan, kurs_usd_aktif):
    hasil_analisis = []
    error_tickers = []
    
    # Progress bar di UI web
    progress_text = "Memindai saham..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, tkr in enumerate(daftar_saham):
        data = dapatkan_data_saham(tkr, periode, kurs_usd_aktif)
        if data:
            hasil_analisis.append(data)
        else:
            error_tickers.append(tkr.replace('.JK', ''))
            
        # Update progress bar
        my_bar.progress((i + 1) / len(daftar_saham), text=f"Memindai {tkr}...")
        # Jeda 0.5 detik per iterasi untuk meminimalisasi risiko diblokir karena rate-limit
        time.sleep(0.5) 
        
    my_bar.empty() # Hilangkan bar setelah selesai
    
    if error_tickers:
        st.warning(f"⚠️ Gagal menarik data historis untuk: **{', '.join(error_tickers)}**. Cek apakah kode saham valid.")
            
    if not hasil_analisis:
        return pd.DataFrame()
        
    df = pd.DataFrame(hasil_analisis)
    df['Sinyal_Investasi'] = 'Pantau'
    df['Sinyal_Trading_Pendek'] = 'Pantau/Tahan' 
    
    # Logika Trading (Aman dari NaN)
    df.loc[(df['RSI_14'] > 75) | (df['Stoch_K'] > 80), 'Sinyal_Trading_Pendek'] = 'JUAL (Overbought)'
    df.loc[(df['RSI_14'] < 35) | (df['Stoch_K'] < 20), 'Sinyal_Trading_Pendek'] = 'BELI (Oversold)'
    df.loc[(df['MACD_Bullish'] == True) & (df['Di_Atas_MA20'] == True) & (df['RSI_14'] >= 35) & (df['RSI_14'] <= 70), 'Sinyal_Trading_Pendek'] = 'BELI (Momentum)'
    
    # Logika Investasi
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

# --- FUNGSI PEMBUAT NARASI ANALISA KOMPREHENSIF ---
def buat_narasi_analisa(df, nama_strategi):
    if df.empty:
        return ""
    
    total = len(df)
    saham_beli_invest = df[df['Sinyal_Investasi'].str.contains('BELI', na=False)]['Ticker'].tolist()
    saham_beli_trading = df[df['Sinyal_Trading_Pendek'].str.contains('BELI', na=False)]['Ticker'].tolist()
    
    # Gabungkan sinyal beli yang unik
    rekomendasi_beli = list(set(saham_beli_invest + saham_beli_trading))
    
    # Analisis Fundamental (Aman dari NaN)
    saham_murah = df[(df['PBV'] < 1.5) & (df['ROE'] > 0.10)]['Ticker'].tolist()
    
    # Analisis Teknikal
    saham_uptrend = df[(df['MACD_Bullish'] == True) & (df['Di_Atas_MA20'] == True)]['Ticker'].tolist()
    saham_oversold = df[(df['RSI_14'] < 35) | (df['Stoch_K'] < 20)]['Ticker'].tolist()
    saham_overbought = df[(df['RSI_14'] > 75) | (df['Stoch_K'] > 80)]['Ticker'].tolist()
    
    narasi = f"### 💡 Ringkasan Eksekutif & Analisa\n"
    narasi += f"Dari **{total} saham** yang dipindai menggunakan strategi **{nama_strategi}**, berikut adalah intisari pergerakan pasarnya saat ini:\n\n"
    
    # 1. Rekomendasi Keseluruhan
    if rekomendasi_beli:
        narasi += f"- 🎯 **Fokus Utama:** Berdasarkan filter yang Anda pilih, saham **{', '.join(rekomendasi_beli)}** masuk ke dalam zona **BELI** dan layak mendapat perhatian khusus untuk dieksekusi.\n"
    else:
        narasi += f"- ⏳ **Fokus Utama:** Belum ada saham yang memenuhi kriteria **BELI** yang kuat untuk strategi ini. Mayoritas masih berada di fase *Pantau* atau *Hold*.\n"
        
    # 2. Narasi Fundamental (Value & Profitability)
    narasi += "- 💰 **Valuasi & Fundamental:** "
    if saham_murah:
        narasi += f"Saham **{', '.join(saham_murah)}** terdeteksi sedang 'salah harga' (Sangat Murah: PBV < 1.5x) namun perusahaannya sehat dengan kemampuan mencetak laba yang tinggi (ROE > 10%).\n"
    else:
        narasi += f"Belum ditemukan saham yang valuasinya sangat terdiskon (PBV < 1.5x) sekaligus memiliki fundamental profitabilitas kuat (ROE > 10%) pada daftar pantauan ini.\n"
        
    # 3. Narasi Teknikal (Momentum)
    narasi += "- 📈 **Momentum Teknikal:** "
    if saham_uptrend:
        narasi += f"Pergerakan harga **{', '.join(saham_uptrend)}** sedang dalam tren naik (*Uptrend*) yang solid, ditandai dengan formasi MACD yang Bullish dan harga bertahan di atas rata-rata 20 harinya.\n"
    else:
        narasi += f"Secara teknikal, mayoritas saham sedang lesu atau berkonsolidasi. Belum ada yang menunjukkan dorongan tren naik yang signifikan.\n"
        
    # 4. Narasi Overbought / Oversold
    if saham_oversold:
        narasi += f"- 🛒 **Peluang Rebound (Buy The Dip):** Saham **{', '.join(saham_oversold)}** sudah masuk ke area jenuh jual (*Oversold*). Harganya sudah didiskon cukup dalam secara teknikal, membuka peluang terjadinya pantulan naik (*technical rebound*) dalam waktu dekat.\n"
    if saham_overbought:
        narasi += f"- ⚠️ **Rawan Koreksi (Take Profit):** Saham **{', '.join(saham_overbought)}** sudah masuk area jenuh beli (*Overbought*). Berhati-hatilah jika baru ingin masuk, atau pertimbangkan untuk merealisasikan keuntungan jika Anda sudah punya barang di bawah.\n"
        
    return narasi

# --- FUNGSI PEMBUAT RINGKASAN AKSI & TARGET HARGA ---
def buat_ringkasan_aksi(df):
    # Ambil saham yang memiliki sinyal BELI di Investasi ATAU Trading
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
        
        # Gabungkan jenis sinyal
        sinyal_list = []
        if 'BELI' in str(row['Sinyal_Investasi']):
            sinyal_list.append("Investasi")
        if 'BELI' in str(row['Sinyal_Trading_Pendek']):
            jenis_trading = str(row['Sinyal_Trading_Pendek']).split('(')[-1].replace(')', '')
            sinyal_list.append(f"Trading ({jenis_trading})")
            
        kategori = " & ".join(sinyal_list)
        
        # Format harga untuk ringkasan
        t_beli = f"Rp {int(row['Target_Beli']):,}" if pd.notnull(row['Target_Beli']) and not np.isnan(row['Target_Beli']) else "-"
        t_jual = f"Rp {int(row['Target_Jual']):,}" if pd.notnull(row['Target_Jual']) and not np.isnan(row['Target_Jual']) else "-"
        
        ringkasan += f"| **{ticker}** | {kategori} | **{t_beli}** | **{t_jual}** |\n"
        
    return ringkasan

# --- ANTARMUKA PENGGUNA (UI) ---
st.title("📈 Web App Screener Saham BEI")
st.markdown("Aplikasi web ini menyaring saham berdasarkan analisis fundamental dan sentimen teknikal secara _real-time_.")

# Sidebar untuk Input
st.sidebar.header("⚙️ Pengaturan Analisis")

# --- KELOMPOK EMITEN (BARU) ---
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

# Panel Informasi Kurs di Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("💵 Pengaturan Kurs (USD/IDR)")
opsi_kurs = st.sidebar.radio("Sumber Kurs:", ["Otomatis (Live API)", "Manual"])

if opsi_kurs == "Otomatis (Live API)":
    kurs_val, kurs_sumber = dapatkan_kurs_usd_idr()
else:
    kurs_val = st.sidebar.number_input("Masukkan Nilai Kurs (Rp):", min_value=10000.0, max_value=25000.0, value=16200.0, step=100.0)
    kurs_sumber = "Manual (Input User)"

st.sidebar.info(f"**Kurs Aktif:** Rp {kurs_val:,.0f}\n\n**Sumber:** {kurs_sumber}")

# Tombol Eksekusi
if st.sidebar.button("Jalankan Pemindaian 🚀", type="primary"):
    
    df_hasil = saring_saham_pilihan(ticker_pantauan, periode, strategi, kurs_val)
    
    if not df_hasil.empty:
        df_hasil = df_hasil.sort_values(by=['Sinyal_Investasi', 'Sinyal_Trading_Pendek'], ascending=[False, False])
        
        # Generate narasi berdasarkan raw data sebelum di-format menjadi string
        narasi_komprehensif = buat_narasi_analisa(df_hasil, pilihan_strategi_label)
        ringkasan_aksi = buat_ringkasan_aksi(df_hasil)
        
        tampil = df_hasil.copy()
        
        # Formatting Tabel Web secara aman (jika data None, ubah jadi '-')
        tampil['PBV'] = tampil['PBV'].apply(lambda x: f"{x:.2f}x" if pd.notnull(x) else "-")
        tampil['ROE'] = tampil['ROE'].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "-")
        tampil['RSI_14'] = tampil['RSI_14'].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else "-")
        tampil['MACD'] = tampil['MACD_Bullish'].apply(lambda x: "Bull 🟢" if x else "Bear 🔴")
        tampil['Harga'] = tampil['Harga'].apply(lambda x: f"Rp{int(x):,}")
        tampil['Harga_Wajar'] = tampil['Harga_Wajar'].apply(lambda x: f"Rp{int(x):,}" if pd.notnull(x) and not np.isnan(x) else "-")
        tampil['Target_Beli'] = tampil['Target_Beli'].apply(lambda x: f"Rp{int(x):,}" if pd.notnull(x) and not np.isnan(x) else "-")
        tampil['Target_Jual'] = tampil['Target_Jual'].apply(lambda x: f"Rp{int(x):,}" if pd.notnull(x) and not np.isnan(x) else "-")
        
        # Kolom dinamis berdasarkan strategi
        kolom = ['Ticker', 'Harga', 'Harga_Wajar', 'Target_Beli', 'Target_Jual', 'PBV', 'ROE', 'RSI_14', 'MACD', 'Sinyal_Investasi', 'Sinyal_Trading_Pendek']
        
        st.success("✅ Pemindaian Selesai!")
        
        # --- TAMPILKAN NARASI DI SINI ---
        st.info(narasi_komprehensif)
        
        # --- TAMPILKAN RINGKASAN AKSI DI SINI ---
        if ringkasan_aksi:
            st.success(ringkasan_aksi)
            
        st.divider()
        
        # Menampilkan dataframe yang responsif
        st.dataframe(tampil[kolom], use_container_width=True, hide_index=True)
        
        # Disclaimer lengkap tentang keanehan/kekosongan data dari Yahoo Finance
        st.info("""
        **Catatan Target Harga & Integritas Data:**
        * **Harga Wajar:** Menggunakan rumus *Graham Number*. Jika kosong (`-`), berarti EPS perusahaan negatif (rugi) atau datanya belum diperbarui oleh Yahoo Finance.
        * **Target Beli & Jual:** Dihitung dari area Support & Resistance teknikal.
        * **Sel Kosong (`-`):** Terjadi karena keterbatasan database gratis Yahoo Finance untuk beberapa saham lapis menengah/kecil di BEI (IDX).
        * **Sistem Koreksi Kurs Aktif:** Aplikasi telah mendeteksi dan mengoreksi laporan keuangan emiten tambang (berbasis Dolar AS) ke Rupiah secara otomatis sehingga nilai PBV, PE, dan Harga Wajar kini terhitung akurat.
        """)
    else:
        st.error("Gagal mengambil data atau tidak ada saham yang valid.")
