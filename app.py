import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time

# --- KONFIGURASI HALAMAN WEB ---
st.set_page_config(page_title="Screener Saham BEI", page_icon="📈", layout="wide")

# --- FUNGSI PENGAMBILAN DATA (DENGAN CACHE) ---
# Menggunakan cache agar web tidak lemot saat pengguna mengganti filter
@st.cache_data(ttl=3600, show_spinner=False)
def dapatkan_data_saham(ticker_symbol, periode="6mo"):
    try:
        # Menambahkan Session dan User-Agent khusus untuk menghindari pemblokiran Anti-Bot Yahoo Finance di Cloud
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        
        ticker = yf.Ticker(ticker_symbol, session=session)
        
        # Penanganan khusus untuk ticker.info yang sering diblokir di Streamlit Cloud
        try:
            info = ticker.info
            if info is None:
                info = {}
        except Exception:
            info = {} # Fallback jika fundamental gagal, tetap bisa lanjut ke teknikal
        
        pe_ratio = info.get('forwardPE') or info.get('trailingPE')
        pbv = info.get('priceToBook')
        roe = info.get('returnOnEquity')
        der = info.get('debtToEquity') 
        div_yield = info.get('dividendYield') or info.get('trailingAnnualDividendYield')
        ev_ebitda = info.get('enterpriseToEbitda')
        eps_growth = info.get('earningsQuarterlyGrowth') or info.get('earningsGrowth')
        
        eps = info.get('trailingEps')
        bvps = info.get('bookValue')
        
        if der is not None: der = der / 100.0
            
        harga_wajar_graham = np.nan
        if eps and bvps and eps > 0 and bvps > 0:
            harga_wajar_graham = np.sqrt(22.5 * eps * bvps)
            
        hist = ticker.history(period=periode)
        if hist.empty or len(hist) < 30:
            return None
        
        harga_terakhir = hist['Close'].iloc[-1]
        
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
        # Jika terjadi error fatal di luar dugaan, kita bisa melihatnya di log console server
        print(f"Error fetching {ticker_symbol}: {str(e)}")
        return None

def saring_saham_pilihan(daftar_saham, periode, strategi_pilihan):
    hasil_analisis = []
    error_tickers = []
    
    # Progress bar di UI web
    progress_text = "Memindai saham..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, tkr in enumerate(daftar_saham):
        data = dapatkan_data_saham(tkr, periode)
        if data:
            hasil_analisis.append(data)
        else:
            error_tickers.append(tkr.replace('.JK', ''))
            
        # Update progress bar
        my_bar.progress((i + 1) / len(daftar_saham), text=f"Memindai {tkr}...")
        # Jeda 0.5 detik per iterasi untuk meminimalisasi risiko diblokir karena rate-limit
        time.sleep(0.5) 
        
    my_bar.empty() # Hilangkan bar setelah selesai
    
    # Beri peringatan spesifik jika ada saham yang gagal ditarik
    if error_tickers:
        st.warning(f"⚠️ Gagal menarik data untuk: **{', '.join(error_tickers)}**. Data mungkin kosong atau diblokir oleh server Yahoo.")
            
    if not hasil_analisis:
        return pd.DataFrame()
        
    df = pd.DataFrame(hasil_analisis)
    df['Sinyal_Investasi'] = 'Pantau'
    df['Sinyal_Trading_Pendek'] = 'Pantau/Tahan' 
    
    # Logika Trading
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

# --- ANTARMUKA PENGGUNA (UI) ---
st.title("📈 Web App Screener Saham BEI")
st.markdown("Aplikasi web ini menyaring saham berdasarkan analisis fundamental dan sentimen teknikal secara _real-time_.")

# Sidebar untuk Input
st.sidebar.header("⚙️ Pengaturan Analisis")

daftar_ticker = st.sidebar.text_area(
    "Daftar Kode Saham (Pisahkan dengan koma):", 
    "PKPK, BSDE, PWON, CTRA, INDF, ICBP, PGAS, ADRO, EXCL, TLKM, ITMG, BBCA"
)
ticker_pantauan = [t.strip() + ".JK" for t in daftar_ticker.split(",")]

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

# Tombol Eksekusi
if st.sidebar.button("Jalankan Pemindaian 🚀", type="primary"):
    
    df_hasil = saring_saham_pilihan(ticker_pantauan, periode, strategi)
    
    if not df_hasil.empty:
        df_hasil = df_hasil.sort_values(by=['Sinyal_Investasi', 'Sinyal_Trading_Pendek'], ascending=[False, False])
        tampil = df_hasil.copy()
        
        # Formatting Tabel Web
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
        
        # Menampilkan dataframe yang responsif
        st.dataframe(tampil[kolom], use_container_width=True, hide_index=True)
        
        st.info("**Catatan Target Harga:** \n* **Harga Wajar:** Graham Number (Dianggap diskon jika Harga < Harga Wajar). \n* **Target Beli:** Area Support Kuat. \n* **Target Jual:** Area Resisten / Take Profit.")
    else:
        st.error("Gagal mengambil data atau tidak ada saham yang valid.")
