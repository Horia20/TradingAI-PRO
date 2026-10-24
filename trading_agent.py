"""
Screener & asistent de tranzacționare v3
  - candidați (scor tehnic + ML) și moment de intrare
  - dividende (istoric, calitate, data ex-dividend)
  - portofoliu: semnale de vânzare + predicții AI pe termen scurt / mediu / lung
  - intraday (interval 1–60 min, actualizare automată)
  - macro: inflație, șomaj, dobânzi, sentiment, VIX -> regim de piață + calendar economic TradingView
  - import CSV pentru acțiuni care nu există pe Yahoo (ex. BVB)

Instalare:
    pip install streamlit yfinance pandas numpy scikit-learn xgboost altair requests

Rulare:
    streamlit run trading_agent.py
"""

import io
import json
import os
import re
import xml.etree.ElementTree as ET

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

OUT_CANDIDATES = "candidates.csv"
PORT_FILE = "portofoliu.csv"
CUSTOM_DIR = "date_custom"          # aici se salvează CSV-urile importate (ex. BVB)
LOG_FILE = "semnale_log.csv"        # memoria de piață — istoricul semnalelor date de agent

# cele mai lichide acțiuni de pe piața principală BVB — preluate acum automat (fără import CSV)
BVB_TICKERS = ["TLV", "SNP", "SNG", "TGN", "BRD", "DIGI", "EL", "FP", "H2O", "M",
              "ONE", "PE", "SFG", "ATB", "AQ", "WINE", "TEL", "BVB"]

UNIVERSES = {
    "SUA – large cap": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "JPM", "V",
        "MA", "UNH", "XOM", "LLY", "COST", "WMT", "HD", "PG", "NFLX", "AMD",
        "CRM", "ORCL", "ADBE", "QCOM", "INTC", "CSCO", "PEP", "KO", "MRK", "ABBV",
        "BAC", "DIS", "PYPL", "UBER", "PLTR", "CRWD", "PANW", "NOW", "AMAT", "MU",
    ],
    "SUA – dividende": [
        "JNJ", "PG", "KO", "PEP", "MCD", "VZ", "T", "CVX", "XOM", "IBM", "MMM",
        "CAT", "HD", "LOW", "ABBV", "MRK", "PFE", "O", "MO", "PM", "SBUX", "TXN",
        "AVGO", "JPM", "BAC", "WFC", "USB", "DUK", "SO", "NEE", "LMT", "UPS",
    ],
    "Europa": [
        "SIE.DE", "SAP.DE", "ALV.DE", "AIR.PA", "MC.PA", "OR.PA", "SU.PA", "BAS.DE",
        "BMW.DE", "MBG.DE", "ASML.AS", "ADYEN.AS", "ENEL.MI", "ISP.MI", "RMS.PA",
        "SAN.PA", "TTE.PA", "DTE.DE", "IFX.DE",
    ],
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD"],
    "Indici & Forex": ["^GSPC", "^IXIC", "^GDAXI", "EURUSD=X", "GBPUSD=X"],
    "🇷🇴 BVB – principal": BVB_TICKERS,
}

FEATURES = [
    "ret_5", "ret_20", "ret_60", "dist_sma50", "dist_sma200", "sma50_200",
    "RSI14", "atr_pct", "vol_ratio", "dist_high", "volat_20",
    "macd_hist", "stoch_k", "roc12", "cci20",
]

# orizonturi pentru predicția portofoliului (zile de tranzacționare)
HORIZONS = {"scurt": (10, "≈ 2 săptămâni"), "mediu": (60, "≈ 3 luni"), "lung": (180, "≈ 9 luni")}

LINKS = {
    "📅 Calendar dividende (Investing.com)": "https://www.investing.com/dividends-calendar/",
    "🗓️ Calendar economic (Investing.com)": "https://www.investing.com/economic-calendar/",
    "📰 Analize & prognoze (Investing.com)": "https://www.investing.com/analysis",
    "🧰 Instrumente (Investing.com)": "https://www.investing.com/tools/",
    "🇷🇴 Bursa de Valori București": "https://bvb.ro/",
    "📈 TradingView": "https://www.tradingview.com/",
}
BVB_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Streamlit-trading-agent",
              "Referer": "https://www.bvb.ro/"}
BVB_PERIOD_DAYS = {"1mo": 31, "3mo": 93, "6mo": 186, "1y": 370, "2y": 740, "5y": 1900, "10y": 3800}


# ------------------------------------------------------------------
# Date de preț (zilnice)
# ------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def download(ticker: str, period: str = "5y"):
    """Istoric zilnic din Yahoo Finance. Returnează OHLCV sau None."""
    try:
        df = yf.download(ticker, period=period, auto_adjust=True,
                         progress=False, threads=False)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    need = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in need):
        return None
    df = df[need].apply(pd.to_numeric, errors="coerce").dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df if len(df) > 300 else None


def _custom():
    try:
        return st.session_state.get("custom_prices", {})
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def bvb_history(ticker: str, period: str = "5y"):
    """
    Istoric OHLCV direct de la Bursa de Valori București, prin feed-ul propriu
    wapi.bvb.ro — exact ce folosesc și graficele publice de pe bvb.ro — fără cont
    sau cheie API. Merge atât pentru acțiuni (ex. TLV, SNP), cât și pentru indici
    (BET, BET-XT). Dacă simbolul nu există la BVB, întoarce None, fără eroare.
    """
    days = BVB_PERIOD_DAYS.get(period, 1900)
    now = int(pd.Timestamp.utcnow().timestamp())
    frm = now - days * 86400
    try:
        r = requests.get("https://wapi.bvb.ro/api/history", headers=BVB_HEADERS, timeout=15,
                         params={"symbol": ticker.strip().upper(), "resolution": "1D",
                                 "from": frm, "to": now, "ajust": 1,
                                 "countback": int(days * 1.6) + 30, "currencyCode": "RON"})
    except Exception:
        return None
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except Exception:
        return None
    if data.get("s") != "ok":
        return None
    t = data.get("t") or []
    if len(t) < 50:
        return None
    try:
        df = pd.DataFrame({"Open": data["o"], "High": data["h"], "Low": data["l"],
                           "Close": data["c"], "Volume": data.get("v") or [0] * len(t)},
                          index=pd.to_datetime(t, unit="s"))
    except Exception:
        return None
    df.index.name = "Date"
    return df.sort_index()


def get_prices(ticker: str, period: str = "5y"):
    """Întâi CSV-urile importate manual, apoi Yahoo, apoi (automat) feed-ul public BVB."""
    cp = _custom().get(ticker)
    if cp is not None:
        return cp
    d = download(ticker, period)
    if d is not None:
        return d
    return bvb_history(ticker, period)


def _num(s):
    if not pd.api.types.is_numeric_dtype(s):
        s = s.astype(str).str.replace("\u00a0", "", regex=False).str.replace(" ", "", regex=False)
        both = s.str.contains(",", regex=False) & s.str.contains(".", regex=False)
        s = s.where(~both, s.str.replace(".", "", regex=False))      # 1.234,56 -> 1234,56
        s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def parse_price_csv(src):
    """Citește un CSV cu istoric (Data, Deschidere, Max, Min, Închidere, Volum) în orice format uzual."""
    try:
        if hasattr(src, "read"):
            data = src.read()
        else:
            with open(src, "rb") as fh:
                data = fh.read()
        if isinstance(data, bytes):
            try:
                data = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                data = data.decode("latin-1")
        first = data.splitlines()[0] if data else ""
        sep = max([";", "\t", ","], key=first.count)
        raw = pd.read_csv(io.StringIO(data), sep=sep)
    except Exception:
        return None
    low = {c: str(c).lower().strip() for c in raw.columns}

    def find(*keys):
        for k in keys:
            for c, l in low.items():
                if k in l:
                    return c
        return None

    dcol = find("date", "data")
    ccol = find("close", "inchid", "închid", "last", "pret", "preț", "price")
    if dcol is None or ccol is None:
        return None
    df = pd.DataFrame({"Close": _num(raw[ccol])})
    dates = raw[dcol].astype(str).str.strip()
    iso = dates.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.9      # 2025-08-04 (nu zi/lună inversate)
    df.index = pd.to_datetime(dates, errors="coerce") if iso else pd.to_datetime(dates, dayfirst=True, errors="coerce")
    for name, keys in (("Open", ("open", "deschid")), ("High", ("high", "max")),
                       ("Low", ("low", "min")), ("Volume", ("vol",))):
        col = find(*keys)
        df[name] = _num(raw[col]).values if col is not None else (0.0 if name == "Volume" else df["Close"].values)
    df = df[df.index.notna()].sort_index().dropna(subset=["Close"])
    df = df[~df.index.duplicated(keep="last")]
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "Date"
    return df if len(df) >= 220 else None


def load_custom_prices():
    out = {}
    if os.path.isdir(CUSTOM_DIR):
        for fn in os.listdir(CUSTOM_DIR):
            if fn.lower().endswith(".csv"):
                d = parse_price_csv(os.path.join(CUSTOM_DIR, fn))
                if d is not None:
                    out[fn[:-4].upper()] = d
    return out


# ------------------------------------------------------------------
# Date intraday (Yahoo — de regulă întârziate până la ~15 min)
# ------------------------------------------------------------------
@st.cache_data(ttl=30, show_spinner=False)
def intraday(ticker: str, interval: str, period: str):
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False, threads=False)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    need = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in need):
        return None
    df = df[need].apply(pd.to_numeric, errors="coerce").dropna(subset=["Close"])
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("Europe/Bucharest").tz_localize(None)
    df.index = idx
    return df


def intraday_features(df):
    d = df.copy()
    day = pd.Series(d.index.date, index=d.index)
    tp = (d["High"] + d["Low"] + d["Close"]) / 3
    vol = d["Volume"].groupby(day).cumsum().replace(0, np.nan)
    d["VWAP"] = (tp * d["Volume"]).groupby(day).cumsum() / vol
    d["EMA9"] = d["Close"].ewm(span=9, adjust=False).mean()
    d["EMA21"] = d["Close"].ewm(span=21, adjust=False).mean()
    d["RSI14"] = rsi(d["Close"])
    d["ATR14"] = atr(d)
    return d


def intraday_signal(d):
    l = d.iloc[-1]
    ref = l["VWAP"] if pd.notna(l["VWAP"]) else l["EMA21"]
    up = l["Close"] > ref and l["EMA9"] > l["EMA21"]
    down = l["Close"] < ref and l["EMA9"] < l["EMA21"]
    a = float(l["ATR14"]) if pd.notna(l["ATR14"]) else 0.0
    c = float(l["Close"])
    if up and l["RSI14"] < 75:
        return "🟢 Bias long: preț peste VWAP, EMA9 > EMA21", c - 1.5 * a, c + 2.5 * a
    if up:
        return "🟠 Trend sus, dar întins (RSI mare) — nu urmări prețul", c - 1.5 * a, c + 2.5 * a
    if down:
        return "🔴 Bias descendent — evită pozițiile long", None, None
    return "🟡 Fără direcție clară (preț în jurul VWAP)", None, None


# ------------------------------------------------------------------
# Macro (FRED — date publice, gratuite) + widgeturi TradingView
# ------------------------------------------------------------------
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fred(series_id: str):
    try:
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                         params={"id": series_id}, timeout=20,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception:
        return None
    if df.shape[1] < 2:
        return None
    df = df.iloc[:, :2]
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().set_index("date")["value"]
    return s if len(s) > 14 else None


def macro_snapshot():
    """Indicatori macro SUA -> scor de regim (risk-on / mixt / prudență)."""
    rows, score = [], 0
    icon = {1: "🟢", 0: "🟡", -1: "🔴"}

    def add(name, value, trend, s, note):
        nonlocal score
        score += s
        rows.append({"indicator": name, "valoare": value, "tendință": trend,
                     "semnal": icon[s], "ce înseamnă": note})

    cpi = fred("CPIAUCSL")
    if cpi is not None:
        yoy = (cpi.pct_change(12) * 100).dropna()
        now, prev = yoy.iloc[-1], yoy.iloc[-4]
        rising, falling = now > prev + 0.1, now < prev - 0.1
        s = 1 if (now < 3 and not rising) else (-1 if (now > 4 or (rising and now > 3)) else 0)
        add("Inflație SUA (CPI, an/an)", f"{now:.1f}%", "↑" if rising else "↓" if falling else "→", s,
            "Inflația mică și stabilă susține acțiunile; inflația mare forțează dobânzi mari.")
    ur = fred("UNRATE")
    if ur is not None:
        m3 = ur.rolling(3).mean()
        sahm = m3.iloc[-1] - m3.iloc[-13:-1].min()
        s = -1 if sahm >= 0.5 else (1 if sahm <= 0.2 else 0)
        add("Șomaj SUA", f"{ur.iloc[-1]:.1f}%", f"{ur.iloc[-1] - ur.iloc[-4]:+.1f} pp / 3 luni", s,
            "Creșterea rapidă a șomajului (regula Sahm ≥ 0,5) a anunțat aproape toate recesiunile.")
    ff = fred("FEDFUNDS")
    if ff is not None:
        ch = ff.iloc[-1] - ff.iloc[-4]
        s = 1 if ch < -0.1 else (-1 if ch > 0.1 else 0)
        add("Dobânda Fed", f"{ff.iloc[-1]:.2f}%", f"{ch:+.2f} pp / 3 luni", s,
            "Dobânzi în scădere = lichiditate mai multă; în creștere = presiune pe evaluări.")
    yc = fred("T10Y2Y")
    if yc is not None:
        v = yc.iloc[-1]
        s = 1 if v > 0.2 else (-1 if v < 0 else 0)
        add("Curba randamentelor 10Y–2Y", f"{v:.2f} pp", f"{v - yc.iloc[-63]:+.2f} pp / 3 luni", s,
            "Curba inversată (negativă) a precedat multe recesiuni.")
    um = fred("UMCSENT")
    if um is not None:
        s = 1 if um.iloc[-1] > um.tail(12).mean() else -1
        add("Sentiment consumatori (UMich)", f"{um.iloc[-1]:.1f}", f"media 12 luni {um.tail(12).mean():.1f}", s,
            "Sentimentul peste medie susține consumul.")
    pay = fred("PAYEMS")
    if pay is not None:
        chg = pay.diff().tail(3).mean()
        s = 1 if chg > 100 else (-1 if chg < 0 else 0)
        add("Locuri de muncă noi (nonfarm, medie 3 luni)", f"{chg:,.0f} mii", "", s,
            "Creștere solidă a angajărilor = economie rezistentă.")
    vix = get_prices("^VIX")
    if vix is not None:
        v = float(vix["Close"].iloc[-1])
        s = 1 if v < 18 else (-1 if v > 25 else 0)
        add("VIX (frica pe piață)", f"{v:.1f}", f"{v - float(vix['Close'].iloc[-6]):+.1f} / 5 zile", s,
            "Sub 18 = calm; peste 25 = stres.")
    if not rows:
        return pd.DataFrame(), None, None
    regime = "Favorabil (risk-on)" if score >= 3 else ("Prudență (risk-off)" if score <= -2 else "Mixt")
    return pd.DataFrame(rows), score, regime


def macro_eu_ro():
    rows = []
    for name, sid, kind in [
        ("Inflație zona euro (HICP, an/an)", "CP0000EZ19M086NEST", "yoy"),
        ("Șomaj zona euro", "LRHUTTTTEZM156S", "lvl"),
        ("Inflație România (HICP, an/an)", "CP0000ROM086NEST", "yoy"),
        ("Șomaj România", "LRHUTTTTROM156S", "lvl"),
    ]:
        s = fred(sid)
        if s is None:
            continue
        val = (s.pct_change(12).dropna().iloc[-1] * 100) if kind == "yoy" else s.iloc[-1]
        rows.append({"indicator": name, "valoare": f"{val:.1f}%", "ultima_dată": s.index[-1].date()})
    return pd.DataFrame(rows)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fmp_dividend_calendar(api_key: str, days_ahead: int = 30):
    """Calendar de dividende (ex-dividend) din Financial Modeling Prep — necesită cheie API gratuită."""
    if not api_key:
        return None, "Lipsește cheia API."
    today = pd.Timestamp.today().normalize()
    frm, to = today.date(), (today + pd.Timedelta(days=days_ahead)).date()
    try:
        r = requests.get("https://financialmodelingprep.com/api/v3/stock_dividend_calendar",
                         params={"from": frm, "to": to, "apikey": api_key}, timeout=20)
    except Exception as e:
        return None, f"Eroare de rețea: {e}"
    if r.status_code == 401 or r.status_code == 403:
        return None, "Cheie API invalidă sau expirată."
    if r.status_code != 200:
        return None, f"Eroare server FMP (cod {r.status_code})."
    try:
        data = r.json()
    except Exception:
        return None, "Răspuns neașteptat de la FMP."
    if isinstance(data, dict) and data.get("Error Message"):
        return None, data["Error Message"]
    if not data:
        return pd.DataFrame(), None
    df = pd.DataFrame(data)
    ren = {"symbol": "ticker", "date": "ex_dividend", "dividend": "dividend",
           "paymentDate": "data_plata", "recordDate": "data_inregistrare",
           "declarationDate": "data_anunt", "adjDividend": "dividend_ajustat"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    keep = [c for c in ["ticker", "ex_dividend", "dividend", "data_plata",
                        "data_inregistrare", "data_anunt"] if c in df.columns]
    df = df[keep].dropna(subset=["ticker", "ex_dividend"])
    df["ex_dividend"] = pd.to_datetime(df["ex_dividend"], errors="coerce").dt.date
    return df.sort_values("ex_dividend").reset_index(drop=True), None


def tv_widget(script, config, height):
    """Widget gratuit TradingView (calendar economic, grafic avansat etc.)."""
    html = f"""
    <div class="tradingview-widget-container" style="height:{height}px;width:100%">
      <div class="tradingview-widget-container__widget" style="height:calc(100% - 32px);width:100%"></div>
      <script type="text/javascript"
        src="https://s3.tradingview.com/external-embedding/{script}" async>
        {json.dumps(config)}
      </script>
    </div>"""
    components.html(html, height=height)


def to_tv_symbol(t):
    t = t.strip().upper()
    if ":" in t:
        return t
    sfx = {".DE": "XETR", ".PA": "EURONEXT", ".AS": "EURONEXT", ".MI": "MIL", ".L": "LSE"}
    for s, ex in sfx.items():
        if t.endswith(s):
            return f"{ex}:{t[:-len(s)]}"
    if t.endswith("-USD"):
        return f"BITSTAMP:{t[:-4]}USD"
    if t.endswith("=X"):
        return f"FX:{t[:-2]}"
    return t


EU_SUFFIXES = (".DE", ".PA", ".AS", ".MI", ".L")
RELEVANT_INDICES = {
    "SUA": [("S&P 500", "^GSPC"), ("Nasdaq Composite", "^IXIC")],
    "Europa": [("STOXX Europe 600", "^STOXX"), ("DAX (Germania)", "^GDAXI"),
              ("CAC 40 (Franța)", "^FCHI"), ("FTSE 100 (UK)", "^FTSE")],
    "Crypto": [("S&P 500", "^GSPC")],
    "BVB": [("Indice BET", "BET"), ("Indice BET-XT", "BET-XT")],
    "Altele": [("S&P 500", "^GSPC")],
}


def market_of(ticker: str) -> str:
    """Clasifică un ticker în SUA / Europa / BVB / Crypto / Altele, fără apel de rețea."""
    t = ticker.strip().upper()
    if t in _custom() or t in BVB_TICKERS:
        return "BVB"
    if t.endswith(EU_SUFFIXES):
        return "Europa"
    if t.endswith("-USD"):
        return "Crypto"
    if t.endswith("=X") or t.startswith("^"):
        return "Altele"
    return "SUA"


def decizie_din_scor(score):
    """BUY / HOLD / AVOID dintr-un scor 0–100 (scor_final sau scor_global, oricare e disponibil)."""
    if pd.isna(score):
        return "–", "Date insuficiente pentru o decizie."
    if score >= 70:
        return "🟢 BUY", "Semnale puternice pe majoritatea componentelor analizate."
    if score >= 45:
        return "🟡 HOLD", "Semnale mixte — nici clar favorabile, nici clar nefavorabile."
    return "🔴 AVOID", "Semnale majoritar slabe — nu pare un moment bun de cumpărare."


@st.cache_data(ttl=120, show_spinner=False)
def yahoo_news(ticker: str):
    try:
        r = requests.get("https://feeds.finance.yahoo.com/rss/2.0/headline",
                         params={"s": ticker, "region": "US", "lang": "en-US"},
                         timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        root = ET.fromstring(r.content)
        out = []
        for i in root.iter("item"):
            ts = pd.to_datetime(i.findtext("pubDate"), utc=True, errors="coerce")
            ts = ts.tz_localize(None) if pd.notna(ts) else pd.NaT
            out.append({"titlu": i.findtext("title"), "link": i.findtext("link"),
                        "data": i.findtext("pubDate"), "ts": ts})
        return out[:12]
    except Exception:
        return []


# ------------------------------------------------------------------
# Dividende
# ------------------------------------------------------------------
def _to_ts(x):
    try:
        t = pd.Timestamp(x)
        return t.tz_localize(None) if t.tzinfo else t
    except Exception:
        return pd.NaT


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def earnings_date(ticker: str):
    """Următoarea dată de raportare a rezultatelor (sau NaT)."""
    try:
        cal = yf.Ticker(ticker).calendar
        if isinstance(cal, dict):
            er = cal.get("Earnings Date")
            if er is None:
                return pd.NaT
            if not isinstance(er, (list, tuple)):
                er = [er]
            now = pd.Timestamp.today().normalize()
            fut = [d for d in (_to_ts(x) for x in er) if pd.notna(d) and d >= now]
            return min(fut) if fut else pd.NaT
    except Exception:
        pass
    return pd.NaT


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def dividend_info(ticker: str):
    """Istoric dividende, calitate și următoarea dată ex-dividend."""
    out = {"div_annual": np.nan, "freq": 0, "crestere_%": np.nan, "ani_fara_reducere": 0,
           "taiat": False, "payout": np.nan, "ex_next": pd.NaT, "ex_estimat": False}
    try:
        tk = yf.Ticker(ticker)
        divs = tk.dividends
    except Exception:
        return out
    if divs is None or len(divs) == 0:
        return out
    divs = divs.copy()
    divs.index = pd.to_datetime(divs.index).tz_localize(None)
    now = pd.Timestamp.today().normalize()

    last12 = divs[divs.index > now - pd.Timedelta(days=365)]
    prev12 = divs[(divs.index <= now - pd.Timedelta(days=365)) &
                  (divs.index > now - pd.Timedelta(days=730))]
    if last12.empty:
        return out
    out["div_annual"] = float(last12.sum())
    out["freq"] = int(len(last12))
    if len(prev12) > 0 and prev12.mean() > 0:
        g = last12.mean() / prev12.mean() - 1
        out["crestere_%"] = round(g * 100, 1)
        out["taiat"] = bool(g < -0.05)

    ym = divs.groupby(divs.index.year).mean()
    ym = ym[ym.index < now.year].values
    n = 0
    for i in range(len(ym) - 1, 0, -1):
        if ym[i] >= ym[i - 1] * 0.999:
            n += 1
        else:
            break
    out["ani_fara_reducere"] = n

    try:
        cal = tk.calendar
        if isinstance(cal, dict) and cal.get("Ex-Dividend Date") is not None:
            ex = _to_ts(cal["Ex-Dividend Date"])
            if pd.notna(ex) and ex >= now:
                out["ex_next"] = ex
    except Exception:
        pass
    if pd.isna(out["ex_next"]):
        gaps = divs.index.to_series().diff().dropna().tail(4)
        if len(gaps) >= 2:
            interval = gaps.median()
            if interval > pd.Timedelta(days=20):
                nxt = divs.index[-1] + interval
                while nxt < now:
                    nxt += interval
                out["ex_next"], out["ex_estimat"] = nxt, True
    try:
        po = tk.info.get("payoutRatio")
        if po is not None:
            out["payout"] = float(po)
    except Exception:
        pass
    return out


# ------------------------------------------------------------------
# Indicatori (pandas pur)
# ------------------------------------------------------------------
def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, n=14):
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - pc).abs(),
                    (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def add_features(df):
    d = df.copy()
    c, h, l = d["Close"], d["High"], d["Low"]
    d["SMA20"] = c.rolling(20).mean()
    d["SMA50"] = c.rolling(50).mean()
    d["SMA100"] = c.rolling(100).mean()
    d["SMA200"] = c.rolling(200).mean()
    d["EMA20"] = c.ewm(span=20, adjust=False).mean()
    d["EMA50"] = c.ewm(span=50, adjust=False).mean()
    d["RSI14"] = rsi(c)
    d["ATR14"] = atr(d)
    d["ret_5"] = c.pct_change(5)
    d["ret_20"] = c.pct_change(20)
    d["ret_60"] = c.pct_change(60)
    d["dist_sma50"] = c / d["SMA50"] - 1
    d["dist_sma200"] = c / d["SMA200"] - 1
    d["sma50_200"] = d["SMA50"] / d["SMA200"] - 1
    d["atr_pct"] = d["ATR14"] / c
    d["vol_ratio"] = (d["Volume"] / d["Volume"].rolling(20).mean()).fillna(1.0)
    d["dist_high"] = c / c.rolling(252).max() - 1
    d["volat_20"] = c.pct_change().rolling(20).std()
    d["high20"] = c.rolling(20).max().shift(1)
    # -- oscilatoare suplimentare (stil "rating tehnic" TradingView) --
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    d["macd_hist"] = macd - signal
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    k = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    d["stoch_k"] = k.rolling(3).mean()
    d["stoch_d"] = d["stoch_k"].rolling(3).mean()
    d["roc12"] = c.pct_change(12) * 100
    tp = (h + l + c) / 3
    d["cci20"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True))
    return d.replace([np.inf, -np.inf], np.nan)


def tech_score(row):
    """Scor tehnic 0–100 (trend + momentum + RSI + volum + apropiere de maxim)."""
    s, why = 0, []
    if row["Close"] > row["SMA200"]:
        s += 20; why.append("peste SMA200")
    if row["SMA50"] > row["SMA200"]:
        s += 15; why.append("SMA50>SMA200")
    if row["Close"] > row["SMA50"]:
        s += 10; why.append("peste SMA50")
    if row["ret_60"] > 0:
        s += 10; why.append("momentum 3 luni+")
    if row["ret_20"] > 0:
        s += 10; why.append("momentum 1 lună+")
    if 45 <= row["RSI14"] <= 70:
        s += 15; why.append("RSI sănătos")
    elif row["RSI14"] > 75:
        s -= 10; why.append("supracumpărat")
    if row["vol_ratio"] > 1.2:
        s += 10; why.append("volum ridicat")
    if row["dist_high"] > -0.10:
        s += 10; why.append("aproape de maxim 52s")
    return max(0, min(100, s)), ", ".join(why)


def entry_signal(r):
    """Când să cumperi: returnează (etichetă, zonă_jos, zonă_sus)."""
    c, a = r["Close"], r["ATR14"]
    lo, hi = r["SMA50"] - 0.5 * a, r["SMA50"] + 1.0 * a
    up = c > r["SMA200"] and r["SMA50"] > r["SMA200"]
    if not up:
        return "Evită – trend slab", lo, hi
    if r["RSI14"] > 72 or r["dist_sma50"] > 0.10:
        return "Prea întins – așteaptă retragere", lo, hi
    if lo <= c <= hi or r["RSI14"] <= 50:
        return "Zonă bună de intrare", lo, hi
    if pd.notna(r["high20"]) and c >= r["high20"] and r["vol_ratio"] > 1.2:
        return "Breakout confirmat", lo, hi
    return "Acceptabil – intră în trepte", lo, hi


def tv_style_rating(row):
    """
    Rating tehnic calculat de noi, cu aceeași metodologie publică descrisă de
    TradingView pentru „Technical Rating”: numără semnalele de cumpărare/vânzare
    date de mediile mobile și de oscilatoare, apoi clasifică procentul de semnale
    de cumpărare. NU este preluat de pe TradingView — e calculat local, din
    prețurile deja descărcate, deci nu are nicio limitare de acces.
    """
    c = row["Close"]
    mas = [row.get(k) for k in ("SMA20", "SMA50", "SMA100", "SMA200", "EMA20", "EMA50")]
    ma_buy = sum(1 for m in mas if pd.notna(m) and c > m)
    ma_n = sum(1 for m in mas if pd.notna(m))

    osc_buy, osc_n = 0, 0
    rsi_v = row.get("RSI14")
    if pd.notna(rsi_v):
        osc_n += 1
        osc_buy += 1 if rsi_v < 40 else (0.5 if rsi_v < 55 else 0)
    macd_v = row.get("macd_hist")
    if pd.notna(macd_v):
        osc_n += 1
        osc_buy += 1 if macd_v > 0 else 0
    k_v, d_v = row.get("stoch_k"), row.get("stoch_d")
    if pd.notna(k_v) and pd.notna(d_v):
        osc_n += 1
        osc_buy += 1 if (k_v > d_v and k_v < 80) else 0
    roc_v = row.get("roc12")
    if pd.notna(roc_v):
        osc_n += 1
        osc_buy += 1 if roc_v > 0 else 0
    cci_v = row.get("cci20")
    if pd.notna(cci_v):
        osc_n += 1
        osc_buy += 1 if cci_v > 0 else 0

    tot_n, tot_buy = ma_n + osc_n, ma_buy + osc_buy
    if tot_n == 0:
        return np.nan, "Date insuficiente"
    frac = tot_buy / tot_n
    if frac >= 0.80:
        label = "Cumpărare puternică"
    elif frac >= 0.60:
        label = "Cumpărare"
    elif frac <= 0.20:
        label = "Vânzare puternică"
    elif frac <= 0.40:
        label = "Vânzare"
    else:
        label = "Neutru"
    return round(frac * 100, 1), label


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def yahoo_fundamentals(ticker: str):
    """Fundamente + consens de analiști, din datele publice Yahoo Finance (gratuite, fără scraping)."""
    out = {"PE": np.nan, "fwd_PE": np.nan, "ROE_%": np.nan, "marja_profit_%": np.nan,
           "datorie_capital": np.nan, "creștere_venituri_%": np.nan,
           "recomandare_analiști": None, "nr_analiști": np.nan,
           "preț_țintă_mediu": np.nan, "potențial_țintă_%": np.nan,
           "P/B": np.nan, "ROA_%": np.nan, "venituri": np.nan, "profit_net": np.nan,
           "cash_flow_liber": np.nan, "capitalizare": np.nan, "sector": None, "nume": None}
    try:
        info = yf.Ticker(ticker).get_info()
    except Exception:
        return out
    if not info:
        return out
    def pct(key):
        v = info.get(key)
        return round(v * 100, 1) if isinstance(v, (int, float)) else np.nan

    g = info.get
    px = g("currentPrice") or g("regularMarketPrice")
    tgt = g("targetMeanPrice")
    out.update({
        "PE": g("trailingPE"), "fwd_PE": g("forwardPE"),
        "ROE_%": pct("returnOnEquity"),
        "marja_profit_%": pct("profitMargins"),
        "datorie_capital": g("debtToEquity"),
        "creștere_venituri_%": pct("revenueGrowth"),
        "recomandare_analiști": g("recommendationKey"),
        "nr_analiști": g("numberOfAnalystOpinions"),
        "preț_țintă_mediu": tgt,
        "potențial_țintă_%": round((tgt / px - 1) * 100, 1) if tgt and px else np.nan,
        "P/B": g("priceToBook"), "ROA_%": pct("returnOnAssets"), "venituri": g("totalRevenue"),
        "profit_net": g("netIncomeToCommon"), "cash_flow_liber": g("freeCashflow"),
        "capitalizare": g("marketCap"), "sector": g("sector"), "nume": g("longName") or g("shortName"),
    })
    return out


def _big(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    for lim, suf in ((1e12, "T"), (1e9, "mld"), (1e6, "mil")):
        if abs(x) >= lim:
            return f"{x / lim:.1f} {suf}"
    return f"{x:,.0f}"


REC_MAP = {"strong_buy": 100, "buy": 75, "hold": 50, "sell": 25, "strong_sell": 0}
REC_LABEL = {"strong_buy": "Cumpărare puternică", "buy": "Cumpărare", "hold": "Neutru",
            "sell": "Vânzare", "strong_sell": "Vânzare puternică", None: "–"}


def fundamental_score(df):
    """Scor 0–100 relativ la restul acțiunilor din listă (percentile): PE mic, ROE mare,
    marjă mare, datorie mică, creștere mare -> scor mai bun."""
    d = df.copy()
    pe_inv = -d["PE"].clip(lower=0)
    debt_inv = -d["datorie_capital"]
    parts, n = pd.Series(0.0, index=d.index), 0
    for col, inv in [("ROE_%", None), ("marja_profit_%", None), ("creștere_venituri_%", None),
                     ("PE", pe_inv), ("datorie_capital", debt_inv)]:
        s = inv if inv is not None else d[col]
        r = s.rank(pct=True, na_option="keep")
        if r.notna().any():
            parts = parts.add(r.fillna(r.mean()), fill_value=0)
            n += 1
    return (parts / max(n, 1) * 100).round(1) if n else pd.Series(np.nan, index=d.index)


def letter_grade(score):
    if pd.isna(score):
        return "–"
    for lim, g in [(90, "A+"), (80, "A"), (70, "B+"), (60, "B"), (50, "C+"),
                  (40, "C"), (30, "D"), (0, "F")]:
        if score >= lim:
            return g
    return "F"


def rating_scan(view, progress=None):
    """Combină tehnic (local) + fundamental (Yahoo) + consens analiști (Yahoo) + momentum într-un scor global."""
    rows = []
    for i, t in enumerate(view["ticker"]):
        if progress:
            progress.progress((i + 1) / len(view), text=f"Rating {t}…")
        df = get_prices(t)
        if df is None:
            continue
        f = add_features(df)
        last = f.iloc[-1]
        tv_pct, tv_label = tv_style_rating(last)
        fund = yahoo_fundamentals(t)
        rec = fund["recomandare_analiști"]
        rows.append({"ticker": t, **fund, "rating_tehnic_%": tv_pct, "rating_tehnic": tv_label,
                     "analiști_verdict": REC_LABEL.get(rec, rec), "_rec_num": REC_MAP.get(rec, np.nan)})
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d["scor_fundamental"] = fundamental_score(d)
    mom = view.set_index("ticker")["scor_tehnic"].reindex(d["ticker"]).values
    parts = pd.concat([
        d["rating_tehnic_%"], d["scor_fundamental"], d["_rec_num"],
        pd.Series(mom, index=d.index),
    ], axis=1)
    d["scor_global"] = parts.mean(axis=1, skipna=True).round(1)
    d["nota"] = d["scor_global"].apply(letter_grade)
    d = d.drop(columns=["_rec_num"])
    cols = ["ticker", "nota", "scor_global", "rating_tehnic", "rating_tehnic_%", "scor_fundamental",
            "analiști_verdict", "nr_analiști", "preț_țintă_mediu", "potențial_țintă_%",
            "PE", "fwd_PE", "ROE_%", "marja_profit_%", "datorie_capital", "creștere_venituri_%"]
    return d[cols].sort_values("scor_global", ascending=False).reset_index(drop=True)


def cheap_flags(row):
    """Semnale de atenție specifice acțiunilor cu preț mic — un preț mic NU înseamnă automat ieftin/bun."""
    f = []
    if pd.isna(row.get("PE")) or row.get("PE") is None:
        f.append("fără P/E (posibil fără profit)")
    elif row.get("PE") is not None and row.get("PE") < 0:
        f.append("P/E negativ (companie pe pierdere)")
    if pd.notna(row.get("creștere_venituri_%")) and row.get("creștere_venituri_%") < 0:
        f.append("venituri în scădere")
    dc = row.get("datorie_capital")
    if pd.notna(dc) and dc > 150:
        f.append("datorie mare față de capital")
    if row.get("vol_ratio", 1) < 0.5:
        f.append("volum redus (lichiditate slabă)")
    if pd.notna(row.get("sub_max52_%")) and row.get("sub_max52_%") < -50:
        f.append("preț la >50% sub maximul anual — verifică de ce a scăzut atât")
    return ", ".join(f) if f else "fără semnale roșii evidente"


def cheap_scan(view, max_price, progress=None):
    """Combină scorul tehnic+ML (din scan) cu ratingul fundamental+analiști (rating_scan) pentru acțiuni ieftine."""
    sub = view[view["pret"] <= max_price].copy()
    if sub.empty:
        return sub
    rt = rating_scan(sub, progress)
    if rt.empty:
        return pd.DataFrame()
    d = sub.merge(rt, on="ticker", how="inner")
    d["scor_combo"] = d[["scor_final", "scor_global"]].mean(axis=1, skipna=True).round(1)
    d["riscuri"] = d.apply(cheap_flags, axis=1)
    cols = ["ticker", "pret", "scor_combo", "nota", "moment_intrare", "prob_ML", "scor_tehnic",
            "rating_tehnic", "scor_fundamental", "analiști_verdict", "preț_țintă_mediu",
            "potențial_țintă_%", "PE", "creștere_venituri_%", "riscuri"]
    return d[cols].sort_values("scor_combo", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=1800, show_spinner=False)
def fmp_news(ticker: str, api_key: str, limit: int = 8):
    if not api_key:
        return [], None
    try:
        r = requests.get("https://financialmodelingprep.com/api/v3/stock_news",
                         params={"tickers": ticker, "limit": limit, "apikey": api_key}, timeout=15)
    except Exception as e:
        return [], f"Eroare de rețea: {e}"
    if r.status_code in (401, 403):
        return [], "Cheie API invalidă sau expirată."
    if r.status_code != 200:
        return [], f"Eroare server FMP (cod {r.status_code})."
    try:
        data = r.json()
    except Exception:
        return [], "Răspuns neașteptat de la FMP."
    if not isinstance(data, list):
        return [], None
    return [{"titlu": n.get("title"), "sursa": n.get("site"), "data": n.get("publishedDate"),
            "link": n.get("url")} for n in data], None


# ------------------------------------------------------------------
# Model ML
# ------------------------------------------------------------------
def make_model():
    return XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
        eval_metric="logloss", verbosity=0, n_jobs=-1,
    )


def build_dataset(data, horizon, target):
    parts = []
    for t, df in data.items():
        f = add_features(df)
        f["fwd"] = f["Close"].shift(-horizon) / f["Close"] - 1
        f["ticker"] = t
        parts.append(f.dropna(subset=FEATURES + ["fwd"])[FEATURES + ["fwd", "ticker"]])
    if not parts:
        return pd.DataFrame()
    ds = pd.concat(parts)
    ds["label"] = (ds["fwd"] >= target).astype(int)
    return ds


def _non_overlapping(df, horizon):
    """
    Păstrează, din fiecare acțiune, doar un exemplu la fiecare `horizon` zile.
    Cu orizontul de 10 zile, exemplul din ziua 1 și cel din ziua 2 se suprapun
    aproape complet (9 din 10 zile sunt comune) — nu sunt observații independente.
    Metricile de evaluare (AUC, randamente) trebuie calculate pe eșantioane care
    NU se suprapun, altfel par mult mai bune decât sunt în realitate — mai ales
    la orizonturi lungi, unde o singură mișcare de piață poate „umple" zeci de
    exemple suprapuse cu același rezultat.
    """
    parts = [g.sort_index().iloc[::max(horizon, 1)] for _, g in df.groupby("ticker")]
    return pd.concat(parts) if parts else df


@st.cache_resource(show_spinner=False)
def train_model(tickers: tuple, horizon: int, target: float, period: str = "5y"):
    """Antrenare pe trecut, test pe perioada cea mai recentă (split în timp, cu gap)."""
    data = {t: get_prices(t, period) for t in tickers}
    data = {t: d for t, d in data.items() if d is not None}
    ds = build_dataset(data, horizon, target)
    if len(ds) < 2000 or ds["label"].nunique() < 2:
        return None
    dates = np.sort(ds.index.unique())
    k = int(len(dates) * 0.75)
    cut, gap_cut = dates[k], dates[max(k - horizon, 0)]
    train, test = ds[ds.index <= gap_cut], ds[ds.index > cut]
    if train["label"].nunique() < 2 or len(test) < 100:
        return None

    # evaluarea se face DOAR pe exemple de test nesuprapuse — vezi _non_overlapping
    test_eff = _non_overlapping(test, horizon)
    m = make_model().fit(train[FEATURES], train["label"])
    n_eff = len(test_eff)
    if n_eff >= 15 and test_eff["label"].nunique() > 1:
        proba_eff = m.predict_proba(test_eff[FEATURES])[:, 1]
        top = test_eff[proba_eff >= np.quantile(proba_eff, 0.9)]
        auc = roc_auc_score(test_eff["label"], proba_eff)
        base_rate, top_hit = test_eff["label"].mean(), top["label"].mean()
        top_ret, all_ret = top["fwd"].mean(), test_eff["fwd"].mean()
    else:
        auc = base_rate = top_hit = top_ret = all_ret = np.nan
    metrics = {
        "auc": auc, "base_rate": base_rate, "base_all": ds["label"].mean(),
        "top_hit": top_hit, "top_ret": top_ret, "all_ret": all_ret,
        "n_test": len(test), "n_test_eff": n_eff, "n_train": len(train), "n_tickers": len(data),
    }
    final = make_model().fit(ds[FEATURES], ds["label"])
    return {"model": final, "metrics": metrics}


# ------------------------------------------------------------------
# Scanare candidați
# ------------------------------------------------------------------
def scan(tickers, trained, progress=None):
    rows = []
    for i, t in enumerate(tickers):
        if progress:
            progress.progress((i + 1) / len(tickers), text=f"Analizez {t}…")
        df = get_prices(t)
        if df is None:
            continue
        f = add_features(df)
        last = f.iloc[-1]
        if last[FEATURES + ["SMA50", "SMA200", "ATR14"]].isna().any():
            continue
        score, why = tech_score(last)
        entry, lo, hi = entry_signal(last)
        prob = np.nan
        if trained is not None:
            prob = float(trained["model"].predict_proba(last[FEATURES].to_frame().T.astype(float))[0, 1])
        close, a = float(last["Close"]), float(last["ATR14"])
        rows.append({
            "ticker": t, "data": f.index[-1].date(), "pret": round(close, 2),
            "scor_tehnic": score, "prob_ML": prob,
            "moment_intrare": entry,
            "zona_cumparare": f"{lo:.2f} – {hi:.2f}",
            "ret_1l_%": round(float(last["ret_20"]) * 100, 1),
            "RSI": round(float(last["RSI14"]), 1),
            "sub_max52_%": round(float(last["dist_high"]) * 100, 1),
            "stop_loss": round(close - 2 * a, 2),
            "target": round(close + 3 * a, 2),
            "motive": why,
        })
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    if trained is not None:
        res["scor_final"] = (0.5 * res["scor_tehnic"] + 0.5 * res["prob_ML"].rank(pct=True) * 100).round(1)
    else:
        res["scor_final"] = res["scor_tehnic"].astype(float)
    res["semnal"] = pd.cut(res["scor_final"], [-1, 45, 70, 101],
                           labels=["Slab", "Neutru", "Interesant"]).astype(str)
    return res.sort_values("scor_final", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------
# Memorie de piață — jurnalul semnalelor date de agent, ca să poată fi verificate ulterior
# ------------------------------------------------------------------
def log_semnale(res):
    """Adaugă în jurnal semnalul de azi pentru fiecare ticker (un rând pe zi per ticker, fără duplicate)."""
    if res is None or res.empty:
        return
    today = pd.Timestamp.today().normalize().date()
    new = res[["ticker", "pret", "scor_final", "prob_ML", "moment_intrare"]].copy()
    new.insert(0, "data_semnal", today)
    new["decizie"] = new["scor_final"].apply(lambda s: decizie_din_scor(s)[0])
    try:
        old = pd.read_csv(LOG_FILE, parse_dates=["data_semnal"]) if os.path.exists(LOG_FILE) else pd.DataFrame()
    except Exception:
        old = pd.DataFrame()
    if not old.empty:
        old["data_semnal"] = pd.to_datetime(old["data_semnal"]).dt.date
        old = old[~((old["data_semnal"] == today) & (old["ticker"].isin(new["ticker"])))]
    combined = pd.concat([old, new], ignore_index=True)
    try:
        combined.to_csv(LOG_FILE, index=False)
    except Exception:
        pass


def evaluate_log(min_days=10):
    """Pentru fiecare semnal mai vechi de min_days zile, verifică prețul de azi și calculează randamentul real."""
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame(), pd.DataFrame()
    try:
        log = pd.read_csv(LOG_FILE, parse_dates=["data_semnal"])
    except Exception:
        return pd.DataFrame(), pd.DataFrame()
    if log.empty:
        return pd.DataFrame(), pd.DataFrame()
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=min_days * 1.45)  # ~min_days zile lucrătoare
    old_enough = log[log["data_semnal"] <= cutoff].copy()
    if old_enough.empty:
        return pd.DataFrame(), log
    rows = []
    for t in old_enough["ticker"].unique():
        df = get_prices(t)
        if df is None:
            continue
        last_price, last_date = float(df["Close"].iloc[-1]), df.index[-1].date()
        sub = old_enough[old_enough["ticker"] == t]
        for _, r in sub.iterrows():
            ret = round((last_price / r["pret"] - 1) * 100, 2)
            rows.append({"ticker": t, "data_semnal": r["data_semnal"].date(), "decizie": r["decizie"],
                        "preț_atunci": r["pret"], "preț_azi": round(last_price, 2),
                        "zile_trecute": (pd.Timestamp.today().normalize() - r["data_semnal"]).days,
                        "randament_%": ret, "corect": (ret > 0) if r["decizie"] == "🟢 BUY"
                        else ((ret < 0) if r["decizie"] == "🔴 AVOID" else np.nan)})
    return pd.DataFrame(rows), log


# ------------------------------------------------------------------
# Dividende: scor și scanare
# ------------------------------------------------------------------
def div_flags(r):
    f = []
    if r["randament_%"] > 10:
        f.append("randament >10% (posibilă capcană)")
    if pd.notna(r["payout"]) and r["payout"] > 0.9:
        f.append("payout >90%")
    if r["taiat"]:
        f.append("dividend redus")
    if r["sub_max52_%"] < -30:
        f.append("preț -30% față de maxim")
    return ", ".join(f)


def div_score(r):
    s, y = 0, r["randament_%"]
    if 1.5 <= y <= 8:
        s += 30
    elif 8 < y <= 12:
        s += 15
    if r["ani_fara_reducere"] >= 5:
        s += 25
    elif r["ani_fara_reducere"] >= 2:
        s += 15
    elif r["ani_fara_reducere"] >= 1:
        s += 8
    po = r["payout"]
    if pd.isna(po):
        s += 5
    elif po <= 0.6:
        s += 20
    elif po <= 0.8:
        s += 10
    elif po > 1:
        s -= 10
    if not r["taiat"]:
        s += 10
    if r["scor_tehnic"] >= 60:
        s += 15
    return max(0, min(100, s))


def dividend_scan(res, progress=None):
    rows = []
    for i, t in enumerate(res["ticker"]):
        if progress:
            progress.progress((i + 1) / len(res), text=f"Dividende {t}…")
        info = dividend_info(t)
        if pd.isna(info["div_annual"]):
            continue
        info = dict(info)
        info["ticker"] = t
        rows.append(info)
    if not rows:
        return pd.DataFrame()
    d = res.merge(pd.DataFrame(rows), on="ticker")
    d["randament_%"] = (d["div_annual"] / d["pret"] * 100).round(2)
    d = d[d["randament_%"] < 30]
    today = pd.Timestamp.today().normalize()
    d["zile_pana_ex"] = (pd.to_datetime(d["ex_next"]) - today).dt.days
    d["ex_dividend"] = pd.to_datetime(d["ex_next"]).dt.date
    d["riscuri"] = d.apply(div_flags, axis=1)
    d["scor_dividend"] = d.apply(div_score, axis=1)
    cols = ["ticker", "pret", "randament_%", "div_annual", "freq", "crestere_%", "ani_fara_reducere",
            "payout", "ex_dividend", "zile_pana_ex", "ex_estimat", "moment_intrare",
            "zona_cumparare", "scor_dividend", "scor_tehnic", "riscuri"]
    return d[cols].sort_values("scor_dividend", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------
# Portofoliu: când vând + predicții AI
# ------------------------------------------------------------------
def empty_portfolio():
    return pd.DataFrame({"ticker": pd.Series(dtype="str"),
                         "pret_cumparare": pd.Series(dtype="float"),
                         "cantitate": pd.Series(dtype="float"),
                         "data_cumparare": pd.Series(dtype="datetime64[ns]")})


def load_portfolio():
    if os.path.exists(PORT_FILE):
        try:
            p = pd.read_csv(PORT_FILE, parse_dates=["data_cumparare"])
            return p[["ticker", "pret_cumparare", "cantitate", "data_cumparare"]]
        except Exception:
            pass
    return empty_portfolio()


def analyze_portfolio(pf):
    rows = []
    for _, p in pf.iterrows():
        t = str(p["ticker"]).strip().upper()
        if not t or t == "NAN" or pd.isna(p["pret_cumparare"]):
            continue
        df = get_prices(t)
        if df is None:
            rows.append({"ticker": t, "actiune": "⚪ Fără date (importă CSV sau vezi TradingView)"})
            continue
        f = add_features(df).dropna(subset=["SMA50", "SMA200", "ATR14", "RSI14"])
        if f.empty:
            continue
        last = f.iloc[-1]
        c, a = float(last["Close"]), float(last["ATR14"])
        bd = pd.Timestamp(p["data_cumparare"]) if pd.notna(p["data_cumparare"]) \
            else f.index[-1] - pd.Timedelta(days=60)
        since = f[f.index >= bd]
        if since.empty:
            since = f.tail(1)
        peak, a0 = float(since["Close"].max()), float(since["ATR14"].iloc[0])
        buy = float(p["pret_cumparare"])
        stop = max(peak - 3 * a, buy - 2 * a0)
        target = buy + 3 * a0

        hits = []
        if c <= stop:
            hits.append((3, "stop atins"))
        if c < last["SMA200"]:
            hits.append((3, "sub SMA200 (trend rupt)"))
        if c >= target:
            hits.append((2, "țintă atinsă (+3×ATR)"))
        if last["RSI14"] > 75:
            hits.append((2, "RSI supracumpărat"))
        if c < last["SMA50"]:
            hits.append((1, "sub SMA50"))
        level = max([h[0] for h in hits], default=0)
        label = ["🟢 Ține", "🟡 Atenție – redu riscul", "🟠 Ia profit parțial", "🔴 VINDE"][level]
        qty = p["cantitate"] if pd.notna(p["cantitate"]) else np.nan
        rows.append({
            "ticker": t, "pret_cumparare": buy, "pret_acum": round(c, 2),
            "P/L_%": round((c / buy - 1) * 100, 1),
            "P/L_valoare": round((c - buy) * qty, 2) if pd.notna(qty) else np.nan,
            "stop_curent": round(stop, 2), "target": round(target, 2),
            "actiune": label, "motive": ", ".join(h[1] for h in hits) or "trend intact",
        })
    return pd.DataFrame(rows)


def risk_manager(pf):
    """
    Analiză de risc la nivel de portofoliu: concentrare pe poziție/piață, corelație
    între acțiunile deținute (diversificare reală sau doar aparentă) și o sugestie
    de dimensionare a pozițiilor, ponderată invers cu volatilitatea fiecăreia
    (acțiunile mai volatile primesc o pondere sugerată mai mică, la risc egal).
    """
    rows, closes = [], {}
    for _, p in pf.iterrows():
        t = str(p["ticker"]).strip().upper()
        if not t or t == "NAN" or pd.isna(p["pret_cumparare"]) or pd.isna(p.get("cantitate")):
            continue
        df = get_prices(t)
        if df is None:
            continue
        val = float(df["Close"].iloc[-1]) * float(p["cantitate"])
        vol = float(df["Close"].pct_change().tail(60).std() * np.sqrt(252) * 100)  # volatilitate anualizată %
        rows.append({"ticker": t, "valoare": val, "piață": market_of(t), "volatilitate_anuală_%": round(vol, 1)})
        closes[t] = df["Close"].pct_change().tail(250)
    if not rows:
        return pd.DataFrame(), pd.DataFrame(), None, []
    pos = pd.DataFrame(rows)
    total = pos["valoare"].sum()
    pos["pondere_%"] = round(pos["valoare"] / total * 100, 1)
    inv_vol = 1 / pos["volatilitate_anuală_%"].clip(lower=1)
    pos["pondere_sugerată_%"] = round(inv_vol / inv_vol.sum() * 100, 1)

    flags = []
    conc = pos.sort_values("pondere_%", ascending=False)
    if len(conc) and conc.iloc[0]["pondere_%"] > 30:
        flags.append(f"{conc.iloc[0]['ticker']} reprezintă {conc.iloc[0]['pondere_%']:.0f}% din portofoliu — "
                     "concentrare mare pe o singură poziție.")
    piata_conc = pos.groupby("piață")["pondere_%"].sum()
    for piata, pct in piata_conc.items():
        if pct > 70 and len(piata_conc) > 1:
            flags.append(f"{pct:.0f}% din portofoliu e pe piața {piata} — expunere redusă pe alte piețe.")

    corr = None
    if len(closes) >= 2:
        ret_df = pd.DataFrame(closes).dropna(how="all")
        if ret_df.shape[1] >= 2 and ret_df.shape[0] > 20:
            corr = ret_df.corr().round(2)
            pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
            high = pairs[pairs > 0.8]
            for (a, b), v in high.items():
                flags.append(f"{a} și {b} se mișcă aproape identic (corelație {v:.2f}) — a deține ambele "
                             "aduce mai puțină diversificare decât pare.")
    if not flags:
        flags.append("Niciun semnal de concentrare sau corelație ridicată — diversificarea arată rezonabil.")
    return pos.sort_values("pondere_%", ascending=False), corr, total, flags


def confidence(auc):
    if pd.isna(auc) or auc < 0.53:
        return "Scăzută"
    return "Medie" if auc < 0.60 else "Moderată"


def ai_forecast(pf_tickers, progress=None):
    """Antrenează câte un model pentru fiecare orizont și prezice pentru fiecare poziție."""
    base = list(dict.fromkeys(UNIVERSES["SUA – large cap"] + UNIVERSES["SUA – dividende"]))
    custom = _custom()
    train_list = tuple(sorted(set(base) | {t for t in pf_tickers if t not in custom}))
    models, metrics = {}, []
    for i, (k, (h, label)) in enumerate(HORIZONS.items()):
        if progress:
            progress.progress((i + 1) / (len(HORIZONS) + 1), text=f"Antrenez modelul {k} ({label})…")
        tm = train_model(train_list, h, 0.0, "10y")
        if tm is None:
            continue
        models[k] = tm
        m = tm["metrics"]
        metrics.append({"orizont": f"{k} ({label})", "AUC test": round(m["auc"], 3),
                        "% zile pozitive (istoric)": round(m["base_all"] * 100, 1),
                        "succes top 10% semnale": round(m["top_hit"] * 100, 1),
                        "încredere": confidence(m["auc"])})
    rows = []
    for t in pf_tickers:
        df = get_prices(t, "10y")
        if df is None:
            continue
        f = add_features(df)
        last = f.iloc[-1]
        if last[FEATURES].isna().any():
            continue
        sigma = float(f["Close"].pct_change().tail(60).std())
        c = float(last["Close"])
        row = {"ticker": t, "pret": round(c, 2)}
        score = 0
        for k, (h, _) in HORIZONS.items():
            if k not in models:
                continue
            p = float(models[k]["model"].predict_proba(last[FEATURES].to_frame().T.astype(float))[0, 1])
            delta = p - models[k]["metrics"]["base_all"]
            v, s = ("Favorabil", 1) if delta >= 0.07 else (("Nefavorabil", -1) if delta <= -0.07 else ("Neutru", 0))
            score += s
            lo, hi = c * np.exp(-sigma * np.sqrt(h)), c * np.exp(sigma * np.sqrt(h))
            row[f"{k}_P(urcare)_%"] = round(p * 100, 1)
            row[f"{k}_verdict"] = {"Favorabil": "🟢 ", "Neutru": "🟡 ", "Nefavorabil": "🔴 "}[v] + v
            row[f"{k}_interval_±1σ"] = f"{lo:.2f} – {hi:.2f}"
        row["imagine_generala"] = "Optimist" if score >= 2 else ("Prudent" if score <= -2 else "Mixt")
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(metrics)


# ------------------------------------------------------------------
# Backtest real al strategiei (scorul tehnic, nu ML — vezi explicație în UI)
# ------------------------------------------------------------------
def backtest_ticker(df, hold_days=20, prag_scor=70):
    """
    Simulează istoric: de fiecare dată când scorul tehnic atinge pragul (implicit 70,
    același prag folosit pentru 🟢 BUY în tabul Candidați), 'cumpără' la închiderea
    acelei zile și urmărește ziua următoare dacă iese prin stop (preț − 2×ATR),
    prin țintă (preț + 3×ATR) sau prin limita de timp (hold_days). Nu suprapune
    tranzacții pe aceeași acțiune — următorul semnal e luat în calcul abia după ce
    tranzacția anterioară s-a închis.
    """
    f = add_features(df).dropna(subset=["SMA50", "SMA200", "ATR14", "RSI14"])
    if len(f) < 60:
        return pd.DataFrame()
    trades, n, i = [], len(f), 0
    while i < n - 1:
        row = f.iloc[i]
        score, _ = tech_score(row)
        if score >= prag_scor:
            entry_price, entry_date = float(row["Close"]), f.index[i]
            atr = float(row["ATR14"])
            stop, target = entry_price - 2 * atr, entry_price + 3 * atr
            limit = min(n - 1, i + hold_days)
            exit_price, exit_date, motiv, j = None, None, None, i + 1
            while j <= limit:
                r2 = f.iloc[j]
                if float(r2["Low"]) <= stop:
                    exit_price, exit_date, motiv = stop, f.index[j], "stop"
                    break
                if float(r2["High"]) >= target:
                    exit_price, exit_date, motiv = target, f.index[j], "țintă"
                    break
                j += 1
            if exit_price is None:
                exit_price, exit_date, motiv = float(f.iloc[limit]["Close"]), f.index[limit], "timp"
                j = limit
            trades.append({"intrare": entry_date.date(), "ieșire": exit_date.date(),
                           "preț_intrare": round(entry_price, 2), "preț_ieșire": round(exit_price, 2),
                           "motiv": motiv, "randament_%": round((exit_price / entry_price - 1) * 100, 2)})
            i = j + 1
        else:
            i += 1
    return pd.DataFrame(trades)


def backtest_scan(tickers, hold_days, prag_scor, progress=None):
    all_trades, per_ticker = [], []
    for k, t in enumerate(tickers):
        if progress:
            progress.progress((k + 1) / len(tickers), text=f"Backtest {t}…")
        df = get_prices(t, "5y")
        if df is None:
            continue
        tr = backtest_ticker(df, hold_days, prag_scor)
        if tr.empty:
            continue
        tr.insert(0, "ticker", t)
        all_trades.append(tr)
        bh = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
        per_ticker.append({"ticker": t, "tranzacții": len(tr),
                           "rată_succes_%": round((tr["randament_%"] > 0).mean() * 100, 1),
                           "randament_mediu_%": round(tr["randament_%"].mean(), 2),
                           "randament_cumulat_%": round(((1 + tr["randament_%"] / 100).prod() - 1) * 100, 1),
                           "buy_&_hold_%": round(bh, 1)})
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    summary_df = pd.DataFrame(per_ticker)
    return trades_df, summary_df


def equity_curve_chart(trades_df):
    """Grafic cu capitalul cumulat dacă ai fi pus aceeași sumă în fiecare tranzacție, pe rând."""
    t = trades_df.sort_values("intrare").copy()
    t["capital"] = (1 + t["randament_%"] / 100).cumprod() * 100
    t = t.reset_index(drop=True)
    t["#tranzacție"] = range(1, len(t) + 1)
    return alt.Chart(t).mark_line(point=True).encode(
        x=alt.X("#tranzacție:Q", title="Număr tranzacție (în ordine cronologică)"),
        y=alt.Y("capital:Q", title="Capital (start = 100)", scale=alt.Scale(zero=False)),
        tooltip=["ticker", "intrare", "randament_%", "motiv"],
    ).properties(height=350)


# ------------------------------------------------------------------
# Grafic zilnic
# ------------------------------------------------------------------
def price_chart(ticker, levels=None):
    df = get_prices(ticker)
    if df is None:
        return None
    f = add_features(df).tail(300).reset_index()
    f = f.rename(columns={f.columns[0]: "Data"})
    long = f.melt("Data", ["Close", "SMA50", "SMA200"], var_name="serie", value_name="valoare").dropna()
    ch = alt.Chart(long).mark_line().encode(
        x="Data:T", y=alt.Y("valoare:Q", scale=alt.Scale(zero=False), title="Preț"),
        color=alt.Color("serie:N", scale=alt.Scale(
            domain=["Close", "SMA50", "SMA200"], range=["#1f77b4", "orange", "red"])),
    )
    for val, col in (levels or []):
        ch = ch + alt.Chart(pd.DataFrame({"y": [val]})).mark_rule(strokeDash=[5, 4], color=col).encode(y="y:Q")
    return ch.properties(height=380)


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Sensibilitate macro per acțiune (dobânzi, inflație așteptată, sentiment)
# ------------------------------------------------------------------
def _ols(y, X):
    Xc = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    resid = y - Xc @ beta
    dof = max(len(y) - Xc.shape[1], 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(Xc.T @ Xc)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / sst if sst > 0 else np.nan
    return beta, se, r2


def _norm_idx(s):
    s = s.copy()
    s.index = (pd.to_datetime(s.index) + pd.Timedelta(hours=12)).normalize()
    return s[~s.index.duplicated(keep="last")].sort_index()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def macro_sensitivity(ticker: str):
    """
    Cât de sensibilă e o acțiune la dobânzi, inflație așteptată și sentimentul pieței.
    Regresie pe ~2 ani de randamente zilnice: randamentul acțiunii față de (S&P 500 ca
    element de control) + variația randamentului Treasury 10Y + variația inflației
    așteptate (breakeven 10Y, FRED) + îmbunătățirea sentimentului (minus variația VIX).
    Sensibilitatea înmulțită cu mișcarea reală din ultimele 20 de zile dă efectul macro recent.
    """
    out = {"ok": False, "motiv": ""}
    px = get_prices(ticker, "5y")
    if px is None:
        out["motiv"] = "nu am prețuri pentru acest ticker"
        return out
    spx, tnx, vix = get_prices("^GSPC", "5y"), get_prices("^TNX", "5y"), get_prices("^VIX", "5y")
    if spx is None or tnx is None or vix is None:
        out["motiv"] = "nu am putut descărca S&P 500, randamentul 10Y sau VIX"
        return out
    d = pd.concat({"y": _norm_idx(px["Close"]), "mkt": _norm_idx(spx["Close"]),
                   "rates": _norm_idx(tnx["Close"]), "vix": _norm_idx(vix["Close"])}, axis=1).dropna().sort_index()
    be = fred("T10YIE")
    if be is not None:
        d["infl"] = _norm_idx(be).reindex(d.index, method="ffill")
    d = d.dropna().tail(520)
    if len(d) < 150:
        out["motiv"] = "istoric comun prea scurt (sub ~150 de zile)"
        return out
    X = pd.DataFrame({"mkt": d["mkt"].pct_change(), "rates": d["rates"].diff(), "sent": -d["vix"].diff()})
    if "infl" in d:
        X["infl"] = d["infl"].diff()
    Z = pd.concat([d["y"].pct_change().rename("y"), X], axis=1).dropna()
    if len(Z) < 120:
        out["motiv"] = "prea puține zile comune de tranzacționare"
        return out
    names = list(X.columns)
    beta, se, r2 = _ols(Z["y"].values, Z[names].values)

    def d20(s):
        return float(s.iloc[-1] - s.iloc[-21])
    moves = {"rates": d20(d["rates"]), "sent": -d20(d["vix"])}
    if "infl" in d:
        moves["infl"] = d20(d["infl"])
    meta = {"rates": ("Dobânzi (randament Treasury 10Y)", "pp", 0.25, "+0,25 pp"),
            "infl": ("Inflație așteptată (breakeven 10Y)", "pp", 0.25, "+0,25 pp"),
            "sent": ("Sentiment de piață (VIX)", "pct", -5.0, "VIX +5 puncte (mai multă frică)")}
    rows, total = [], 0.0
    for k in ("rates", "infl", "sent"):
        if k not in names:
            continue
        i = names.index(k) + 1
        b, t = float(beta[i]), (float(beta[i] / se[i]) if se[i] > 0 else 0.0)
        w = float(np.clip(abs(t) - 1.0, 0.0, 1.0))          # sensibilitățile nesigure contează mai puțin
        contrib = b * moves[k] * 100
        total += w * contrib
        label, _, scen_move, scen_txt = meta[k]
        if abs(t) < 1.5:
            sens = "neclară (nesemnificativă statistic)"
        else:
            fort = "semnificativă" if abs(t) >= 2 else "probabilă"
            if k == "sent":
                sens = f"{fort}: " + ("câștigă când sentimentul se îmbunătățește" if b > 0
                                      else "câștigă când frica crește (defensivă)")
            else:
                what = "dobânzile" if k == "rates" else "inflația așteptată"
                sens = f"{fort}: " + (f"pierde când {what} cresc" if b < 0 else f"câștigă când {what} cresc")
        mv = moves[k]
        mv_txt = f"{mv:+.2f} pp" if k != "sent" else f"VIX {-mv:+.1f} puncte"
        rows.append({"factor": label, "mișcare ultimele 20 zile": mv_txt, "sensibilitate": sens,
                     "scenariu": scen_txt, "efect scenariu pe acțiune": f"{b * scen_move * 100:+.2f}%",
                     "contribuție recentă (20 zile)": f"{contrib:+.2f}%", "t-stat": round(t, 1)})
    vol20 = float(Z["y"].std() * np.sqrt(20) * 100)
    score = float(50 + 50 * np.tanh(2 * total / vol20)) if vol20 > 0 else 50.0
    verdict = "Vânt din spate" if score >= 60 else ("Vânt împotrivă" if score <= 40 else "Neutru")
    out.update(ok=True, tabel=pd.DataFrame(rows), scor=int(round(score)), verdict=verdict,
               efect_total=round(total, 2), beta_piata=round(float(beta[1 + names.index("mkt")]), 2),
               r2=round(float(r2), 2), n=len(Z), fara_inflatie=("infl" not in names))
    return out


def render_macro_detail(ms):
    """Afișează rezultatul macro_sensitivity într-un format ușor de citit."""
    if not ms or not ms.get("ok"):
        st.info(f"Sensibilitatea macro nu poate fi calculată: {ms.get('motiv', 'date insuficiente') if ms else 'date insuficiente'}.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scor macro (0–100)", ms["scor"], help="50 = neutru. Peste 60 = mișcările macro recente au fost favorabile "
                                                   "acestei acțiuni, sub 40 = nefavorabile.")
    c2.metric("Verdict", ms["verdict"])
    c3.metric("Efect macro recent (20 zile)", f"{ms['efect_total']:+.2f}%")
    c4.metric("Beta față de S&P 500", ms["beta_piata"])
    st.dataframe(ms["tabel"], use_container_width=True, hide_index=True)
    st.caption(f"Calculat pe {ms['n']} zile de tranzacționare (R² = {ms['r2']}). Factorii sunt americani; pentru acțiuni "
               "europene sau BVB e doar orientativ. Sensibilitățile cu |t| sub 1,5 sunt zgomot statistic și nu intră "
               "în scor. Istoricul nu garantează viitorul — e un context, nu un semnal de tranzacționare."
               + (" Inflația așteptată (FRED) nu a putut fi descărcată, deci a fost omisă." if ms.get("fara_inflatie") else ""))


def macro_concluzie(img, vant):
    if vant in (None, "–"):
        return img
    if img == "Optimist":
        return {"Vânt din spate": "Optimist + macro favorabil",
                "Vânt împotrivă": "Optimist, dar macro împotrivă — poziție mai mică"}.get(vant, "Optimist, macro neutru")
    if img == "Prudent":
        return {"Vânt din spate": "Prudent tehnic, dar macro ajută",
                "Vânt împotrivă": "Prudent + macro nefavorabil — evită sau redu"}.get(vant, "Prudent, macro neutru")
    return f"Mixt · macro: {vant.lower()}"


# ------------------------------------------------------------------
# Plan de tranzacționare (swing, grafic zilnic): zonă de intrare, stop, ținte
# ------------------------------------------------------------------
def trade_plan(f):
    """
    Zonă de intrare + stop-loss + 3 ținte pe baza ATR și a structurii recente.
    - Dacă prețul nu e întins: zona e în jurul prețului curent (ordin limită puțin sub piață).
    - Dacă e întins: zona e la o retragere spre media pe 20 de zile.
    - Stopul stă sub minimul ultimelor 15 zile, dar între 1,5 și 2,5 × ATR sub intrare.
    - Țintele sunt multipli ai riscului: 1,5R, 2,5R și 4R.
    """
    last = f.iloc[-1]
    c, a = float(last["Close"]), float(last["ATR14"])
    sig, _, _ = entry_signal(last)
    if sig.startswith("Evită") or not np.isfinite(a) or a <= 0:
        return {"valid": False, "semnal": sig}
    acum = not sig.startswith("Prea întins")
    ema20 = float(last["EMA20"])
    ref = c if acum else min(c, ema20)
    zlo, zhi = ref - 0.5 * a, ref + 0.25 * a
    if not acum:
        zhi = min(zhi, c)
    mid = (zlo + zhi) / 2
    swing = float(f["Low"].tail(15).min())
    stop = max(mid - 2.5 * a, min(mid - 1.5 * a, swing - 0.25 * a))
    R = mid - stop
    return {"valid": True, "semnal": sig, "acum": acum, "zona_lo": zlo, "zona_hi": zhi, "mid": mid,
            "stop": stop, "R": R, "t1": mid + 1.5 * R, "t2": mid + 2.5 * R, "t3": mid + 4 * R,
            "risc_%": R / mid * 100, "max52": float(f["Close"].tail(252).max()), "pret": c}


def render_trade_plan(pl, capital, risk_pct):
    if not pl.get("valid"):
        st.warning(f"Fără plan de intrare acum: {pl.get('semnal', 'trend slab')}. Nu forța o cumpărare când trendul "
                   "e împotrivă — așteaptă să revină peste media pe 200 de zile.")
        return
    shares = int((capital * risk_pct / 100) // pl["R"]) if pl["R"] > 0 else 0
    st.write(f"**{pl['semnal']}** — " + ("prețul e într-o zonă rezonabilă acum; pune un ordin limită în zonă."
                                         if pl["acum"] else "așteaptă o retragere în zonă înainte să intri."))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Zonă de intrare", f"{pl['zona_lo']:.2f} – {pl['zona_hi']:.2f}")
    c2.metric("Stop-loss", f"{pl['stop']:.2f}", f"-{pl['risc_%']:.1f}% față de intrare", delta_color="inverse")
    c3.metric("Țintă 1 / 2 / 3", f"{pl['t1']:.2f} / {pl['t2']:.2f} / {pl['t3']:.2f}")
    c4.metric("Acțiuni (risc %.1f%% din capital)" % risk_pct, shares,
              f"valoare ≈ {shares * pl['mid']:,.0f}")
    p1 = (pl["t1"] / pl["mid"] - 1) * 100
    p2 = (pl["t2"] / pl["mid"] - 1) * 100
    p3 = (pl["t3"] / pl["mid"] - 1) * 100
    st.caption(f"Țintele înseamnă +{p1:.1f}% / +{p2:.1f}% / +{p3:.1f}% față de intrare (1,5R / 2,5R / 4R). Sugestie de "
               "ieșire: vinde o treime la Țintă 1 și mută stopul la prețul de intrare; altă treime la Țintă 2; "
               "restul cu stop urcat sub minimele recente. "
               + (f"Atenție: Țintă 3 depășește maximul pe 52 de săptămâni ({pl['max52']:.2f}) — rezistență probabilă. "
                  if pl["t3"] > pl["max52"] else ""))


# ------------------------------------------------------------------
# Toți indicatorii tehnici + niveluri cheie
# ------------------------------------------------------------------
def key_levels(df):
    last = df.iloc[-1]
    H, L, C = float(last["High"]), float(last["Low"]), float(last["Close"])
    P = (H + L + C) / 3
    lv = {"Pivot (ziua anterioară)": P, "Rezistență R1": 2 * P - L, "Rezistență R2": P + (H - L),
          "Suport S1": 2 * P - H, "Suport S2": P - (H - L)}
    y = df.tail(252)
    hi, lo = float(y["High"].max()), float(y["Low"].min())
    lv["Maxim 52 săptămâni"], lv["Minim 52 săptămâni"] = hi, lo
    for name, r in (("Fibonacci 38,2%", 0.382), ("Fibonacci 50%", 0.5), ("Fibonacci 61,8%", 0.618)):
        lv[name] = hi - (hi - lo) * r
    return lv


def all_indicators(df):
    """Tabel cu ~30 de indicatori (medii mobile, oscilatoare, volum, volatilitate) și semnalul fiecăruia."""
    d = df.copy()
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
    price = float(c.iloc[-1])
    rows = []

    def add(grup, nume, val, sig, nota=""):
        rows.append({"grup": grup, "indicator": nume, "valoare": round(float(val), 3) if pd.notna(val) else np.nan,
                     "semnal": sig, "notă": nota})

    for n in (10, 20, 30, 50, 100, 200):
        if len(c) > n:
            s = float(c.rolling(n).mean().iloc[-1])
            add("Medii mobile", f"SMA {n}", s, "Cumpără" if price > s else "Vinde", "preț peste medie = Cumpără")
            e = float(c.ewm(span=n, adjust=False).mean().iloc[-1])
            add("Medii mobile", f"EMA {n}", e, "Cumpără" if price > e else "Vinde")
    r = float(rsi(c).iloc[-1])
    add("Oscilatoare", "RSI (14)", r, "Cumpără" if r < 30 else ("Vinde" if r > 70 else "Neutru"),
        "sub 30 = supravândut, peste 70 = supracumpărat")
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    k = (100 * (c - ll) / (hh - ll).replace(0, np.nan)).rolling(3).mean()
    dd = k.rolling(3).mean()
    kv, dv = float(k.iloc[-1]), float(dd.iloc[-1])
    add("Oscilatoare", "Stochastic %K (14,3,3)", kv,
        "Cumpără" if (kv < 20 and kv > dv) else ("Vinde" if (kv > 80 and kv < dv) else "Neutru"))
    tp = (h + l + c) / 3
    cci = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True))
    cv = float(cci.iloc[-1])
    add("Oscilatoare", "CCI (20)", cv, "Cumpără" if cv < -100 else ("Vinde" if cv > 100 else "Neutru"))
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=d.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=d.index)
    pc = c.shift()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / atr14
    mdi = 100 * mdm.ewm(alpha=1 / 14, adjust=False).mean() / atr14
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = float(dx.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
    add("Oscilatoare", "ADX (14)", adx,
        "Cumpără" if (adx > 25 and pdi.iloc[-1] > mdi.iloc[-1]) else ("Vinde" if (adx > 25 and pdi.iloc[-1] < mdi.iloc[-1]) else "Neutru"),
        "peste 25 = trend puternic; direcția dată de +DI vs −DI")
    ao = ((h + l) / 2).rolling(5).mean() - ((h + l) / 2).rolling(34).mean()
    aov = float(ao.iloc[-1])
    add("Oscilatoare", "Awesome Oscillator", aov,
        "Cumpără" if (aov > 0 and aov > ao.iloc[-2]) else ("Vinde" if (aov < 0 and aov < ao.iloc[-2]) else "Neutru"))
    mom = float(c.iloc[-1] - c.iloc[-11])
    add("Oscilatoare", "Momentum (10)", mom, "Cumpără" if mom > 0 else "Vinde")
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    add("Oscilatoare", "MACD (12,26,9)", macd.iloc[-1], "Cumpără" if macd.iloc[-1] > sig.iloc[-1] else "Vinde",
        f"linie de semnal {float(sig.iloc[-1]):.3f}")
    wr = float(-100 * (hh.iloc[-1] - price) / (hh.iloc[-1] - ll.iloc[-1])) if hh.iloc[-1] != ll.iloc[-1] else np.nan
    add("Oscilatoare", "Williams %R (14)", wr, "Cumpără" if wr < -80 else ("Vinde" if wr > -20 else "Neutru"))
    roc = float((c.iloc[-1] / c.iloc[-13] - 1) * 100)
    add("Oscilatoare", "ROC (12) %", roc, "Cumpără" if roc > 0 else "Vinde")
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    bw = (4 * sd20 / ma20).iloc[-1]
    pb = float((price - (ma20 - 2 * sd20).iloc[-1]) / (4 * sd20.iloc[-1])) if sd20.iloc[-1] > 0 else np.nan
    add("Oscilatoare", "Bollinger %B (20,2)", pb, "Cumpără" if pb < 0.05 else ("Vinde" if pb > 0.95 else "Neutru"),
        f"lățimea benzilor {bw * 100:.1f}% din preț")
    rmf = tp * v
    pos_f = rmf.where(tp > tp.shift(), 0.0).rolling(14).sum()
    neg_f = rmf.where(tp < tp.shift(), 0.0).rolling(14).sum()
    mfi = float((100 - 100 / (1 + pos_f / neg_f.replace(0, np.nan))).iloc[-1])
    if pd.notna(mfi):
        add("Volum", "MFI (14)", mfi, "Cumpără" if mfi < 20 else ("Vinde" if mfi > 80 else "Neutru"))
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    add("Volum", "OBV vs media lui pe 20 zile", float(obv.iloc[-1]),
        "Cumpără" if obv.iloc[-1] > obv.rolling(20).mean().iloc[-1] else "Vinde", "acumulare vs distribuție")
    vr = float(v.iloc[-1] / v.rolling(20).mean().iloc[-1]) if v.rolling(20).mean().iloc[-1] > 0 else np.nan
    add("Volum", "Volum față de media pe 20 zile", vr,
        "Cumpără" if (vr > 1.5 and c.iloc[-1] > c.iloc[-2]) else ("Vinde" if (vr > 1.5 and c.iloc[-1] < c.iloc[-2]) else "Neutru"),
        "volum mare confirmă mișcarea")
    add("Volatilitate", "ATR (14) % din preț", float(atr14.iloc[-1] / price * 100), "Neutru", "cât se mișcă într-o zi, în medie")
    t = pd.DataFrame(rows)
    nb, ns, nn = (t["semnal"] == "Cumpără").sum(), (t["semnal"] == "Vinde").sum(), (t["semnal"] == "Neutru").sum()
    net = (nb - ns) / max(nb + ns + nn, 1)
    verdict = ("Cumpărare puternică" if net >= 0.5 else "Cumpărare" if net >= 0.1 else
               "Vânzare puternică" if net <= -0.5 else "Vânzare" if net <= -0.1 else "Neutru")
    return t, {"cumpără": int(nb), "vinde": int(ns), "neutru": int(nn), "verdict": verdict}


# ------------------------------------------------------------------
# Știri + ton (cuvinte-cheie în titlu)
# ------------------------------------------------------------------
POS_WORDS = {"beat", "beats", "surge", "surges", "soar", "soars", "jump", "jumps", "rally", "rallies", "record",
             "upgrade", "upgrades", "raises", "outperform", "strong", "growth", "profit", "wins", "approval",
             "approved", "buyback", "bullish", "gain", "gains", "rise", "rises", "higher", "expands", "partnership",
             "deal", "boost", "boosts", "tops", "optimism"}
NEG_WORDS = {"miss", "misses", "plunge", "plunges", "fall", "falls", "drop", "drops", "slump", "slumps", "downgrade",
             "downgrades", "cut", "cuts", "lawsuit", "probe", "investigation", "recall", "weak", "loss", "losses",
             "warning", "warns", "bearish", "decline", "declines", "lower", "fraud", "layoffs", "bankruptcy", "delay",
             "delays", "halt", "selloff", "tumble", "tumbles", "slides", "risk", "fears", "concerns"}


def headline_sentiment(title):
    words = re.findall(r"[a-z]+", (title or "").lower())
    s = sum(w in POS_WORDS for w in words) - sum(w in NEG_WORDS for w in words)
    return 1 if s > 0 else (-1 if s < 0 else 0)


def _age_text(ts):
    if ts is None or pd.isna(ts):
        return ""
    mins = (pd.Timestamp.now(tz="UTC").tz_localize(None) - ts).total_seconds() / 60
    if mins < 0:
        return "acum"
    if mins < 90:
        return f"acum {int(mins)} min"
    if mins < 48 * 60:
        return f"acum {int(mins // 60)} h"
    return f"acum {int(mins // 1440)} zile"


def all_news(ticker, fmp_key="", limit=12):
    items = []
    for n in yahoo_news(ticker):
        items.append({"titlu": n["titlu"], "link": n["link"], "sursa": "Yahoo Finance", "ts": n.get("ts"),
                      "acum": _age_text(n.get("ts"))})
    if fmp_key:
        fn, _ = fmp_news(ticker, fmp_key, limit)
        for n in fn:
            ts = pd.to_datetime(n["data"], errors="coerce")
            items.append({"titlu": n["titlu"], "link": n["link"], "sursa": n["sursa"] or "FMP", "ts": ts,
                          "acum": str(n["data"] or "")})
    seen, out = set(), []
    for n in items:
        key = (n["titlu"] or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        n["sent"] = headline_sentiment(n["titlu"])
        out.append(n)
    out.sort(key=lambda n: (pd.isna(n["ts"]) if n["ts"] is not None else True,
                            -(n["ts"].value if (n["ts"] is not None and pd.notna(n["ts"])) else 0)))
    return out[:limit]


def news_panel(ticker, fmp_key=""):
    items = all_news(ticker, fmp_key)
    if not items:
        st.info("Nicio știre găsită acum pentru acest simbol (Yahoo nu are știri pentru toate acțiunile, mai ales "
                "cele de la BVB). Verifică și pe Investing.com sau TradingView.")
        return
    pos, neg = sum(n["sent"] > 0 for n in items), sum(n["sent"] < 0 for n in items)
    tone = "pozitiv" if pos > neg + 1 else ("negativ" if neg > pos + 1 else "mixt / neutru")
    st.markdown(f"**Ton general al știrilor recente: {tone}** ({pos} pozitive, {neg} negative din {len(items)})")
    for n in items:
        emo = {1: "🟢", -1: "🔴", 0: "⚪"}[n["sent"]]
        st.markdown(f"- {emo} [{n['titlu']}]({n['link']}) · {n['sursa']} · {n['acum']}")
    st.caption("Tonul e estimat din cuvinte-cheie din titlu (de exemplu «upgrade», «miss»), deci e orientativ — "
               "citește știrea înainte să tranzacționezi pe ea. Se actualizează la câteva minute.")


# ------------------------------------------------------------------
# Intraday: scanner zilnic + plan de tranzacție
# ------------------------------------------------------------------
INTRA_PERIOD = {"1m": "5d", "5m": "1mo", "15m": "1mo", "30m": "1mo", "60m": "3mo"}
INTRA_OR_BARS = {"1m": 30, "5m": 6, "15m": 2, "30m": 1, "60m": 1}
INTRA_BARS_HOUR = {"1m": 60, "5m": 12, "15m": 4, "30m": 2, "60m": 1}


def _tz_bucharest(idx):
    idx = pd.to_datetime(idx)
    if idx.tz is not None:
        idx = idx.tz_convert("Europe/Bucharest").tz_localize(None)
    return idx


@st.cache_data(ttl=60, show_spinner=False)
def intraday_batch(tickers: tuple, interval: str, period: str):
    """Descarcă dintr-o singură cerere date intraday pentru mai multe tickere."""
    out = {}
    if not tickers:
        return out
    try:
        raw = yf.download(list(tickers), period=period, interval=interval, auto_adjust=True,
                          progress=False, threads=True, group_by="ticker")
    except Exception:
        return out
    if raw is None or raw.empty:
        return out
    need = ["Open", "High", "Low", "Close", "Volume"]
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                sub = raw[t] if t in set(raw.columns.get_level_values(0)) else raw.xs(t, axis=1, level=1)
            else:
                sub = raw
            sub = sub[need].apply(pd.to_numeric, errors="coerce").dropna(subset=["Close"]).copy()
        except Exception:
            continue
        if len(sub) < 40:
            continue
        sub.index = _tz_bucharest(sub.index)
        out[t] = sub
    return out


def intraday_setup(d, daily, interval, capital, risk_pct, allow_short=True):
    """
    Analizează sesiunea curentă (sau ultima disponibilă) și construiește planul de tranzacție.
    Returnează (info, f) sau (None, None) dacă datele nu ajung.
    Unitatea de risc U = max(2×mărimea medie a unei lumânări, 20% din ATR zilnic).
    """
    if d is None or len(d) < 40:
        return None, None
    f = intraday_features(d)
    f["ATRb"] = (f["High"] - f["Low"]).rolling(14).mean()
    dts = pd.Series(f.index.date, index=f.index)
    day = dts.iloc[-1]
    today, prev = f[dts == day], f[dts < day]
    if len(today) < 3 or prev.empty:
        return None, None
    pdt = pd.Series(prev.index.date, index=prev.index)
    prev_day = prev[pdt == pdt.iloc[-1]]
    pdh, pdl, pdc = float(prev_day["High"].max()), float(prev_day["Low"].min()), float(prev_day["Close"].iloc[-1])
    last = f.iloc[-1]
    price, vwap = float(last["Close"]), last["VWAP"]
    atrb = float(last["ATRb"]) if pd.notna(last["ATRb"]) else np.nan
    if not np.isfinite(atrb) or atrb <= 0 or pd.isna(vwap):
        return None, None
    vwap = float(vwap)
    gap = (float(today["Open"].iloc[0]) / pdc - 1) * 100
    tl = f.index[-1].time()
    cum = []
    for dd in sorted(set(prev.index.date))[-5:]:
        sub = prev[pdt == dd]
        sub = sub[[x <= tl for x in sub.index.time]]
        if len(sub):
            cum.append(float(sub["Volume"].sum()))
    rvol = float(today["Volume"].sum()) / float(np.mean(cum)) if cum and np.mean(cum) > 0 else np.nan

    atr_d, sma50, sma200, c_d, dvol = np.nan, np.nan, np.nan, np.nan, np.nan
    if daily is not None and len(daily) > 60:
        fd = add_features(daily)
        ld = fd.iloc[-1]
        atr_d, sma50, sma200, c_d = float(ld["ATR14"]), float(ld["SMA50"]), float(ld["SMA200"]), float(ld["Close"])
        dvol = float((daily["Close"] * daily["Volume"]).tail(20).mean())
    atr_pct_d = atr_d / price if np.isfinite(atr_d) else np.nan
    U = max(2.0 * atrb, 0.2 * atr_d) if np.isfinite(atr_d) else 2.0 * atrb

    n_or = INTRA_OR_BARS.get(interval, 6)
    or_ready = len(today) >= n_or
    orb = today.iloc[:n_or]
    orh, orl = float(orb["High"].max()), float(orb["Low"].min())
    day_hi, day_lo = float(today["High"].max()), float(today["Low"].min())
    ema9, ema21, rsi_v = float(last["EMA9"]), float(last["EMA21"]), float(last["RSI14"])

    def side(dr):
        s, why, broke = 0.0, [], False
        if np.isfinite(rvol):
            s += float(np.clip((rvol - 0.8) * 20, 0, 25))
            if rvol >= 1.5:
                why.append(f"volum relativ {rvol:.1f}×")
        g = gap * dr
        if g > 0:
            s += float(np.clip(g * 5, 0, 10))
        if g >= 1:
            why.append(f"gap {gap:+.1f}% în direcția setup-ului")
        if np.isfinite(atr_pct_d):
            s += float(np.clip((atr_pct_d * 100 - 0.8) * 5, 0, 15))
        if dr * (price - vwap) > 0:
            s += 10
            why.append("de partea bună a VWAP")
        if dr * (ema9 - ema21) > 0:
            s += 8
            why.append("EMA9/EMA21 aliniate")
        if (dr == 1 and 48 <= rsi_v <= 72) or (dr == -1 and 28 <= rsi_v <= 52):
            s += 7
        if or_ready:
            if (dr == 1 and price > orh) or (dr == -1 and price < orl):
                s += 15
                broke = True
                why.append("a ieșit din intervalul de deschidere")
            elif orl <= price <= orh:
                s += 5
        if np.isfinite(sma50) and np.isfinite(sma200):
            if (dr == 1 and c_d > sma50 > sma200) or (dr == -1 and c_d < sma50 < sma200):
                s += 10
                why.append("trend zilnic aliniat")
        ext = dr * (price - vwap) / U
        if ext > 1.5:
            s -= 10
            why.append("preț întins față de VWAP")
        return s, why, broke, ext

    cands = [(1,) + side(1)]
    if allow_short:
        cands.append((-1,) + side(-1))
    dr, score, why, broke, ext = max(cands, key=lambda x: x[1])
    score = float(np.clip(score, 0, 100))

    edge = orh if dr == 1 else orl
    setup, zlo, zhi, acum_flag = None, None, None, False
    if or_ready and broke and dr * (price - edge) <= 0.6 * U and (not np.isfinite(rvol) or rvol >= 1.2):
        setup = "Breakout din intervalul de deschidere"
        zlo, zhi = (edge - 0.15 * U, edge + 0.6 * U) if dr == 1 else (edge - 0.6 * U, edge + 0.15 * U)
    elif or_ready and broke:
        setup = "Breakout depășit — așteaptă retestul nivelului"
        zlo, zhi = (edge - 0.15 * U, edge + 0.4 * U) if dr == 1 else (edge - 0.4 * U, edge + 0.15 * U)
    elif dr * (price - vwap) > 0 and dr * (ema9 - ema21) > 0:
        setup = ("Continuare de trend — pullback la VWAP" if ext <= 1.0
                 else "Întins — așteaptă pullback la VWAP")
        zlo, zhi = (vwap - 0.25 * U, vwap + 0.5 * U) if dr == 1 else (vwap - 0.5 * U, vwap + 0.25 * U)

    info = {"direcție": "LONG" if dr == 1 else "SHORT", "scor": int(round(score)), "preț": price, "vwap": vwap,
            "rvol": rvol, "gap": gap, "atr_zi_%": atr_pct_d * 100 if np.isfinite(atr_pct_d) else np.nan,
            "orh": orh, "orl": orl, "or_ready": or_ready, "pdh": pdh, "pdl": pdl, "pdc": pdc,
            "rsi": rsi_v, "sesiune": day, "bara": f.index[-1], "dvol": dvol, "U": U, "motive": why,
            "plan": False, "setup": setup or "Fără setup clar (preț în jurul VWAP / fără direcție)"}
    if setup is None or score < 35:
        return info, f
    acum_flag = zlo <= price <= zhi
    entry = price if acum_flag else (zlo + zhi) / 2
    swing = float(f["Low"].tail(12).min()) if dr == 1 else float(f["High"].tail(12).max())
    struct = swing - dr * 0.1 * atrb
    dist = dr * (entry - struct)
    stop = struct if 0.6 * U <= dist <= 1.6 * U else entry - dr * U
    R = abs(entry - stop)
    if R <= 0:
        return info, f
    t1, t2, t3 = (entry + dr * R * k for k in (1, 2, 3))
    levels = [x for x in (pdh, day_hi, orh) if x > entry] if dr == 1 else [x for x in (pdl, day_lo, orl) if x < entry]
    resist = (min(levels) if dr == 1 else max(levels)) if levels else None
    shares = int((capital * risk_pct / 100) // R)
    realism = ""
    if np.isfinite(atr_d) and atr_d > 0 and (2 * R) / atr_d > 0.6:
        realism = "Țintă 2 e ambițioasă pentru o singură sesiune (peste 60% din mișcarea zilnică medie)."
    info.update(plan=True, zona_lo=zlo, zona_hi=zhi, în_zonă=bool(acum_flag), intrare=entry, stop=stop, R=R,
                t1=t1, t2=t2, t3=t3, rezistență=resist, acțiuni=shares, valoare=shares * entry,
                risc_bani=shares * R, realism=realism, bare_timp=INTRA_BARS_HOUR.get(interval, 12))
    return info, f


def intraday_scan(tickers, interval, capital, risk_pct, min_dvol, min_atr_pct, allow_short, max_n, progress=None):
    """Filtrează după lichiditate și volatilitate zilnică, apoi analizează intraday și ordonează după scor."""
    pre = []
    for k, t in enumerate(tickers):
        if progress:
            progress.progress(0.25 * (k + 1) / max(len(tickers), 1), text=f"Filtrez {t}…")
        df = get_prices(t)
        if df is None or len(df) < 120:
            continue
        price = float(df["Close"].iloc[-1])
        a = float(atr(df).iloc[-1])
        dvol = float((df["Close"] * df["Volume"]).tail(20).mean())
        if price <= 0 or dvol < min_dvol * 1e6 or (a / price * 100) < min_atr_pct:
            continue
        pre.append((t, dvol))
    pre.sort(key=lambda x: -x[1])
    sel = [t for t, _ in pre[:max_n]]
    meta = {"filtrate": len(pre), "analizate": len(sel), "total": len(tickers)}
    if not sel:
        return pd.DataFrame(), meta
    if progress:
        progress.progress(0.35, text=f"Descarc date intraday ({interval}) pentru {len(sel)} acțiuni…")
    data = intraday_batch(tuple(sel), interval, INTRA_PERIOD[interval])
    rows = []
    for k, t in enumerate(sel):
        if progress:
            progress.progress(0.5 + 0.5 * (k + 1) / len(sel), text=f"Analizez {t}…")
        d = data.get(t)
        if d is None:
            continue
        info, _ = intraday_setup(d, get_prices(t), interval, capital, risk_pct, allow_short)
        if info is None or not info["plan"]:
            continue
        rows.append({
            "ticker": t, "direcție": info["direcție"], "scor": info["scor"], "setup": info["setup"],
            "preț": round(info["preț"], 2), "în_zonă_acum": "✅" if info["în_zonă"] else "⏳ așteaptă",
            "zonă_intrare": f"{info['zona_lo']:.2f} – {info['zona_hi']:.2f}", "stop": round(info["stop"], 2),
            "țintă_1": round(info["t1"], 2), "țintă_2": round(info["t2"], 2), "țintă_3": round(info["t3"], 2),
            "acțiuni": info["acțiuni"], "RVOL": round(info["rvol"], 1) if np.isfinite(info["rvol"]) else np.nan,
            "gap_%": round(info["gap"], 1), "mișcare_zilnică_%": round(info["atr_zi_%"], 1) if np.isfinite(info["atr_zi_%"]) else np.nan,
            "motive": ", ".join(info["motive"]), "_sesiune": info["sesiune"], "_bara": info["bara"]})
    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.sort_values("scor", ascending=False).reset_index(drop=True)
        meta["sesiune"] = df_out["_sesiune"].mode().iloc[0]
        meta["ultima_bara"] = df_out["_bara"].max()
    return df_out, meta


def candle_chart(f, n=120, levels=None):
    x = f.tail(n).copy()
    x.index.name = "t"
    x = x.reset_index()
    x["col"] = np.where(x["Close"] >= x["Open"], "up", "down")
    color = alt.Color("col:N", scale=alt.Scale(domain=["up", "down"], range=["#26a69a", "#ef5350"]), legend=None)
    base = alt.Chart(x).encode(x=alt.X("t:T", title=None))
    rule = base.mark_rule().encode(
        y=alt.Y("Low:Q", scale=alt.Scale(zero=False), title="Preț"), y2="High:Q", color=color)
    bar = base.mark_bar(size=4).encode(y="Open:Q", y2="Close:Q", color=color)
    lines = alt.Chart(x.melt("t", ["VWAP", "EMA9", "EMA21"], var_name="serie", value_name="v").dropna()) \
        .mark_line().encode(
            x="t:T", y=alt.Y("v:Q", title=None, scale=alt.Scale(zero=False)),
            color=alt.Color("serie:N", scale=alt.Scale(domain=["VWAP", "EMA9", "EMA21"],
                                                       range=["#9c27b0", "#ff9800", "#2196f3"])))
    layers = [rule, bar, lines]
    for val, col in (levels or []):
        layers.append(alt.Chart(pd.DataFrame({"y": [float(val)]})).mark_rule(strokeDash=[5, 4], color=col)
                      .encode(y=alt.Y("y:Q", scale=alt.Scale(zero=False))))
    return alt.layer(*layers).resolve_scale(color="independent").properties(height=420)


def intraday_panel(sym, interval, capital=10000.0, risk_pct=0.5, allow_short=True, fmp_key=""):
    sym = sym.strip().upper()
    d = intraday(sym, interval, INTRA_PERIOD[interval])
    if d is None or len(d) < 30:
        st.warning("Nu am primit date intraday pentru acest ticker (piață închisă, ticker greșit sau fără date la acest interval).")
        return
    info, f = intraday_setup(d, get_prices(sym), interval, capital, risk_pct, allow_short)
    if info is None:
        f = intraday_features(d)
        st.info("Nu pot construi un plan complet (lipsesc zile anterioare sau volum). Arăt doar graficul.")
        st.altair_chart(candle_chart(f), use_container_width=True)
        return
    now = pd.Timestamp.now(tz="Europe/Bucharest").tz_localize(None)
    lag = (now - info["bara"]).total_seconds() / 60
    if lag > 30:
        st.info(f"Piața pare închisă sau datele sunt vechi: ultima bară e din {info['bara']:%d.%m %H:%M} "
                f"(acum {lag / 60:.1f} h). Analiza de mai jos e pe sesiunea din {info['sesiune']:%d.%m} — "
                "folosește-o ca pregătire pentru următoarea deschidere, nu ca semnal live.")
    else:
        st.caption(f"Date la zi (întârziere ≈ {max(lag, 0):.0f} min) · sesiunea {info['sesiune']:%d.%m}")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Ultimul preț", f"{info['preț']:.2f}", f"{(info['preț'] / info['pdc'] - 1) * 100:+.2f}% vs închidere ieri")
    m2.metric("VWAP", f"{info['vwap']:.2f}")
    m3.metric("Volum relativ", "–" if pd.isna(info["rvol"]) else f"{info['rvol']:.1f}×",
              help="Volumul de azi până acum, față de media ultimelor zile la aceeași oră. Peste 1,5 = activitate neobișnuită.")
    m4.metric("Gap deschidere", f"{info['gap']:+.1f}%")
    m5.metric("RSI 14", f"{info['rsi']:.0f}")
    calit = "foarte bun" if info["scor"] >= 70 else "bun" if info["scor"] >= 55 else "mediu" if info["scor"] >= 40 else "slab"
    st.markdown(f"### {'🟢' if info['direcție'] == 'LONG' else '🔴'} {info['direcție']} · {info['setup']} · "
                f"calitate {info['scor']}/100 ({calit})")
    levels = [(info["orh"], "#9e9e9e"), (info["orl"], "#9e9e9e"), (info["pdh"], "#ff9800"), (info["pdl"], "#ff9800")]
    if info["plan"]:
        levels += [(info["zona_lo"], "green"), (info["zona_hi"], "green"), (info["stop"], "red"),
                   (info["t1"], "#7b1fa2"), (info["t2"], "#7b1fa2"), (info["t3"], "#7b1fa2")]
        sign = 1 if info["direcție"] == "LONG" else -1
        pct = lambda x: (x / info["intrare"] - 1) * 100
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Zonă de intrare", f"{info['zona_lo']:.2f} – {info['zona_hi']:.2f}",
                  "✅ prețul e în zonă acum" if info["în_zonă"] else "⏳ prețul e în afara zonei — așteaptă",
                  delta_color="off")
        c2.metric("Stop-loss", f"{info['stop']:.2f}", f"{pct(info['stop']):+.2f}%", delta_color="inverse")
        c3.metric("Ținte 1 / 2 / 3", f"{info['t1']:.2f} / {info['t2']:.2f} / {info['t3']:.2f}",
                  f"{pct(info['t1']):+.2f}% / {pct(info['t2']):+.2f}% / {pct(info['t3']):+.2f}%", delta_color="off")
        c4.metric(f"Mărime poziție (risc {risk_pct:.1f}%)", f"{info['acțiuni']} acțiuni",
                  f"valoare ≈ {info['valoare']:,.0f} · risc ≈ {info['risc_bani']:,.0f}", delta_color="off")
        verb = "sub" if sign == 1 else "peste"
        st.markdown(
            f"**Plan:** intră în zona de intrare (ordin limită; nu urmări prețul dacă a plecat deja din zonă). "
            f"La **Țintă 1** vinde o treime și mută stopul la prețul de intrare; la **Țintă 2** încă o treime; "
            f"restul la **Țintă 3** sau cu stop urcat. Risc/câștig la Țintă 2 = 1 : 2.\n\n"
            f"**Invalidare:** ieși dacă o lumânare de {interval} închide {verb} VWAP ({info['vwap']:.2f}) sau dacă se atinge stopul. "
            f"**Ieșire la timp:** dacă după ~{info['bare_timp']} lumânări ({interval}) nu s-a atins Țintă 1, închide poziția; "
            "în orice caz, închide cu cel puțin 15 minute înainte de sfârșitul sesiunii (nu păstra poziții intraday peste noapte).")
        if info["rezistență"]:
            st.caption(f"Următorul nivel important în drumul spre țintă: {info['rezistență']:.2f} "
                       "(maxim/minim de ieri, al zilei sau al intervalului de deschidere) — acolo prețul poate ezita.")
        if info["realism"]:
            st.warning(info["realism"])
    else:
        st.info("Niciun plan de intrare acum: fie nu există un setup clar, fie scorul e prea mic. "
                "Răbdarea e și ea o decizie — nu forța tranzacții.")
    if info["motive"]:
        st.caption("De ce: " + ", ".join(info["motive"]))
    if not info["or_ready"]:
        st.caption("Intervalul de deschidere (prima jumătate de oră) încă se formează — scorul se va schimba.")
    st.altair_chart(candle_chart(f, levels=levels), use_container_width=True)
    st.caption("Linii: verde = zonă de intrare · roșu = stop · mov = ținte · gri = intervalul de deschidere · "
               "portocaliu = maximul/minimul de ieri · violet/portocaliu/albastru curbe = VWAP/EMA9/EMA21.")
    with st.expander("📰 Știri recente pentru acest simbol", expanded=False):
        news_panel(sym, fmp_key)


def main():
    st.set_page_config(page_title="Tranzacționare AI", layout="wide")
    st.title("⚡ Tranzacționare AI")

    if "custom_prices" not in st.session_state:
        st.session_state["custom_prices"] = load_custom_prices()

    sb = st.sidebar
    sb.header("Setări")
    chosen = sb.multiselect("Piețe", list(UNIVERSES), default=["SUA – large cap", "SUA – dividende"])
    custom = sb.text_area("Tickere proprii (separate prin virgulă)", "",
                          help="Format Yahoo Finance, ex: NVDA, SAP.DE, BTC-USD. Acțiunile BVB importate din CSV se scriu cu numele dat la import.")
    horizon = sb.number_input("Orizont de predicție (zile de tranzacționare)", min_value=2, max_value=250,
                              value=10, step=1,
                              help="Peste ~60 de zile, modelul are tot mai puține exemple independente de "
                                   "antrenare (ferestrele se suprapun), deci AUC-ul devine tot mai puțin "
                                   "de încredere — verifică mereu «Cât de bun este modelul» de mai sus.")
    target = sb.slider("Câștig țintă în orizont (%)", 2, 15, 5) / 100
    max_price = sb.number_input("Preț maxim (0 = fără filtru)", 0.0, value=0.0)
    use_ml = sb.checkbox("Folosește modelul ML", value=True)
    sb.divider()
    capital = sb.number_input("Capital total", 100.0, value=10000.0, step=500.0)
    risk_pct = sb.slider("Risc pe tranzacție (% din capital)", 0.25, 3.0, 1.0, 0.25)
    run = sb.button("🔍 Scanează", type="primary")

    sb.divider()
    sb.header("Integrare Investing.com (opțional)")
    fmp_key = sb.text_input(
        "Cheie API Financial Modeling Prep", type="password",
        help="Gratuită pe site.financialmodelingprep.com/developer/docs → Pricing → Free. "
             "O folosesc pentru calendarul de dividende global.")
    eco_embed = sb.text_area(
        "Cod widget „Economic Calendar” de la Investing.com", height=80,
        help="Mergi pe investing.com/webmaster-tools/economic-calendar, alege țările și "
             "aspectul dorit, apoi copiază tot codul generat și lipește-l aici.")
    if fmp_key:
        st.session_state["fmp_key"] = fmp_key
    if eco_embed:
        st.session_state["eco_embed"] = eco_embed

    tickers = []
    for name in chosen:
        tickers += UNIVERSES[name]
    tickers += [t.strip().upper() for t in custom.replace("\n", ",").split(",") if t.strip()]
    tickers += list(_custom().keys())
    tickers = list(dict.fromkeys(tickers))

    if run:
        if not tickers:
            st.warning("Alege cel puțin o piață sau introdu tickere.")
        else:
            trained = None
            if use_ml:
                with st.spinner("Descarc istoricul și antrenez modelul (prima rulare durează 1–3 minute)…"):
                    trained = train_model(tuple(t for t in tickers if t not in _custom()), horizon, target, "5y")
                if trained is None:
                    st.warning("Date insuficiente pentru model — folosesc doar scorul tehnic.")
            prog = st.progress(0.0)
            res = scan(tickers, trained, prog)
            prog.empty()
            st.session_state.update(res=res, trained=trained, capital=capital, div=None)
            if not res.empty:
                res.to_csv(OUT_CANDIDATES, index=False)
                log_semnale(res)

    res = st.session_state.get("res")
    trained = st.session_state.get("trained")

    if trained:
        m = trained["metrics"]
        with st.expander("🧪 Cât de bun este modelul ML al candidaților? (testat pe date nevăzute)"):
            st.caption(f"Calculat pe **{m['n_test_eff']} exemple independente** (nesuprapuse) din perioada de "
                       f"test — din {m['n_test']} zile disponibile acolo. La orizont mare, exemplele consecutive "
                       "se suprapun aproape complet, deci sunt rărite la un exemplu la fiecare "
                       f"{horizon} zile per acțiune, ca numerele de mai jos să nu fie umflate artificial.")
            if m["n_test_eff"] < 30 or pd.isna(m["auc"]):
                st.warning(f"Doar {m['n_test_eff']} exemple independente — prea puține pentru o concluzie de "
                           "încredere la acest orizont. Scade orizontul de predicție sau scanează mai multe "
                           "acțiuni, ca să ai o perioadă de test mai lungă.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("AUC", f"{m['auc']:.3f}", help="0.50 = aruncat cu banul")
                c2.metric("Rată de succes de bază", f"{m['base_rate']*100:.1f}%")
                c3.metric("Succes în top 10% semnale", f"{m['top_hit']*100:.1f}%")
                c4.metric("Randament mediu top 10%", f"{m['top_ret']*100:.2f}%",
                          delta=f"{(m['top_ret']-m['all_ret'])*100:+.2f}% vs medie")
                if not (m["auc"] > 0.55 and m["top_hit"] > m["base_rate"] * 1.15):
                    st.warning("Modelul nu arată un avantaj clar față de hazard. Folosește scorul tehnic ca filtru.")

    tabs = st.tabs(["🔍 Analiză acțiune", "🏆 Candidați", "⏱️ Intraday", "💼 Portofoliu", "⭐ Rating",
                    "💎 Ieftine", "💰 Dividende", "🏅 Topuri", "📐 Backtest", "🧠 Memorie", "🌍 Macro",
                    "📊 Grafic", "📋 Date"])
    (t_deep, t_cand, t_intra, t_port, t_rating, t_cheap, t_div, t_top, t_back, t_mem, t_macro, t_chart,
     t_all) = tabs

    have = res is not None and not res.empty
    view = None
    if have:
        view = res if max_price <= 0 else res[res["pret"] <= max_price]
    no_scan_msg = "Alege piețele din stânga și apasă **Scanează**."
    if res is not None and res.empty:
        st.error("Nu s-au putut descărca date. Verifică conexiunea și tickerele.")

    # ---------------- Candidați
    with t_cand:
        if not have:
            st.info(no_scan_msg)
        else:
            c1, c2 = st.columns([2, 1])
            only_now = c1.checkbox("Arată doar cei aflați acum în zonă bună de intrare / breakout", value=False)
            piata_sel = c2.selectbox("Piață", ["Toate", "SUA", "Europa", "BVB", "Crypto", "Altele"], key="cand_market")
            top = view.copy()
            top["decizie"] = top["scor_final"].apply(lambda s: decizie_din_scor(s)[0])
            if piata_sel != "Toate":
                top = top[top["ticker"].apply(market_of) == piata_sel]
            top_i = top[top["semnal"] == "Interesant"]
            if only_now:
                top_i = top_i[top_i["moment_intrare"].isin(["Zonă bună de intrare", "Breakout confirmat"])]
            shown = top_i if not top_i.empty else top.head(10)
            st.caption(f"{len(shown)} candidați afișați din {len(top)} analizați ({piata_sel}). "
                       "«decizie» e orientativă, calculată doar din scorul tehnic+ML — pentru decizia completă, "
                       "vezi tabul 🔍 Analiză acțiune.")
            st.dataframe(shown, use_container_width=True,
                         column_config={"prob_ML": st.column_config.ProgressColumn(
                             "prob_ML", min_value=0.0, max_value=1.0, format="%.2f")})

    # ---------------- Rating / scor global
    with t_rating:
        st.caption(
            "Scor compus din patru surse **gratuite și publice**, calculat aici, nu copiat de pe un site cu plată:\n"
            "- **Rating tehnic** — calculat local din prețuri, cu aceeași logică descrisă public de TradingView "
            "(numără semnalele de cumpărare/vânzare din 6 medii mobile și 5 oscilatoare).\n"
            "- **Scor fundamental** — P/E, ROE, marjă de profit, datorie și creștere a veniturilor, din Yahoo "
            "Finance, comparate relativ (percentile) cu restul acțiunilor scanate.\n"
            "- **Consens analiști** — recomandarea medie (Strong Buy…Strong Sell) și prețul țintă mediu, agregate "
            "de Yahoo Finance din rapoartele publice ale caselor de brokeraj.\n"
            "- **Momentum** — scorul tehnic deja calculat în tabul Candidați.\n\n"
            "Nu reproduce Seeking Alpha Quant, InvestingPro, TipRanks, Simply Wall St sau WallStreetZen — acelea "
            "sunt produse cu plată sau care interzic extragerea automată a datelor lor.")
        if not have:
            st.info(no_scan_msg)
        else:
            if st.button("⭐ Calculează ratingul global"):
                prog = st.progress(0.0)
                st.session_state["rating"] = rating_scan(view, prog)
                prog.empty()
            rt = st.session_state.get("rating")
            if rt is not None and rt.empty:
                st.warning("Nu am putut calcula ratinguri pentru lista curentă.")
            elif rt is not None:
                st.dataframe(rt, use_container_width=True, column_config={
                    "scor_global": st.column_config.ProgressColumn(
                        "scor_global", min_value=0.0, max_value=100.0, format="%.0f")})
                st.caption("`datorie_capital` = datorie/capital propriu (debt/equity); valori lipsă apar ca „–” "
                           "când Yahoo nu are date pentru acea companie (frecvent la acțiuni mai mici).")

    # ---------------- Ieftine cu scor mare
    with t_cheap:
        st.caption(
            "Combină **toate** scorurile deja calculate — tehnic, predicție ML, fundamental, consens analiști — "
            "pentru acțiunile sub un preț pe care îl alegi. Un preț mic **nu înseamnă automat ieftin sau bun**: "
            "poate fi mic pentru că afacerea merge prost. De asta, lista de mai jos arată și o coloană de riscuri — "
            "citește-o înainte să te uiți la scor.")
        if not have:
            st.info(no_scan_msg)
        else:
            max_p = st.slider("Preț maxim", 1.0, 100.0, 20.0, 1.0, key="cheap_max_price")
            n_sub = int((view["pret"] <= max_p).sum())
            st.write(f"{n_sub} acțiuni din lista scanată sunt sub {max_p:.0f}.")
            if n_sub == 0:
                st.info("Nicio acțiune din scanarea curentă nu e sub acest preț. Mărește pragul sau adaugă "
                        "mai multe piețe/tickere din bara din stânga.")
            elif st.button("💎 Găsește cele mai bune acțiuni ieftine"):
                prog = st.progress(0.0)
                st.session_state["cheap"] = cheap_scan(view, max_p, prog)
                prog.empty()
            cp = st.session_state.get("cheap")
            if cp is not None and cp.empty:
                st.warning("Nu am putut calcula scoruri pentru acțiunile sub acest preț.")
            elif cp is not None:
                st.dataframe(cp, use_container_width=True, column_config={
                    "scor_combo": st.column_config.ProgressColumn(
                        "scor_combo", min_value=0.0, max_value=100.0, format="%.0f"),
                    "prob_ML": st.column_config.ProgressColumn(
                        "prob_ML", min_value=0.0, max_value=1.0, format="%.2f")})
                st.caption("`scor_combo` = media dintre scorul tehnic+ML (Candidați) și scorul global "
                           "(Rating). Verifică mereu `riscuri` — un scor mare cu riscuri multe înseamnă "
                           "că modelul vede un trend bun, dar afacerea are probleme de fond.")

    # ---------------- Analiză acțiune (pagina principală pentru o singură acțiune)
    with t_deep:
        st.caption("Alege o acțiune și vezi tot într-un loc: decizie, analiză tehnică (toți indicatorii), fundamentală, "
                   "plan cu stop-loss și take-profit, intraday, știri, macro și predicție AI.")
        opts = sorted(set((view["ticker"].tolist() if have else []) + list(_custom()) + BVB_TICKERS))
        c1, c2, c3 = st.columns([2, 2, 1])
        pick = c1.selectbox("Din lista scanată / BVB", opts, key="deep_pick")
        manual = c2.text_input("...sau scrie orice ticker (NVDA, SAP.DE, TLV)", "", key="deep_manual").strip().upper()
        c3.markdown("&nbsp;")
        if c3.button("🔍 Analizează", key="deep_go", type="primary"):
            st.session_state["deep_active"] = manual if manual else pick
        ticker = st.session_state.get("deep_active")
        if not ticker:
            st.info("Alege un ticker și apasă **Analizează**.")
        else:
            df = get_prices(ticker, "10y")
            if df is None:
                st.error(f"Nu am găsit date de preț pentru {ticker} (nici pe Yahoo, nici la BVB). "
                         "Verifică simbolul sau importă manual un CSV în tabul Portofoliu.")
            else:
                f = add_features(df)
                last = f.iloc[-1]
                px_now = float(last["Close"])
                chg_d = (px_now / float(f["Close"].iloc[-2]) - 1) * 100
                tscore, twhy = tech_score(last)
                tv_pct, tv_label = tv_style_rating(last)
                fund = yahoo_fundamentals(ticker)
                mkt = market_of(ticker)
                cur = " RON" if mkt == "BVB" else ""
                plan = trade_plan(f)
                ind_t, ind_s = all_indicators(df)
                ms_deep = macro_sensitivity(ticker)
                ai_pair = st.session_state.get("deep_ai", {}).get(ticker)

                parts = [("Rating tehnic", tv_pct)]
                if fund.get("recomandare_analiști") in REC_MAP:
                    parts.append(("Analiști", float(REC_MAP[fund["recomandare_analiști"]])))
                if ms_deep.get("ok"):
                    parts.append(("Macro", float(ms_deep["scor"])))
                if ai_pair is not None and not ai_pair[0].empty:
                    parts.append(("Predicție AI", {"Optimist": 90.0, "Mixt": 50.0, "Prudent": 15.0}.get(
                        ai_pair[0].iloc[0]["imagine_generala"], 50.0)))
                final_score = float(np.mean([p[1] for p in parts]))
                dec_label, dec_reason = decizie_din_scor(final_score)

                st.markdown(f"## {ticker} · {mkt} · {px_now:.2f}{cur}  ({chg_d:+.2f}% ultima zi)")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Decizie", dec_label, f"scor {final_score:.0f}/100", delta_color="off")
                m2.metric("Rating tehnic", tv_label, f"{ind_s['cumpără']} cumpără · {ind_s['vinde']} vinde", delta_color="off")
                m3.metric("Analiști", REC_LABEL.get(fund.get("recomandare_analiști"), "–"))
                m4.metric("Macro", ms_deep.get("verdict", "–") if ms_deep.get("ok") else "–")

                s_sum, s_tech, s_fund, s_plan, s_intra, s_news, s_mkt = st.tabs(
                    ["📌 Rezumat", "📈 Tehnic", "🏦 Fundamental", "🎯 Stop-loss & Take-profit", "⏱️ Intraday",
                     "📰 Știri", "🌍 Piață, macro & AI"])

                with s_sum:
                    st.markdown(f"### {dec_label}")
                    st.write(dec_reason)
                    st.dataframe(pd.DataFrame([{"componentă": n, "scor (0–100)": round(v)} for n, v in parts]),
                                 use_container_width=True, hide_index=True)
                    if ai_pair is None:
                        st.caption("Decizia nu include încă predicția AI — rulează-o în secțiunea «Piață, macro & AI».")
                    lv = ([(plan["zona_lo"], "green"), (plan["zona_hi"], "green"), (plan["stop"], "red"),
                           (plan["t1"], "purple"), (plan["t2"], "purple"), (plan["t3"], "purple")]
                          if plan.get("valid") else [(px_now - 2 * float(last["ATR14"]), "red")])
                    ch = price_chart(ticker, lv)
                    if ch is not None:
                        st.altair_chart(ch, use_container_width=True)
                    st.caption("Verde = zonă de intrare · roșu = stop-loss · mov = ținte (Take-profit 1/2/3).")
                    if plan.get("valid"):
                        st.write(f"**Plan rapid:** intrare {plan['zona_lo']:.2f}–{plan['zona_hi']:.2f}{cur} · stop {plan['stop']:.2f} · "
                                 f"ținte {plan['t1']:.2f} / {plan['t2']:.2f} / {plan['t3']:.2f}. Detalii în «Stop-loss & Take-profit».")
                    else:
                        st.write(f"**Plan:** {plan.get('semnal', 'fără plan')} — nu intra acum.")
                    st.caption("Nu e sfat financiar — verifică compania și folosește mereu stop-loss.")

                with s_tech:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Scor tehnic", f"{tscore}/100")
                    c2.metric("RSI 14", f"{last['RSI14']:.1f}")
                    c3.metric("Față de SMA200", f"{last['dist_sma200'] * 100:+.1f}%")
                    c4.metric("Față de maxim 52s", f"{last['dist_high'] * 100:+.1f}%")
                    st.write(f"**Rating tehnic (stil TradingView): {tv_label}** ({tv_pct}% semnale de cumpărare) · "
                             f"**Verdict toți indicatorii: {ind_s['verdict']}**")
                    st.caption(f"Motive scor tehnic: {twhy}")
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Semnale de cumpărare", ind_s["cumpără"])
                    k2.metric("Neutre", ind_s["neutru"])
                    k3.metric("Semnale de vânzare", ind_s["vinde"])
                    ind_show = ind_t.copy()
                    ind_show["semnal"] = ind_show["semnal"].map({"Cumpără": "🟢 Cumpără", "Vinde": "🔴 Vinde", "Neutru": "⚪ Neutru"})
                    st.dataframe(ind_show, use_container_width=True, hide_index=True)
                    lvl = key_levels(df)
                    lvl_df = pd.DataFrame({"nivel": list(lvl.keys()), "preț": [round(v, 2) for v in lvl.values()]})
                    lvl_df["față de preț"] = [f"{(v / px_now - 1) * 100:+.1f}%" for v in lvl.values()]
                    st.markdown("**Niveluri cheie (suport / rezistență)**")
                    st.dataframe(lvl_df.sort_values("preț", ascending=False), use_container_width=True, hide_index=True)
                    st.caption("Fiecare indicator votează după reguli standard (de exemplu RSI sub 30 = supravândut). "
                               "Un singur indicator nu spune mult; contează cât de mulți se aliniază.")

                with s_fund:
                    if mkt == "BVB":
                        st.info("Yahoo Finance nu are date fundamentale pentru acțiuni BVB — verifică raportările companiei pe bvb.ro.")
                    if fund.get("nume"):
                        st.write(f"**{fund['nume']}** · {fund.get('sector') or '–'}  ·  capitalizare {_big(fund.get('capitalizare'))}")
                    def fm(v, suf="", nd=1):
                        return f"{v:.{nd}f}{suf}" if v is not None and pd.notna(v) else "–"
                    a1, a2, a3, a4 = st.columns(4)
                    a1.metric("P/E", fm(fund["PE"]))
                    a2.metric("P/E estimat", fm(fund["fwd_PE"]))
                    a3.metric("P/B", fm(fund.get("P/B")))
                    a4.metric("Datorie / capital", fm(fund["datorie_capital"], nd=0))
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("ROE", fm(fund["ROE_%"], "%"))
                    b2.metric("ROA", fm(fund.get("ROA_%"), "%"))
                    b3.metric("Marjă de profit", fm(fund["marja_profit_%"], "%"))
                    b4.metric("Creștere venituri", fm(fund["creștere_venituri_%"], "%"))
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Venituri (12 luni)", _big(fund.get("venituri")))
                    c2.metric("Profit net", _big(fund.get("profit_net")))
                    c3.metric("Cash flow liber", _big(fund.get("cash_flow_liber")))
                    st.write(f"**Verdict analiști:** {REC_LABEL.get(fund['recomandare_analiști'], '–')} "
                             f"({int(fund['nr_analiști']) if pd.notna(fund['nr_analiști']) else 0} analiști) · "
                             f"preț țintă mediu: {fm(fund['preț_țintă_mediu'], nd=2)}"
                             + (f" ({fund['potențial_țintă_%']:+.1f}% față de prețul de acum)" if pd.notna(fund["potențial_țintă_%"]) else ""))
                    st.caption("Date Yahoo Finance; pot lipsi sau avea întârzieri, mai ales la companii mici. "
                               "Un P/E mic nu înseamnă automat ieftin — compară cu sectorul și cu creșterea.")

                with s_plan:
                    render_trade_plan(plan, capital, risk_pct)
                    if plan.get("valid"):
                        shares = int((capital * risk_pct / 100) // plan["R"]) if plan["R"] > 0 else 0
                        rws = [("Stop-loss", plan["stop"]), ("Take-profit 1", plan["t1"]),
                               ("Take-profit 2", plan["t2"]), ("Take-profit 3", plan["t3"])]
                        st.dataframe(pd.DataFrame([{
                            "nivel": n, "preț": round(p, 2), "față de intrare": f"{(p / plan['mid'] - 1) * 100:+.1f}%",
                            f"profit / pierdere ({shares} acțiuni)": f"{(p - plan['mid']) * shares:+,.0f}"} for n, p in rws]),
                            use_container_width=True, hide_index=True)
                        st.caption("Calculat pentru capitalul și riscul din bara din stânga, cu intrarea la mijlocul zonei.")
                    with st.expander("Deții deja această acțiune? Vezi ce să faci cu ea"):
                        h1, h2, h3 = st.columns(3)
                        bp = h1.number_input("Prețul tău de cumpărare", 0.0, value=0.0, key="deep_bp")
                        bq = h2.number_input("Cantitate", 0.0, value=0.0, key="deep_bq")
                        bd = h3.text_input("Data cumpărării (AAAA-LL-ZZ, opțional)", "", key="deep_bd")
                        if bp > 0:
                            row = pd.DataFrame([{"ticker": ticker, "pret_cumparare": bp, "cantitate": bq if bq > 0 else np.nan,
                                                 "data_cumparare": pd.to_datetime(bd, errors="coerce") if bd else pd.NaT}])
                            st.dataframe(analyze_portfolio(row), use_container_width=True, hide_index=True)
                            st.caption("«stop_curent» urcă odată cu prețul (trailing) și nu coboară; «VINDE» apare când stopul "
                                       "sau trendul de lungă durată sunt rupte.")

                with s_intra:
                    if mkt == "BVB":
                        st.info("Datele intraday BVB nu sunt disponibile gratuit prin Yahoo. Folosește graficul TradingView de mai jos "
                                "(poate avea întârziere) sau datele brokerului tău.")
                    else:
                        i1, i2, i3, i4 = st.columns(4)
                        iv = i1.selectbox("Interval", ["1m", "5m", "15m", "30m", "60m"], index=1, key="deep_iv")
                        risk_ii = i2.number_input("Risc / tranzacție (% capital)", 0.1, 2.0, 0.5, 0.1, key="deep_risk_i")
                        short_ii = i3.checkbox("Include SHORT (CFD)", value=False, key="deep_short")
                        rf = i4.selectbox("Actualizare", ["Oprită", "30 s", "60 s", "120 s"], key="deep_rf")
                        fk_i = st.session_state.get("fmp_key", "")
                        if hasattr(st, "fragment"):
                            ev = None if rf == "Oprită" else int(rf.split()[0])
                            st.fragment(run_every=ev)(intraday_panel)(ticker, iv, capital, risk_ii, short_ii, fk_i)
                        else:
                            st.button("🔄 Actualizează", key="deep_refresh_btn")
                            intraday_panel(ticker, iv, capital, risk_ii, short_ii, fk_i)
                    if st.checkbox("Arată graficul live TradingView", value=False, key="deep_tv"):
                        tv_s = f"BVB:{ticker}" if mkt == "BVB" else to_tv_symbol(ticker)
                        tv_widget("embed-widget-advanced-chart.js", {
                            "autosize": True, "symbol": tv_s, "interval": "15", "timezone": "Europe/Bucharest",
                            "theme": "light", "style": "1", "locale": "ro", "allow_symbol_change": True,
                            "studies": ["STD;VWAP"], "support_host": "https://www.tradingview.com"}, 560)

                with s_news:
                    fk = st.session_state.get("fmp_key", "")
                    if hasattr(st, "fragment"):
                        st.fragment(run_every=120)(news_panel)(ticker, fk)
                    else:
                        news_panel(ticker, fk)

                with s_mkt:
                    st.markdown("**Indici bursieri relevanți (variație pe 1 an)**")
                    idx_list = RELEVANT_INDICES.get(mkt, [])
                    if not idx_list:
                        st.caption("Niciun indice de referință disponibil pentru această piață.")
                    else:
                        stock_1y = (px_now / float(f["Close"].iloc[max(0, len(f) - 252)]) - 1) * 100
                        rows = [{"instrument": ticker, "variație 1 an": f"{stock_1y:+.1f}%"}]
                        for name, sym in idx_list:
                            idf = get_prices(sym, "2y")
                            if idf is None:
                                rows.append({"instrument": name, "variație 1 an": "indisponibil"})
                                continue
                            chg = (idf["Close"].iloc[-1] / idf["Close"].iloc[max(0, len(idf) - 252)] - 1) * 100
                            rows.append({"instrument": name, "variație 1 an": f"{chg:+.1f}%"})
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                        st.caption("O acțiune care crește mai puțin decât indicele ei, deși indicele urcă, are forță relativă slabă.")
                    _, msc, mreg = macro_snapshot()
                    if mreg:
                        st.info(f"Regim macro SUA: **{mreg}** (scor {msc:+d}) — detalii în tabul Macro.")
                    st.markdown("**Sensibilitatea acțiunii la dobânzi, inflație și sentiment**")
                    render_macro_detail(ms_deep)
                    st.markdown("**Predicție AI (scurt / mediu / lung)**")
                    if st.button("🤖 Rulează predicția AI pentru această acțiune", key="deep_ai_btn"):
                        prog = st.progress(0.0)
                        res_ai = ai_forecast([ticker], prog)
                        prog.empty()
                        st.session_state.setdefault("deep_ai", {})[ticker] = res_ai
                        (getattr(st, "rerun", None) or st.experimental_rerun)()
                    if ai_pair is not None:
                        ai_df, ai_met = ai_pair
                        if ai_df.empty:
                            st.caption("Nu am putut calcula o predicție AI (date insuficiente).")
                        else:
                            st.dataframe(ai_df, use_container_width=True, hide_index=True)
                            st.dataframe(ai_met, use_container_width=True, hide_index=True)
                            if mkt == "BVB":
                                st.caption("Modelul AI e antrenat pe acțiuni din SUA — pentru BVB are mai puțină încredere.")
                    else:
                        st.caption("Prima rulare antrenează trei modele și poate dura câteva minute; apoi rezultatul rămâne în decizie.")

    # ---------------- Topuri sintetice
    with t_top:
        st.caption("Rezumat rapid din analizele deja rulate în celelalte taburi. Dacă o secțiune lipsește, "
                   "înseamnă că nu ai rulat încă acel calcul (scanare, rating, dividende sau acțiuni ieftine).")
        rt = st.session_state.get("rating")
        dv = st.session_state.get("div")
        cp = st.session_state.get("cheap")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🚀 Top 5 Recomandate")
            if rt is not None and not rt.empty:
                st.dataframe(rt.head(5)[["ticker", "nota", "scor_global", "rating_tehnic", "analiști_verdict"]],
                             use_container_width=True, hide_index=True)
            elif have:
                st.dataframe(view.sort_values("scor_final", ascending=False)
                             .head(5)[["ticker", "scor_final", "moment_intrare"]],
                             use_container_width=True, hide_index=True)
                st.caption("Rating complet indisponibil — arăt topul după scorul tehnic+ML. "
                           "Rulează ⭐ Rating pentru topul complet.")
            else:
                st.info("Scanează mai întâi în tabul Candidați.")

            st.markdown("#### 📈 Top 5 Predicții AI (scor ML)")
            if have and view["prob_ML"].notna().any():
                st.dataframe(view.sort_values("prob_ML", ascending=False)
                             .head(5)[["ticker", "prob_ML", "pret", "moment_intrare"]],
                             use_container_width=True, hide_index=True)
            else:
                st.info("Activează „Folosește modelul ML” în bara din stânga și scanează din nou.")

            st.markdown("#### 🏅 Top 5 Dividende")
            if dv is not None and not dv.empty:
                st.dataframe(dv.head(5)[["ticker", "randament_%", "scor_dividend", "ex_dividend"]],
                             use_container_width=True, hide_index=True)
            else:
                st.info("Rulează „Analizează dividende” în tabul 💰 Dividende.")

        with c2:
            st.markdown("#### 💎 Top 5 Ieftine")
            if cp is not None and not cp.empty:
                st.dataframe(cp.head(5)[["ticker", "pret", "scor_combo", "nota"]],
                             use_container_width=True, hide_index=True)
            else:
                st.info("Rulează căutarea în tabul 💎 Ieftine.")

            st.markdown("#### 🇪🇺 Top 5 Europa")
            if have:
                eu = view[view["ticker"].apply(market_of) == "Europa"].sort_values("scor_final", ascending=False)
                if not eu.empty:
                    st.dataframe(eu.head(5)[["ticker", "pret", "scor_final", "moment_intrare"]],
                                 use_container_width=True, hide_index=True)
                else:
                    st.info("Nicio acțiune europeană în scanarea curentă — bifează „Europa” în stânga.")
            else:
                st.info("Scanează mai întâi în tabul Candidați.")

            st.markdown("#### 🇷🇴 Top 5 BVB")
            if have:
                bv = view[view["ticker"].apply(market_of) == "BVB"].sort_values("scor_final", ascending=False)
                if not bv.empty:
                    st.dataframe(bv.head(5)[["ticker", "pret", "scor_final", "moment_intrare"]],
                                 use_container_width=True, hide_index=True)
                else:
                    st.info("Nicio acțiune BVB în scanarea curentă — bifează „🇷🇴 BVB – principal” în stânga.")
            else:
                st.info("Scanează mai întâi în tabul Candidați.")

    # ---------------- Backtest strategie
    with t_back:
        st.caption(
            "**Ce testează**: de fiecare dată când, în istoric, scorul tehnic al unei acțiuni a atins pragul "
            "de cumpărare, aplicația simulează o tranzacție cu aceleași reguli de stop-loss și țintă pe care "
            "le vezi deja în Candidați, și urmărește ce s-ar fi întâmplat. **Ce NU testează**: predicția ML "
            "(aceea ar avea nevoie de reantrenare la fiecare punct istoric, ca să nu vadă viitorul — mult "
            "mai lent; dacă vrei, pot construi și asta separat). Rezultatele de mai jos sunt despre scorul "
            "tehnic simplu (tendință + RSI + volum), care nu se schimbă în timp, deci poate fi testat corect.")
        if not have:
            st.info(no_scan_msg)
        else:
            c1, c2, c3 = st.columns(3)
            hold_days = c1.number_input("Țin maxim (zile)", 3, 120, 20, 1)
            prag = c2.slider("Prag scor pentru semnal BUY", 50, 90, 70, 5)
            n_tick = c3.slider("Pe câte acțiuni (din cele scanate)", 3, min(40, len(view)), min(15, len(view)))
            if st.button("📐 Rulează backtest"):
                prog = st.progress(0.0)
                trades, summary = backtest_scan(view["ticker"].head(n_tick).tolist(), hold_days, prag, prog)
                prog.empty()
                st.session_state["backtest"] = (trades, summary)
            res_bt = st.session_state.get("backtest")
            if res_bt is not None:
                trades, summary = res_bt
                if trades.empty:
                    st.warning("Niciun semnal de cumpărare nu a apărut în istoricul acestor acțiuni, la acest "
                               "prag — încearcă un prag mai mic sau mai multe acțiuni.")
                else:
                    wr = (trades["randament_%"] > 0).mean() * 100
                    avg = trades["randament_%"].mean()
                    cum = ((1 + trades["randament_%"] / 100).prod() - 1) * 100
                    bh_avg = summary["buy_&_hold_%"].mean()
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Tranzacții simulate", len(trades))
                    m2.metric("Rată de succes", f"{wr:.0f}%")
                    m3.metric("Randament mediu / tranzacție", f"{avg:+.2f}%")
                    m4.metric("Randament cumulat (compus)", f"{cum:+.1f}%",
                             delta=f"{cum - bh_avg:+.1f}% vs buy&hold mediu")
                    st.altair_chart(equity_curve_chart(trades), use_container_width=True)
                    st.caption("Graficul presupune că ai fi pus aceeași sumă în fiecare tranzacție, una după "
                               "alta (nu simultan) — e o simplificare, nu o simulare de portofoliu real cu "
                               "mai multe poziții deodată.")
                    st.markdown("**Pe fiecare acțiune**")
                    st.dataframe(summary.sort_values("randament_cumulat_%", ascending=False),
                                use_container_width=True, hide_index=True)
                    st.markdown("**Toate tranzacțiile simulate**")
                    st.dataframe(trades, use_container_width=True, hide_index=True)
                    if wr < 45 or cum < bh_avg:
                        st.warning("Pe acest eșantion, strategia nu bate clar simpla cumpărare-și-ținere. "
                                   "Nu generaliza dintr-un singur test — încearcă praguri și perioade diferite.")

    # ---------------- Memorie de piață
    with t_mem:
        st.caption(
            "De fiecare dată când apeși **🔍 Scanează**, agentul își notează singur, într-un fișier local "
            f"(`{LOG_FILE}`), ce decizie a dat pentru fiecare acțiune și la ce preț. După câteva zile, poți "
            "verifica aici dacă deciziile lui chiar s-au confirmat — e felul lui de a ține minte piața, "
            "nu doar de a o prezice o singură dată.")
        if not os.path.exists(LOG_FILE):
            st.info("Jurnalul e gol — apasă „Scanează” cel puțin o dată; de azi încolo, fiecare scanare "
                    "adaugă un rând nou aici, automat.")
        else:
            min_days = st.slider("Verifică semnalele mai vechi de (zile)", 3, 60, 10, key="mem_days")
            if st.button("🧠 Verifică jurnalul"):
                ev, log_raw = evaluate_log(min_days)
                st.session_state["mem_eval"] = (ev, log_raw)
            res_mem = st.session_state.get("mem_eval")
            if res_mem is not None:
                ev, log_raw = res_mem
                st.markdown(f"**{len(log_raw)}** semnale în jurnal în total, din **{log_raw['ticker'].nunique()}** "
                           "acțiuni diferite.")
                if ev.empty:
                    st.info(f"Niciun semnal mai vechi de {min_days} zile încă — revino peste câteva zile.")
                else:
                    buys = ev[ev["decizie"] == "🟢 BUY"]
                    avoids = ev[ev["decizie"] == "🔴 AVOID"]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Semnale BUY verificate", len(buys),
                             f"{buys['corect'].mean()*100:.0f}% confirmate" if len(buys) else None)
                    c2.metric("Semnale AVOID verificate", len(avoids),
                             f"{avoids['corect'].mean()*100:.0f}% confirmate" if len(avoids) else None)
                    c3.metric("Randament mediu după BUY", f"{buys['randament_%'].mean():+.2f}%" if len(buys) else "–")
                    st.dataframe(ev.sort_values("data_semnal", ascending=False), use_container_width=True,
                                hide_index=True)
                    st.caption("„Confirmate” = prețul chiar a crescut după un semnal BUY (sau a scăzut/stagnat "
                               "după un AVOID). Cu puține semnale verificate, procentul poate varia mult de la "
                               "o săptămână la alta — are sens doar pe termen mai lung, cu zeci de semnale.")

    # ---------------- Dividende
    with t_div:
        st.subheader("🌍 Calendar de dividende global (toate acțiunile, nu doar cele scanate)")
        fmp_key = st.session_state.get("fmp_key", "")
        if not fmp_key:
            st.info(
                "Pentru un calendar complet de dividende (nu doar pentru acțiunile scanate), ai nevoie de o "
                "cheie API gratuită de la Financial Modeling Prep:\n\n"
                "1. Intră pe **site.financialmodelingprep.com/developer/docs** → Pricing → planul **Free**.\n"
                "2. Creează cont, copiază cheia API.\n"
                "3. Lipește cheia în bara din stânga, la „Integrare Investing.com”.\n\n"
                "Planul gratuit are o limită zilnică de cereri — o cerere pe zi pentru acest calendar e suficientă.")
        else:
            days_ahead = st.slider("Zile înainte (până la ~6 luni)", 7, 180, 30, key="fmp_days")
            if st.button("📅 Încarcă calendarul de dividende"):
                cal, err = fmp_dividend_calendar(fmp_key, days_ahead)
                st.session_state["fmp_cal"] = (cal, err)
            res_cal = st.session_state.get("fmp_cal")
            if res_cal is not None:
                cal, err = res_cal
                if err:
                    st.error(f"Nu am putut încărca calendarul: {err}")
                elif cal.empty:
                    st.info("Niciun eveniment în intervalul ales.")
                else:
                    q = st.text_input("Filtrează după ticker (opțional)", "").strip().upper()
                    shown = cal[cal["ticker"].str.contains(q)] if q else cal
                    st.dataframe(shown, use_container_width=True)
                    st.caption(f"{len(cal)} evenimente găsite · sursă: Financial Modeling Prep.")
        st.caption("Calendarul complet, cu filtre avansate, rămâne disponibil și direct pe "
                   "[Investing.com](https://www.investing.com/dividends-calendar/).")
        st.divider()
        st.subheader("🔎 Dividende pentru acțiunile scanate (analiză de calitate)")
        if not have:
            st.info(no_scan_msg)
        else:
            st.write("Verific istoricul de dividende pentru acțiunile scanate "
                     "(prima rulare poate dura câteva minute; apoi rezultatele se rețin 6 ore).")
            if st.button("💰 Analizează dividende"):
                prog = st.progress(0.0)
                st.session_state["div"] = dividend_scan(view, prog)
                prog.empty()
            dv = st.session_state.get("div")
            if dv is not None and dv.empty:
                st.warning("Nu am găsit acțiuni cu dividende în lista scanată. Încearcă piața «SUA – dividende».")
            elif dv is not None:
                f1, f2, f3 = st.columns(3)
                min_y = f1.slider("Randament minim (%)", 0.0, 10.0, 1.5, 0.5)
                max_po = f2.slider("Payout maxim (%)", 30, 150, 90, 5)
                days = f3.slider("Ex-dividend în următoarele … zile (0 = oricând)", 0, 90, 0, 5)
                sel = dv[(dv["randament_%"] >= min_y) &
                         (dv["payout"].isna() | (dv["payout"] * 100 <= max_po))]
                if days > 0:
                    sel = sel[sel["zile_pana_ex"].between(0, days)]
                st.dataframe(sel, use_container_width=True)
                st.caption("`ex_estimat = True` înseamnă că data e estimată din ritmul istoric, nu confirmată. "
                           "Verifică data reală la companie, la broker sau în calendarul Investing.com.")
                st.info(
                    "**Cum se folosește:** pentru dividend trebuie să deții acțiunea **înainte** de data ex-dividend. "
                    "În ziua ex-dividend prețul scade de regulă cu aproximativ valoarea dividendului, deci nu "
                    "este câștig «gratuit». Alege companii cu dividend stabil și în creștere "
                    "(`ani_fara_reducere`, payout mic) și cumpără-le când `moment_intrare` arată zonă bună.")

    # ---------------- Portofoliu
    with t_port:
        st.write("Introdu ce ai cumpărat. Primești semnale **ține / ia profit / vinde** și o **predicție AI** "
                 "pe termen scurt, mediu și lung.")
        if "pf" not in st.session_state:
            st.session_state["pf"] = load_portfolio()
        edited = st.data_editor(
            st.session_state["pf"], num_rows="dynamic", use_container_width=True, key="pf_editor",
            column_config={
                "ticker": st.column_config.TextColumn("Ticker (Yahoo)"),
                "pret_cumparare": st.column_config.NumberColumn("Preț cumpărare", min_value=0.0),
                "cantitate": st.column_config.NumberColumn("Cantitate", min_value=0.0),
                "data_cumparare": st.column_config.DateColumn("Data cumpărării"),
            })
        if st.button("💾 Salvează portofoliul"):
            edited.to_csv(PORT_FILE, index=False)
            st.session_state["pf"] = edited
            st.success(f"Salvat în {PORT_FILE}")

        with st.expander("📥 Adaugă acțiuni care nu există pe Yahoo (ex. Bursa de Valori București)"):
            st.write("Descarcă istoricul de preț (CSV) de la BVB sau de la brokerul tău și importă-l aici. "
                     "Trebuie să conțină cel puțin data și prețul de închidere, ~1 an de zile de tranzacționare sau mai mult "
                     "(coloanele Deschidere/Max/Min/Volum sunt opționale). Apoi scrie același nume în tabelul de mai sus.")
            nm = st.text_input("Nume ticker (ex: SNP)", key="imp_name")
            up = st.file_uploader("Fișier CSV", type=["csv"], key="imp_file")
            if st.button("Importă"):
                if not nm or up is None:
                    st.warning("Completează numele și alege fișierul.")
                else:
                    dfc = parse_price_csv(up)
                    if dfc is None:
                        st.error("Nu am putut citi fișierul (lipsesc data/prețul de închidere sau sunt sub 220 de rânduri).")
                    else:
                        os.makedirs(CUSTOM_DIR, exist_ok=True)
                        dfc.to_csv(os.path.join(CUSTOM_DIR, nm.strip().upper() + ".csv"))
                        st.session_state["custom_prices"][nm.strip().upper()] = dfc
                        st.success(f"{nm.strip().upper()} importat ({len(dfc)} zile, până la {dfc.index[-1].date()}).")

        pf = edited.dropna(subset=["ticker", "pret_cumparare"])
        if pf.empty:
            st.info("Adaugă o linie (ticker, preț, cantitate, dată) ca să vezi semnalele.")
        else:
            st.subheader("Semnale de ținere / vânzare")
            st.dataframe(analyze_portfolio(pf), use_container_width=True)
            st.caption("Stop curent = maximul dintre (cel mai mare preț de la cumpărare − 3×ATR) și "
                       "(preț cumpărare − 2×ATR la intrare). Dacă lipsește data cumpărării, folosesc ultimele 60 de zile.")

            st.subheader("🛡️ Risk manager")
            st.caption("Cât de concentrat și de corelat e portofoliul tău — diversificarea reală contează "
                       "mai mult decât numărul de poziții diferite.")
            pos, corr, total, flags = risk_manager(pf)
            if pos.empty:
                st.info("Nu am putut calcula riscul (lipsesc prețuri sau cantități complete).")
            else:
                st.metric("Valoare totală portofoliu (estimată)", f"{total:,.0f}")
                st.dataframe(pos, use_container_width=True, hide_index=True,
                            column_config={"pondere_%": st.column_config.ProgressColumn(
                                "pondere_%", min_value=0.0, max_value=100.0, format="%.1f%%")})
                st.caption("`pondere_sugerată_%` e orientativă, ponderată invers cu volatilitatea fiecărei "
                           "acțiuni (risc egal pe poziție) — nu ține cont de câte ai cumpărat deja, doar de "
                           "cum ai distribui capital nou, în teorie, pentru un risc mai echilibrat.")
                for fl in flags:
                    (st.warning if "concentrare" in fl or "corelație" in fl or "expunere" in fl
                     else st.success)(fl)
                if corr is not None:
                    with st.expander("Matrice de corelație între acțiunile deținute"):
                        st.dataframe(corr, use_container_width=True)
                        st.caption("Valori apropiate de 1 = se mișcă aproape identic (diversificare slabă "
                                   "între ele); apropiate de 0 sau negative = se mișcă independent "
                                   "(diversificare reală).")

            tick_pf = list(dict.fromkeys(str(t).strip().upper() for t in pf["ticker"]))
            st.subheader("🌐 Sensibilitate macro per acțiune (dobânzi, inflație, sentiment)")
            ms_pf = {t: macro_sensitivity(t) for t in tick_pf}
            ms_rows = []
            for t, m in ms_pf.items():
                if m.get("ok"):
                    tb = m["tabel"].set_index("factor")["efect scenariu pe acțiune"]
                    ms_rows.append({"ticker": t, "scor_macro": m["scor"], "vânt_macro": m["verdict"],
                                    "efect_macro_20z": f"{m['efect_total']:+.2f}%",
                                    "dacă dobânzile +0,25pp": tb.get("Dobânzi (randament Treasury 10Y)", "–"),
                                    "dacă inflația așteptată +0,25pp": tb.get("Inflație așteptată (breakeven 10Y)", "–"),
                                    "dacă VIX +5": tb.get("Sentiment de piață (VIX)", "–"),
                                    "beta_piață": m["beta_piata"]})
                else:
                    ms_rows.append({"ticker": t, "scor_macro": np.nan, "vânt_macro": "–", "efect_macro_20z": m.get("motiv", "")})
            st.dataframe(pd.DataFrame(ms_rows), use_container_width=True, hide_index=True, column_config={
                "scor_macro": st.column_config.ProgressColumn("scor_macro", min_value=0, max_value=100, format="%d")})
            st.caption("Scor macro: 50 = neutru. Arată dacă mișcările recente ale dobânzilor, inflației așteptate și sentimentului "
                       "au ajutat sau au încurcat acțiunea, pe baza sensibilității ei istorice. Coloanele «dacă…» sunt scenarii: "
                       "efectul estimat asupra prețului dacă factorul respectiv se mișcă așa.")
            pick_ms = st.selectbox("Detalii pentru", tick_pf, key="pf_ms_pick")
            with st.expander("Detalii sensibilitate macro", expanded=False):
                render_macro_detail(ms_pf.get(pick_ms))

            st.subheader("📍 Plan de tranzacționare: zonă de intrare, stop-loss, ținte")
            plan_rows = []
            for t in tick_pf:
                dfp = get_prices(t)
                if dfp is None or len(dfp) < 210:
                    plan_rows.append({"ticker": t, "semnal": "fără date suficiente"})
                    continue
                pl = trade_plan(add_features(dfp))
                if not pl.get("valid"):
                    plan_rows.append({"ticker": t, "semnal": pl.get("semnal", "–"), "zonă_intrare": "—"})
                    continue
                sh = int((capital * risk_pct / 100) // pl["R"]) if pl["R"] > 0 else 0
                plan_rows.append({"ticker": t, "semnal": pl["semnal"], "preț": round(pl["pret"], 2),
                                  "zonă_intrare": f"{pl['zona_lo']:.2f} – {pl['zona_hi']:.2f}",
                                  "stop_loss": round(pl["stop"], 2), "risc_%": round(pl["risc_%"], 1),
                                  "țintă_1": round(pl["t1"], 2), "țintă_2": round(pl["t2"], 2),
                                  "țintă_3": round(pl["t3"], 2), "acțiuni_pentru_risc": sh})
            st.dataframe(pd.DataFrame(plan_rows), use_container_width=True, hide_index=True)
            st.caption("Plan pentru poziții noi sau pentru adăugiri (swing, grafic zilnic). Pentru ce deții deja, "
                       "stop-ul care contează e «stop_curent» din tabelul de sus. Țintele = 1,5R / 2,5R / 4R, "
                       "unde R e distanța până la stop. «acțiuni_pentru_risc» folosește capitalul și riscul din bara din stânga.")

            st.subheader("🤖 Predicție AI: scurt / mediu / lung")
            st.caption("Probabilitatea ca prețul să fie mai sus peste ≈2 săptămâni / ≈3 luni / ≈9 luni, comparată cu "
                       "frecvența istorică. Modelele se antrenează pe ~70 de acțiuni (10 ani de istoric); prima rulare durează câteva minute.")
            if st.button("🤖 Rulează predicția AI pentru portofoliu"):
                prog = st.progress(0.0)
                ai, met = ai_forecast([str(t).strip().upper() for t in pf["ticker"]], prog)
                prog.empty()
                st.session_state["ai"] = (ai, met)
            if st.session_state.get("ai") is not None:
                ai, met = st.session_state["ai"]
                if ai.empty:
                    st.warning("Nu am putut face predicții (date insuficiente).")
                else:
                    snap, sc, regime = macro_snapshot()
                    if regime:
                        st.info(f"Context macro SUA acum: **{regime}** (scor {sc:+d}). Modelele AI nu folosesc date macro — "
                                "tratează contextul ca filtru suplimentar.")
                    ai_show = ai.copy()
                    msx = {t: ms_pf.get(t) or macro_sensitivity(t) for t in ai_show["ticker"]}
                    ai_show["scor_macro"] = [msx[t]["scor"] if msx[t].get("ok") else np.nan for t in ai_show["ticker"]]
                    ai_show["vânt_macro"] = [msx[t]["verdict"] if msx[t].get("ok") else "–" for t in ai_show["ticker"]]
                    ai_show["concluzie"] = [macro_concluzie(r["imagine_generala"], r["vânt_macro"]) for _, r in ai_show.iterrows()]
                    st.dataframe(ai_show, use_container_width=True)
                    st.markdown("**Cât de mult te poți baza pe fiecare orizont?**")
                    st.dataframe(met, use_container_width=True)
                    st.warning("Intervalul ±1σ arată unde a stat prețul ~68% din timp, în funcție de volatilitatea recentă — "
                               "nu este o garanție. Cu cât orizontul e mai lung, cu atât predicția e mai nesigură; "
                               "dacă încrederea e «Scăzută», ignoră verdictul acelui orizont.")

    # ---------------- Intraday
    with t_intra:
        st.caption("Date Yahoo Finance: de regulă întârziate până la ~15 minute (depinde de bursă). Pentru date în timp "
                   "real reale ai nevoie de datele brokerului tău sau de un abonament la bursă; graficul TradingView "
                   "de la final afișează cotații în direct pentru piețele gratuite.")
        with st.expander("📘 Reguli de bază pentru tranzacționarea intraday (citește-le o dată)", expanded=False):
            st.markdown(
                "- **Riscă puțin pe fiecare tranzacție** (0,25–1% din capital) și **oprește-te pentru azi** dacă pierzi "
                "2–3% din capital într-o zi.\n"
                "- **Nu intra în primele 5–10 minute** după deschidere (spread-uri mari, mișcări haotice); "
                "intervalul de deschidere (prima jumătate de oră) se folosește ca reper, nu ca moment de intrare.\n"
                "- **Intră doar în zona de intrare**, cu ordin limită. Dacă prețul a plecat din zonă, ratezi tranzacția — nu o urmări.\n"
                "- **Stop-loss real la broker**, pus imediat după intrare. Nu îl muta mai departe.\n"
                "- **Închide înainte de final de sesiune**; evită poziții intraday peste noapte și evită rezultatele financiare (vezi tabul Grafic).\n"
                "- **Comisioane și spread** contează mult la tranzacții scurte: un plan cu câștig mic per tranzacție poate ieși pe minus după costuri.\n"
                "- Majoritatea persoanelor care fac day trading pierd bani pe termen lung. Tratează scannerul ca pe o "
                "listă de idei de verificat, nu ca pe un semnal garantat, și exersează întâi cu sume mici sau cont demo.")
        st.subheader("🔎 Scanner: ce merită tranzacționat azi")
        cA, cB, cC, cD = st.columns(4)
        uni = cA.selectbox("Univers", ["Piețele bifate în stânga", "Portofoliul meu", "Lista mea (scrisă mai jos)"], key="intra_uni")
        interval_s = cB.selectbox("Interval lumânări", ["1m", "5m", "15m"], index=1, key="intra_iv_scan")
        risk_i = cC.number_input("Risc / tranzacție (% capital)", 0.1, 2.0, 0.5, 0.1, key="intra_risk")
        max_n = cD.number_input("Max acțiuni analizate", 5, 60, 30, 5, key="intra_maxn")
        e1, e2, e3 = st.columns(3)
        min_dv = e1.number_input("Volum zilnic minim (milioane, în moneda acțiunii)", 1.0, 500.0, 20.0, 5.0, key="intra_dv")
        min_atr = e2.number_input("Mișcare zilnică minimă (ATR %)", 0.5, 8.0, 1.2, 0.1, key="intra_atr",
                                  help="Acțiunile care se mișcă prea puțin într-o zi nu lasă loc de profit după costuri.")
        allow_short = e3.checkbox("Include și setup-uri SHORT (necesită CFD / vânzare în lipsă)", value=False, key="intra_short")
        my_list = st.text_input("Lista mea (separată prin virgulă)", "AAPL, NVDA, TSLA, AMD, META", key="intra_list")
        if st.button("🔎 Caută oportunități acum", type="primary", key="intra_go"):
            if uni == "Portofoliul meu":
                universe = [str(t).strip().upper() for t in edited["ticker"].dropna()]
            elif uni == "Lista mea (scrisă mai jos)":
                universe = [t.strip().upper() for t in my_list.split(",") if t.strip()]
            else:
                universe = list(tickers)
            universe = list(dict.fromkeys(universe))
            if not universe:
                st.warning("Lista de tickere e goală.")
            else:
                prog = st.progress(0.0)
                df_s, meta_s = intraday_scan(universe, interval_s, capital, risk_i, min_dv, min_atr, allow_short, int(max_n), prog)
                prog.empty()
                st.session_state["intra_scan"] = (df_s, meta_s, interval_s)
        scan_res = st.session_state.get("intra_scan")
        scan_tickers = []
        if scan_res is not None:
            df_s, meta_s, iv_used = scan_res
            st.caption(f"Din {meta_s['total']} tickere: {meta_s['filtrate']} au trecut filtrul de lichiditate și mișcare, "
                       f"{meta_s['analizate']} au fost analizate intraday ({iv_used}).")
            if df_s.empty:
                st.warning("Niciun setup clar acum. Poate piața e închisă sau calmă, sau filtrele sunt prea stricte "
                           "(încearcă un volum minim mai mic sau o mișcare zilnică minimă mai mică).")
            else:
                now_b = pd.Timestamp.now(tz="Europe/Bucharest").tz_localize(None)
                lag_s = (now_b - meta_s["ultima_bara"]).total_seconds() / 60
                if lag_s > 30:
                    st.info(f"Piața pare închisă: ultima bară e din {meta_s['ultima_bara']:%d.%m %H:%M}. Lista de mai jos e "
                            f"pe sesiunea din {meta_s['sesiune']:%d.%m} — folosește-o ca pregătire pentru deschidere, "
                            "apoi rulează scanarea din nou după ce piața se deschide și trec cel puțin 30 de minute.")
                show = df_s.drop(columns=[c for c in df_s.columns if c.startswith("_")])
                st.dataframe(show, use_container_width=True, hide_index=True, column_config={
                    "scor": st.column_config.ProgressColumn("scor", min_value=0, max_value=100, format="%d")})
                st.caption("Scor = volum relativ + gap + mișcare zilnică + aliniere cu VWAP/EMA + ieșire din intervalul de "
                           "deschidere + trend zilnic (max 100). «În zonă acum» = prețul e în zona de intrare. "
                           "Alege un ticker mai jos ca să vezi planul complet și graficul.")
                scan_tickers = show["ticker"].tolist()

        st.divider()
        st.subheader("📈 Panou live + plan de tranzacție")
        l1, l2, l3, l4 = st.columns([2, 2, 1, 1])
        picked = l1.selectbox("Din rezultatele scanării", scan_tickers, key="intra_pick") if scan_tickers else None
        manual_i = l2.text_input("...sau alt ticker Yahoo", "" if scan_tickers else "AAPL", key="intra_manual").strip().upper()
        interval = l3.selectbox("Interval", ["1m", "5m", "15m", "30m", "60m"], index=1, key="intra_iv_live")
        refresh = l4.selectbox("Actualizare", ["Oprită", "30 s", "60 s", "120 s"], index=0, key="intra_refresh")
        sym = manual_i if manual_i else (picked or "AAPL")
        fkey = st.session_state.get("fmp_key", "")
        if hasattr(st, "fragment"):
            every = None if refresh == "Oprită" else int(refresh.split()[0])
            st.fragment(run_every=every)(intraday_panel)(sym, interval, capital, risk_i, allow_short, fkey)
        else:
            st.button("🔄 Actualizează", key="intra_refresh_btn")
            intraday_panel(sym, interval, capital, risk_i, allow_short, fkey)
        st.divider()
        st.subheader("Grafic live TradingView (inclusiv BVB)")
        tv_sym = st.text_input("Simbol TradingView", value=to_tv_symbol(sym),
                               help="Exemple: NASDAQ:AAPL, BVB:SNP, BVB:TLV, XETR:SIE, BITSTAMP:BTCUSD")
        tv_widget("embed-widget-advanced-chart.js", {
            "autosize": True, "symbol": tv_sym, "interval": "15", "timezone": "Europe/Bucharest",
            "theme": "light", "style": "1", "locale": "ro", "allow_symbol_change": True,
            "hide_side_toolbar": False, "studies": ["STD;VWAP"], "support_host": "https://www.tradingview.com"}, 620)

    # ---------------- Macro
    with t_macro:
        st.subheader("Regimul macro: ce spun inflația, șomajul, dobânzile și sentimentul")
        snap, sc, regime = macro_snapshot()
        if snap.empty:
            st.warning("Nu am putut descărca datele macro (FRED). Verifică conexiunea la internet.")
        else:
            cA, cB = st.columns([1, 3])
            cA.metric("Regim", regime, delta=f"scor {sc:+d}")
            cB.info("Scorul adună semnalele de mai jos (🟢 +1, 🟡 0, 🔴 −1). ≥ +3 = mediu favorabil acțiunilor, "
                    "≤ −2 = prudență. Este un indicator de context, nu un semnal de tranzacționare.")
            st.dataframe(snap, use_container_width=True)
        eur = macro_eu_ro()
        if not eur.empty:
            st.markdown("**Europa & România (informativ)**")
            st.dataframe(eur, use_container_width=True)
        st.subheader("Calendar economic — evenimentele importante ale săptămânii")
        eco_embed = st.session_state.get("eco_embed", "")
        if eco_embed:
            components.html(eco_embed, height=650, scrolling=True)
        else:
            st.info(
                "Poți afișa aici **calendarul economic oficial Investing.com** (inflație, șomaj, dobânzi, PIB): "
                "intră pe **investing.com/webmaster-tools/economic-calendar**, alege țările, "
                "importanța evenimentelor și aspectul, apoi copiază codul generat și lipește-l în bara din stânga, "
                "la „Integrare Investing.com”. Până atunci, iată varianta TradingView:")
            tv_widget("embed-widget-events.js", {
                "colorTheme": "light", "isTransparent": False, "width": "100%", "height": "600",
                "locale": "ro", "importanceFilter": "-1,0,1", "countryFilter": "us,eu,de,gb,ro"}, 620)
        st.subheader("Resurse externe")
        st.markdown("  \n".join(f"- [{k}]({v})" for k, v in LINKS.items()))

    # ---------------- Grafic & poziție
    with t_chart:
        if not have:
            st.info(no_scan_msg)
        else:
            pick = st.selectbox("Ticker", view["ticker"].tolist())
            r = view[view["ticker"] == pick].iloc[0]
            lo, hi = [float(x) for x in r["zona_cumparare"].split("–")]
            ch = price_chart(pick, [(lo, "green"), (hi, "green"), (r["stop_loss"], "red"), (r["target"], "purple")])
            if ch is not None:
                st.altair_chart(ch, use_container_width=True)
            st.caption("Verde = zonă de cumpărare · Roșu = stop-loss · Mov = țintă")
            risk_amt = st.session_state.get("capital", capital) * risk_pct / 100
            per_share = r["pret"] - r["stop_loss"]
            shares = int(risk_amt // per_share) if per_share > 0 else 0
            a, b, c = st.columns(3)
            a.metric("Acțiuni de cumpărat", shares)
            b.metric("Valoare poziție", f"{shares * r['pret']:,.0f}")
            c.metric("Risc maxim", f"{risk_amt:,.0f}")
            ed = earnings_date(pick)
            if pd.notna(ed):
                days_to = (ed - pd.Timestamp.today().normalize()).days
                msg = f"📅 Următoarele rezultate financiare: {ed.date()} (în {days_to} zile)."
                (st.warning if days_to <= 10 else st.info)(
                    msg + (" Prețul poate face salturi mari după raport — fii atent la momentul intrării."
                           if days_to <= 10 else ""))
            st.write(f"**Moment intrare:** {r['moment_intrare']} · Zonă: {r['zona_cumparare']} · "
                     f"Stop {r['stop_loss']} · Țintă {r['target']}")
            st.caption(f"Motive: {r['motive']}")
            st.markdown("**Știri recente**")
            news_panel(pick, st.session_state.get("fmp_key", ""))

    # ---------------- Toate datele
    with t_all:
        if not have:
            st.info(no_scan_msg)
        else:
            st.dataframe(view, use_container_width=True)
            st.download_button("Descarcă CSV", view.to_csv(index=False), "candidates.csv")

    st.caption("⚠️ Instrument educațional, nu sfat financiar. Performanța trecută nu garantează rezultate viitoare. "
               "Verifică întotdeauna compania (rezultate, știri) și folosește stop-loss.")


def _in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _in_streamlit():
    main()
elif __name__ == "__main__":
    print("Rulează cu:  streamlit run trading_agent.py")