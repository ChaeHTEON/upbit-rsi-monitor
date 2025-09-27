# app.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import ta
from datetime import datetime, timedelta
from pytz import timezone

# -----------------------------
# 페이지/스타일 (UI/UX 유지)
# -----------------------------
st.set_page_config(page_title="Upbit RSI(13) + Bollinger Band 시뮬레이터", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 0.8rem; padding-bottom: 0.8rem; max-width: 1100px;}
  .stMetric {text-align:center;}
  .section-title {font-size:1.05rem; font-weight:700; margin: 0.6rem 0 0.2rem;}
  .hint {color:#6b7280;}
  .success-cell {background-color:#FFF59D; color:#E53935; font-weight:600;}
  .fail-cell {color:#1E40AF; font-weight:600;}
  .neutral-cell {color:#374151; font-weight:600;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 세션 상태 초기화 (필수 키)
# -----------------------------
if "uirevision_key" not in st.session_state:
    st.session_state.uirevision_key = "keep"
if "data_cache" not in st.session_state:
    st.session_state.data_cache = None
if "signals_cache" not in st.session_state:
    st.session_state.signals_cache = None
if "last_params" not in st.session_state:
    st.session_state.last_params = {}

# -----------------------------
# 공통 유틸
# -----------------------------
KST = timezone("Asia/Seoul")

def _session():
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def _upbit_url(interval: str) -> str:
    # interval: minute1 | minute5 | minute15 | minute30 | minute60 | day
    return f"https://api.upbit.com/v1/candles/{interval}"

@st.cache_data(show_spinner=False, ttl=60)
def fetch_upbit_once(symbol="KRW-BTC", interval="minute60", to_iso=None, count=200) -> pd.DataFrame:
    """단일 호출(최대 200개)"""
    params = {"market": symbol, "count": count}
    if to_iso:
        params["to"] = to_iso
    r = _session().get(_upbit_url(interval), headers={"Accept": "application/json"}, params=params, timeout=10)
    r.raise_for_status()
    js = r.json()
    if not js:
        return pd.DataFrame()
    df = pd.DataFrame(js)
    df["candle_date_time_kst"] = pd.to_datetime(df["candle_date_time_kst"])
    df = df.rename(columns={
        "candle_date_time_kst": "time",
        "opening_price": "open",
        "high_price": "high",
        "low_price": "low",
        "trade_price": "close",
        "candle_acc_trade_volume": "volume",
    })
    df = df.sort_values("time").reset_index(drop=True)
    return df[["time", "open", "high", "low", "close", "volume"]]

def fetch_upbit_paged(symbol: str, interval: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """여러 페이지로 이어 받아 start~end 구간 필터"""
    all_chunks = []
    cursor = end_dt
    safety = 0
    while True:
        safety += 1
        if safety > 50:  # 과도한 루프 방지
            break
        df = fetch_upbit_once(symbol, interval, to_iso=cursor.strftime("%Y-%m-%d %H:%M:%S"), count=200)
        if df.empty:
            break
        all_chunks.append(df)
        oldest = df["time"].min()
        if oldest <= start_dt:
            break
        # 분봉/일봉 모두 1분 감소로 충분 (Upbit는 'to'가 그 시각 이전까지 반환)
        cursor = oldest - timedelta(minutes=1)

    if not all_chunks:
        return pd.DataFrame()

    data = pd.concat(all_chunks, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time")
    data = data[(data["time"] >= start_dt) & (data["time"] <= end_dt)].reset_index(drop=True)
    return data

def compute_indicators(df: pd.DataFrame, rsi_len=13, bb_len=20, bb_std=2.0) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = ta.momentum.RSIIndicator(close=out["close"], window=rsi_len).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=bb_len, window_dev=bb_std)
    out["bb_mid"] = bb.bollinger_mavg()
    out["bb_low"] = bb.bollinger_lband()
    out["bb_high"] = bb.bollinger_hband()
    return out

def detect_signals(df: pd.DataFrame) -> pd.DataFrame:
    """기본 로직 유지: RSI(13) ≤ 30 신호. (추가 필터는 미적용, UI/UX 유지 원칙)"""
    if df.empty or "rsi" not in df:
        return pd.DataFrame(columns=["time", "price", "signal"])
    mask = df["rsi"] <= 30
    sig = df.loc[mask, ["time", "close"]].copy()
    sig = sig.rename(columns={"close": "price"})
    sig["signal"] = "BUY"
    return sig[["time", "price", "signal"]]

def make_chart(df: pd.DataFrame, signals: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    # 캔들
    fig.add_trace(
        go.Candlestick(
            x=df["time"],
            open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name="Candles",
        ),
        row=1, col=1
    )
    # 볼린저
    fig.add_trace(go.Scatter(x=df["time"], y=df["bb_high"], name="BB High", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["bb_mid"],  name="BB Mid",  mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["bb_low"],  name="BB Low",  mode="lines"), row=1, col=1)
    # RSI
    fig.add_trace(go.Scatter(x=df["time"], y=df["rsi"], name="RSI(13)", mode="lines"), row=2, col=1)
    # 신호 마커
    if not signals.empty:
        fig.add_trace(
            go.Scatter(
                x=signals["time"], y=signals["price"],
                mode="markers",
                marker=dict(size=10, symbol="star"),
                name="Signal"
            ),
            row=1, col=1
        )
    # 범위슬라이더/뷰 유지
    fig.update_layout(xaxis_rangeslider_visible=False, uirevision=st.session_state.uirevision_key)
    # RSI 보조선 (30/70)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    return fig

# -----------------------------
# 상단 입력 (시간 단위 컨트롤, 기본 24시간)
# -----------------------------
st.title("Upbit RSI(13) + Bollinger Band 시뮬레이터")

symbol = st.text_input("심볼", "KRW-BTC")
interval = st.selectbox("캔들 단위", ["minute1","minute5","minute15","minute30","minute60","day"], index=4)

now_kst = datetime.now(KST)
default_end = now_kst
default_start = now_kst - timedelta(hours=24)

c1, c2 = st.columns(2)
with c1:
    start_date = st.date_input("시작 날짜", default_start.date())
    start_time = st.time_input("시작 시간", default_start.time())
with c2:
    end_date = st.date_input("종료 날짜", default_end.date())
    end_time = st.time_input("종료 시간", default_end.time())

start_dt = datetime.combine(start_date, start_time)
end_dt = datetime.combine(end_date, end_time)

# -----------------------------
# 실행
# -----------------------------
try:
    # 파라미터 변경 시에만 재조회
    params = {"symbol": symbol, "interval": interval, "start": start_dt, "end": end_dt}
    need_fetch = (st.session_state.last_params != params)

    if need_fetch:
        st.session_state.last_params = params
        data = fetch_upbit_paged(symbol, interval, start_dt, end_dt)
        if not data.empty:
            data = compute_indicators(data, rsi_len=13, bb_len=20, bb_std=2.0)
        st.session_state.data_cache = data
        st.session_state.signals_cache = detect_signals(data) if not data.empty else pd.DataFrame(columns=["time","price","signal"])

    df = st.session_state.data_cache if st.session_state.data_cache is not None else pd.DataFrame()
    signals = st.session_state.signals_cache if st.session_state.signals_cache is not None else pd.DataFrame(columns=["time","price","signal"])

    # ③ 요약 & 차트 (구성/명칭 유지)
    st.markdown('<div class="section-title">③ 요약 & 차트</div>', unsafe_allow_html=True)
    if df.empty:
        st.warning("데이터가 없습니다.")
    else:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("데이터 개수", f"{len(df):,}")
        with m2:
            st.metric("신호 개수", f"{len(signals):,}")
        with m3:
            st.metric("기간", f"{df['time'].min().strftime('%Y-%m-%d %H:%M')} ~ {df['time'].max().strftime('%Y-%m-%d %H:%M')}")

        fig = make_chart(df, signals)
        st.plotly_chart(fig, use_container_width=True)

    # ④ 신호 결과 (최신 순) (구성/명칭/컬럼 유지)
    st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)
    if df.empty:
        st.info("신호 없음")
    else:
        if signals.empty:
            st.info("신호 없음")
        else:
            sig_df = signals.sort_values("time", ascending=False).reset_index(drop=True)
            st.dataframe(sig_df, use_container_width=True)

except Exception as e:
    st.error(f"오류: {e}")
