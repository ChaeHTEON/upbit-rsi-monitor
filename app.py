# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import ta
from datetime import datetime, timedelta
import numpy as np

# -----------------------------
# 페이지/스타일
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
</style>
""", unsafe_allow_html=True)

st.title("📊 Upbit RSI(13) + Bollinger Band 시뮬레이터")

# -----------------------------
# 사용자 입력 (UI/UX 정렬 반영)
# -----------------------------
# 1번째 줄
col1, col2 = st.columns(2)
with col1:
    n_candles = st.number_input("측정 캔들 수 (기준 이후 N봉)", min_value=1, value=5)
with col2:
    success_threshold = st.number_input("성공/실패 기준 값(%)", min_value=1, max_value=100, value=10)

# 2번째 줄
col3, col4, col5 = st.columns(3)
with col3:
    use_rsi = st.selectbox("RSI 조건", ["없음", "있음"])
with col4:
    rsi_overbought = st.number_input("RSI 과매수 기준 (매도 조건)", min_value=50, max_value=100, value=70)
with col5:
    rsi_oversold = st.number_input("RSI 과매도 기준 (매수 조건)", min_value=0, max_value=50, value=30)

# 3번째 줄
col6, col7 = st.columns(2)
with col6:
    use_bullish_candles = st.checkbox("양봉 2개 연속 상승 체크")
with col7:
    first_signal_bullish = st.checkbox("첫 신호 양봉으로 표시")

# -----------------------------
# 데이터 불러오기
# -----------------------------
def get_upbit_data(market="KRW-BTC", interval="minute15", count=200):
    url = f"https://api.upbit.com/v1/candles/{interval}"
    params = {"market": market, "count": count}
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    response = session.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data)
    df = df.rename(columns={"trade_price":"close","candle_date_time_kst":"time"})
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df

# -----------------------------
# 시뮬레이션
# -----------------------------
def simulate(df):
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=13).rsi()
    signals = []

    for i in range(len(df)):
        cond_rsi = True
        if use_rsi == "있음":
            cond_rsi = (df["rsi"].iloc[i] >= rsi_overbought) or (df["rsi"].iloc[i] <= rsi_oversold)

        cond_bullish = True
        if use_bullish_candles and i >= 2:
            cond_bullish = (df["close"].iloc[i-1] > df["open"].iloc[i-1]) and (df["close"].iloc[i] > df["open"].iloc[i])

        if cond_rsi and cond_bullish:
            signals.append(i)

    return signals

# -----------------------------
# 실행
# -----------------------------
try:
    df = get_upbit_data()
    signals = simulate(df)

    # 차트
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Candlestick"), row=1, col=1
    )
    fig.add_trace(go.Scatter(x=df["time"], y=df["rsi"], mode="lines", name="RSI(13)"), row=2, col=1)

    for i in signals:
        fig.add_shape(type="line",
                      x0=df["time"].iloc[i], x1=df["time"].iloc[i],
                      y0=df["low"].min(), y1=df["high"].max(),
                      line=dict(color="blue", dash="dot"))

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"오류: {e}")
