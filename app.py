# app.py
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
from pytz import timezone

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
  .neutral-cell {color:#374151; font-weight:600;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 함수 정의
# -----------------------------
def fetch_upbit(symbol="KRW-BTC", interval="minute60", to=None, count=200):
    url = f"https://api.upbit.com/v1/candles/{interval}"
    headers = {"Accept": "application/json"}
    params = {"market": symbol, "count": count}
    if to:
        params["to"] = to
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[500,502,503,504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    res = session.get(url, headers=headers, params=params)
    res.raise_for_status()
    data = res.json()
    df = pd.DataFrame(data)
    df['candle_date_time_kst'] = pd.to_datetime(df['candle_date_time_kst'])
    df = df.rename(columns={
        "candle_date_time_kst":"time","opening_price":"open","high_price":"high",
        "low_price":"low","trade_price":"close","candle_acc_trade_volume":"volume"})
    return df.sort_values("time").reset_index(drop=True)

def get_data(symbol, interval, start, end):
    all_df = []
    dt = end
    while dt > start:
        df = fetch_upbit(symbol, interval, dt.strftime("%Y-%m-%d %H:%M:%S"), 200)
        if df.empty:
            break
        all_df.append(df)
        dt = df["time"].min() - timedelta(minutes=1)
    if not all_df:
        return pd.DataFrame()
    df = pd.concat(all_df).drop_duplicates().sort_values("time")
    return df[(df["time"] >= start) & (df["time"] <= end)]

def simulate(df, rsi_len=13, bb_len=20, bb_std=2):
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], rsi_len).rsi()
    bb = ta.volatility.BollingerBands(df["close"], bb_len, bb_std)
    df["bb_low"] = bb.bollinger_lband()
    df["bb_high"] = bb.bollinger_hband()
    signals = []
    for i in range(len(df)):
        if df["rsi"].iloc[i] <= 30 and df["close"].iloc[i] <= df["bb_low"].iloc[i]:
            signals.append({"time":df["time"].iloc[i],"price":df["close"].iloc[i],"signal":"BUY"})
    return signals

# -----------------------------
# UI
# -----------------------------
st.title("Upbit RSI(13) + Bollinger Band 시뮬레이터")

symbol = st.text_input("심볼", "KRW-BTC")
interval = st.selectbox("캔들 단위", ["minute1","minute5","minute15","minute30","minute60","day"], index=4)

now_kst = datetime.now(timezone("Asia/Seoul"))
default_end = now_kst
default_start = now_kst - timedelta(hours=24)

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("시작 날짜", default_start.date())
    start_time = st.time_input("시작 시간", default_start.time())
with col2:
    end_date = st.date_input("종료 날짜", default_end.date())
    end_time = st.time_input("종료 시간", default_end.time())

start_dt = datetime.combine(start_date, start_time)
end_dt = datetime.combine(end_date, end_time)

if st.button("데이터 불러오기"):
    try:
        df = get_data(symbol, interval, start_dt, end_dt)
        if df.empty:
            st.warning("데이터가 없습니다.")
        else:
            signals = simulate(df)
            # 차트
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3])
            fig.add_trace(go.Candlestick(x=df["time"],open=df["open"],high=df["high"],
                                         low=df["low"],close=df["close"],name="Candles"), row=1,col=1)
            fig.add_trace(go.Scatter(x=df["time"],y=df["rsi"],name="RSI",mode="lines"), row=2,col=1)
            for sig in signals:
                fig.add_trace(go.Scatter(x=[sig["time"]],y=[sig["price"]],
                                         mode="markers",marker=dict(color="red",size=10),
                                         name=sig["signal"]))
            fig.update_layout(xaxis_rangeslider_visible=False, uirevision="keep")
            st.plotly_chart(fig, use_container_width=True)

            # 신호 결과
            st.subheader("신호 결과 (최신 순)")
            if signals:
                sig_df = pd.DataFrame(signals).sort_values("time", ascending=False)
                st.dataframe(sig_df)
            else:
                st.info("신호 없음")
    except Exception as e:
        st.error(f"오류: {e}")
