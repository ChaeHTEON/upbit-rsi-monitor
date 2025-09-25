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
  .neutral-cell {color:#6B7280; font-weight:600;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 유틸 함수
# -----------------------------
def fetch_upbit(market, interval="minute15", count=200):
    url = f"https://api.upbit.com/v1/candles/{interval}"
    headers = {"Accept": "application/json"}
    params = {"market": market, "count": count}
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["candle_date_time_kst"])
    df = df.rename(columns={"opening_price": "open", "high_price": "high",
                            "low_price": "low", "trade_price": "close",
                            "candle_acc_trade_volume": "volume"})
    df = df.sort_values("time")
    return df

def add_indicators(df, bb_window=20, bb_dev=2):
    df["RSI"] = ta.momentum.RSIIndicator(df["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(df["close"], window=bb_window, window_dev=bb_dev)
    df["BB_up"] = bb.bollinger_hband()
    df["BB_mid"] = bb.bollinger_mavg()
    df["BB_low"] = bb.bollinger_lband()
    return df

# -----------------------------
# 사이드바
# -----------------------------
st.sidebar.title("설정")
market = st.sidebar.selectbox("마켓", ["KRW-BTC", "KRW-ETH", "KRW-XRP"], index=0)
interval = st.sidebar.selectbox("봉간격", ["minute1", "minute15", "minute60", "days"], index=1)
count = st.sidebar.slider("캔들 수", 200, 500, 200, step=50)

# -----------------------------
# 데이터 가져오기
# -----------------------------
try:
    df = fetch_upbit(market, interval, count)
    df = add_indicators(df)

    # -----------------------------
    # 매수가 입력 + 최적화뷰 버튼 (차트 위쪽)
    # -----------------------------
    ui_col1, ui_col2 = st.columns([2, 1])
    with ui_col1:
        buy_price = st.number_input(
            "💰 매수가 입력",
            min_value=0,
            value=0,
            step=1,
            format="%d"
        )
    with ui_col2:
        if "opt_view" not in st.session_state:
            st.session_state.opt_view = False
        opt_label = "↩ 되돌아가기" if st.session_state.opt_view else "📈 최적화뷰"
        if st.button(opt_label, key="btn_opt_view"):
            st.session_state.opt_view = not st.session_state.opt_view

    # 차트 컨테이너
    chart_box = st.container()

    # -----------------------------
    # Hovertext 생성
    # -----------------------------
    hovertext = []
    for t, o, h, l, c in zip(
        df["time"].dt.strftime("%Y-%m-%d %H:%M"),
        df["open"], df["high"], df["low"], df["close"]
    ):
        if buy_price > 0:
            pct = (c / buy_price - 1) * 100
            color = "red" if pct > 0 else "blue"
            hovertext.append(
                f"시간: {t}<br>"
                f"시가: {o}<br>고가: {h}<br>저가: {l}<br>종가: {c}<br>"
                f"수익률: <span style='color:{color}'>{pct:.2f}%</span>"
            )
        else:
            hovertext.append(
                f"시간: {t}<br>"
                f"시가: {o}<br>고가: {h}<br>저가: {l}<br>종가: {c}"
            )

    # -----------------------------
    # 차트 그리기
    # -----------------------------
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Candlestick(
        x=df["time"],
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="가격",
        increasing=dict(line=dict(color="red", width=1.1)),
        decreasing=dict(line=dict(color="blue", width=1.1)),
        hovertext=hovertext,
        hoverinfo="text"
    ))

    fig.add_trace(go.Scatter(
        x=df["time"], y=df["BB_up"], mode="lines",
        line=dict(color="#FFB703", width=1.4), name="BB 상단"
    ))
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["BB_low"], mode="lines",
        line=dict(color="#219EBC", width=1.4), name="BB 하단"
    ))
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["BB_mid"], mode="lines",
        line=dict(color="#8D99AE", width=1.1, dash="dot"), name="BB 중앙"
    ))

    # RSI 라인
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["RSI"], mode="lines",
        line=dict(color="#06D6A0", width=1.2, dash="dot"), name="RSI(13)", yaxis="y2"
    ))

    # Layout
    fig.update_layout(
        title=f"{market} · {interval} · RSI(13)+BB",
        xaxis_rangeslider_visible=False,
        dragmode="pan",
        height=600,
        margin=dict(l=30, r=30, t=60, b=40),
        yaxis=dict(title="가격"),
        yaxis2=dict(overlaying="y", side="right", showgrid=False, title="RSI(13)", range=[0, 100]),
        hovermode="x unified",
        uirevision="chart-static",
        legend_orientation="h",
        legend_y=1.05
    )

    # 최적화뷰 적용
    if st.session_state.opt_view and len(df) > 0:
        window_n = max(int(len(df) * 0.15), 200)
        start_idx = max(len(df) - window_n, 0)
        x_start = df.iloc[start_idx]["time"]
        x_end = df.iloc[-1]["time"]
        fig.update_xaxes(range=[x_start, x_end])

    # -----------------------------
    # 차트 출력
    # -----------------------------
    chart_box.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "doubleClick": "reset",
            "responsive": True
        }
    )

except Exception as e:
    st.error(f"오류: {e}")
