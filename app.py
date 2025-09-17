import streamlit as st
import pandas as pd
import requests
import datetime
import plotly.graph_objs as go
import ta

# --- 업비트에서 캔들 데이터 불러오기 ---
def get_ohlcv(market="KRW-BTC", interval="minute1", count=200):
    url = f"https://api.upbit.com/v1/candles/{interval}"
    querystring = {"market": market, "count": count}
    headers = {"Accept": "application/json"}
    res = requests.get(url, headers=headers).json()
    df = pd.DataFrame(res)
    df = df.rename(columns={
        "candle_date_time_kst": "timestamp",
        "trade_price": "close"
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[["timestamp", "close"]]
    df = df.sort_values("timestamp")
    return df

# --- RSI 계산 ---
def add_rsi(df, period=14):
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=period).rsi()
    return df

# --- Streamlit UI ---
st.set_page_config(page_title="Upbit RSI 모니터", layout="wide")
st.title("📈 Upbit RSI 실시간 모니터")

# 사용자 입력
market = st.text_input("코인 선택 (예: KRW-BTC)", "KRW-BTC")
interval = st.selectbox("봉 종류 선택", ["minutes/1", "minutes/3", "minutes/5", "minutes/15", "minutes/30", "minutes/60", "days"], index=0)
count = st.slider("캔들 개수", 50, 200, 100)

# 데이터 불러오기
try:
    df = get_ohlcv(market, interval, count)
    df = add_rsi(df)

    # 가격 차트
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], mode="lines", name="가격"))
    fig_price.update_layout(title=f"{market} 가격 차트", xaxis_title="시간", yaxis_title="가격")
    st.plotly_chart(fig_price, use_container_width=True)

    # RSI 차트
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df["timestamp"], y=df["rsi"], mode="lines", name="RSI"))
    fig_rsi.add_hline(y=70, line=dict(color="red", dash="dot"))
    fig_rsi.add_hline(y=30, line=dict(color="green", dash="dot"))
    fig_rsi.update_layout(title="RSI (14)", xaxis_title="시간", yaxis_title="RSI")
    st.plotly_chart(fig_rsi, use_container_width=True)

    st.dataframe(df.tail(20))
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류 발생: {e}")
