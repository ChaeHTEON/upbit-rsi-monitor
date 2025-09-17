import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
import ta  # ✅ pandas_ta 대신 ta 사용

# -----------------------------
# 업비트 캔들 데이터 불러오기
# -----------------------------
def get_upbit_candles(market="KRW-BTC", interval="minute1", count=100):
    url = f"https://api.upbit.com/v1/candles/{interval}"
    headers = {"Accept": "application/json"}
    params = {"market": market, "count": count}
    res = requests.get(url, headers=headers, params=params)

    if res.status_code != 200:
        raise Exception(f"API Error: {res.text}")

    data = res.json()
    df = pd.DataFrame(data)

    # 컬럼명 정리
    df.rename(columns={
        "candle_date_time_kst": "time",
        "opening_price": "open",
        "high_price": "high",
        "low_price": "low",
        "trade_price": "close",
        "candle_acc_trade_volume": "volume"
    }, inplace=True)

    # 시간순 정렬
    df = df.loc[:, ["time", "open", "high", "low", "close", "volume"]]
    df["time"] = pd.to_datetime(df["time"])
    df = df.iloc[::-1].reset_index(drop=True)

    return df


# -----------------------------
# RSI 계산
# -----------------------------
def add_rsi(df, period=14):
    rsi = ta.momentum.RSIIndicator(close=df["close"], window=period)
    df["RSI"] = rsi.rsi()
    return df


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Upbit RSI 모니터", layout="wide")
st.title("📈 Upbit RSI 실시간 모니터")

# 사용자 입력
coin = st.text_input("코인 선택 (예: KRW-BTC)", value="KRW-BTC")

interval = st.selectbox(
    "봉 종류 선택",
    ["minutes/1", "minutes/3", "minutes/5", "minutes/15", "minutes/30", "minutes/60", "days"]
)

count = st.slider("캔들 개수", min_value=30, max_value=200, value=100)

try:
    df = get_upbit_candles(market=coin, interval=interval, count=count)
    df = add_rsi(df)

    # 가격 차트 (캔들스틱)
    fig_price = go.Figure(data=[go.Candlestick(
        x=df["time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Price"
    )])
    fig_price.update_layout(title=f"{coin} 가격 차트", xaxis_title="시간", yaxis_title="가격", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_price, use_container_width=True)

    # RSI 차트
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df["time"], y=df["RSI"], mode="lines", name="RSI"))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="blue")
    fig_rsi.update_layout(title="RSI (14)", xaxis_title="시간", yaxis_title="RSI")
    st.plotly_chart(fig_rsi, use_container_width=True)

    # 최근 RSI 값
    latest_rsi = df["RSI"].iloc[-1]
    st.metric("현재 RSI", round(latest_rsi, 2))

    # 최근 데이터 테이블
    st.dataframe(df.tail(20))

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류 발생: {e}")
