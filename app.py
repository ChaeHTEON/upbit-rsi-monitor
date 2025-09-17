import streamlit as st
import pandas as pd
import pandas_ta as ta
import requests
import plotly.graph_objects as go

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

    # ✅ timestamp 컬럼 이름 다듬기
    df.rename(columns={
        "candle_date_time_kst": "timestamp",
        "opening_price": "open",
        "high_price": "high",
        "low_price": "low",
        "trade_price": "close",
        "candle_acc_trade_volume": "volume"
    }, inplace=True)

    # ✅ 시간순 정렬
    df = df.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]]
    df = df.iloc[::-1].reset_index(drop=True)

    return df


# -----------------------------
# RSI 계산 함수
# -----------------------------
def calculate_rsi(df, length=14):
    df["RSI"] = ta.rsi(df["close"], length=length)
    return df


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Upbit RSI 모니터", layout="wide")

st.title("📈 Upbit RSI 실시간 모니터")

# 코인 선택
coin = st.text_input("코인 선택 (예: KRW-BTC)", value="KRW-BTC")

# 봉 종류 선택
interval = st.selectbox(
    "봉 종류 선택",
    ["minutes/1", "minutes/3", "minutes/5", "minutes/15", "minutes/30", "minutes/60", "days"]
)

# 캔들 개수 선택
count = st.slider("캔들 개수", min_value=30, max_value=200, value=100)

try:
    df = get_upbit_candles(market=coin, interval=interval, count=count)
    df = calculate_rsi(df)

    # RSI 차트
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["RSI"], mode="lines", name="RSI"))

    # 기준선 추가 (과매수/과매도)
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="blue")

    fig.update_layout(title=f"{coin} RSI 차트", xaxis_title="시간", yaxis_title="RSI")
    st.plotly_chart(fig, use_container_width=True)

    # 최근 RSI 값 표시
    latest_rsi = df["RSI"].iloc[-1]
    st.metric(label=f"{coin} 최신 RSI", value=round(latest_rsi, 2))

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류 발생: {e}")
