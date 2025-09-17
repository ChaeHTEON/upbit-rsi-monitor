import streamlit as st
import pandas as pd
import pandas_ta as ta
import requests
import plotly.graph_objs as go

# 페이지 기본 설정
st.set_page_config(page_title="Upbit RSI 모니터", layout="wide")

st.title("📈 Upbit RSI 실시간 모니터")

# --- 사용자 입력 ---
coin = st.text_input("코인 선택 (예: KRW-BTC)", "KRW-BTC")
interval = st.selectbox("봉 종류 선택", ["1분봉", "3분봉", "5분봉", "10분봉", "15분봉", "30분봉", "60분봉", "240분봉", "일봉"])
count = st.slider("캔들 개수", min_value=50, max_value=200, value=100)

# interval 변환
interval_map = {
    "1분봉": "minute1",
    "3분봉": "minute3",
    "5분봉": "minute5",
    "10분봉": "minute10",
    "15분봉": "minute15",
    "30분봉": "minute30",
    "60분봉": "minute60",
    "240분봉": "minute240",
    "일봉": "day"
}
interval_code = interval_map[interval]

# --- 데이터 가져오기 ---
url = f"https://api.upbit.com/v1/candles/{interval_code}"
params = {"market": coin, "count": count}

try:
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data)
    df = df.rename(columns={
        "candle_date_time_kst": "time",
        "opening_price": "open",
        "high_price": "high",
        "low_price": "low",
        "trade_price": "close",
        "candle_acc_trade_volume": "volume"
    })
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    # RSI 계산
    df["RSI"] = ta.rsi(df["close"], length=14)

    # --- 차트 출력 ---
    fig = go.Figure()

    # 캔들차트
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price"
    ))

    # RSI 보조지표
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["RSI"], mode="lines", name="RSI", yaxis="y2"
    ))

    # 레이아웃
    fig.update_layout(
        title=f"{coin} {interval} RSI 차트",
        xaxis=dict(domain=[0, 1]),
        yaxis=dict(title="Price", side="left"),
        yaxis2=dict(title="RSI", overlaying="y", side="right", range=[0, 100]),
        xaxis_rangeslider_visible=False,
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df[["time", "open", "high", "low", "close", "RSI"]].tail(20))

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류 발생: {e}")
