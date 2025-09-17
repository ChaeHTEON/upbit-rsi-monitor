import requests
import pandas as pd
import pandas_ta as ta
import streamlit as st

# 페이지 기본 설정
st.set_page_config(page_title="Upbit RSI 실시간 모니터", layout="wide")
st.title("📈 Upbit RSI 실시간 모니터")

# 사용자 입력
market = st.text_input("마켓 코드 입력 (예: KRW-BTC, KRW-ETH)", "KRW-BTC")
interval = st.selectbox("봉 종류 선택", ["minutes/1", "minutes/5", "minutes/15", "minutes/30", "days"])
count = st.slider("캔들 개수", 50, 200, 100)
refresh_sec = st.slider("자동 새로고침 간격(초)", 5, 60, 10)

# 데이터 가져오기 함수
@st.cache_data(ttl=60)
def get_data(market, interval, count):
    url = f"https://api.upbit.com/v1/candles/{interval}"
    params = {"market": market, "count": count}
    res = requests.get(url, params=params)
    data = res.json()
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "opening_price": "open",
        "high_price": "high",
        "low_price": "low",
        "trade_price": "close",
        "candle_acc_trade_volume": "volume"
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["rsi"] = ta.rsi(df["close"], length=14)
    return df

# 데이터 불러오기
df = get_data(market, interval, count)

# 가격 차트
st.subheader(f"{market} 가격 차트")
st.line_chart(df.set_index("timestamp")[["close"]])

# RSI 차트
st.subheader("RSI (14)")
st.line_chart(df.set_index("timestamp")[["rsi"]])

# 현재 RSI 값
latest_rsi = df["rsi"].iloc[-1]
st.metric("현재 RSI", round(latest_rsi, 2))

# 경고 표시
if latest_rsi < 30:
    st.error("⚠️ RSI 30 이하: 과매도 구간 (반등 가능성 주의)")
elif latest_rsi > 70:
    st.warning("⚠️ RSI 70 이상: 과매수 구간 (조정 가능성 주의)")
else:
    st.info("📊 RSI 정상 범위")

# 자동 새로고침
st_autorefresh = st.experimental_rerun
