import streamlit as st
import pandas as pd
import requests
import pandas_ta as ta
import plotly.express as px

st.set_page_config(page_title="Upbit RSI 실시간 모니터", layout="wide")

st.title("📈 Upbit RSI 실시간 모니터")

# ---- 사용자 입력 ----
market = st.text_input("코인 선택 (예: KRW-BTC)", "KRW-BTC")
interval = st.selectbox("봉 종류 선택", ["minutes/1", "minutes/3", "minutes/5", "minutes/15", "minutes/30", "minutes/60", "days"])
count = st.slider("캔들 개수", min_value=20, max_value=200, value=100)


# ---- 데이터 불러오기 함수 ----
def get_ohlcv(market="KRW-BTC", interval="minutes/1", count=200):
    try:
        # interval minutes/x 처리
        if "minutes/" in interval:
            unit = interval.split("/")[1]
            url = f"https://api.upbit.com/v1/candles/minutes/{unit}"
        else:
            url = f"https://api.upbit.com/v1/candles/{interval}"

        querystring = {"market": market, "count": count}
        headers = {"Accept": "application/json"}
        res = requests.get(url, headers=headers, params=querystring).json()

        if isinstance(res, dict) and res.get("error"):
            raise Exception(res["error"]["message"])

        df = pd.DataFrame(res)
        df = df.rename(columns={
            "candle_date_time_kst": "timestamp",
            "trade_price": "close"
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[["timestamp", "close"]].sort_values("timestamp")
        return df

    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류 발생: {e}")
        return pd.DataFrame()


# ---- 데이터 가져오기 ----
df = get_ohlcv(market, interval, count)

if not df.empty:
    # RSI 계산
    df["rsi"] = ta.rsi(df["close"], length=14)

    # ---- 가격 차트 ----
    fig_price = px.line(df, x="timestamp", y="close", title=f"{market} 가격 차트")
    st.plotly_chart(fig_price, use_container_width=True)

    # ---- RSI 차트 ----
    fig_rsi = px.line(df, x="timestamp", y="rsi", title="RSI (14)")
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="blue")
    st.plotly_chart(fig_rsi, use_container_width=True)
