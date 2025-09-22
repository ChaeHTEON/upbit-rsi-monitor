# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import ta
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
  .fail-cell {background-color:#FFCDD2; color:#1E40AF; font-weight:600;}
  /* 요청: 중립 연두색 */
  .neutral-cell {background-color:#CCFFCC; color:#1E7D22; font-weight:600;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Upbit RSI(13) + Bollinger Band 시뮬레이터")

# -----------------------------
# 사용자 입력
# -----------------------------
col1, col2 = st.columns(2)
with col1:
    n_candles = st.number_input("측정 캔들 수 (기준 이후 N봉)", min_value=1, value=5)
with col2:
    success_threshold = st.number_input("성공/실패 기준 값(%)", min_value=1, max_value=100, value=10)

col3, col4, col5 = st.columns(3)
with col3:
    use_rsi = st.selectbox("RSI 조건", ["없음", "있음"])
with col4:
    rsi_overbought = st.number_input("RSI 과매수 기준 (매도 조건)", min_value=50, max_value=100, value=70)
with col5:
    rsi_oversold = st.number_input("RSI 과매도 기준 (매수 조건)", min_value=0, max_value=50, value=30)

col6, col7 = st.columns(2)
with col6:
    use_bullish_candles = st.checkbox("양봉 2개 연속 상승 체크")
with col7:
    first_signal_bullish = st.checkbox("첫 신호 양봉으로 표시")

# -----------------------------
# Upbit interval 보정
# -----------------------------
def normalize_interval(interval: str) -> str:
    if interval.startswith("minute") and not interval.startswith("minutes/"):
        unit = "".join(ch for ch in interval if ch.isdigit())
        return f"minutes/{unit}"
    return interval

# -----------------------------
# 데이터 불러오기
# -----------------------------
def get_upbit_data(market="KRW-BTC", interval="minute15", count=200):
    base = "https://api.upbit.com/v1/candles"
    interval_path = normalize_interval(interval)
    url = f"{base}/{interval_path}"
    params = {"market": market, "count": count}

    session = requests.Session()
    retry = Retry(total=3, connect=3, backoff_factor=0.6, status_forcelist=(429, 500, 502, 503, 504))
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    resp = session.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Upbit 응답이 비어 있습니다.")

    df = pd.DataFrame(data)
    time_col = "candle_date_time_kst" if "candle_date_time_kst" in df.columns else "candle_date_time_utc"
    rename_map = {
        "opening_price": "open",
        "high_price": "high",
        "low_price": "low",
        "trade_price": "close",
        time_col: "time",
    }
    df = df.rename(columns=rename_map)

    needed = {"open", "high", "low", "close", "time"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"필수 컬럼 누락: {missing}. 실제 컬럼: {list(df.columns)[:10]}")

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df

# -----------------------------
# 유틸: 결과 테이블 스타일
# -----------------------------
def style_result_col(s: pd.Series):
    styles = []
    for v in s:
        if v == "성공":
            styles.append("background-color:#FFF59D; color:#E53935; font-weight:600;")
        elif v == "실패":
            styles.append("background-color:#FFCDD2; color:#1E40AF; font-weight:600;")
        else:  # 중립 → 연두색
            styles.append("background-color:#CCFFCC; color:#1E7D22; font-weight:600;")
    return styles

# -----------------------------
# 시뮬레이션
# -----------------------------
def simulate(df: pd.DataFrame):
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=13).rsi()

    signals = []
    for i in range(len(df)):
        cond_rsi = True
        if use_rsi == "있음":
            rsi_now = df["rsi"].iloc[i]
            cond_rsi = (rsi_now >= rsi_overbought) or (rsi_now <= rsi_oversold)

        cond_bullish = True
        if use_bullish_candles and i >= 1:
            prev_green = df["close"].iloc[i-1] > df["open"].iloc[i-1]
            curr_green = df["close"].iloc[i] > df["open"].iloc[i]
            cond_bullish = (prev_green and curr_green)

        if cond_rsi and cond_bullish:
            signals.append(i)

    rows = []
    for i in signals:
        base_price = df["close"].iloc[i]
        end_idx = min(i + n_candles, len(df) - 1)
        window = df.iloc[i+1:end_idx+1]
        if len(window) == 0:
            continue

        max_ret = (window["high"].max() - base_price) / base_price * 100.0
        min_ret = (window["low"].min() - base_price) / base_price * 100.0

        hit_time = "-"
        hit_idx = None
        for j in range(1, end_idx - i + 1):
            r = (df["high"].iloc[i + j] - base_price) / base_price * 100.0
            if r >= success_threshold:
                hit_idx = j
                break
        fail_hit_idx = None
        for j in range(1, end_idx - i + 1):
            r = (df["low"].iloc[i + j] - base_price) / base_price * 100.0
            if r <= -success_threshold:
                fail_hit_idx = j
                break

        result = "중립"
        if max_ret >= success_threshold:
            result = "성공"
            if hit_idx is not None:
                delta = df["time"].iloc[i + hit_idx] - df["time"].iloc[i]
                hit_time = f"{int(delta.total_seconds()//60):02d}:{int(delta.total_seconds()%60):02d}"
        elif min_ret <= -success_threshold:
            result = "실패"
            if fail_hit_idx is not None:
                delta = df["time"].iloc[i + fail_hit_idx] - df["time"].iloc[i]
                hit_time = f"{int(delta.total_seconds()//60):02d}:{int(delta.total_seconds()%60):02d}"

        final_ret = (df["close"].iloc[end_idx] - base_price) / base_price * 100.0

        rows.append({
            "신호시간": df["time"].iloc[i].strftime("%Y-%m-%d %H:%M"),
            "기준종가": f"{int(base_price):,}",
            "RSI(13)": round(float(df["rsi"].iloc[i]), 1) if not np.isnan(df["rsi"].iloc[i]) else np.nan,
            "성공기준(%)": f"{success_threshold:.2f}%",
            "결과": result,
            "최종수익률(%)": f"{final_ret:.2f}%",
            "최저수익률(%)": f"{min_ret:.2f}%",
            "최고수익률(%)": f"{max_ret:.2f}%",
            "도달시간": hit_time
        })

    results_df = pd.DataFrame(rows)
    return signals, results_df

# -----------------------------
# 실행
# -----------------------------
try:
    df = get_upbit_data()
    signals, results_df = simulate(df)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Candlestick"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["rsi"], mode="lines", name="RSI(13)"), row=2, col=1)

    y_min, y_max = df["low"].min(), df["high"].max()
    for i in signals:
        t = df["time"].iloc[i]
        hi = df["high"].iloc[i]
        fig.add_shape(type="line",
                      x0=t, x1=t, y0=y_min, y1=y_max,
                      line=dict(color="blue", dash="dot"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[t], y=[hi * 1.0015],
            mode="text", text=["★"], textposition="top center",
            name="신호", showlegend=False
        ), row=1, col=1)

    st.plotly_chart(fig, use_container_width=True)

    if not results_df.empty:
        styled = results_df.style.apply(style_result_col, subset=["결과"])
        st.write(styled)
    else:
        st.info("표시할 신호가 없습니다.")

except Exception as e:
    st.error(f"오류: {e}")
