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
from pytz import timezone
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
  .neutral-cell {color:#FF9800; font-weight:600;}
  table {border-collapse:collapse; width:100%;}
  th, td {border:1px solid #ddd; padding:6px; text-align:center;}
</style>
""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
st.title("📊 코인 시뮬레이션")
st.markdown("<div style='margin-bottom:10px; color:gray;'>※ 차트 점선: 신호~판정 구간, 성공 시 도달 지점에 ⭐ 마커</div>", unsafe_allow_html=True)

# -----------------------------
# 업비트 마켓 로드
# -----------------------------
@st.cache_data(ttl=3600)
def get_upbit_krw_markets():
    url = "https://api.upbit.com/v1/market/all"
    try:
        r = requests.get(url, params={"isDetails": "false"}, timeout=8)
        r.raise_for_status()
        items = r.json()
        rows = []
        for it in items:
            mk = it.get("market", "")
            if mk.startswith("KRW-"):
                sym = mk[4:]
                label = f'{it.get("korean_name","")} ({sym}) — {mk}'
                rows.append((label, mk))
        rows.sort(key=lambda x: (x[1] != "KRW-BTC", x[1]))
        if rows:
            return rows
    except Exception:
        pass
    return [("비트코인 (BTC) — KRW-BTC", "KRW-BTC")]

MARKET_LIST = get_upbit_krw_markets()
default_idx = next((i for i, (_, code) in enumerate(MARKET_LIST) if code == "KRW-BTC"), 0)

# -----------------------------
# 타임프레임
# -----------------------------
TF_MAP = {
    "1분": ("minutes/1", 1),
    "3분": ("minutes/3", 3),
    "5분": ("minutes/5", 5),
    "15분": ("minutes/15", 15),
    "30분": ("minutes/30", 30),
    "60분": ("minutes/60", 60),
    "일봉": ("days", 24 * 60),
}

dup_mode = st.radio("신호 중복 처리", ["중복 포함 (연속 신호 모두)", "중복 제거 (연속 동일 결과 1개)"], horizontal=True)

# 세션 상태 초기화
if "opt_view" not in st.session_state:
    st.session_state.opt_view = False
if "buy_price_text" not in st.session_state:
    st.session_state.buy_price_text = "0"
if "buy_price" not in st.session_state:
    st.session_state.buy_price = 0

# -----------------------------
# ① 기본 설정
# -----------------------------
st.markdown('<div class="section-title">① 기본 설정</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    market_label, market_code = st.selectbox("종목 선택", MARKET_LIST, index=default_idx, format_func=lambda x: x[0])
with c2:
    tf_label = st.selectbox("봉 종류 선택", list(TF_MAP.keys()), index=2)
with c3:
    KST = timezone("Asia/Seoul")
    today_kst = datetime.now(KST).date()
    default_start = today_kst - timedelta(days=1)
    start_date = st.date_input("시작 날짜", value=default_start)
with c4:
    end_date = st.date_input("종료 날짜", value=today_kst)

interval_key, minutes_per_bar = TF_MAP[tf_label]
st.markdown("---")

chart_box = st.container()

# -----------------------------
# 실행
# -----------------------------
try:
    if start_date > end_date:
        st.error("시작 날짜가 종료 날짜보다 이후입니다.")
        st.stop()

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())

    # --- 매수가 입력 포맷 콜백 ---
    def _format_buy_price():
        raw = st.session_state.get("buy_price_text", "0")
        digits = "".join(ch for ch in str(raw) if ch.isdigit())
        val = int(digits) if digits else 0
        st.session_state.buy_price = val
        st.session_state.buy_price_text = f"{val:,}"

    # -----------------------------
    # 차트 상단: (왼) 매수가 입력  |  (오) 최적화뷰 버튼
    # -----------------------------
    with chart_box:
        top_l, top_r = st.columns([7, 3])

        with top_l:
            st.text_input(
                "💰 매수가 입력",
                key="buy_price_text",
                on_change=_format_buy_price
            )
            buy_price = st.session_state.get("buy_price", 0)

        with top_r:
            st.markdown("<div style='margin-top:5px'></div>", unsafe_allow_html=True)
            label = "↩ 되돌아가기" if st.session_state.opt_view else "📈 최적화뷰"
            if st.button(label, key="btn_opt_view_top"):
                st.session_state.opt_view = not st.session_state.opt_view

    # -----------------------------
    # 차트 데이터 및 수익률 (예시 데이터)
    # -----------------------------
    df = pd.DataFrame({
        "time": pd.date_range(start=start_dt, end=end_dt, freq="min")[:100],
        "open": np.random.rand(100)*100,
        "high": np.random.rand(100)*100,
        "low": np.random.rand(100)*100,
        "close": np.random.rand(100)*100
    }).reset_index(drop=True)
    if buy_price > 0:
        df["수익률(%)"] = (df["close"]/buy_price - 1) * 100
    else:
        df["수익률(%)"] = np.nan

    n = len(df)
    if n == 0:
        st.info("표시할 데이터가 없습니다.")
        st.stop()

    fig = make_subplots(rows=1, cols=1)
    if buy_price > 0:
        pct = df["수익률(%)"].fillna(0).astype(float).to_numpy()
        colors = np.where(pct > 0, "red", "blue").tolist()
        hovertext = [f"<span style='color:{c}'>수익률: {v:.2f}%</span>" for v, c in zip(pct, colors)]
    else:
        hovertext = ["수익률: 0.00%" for _ in range(n)]

    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="가격",
        increasing=dict(line=dict(color="red", width=1.1)),
        decreasing=dict(line=dict(color="blue", width=1.1)),
        hovertext=hovertext,
        hoverinfo="text"
    ))

    if buy_price > 0:
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["close"], mode="markers",
            marker=dict(opacity=0),
            showlegend=False,
            hovertext=hovertext,
            hoverinfo="text",
            name="PnL Hover"
        ))

    # -----------------------------
    # ③ 요약 & 차트
    # -----------------------------
    st.markdown('<div class="section-title">③ 요약 & 차트</div>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # ④ 신호 결과 (최신 순)
    # -----------------------------
    st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)
    res = pd.DataFrame({"시간": df["time"], "수익률(%)": df["수익률(%)"].round(2)})
    st.dataframe(res)

except Exception as e:
    st.error(f"오류: {e}")
