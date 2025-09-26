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
# ② 조건 설정
# -----------------------------
st.markdown('<div class="section-title">② 조건 설정</div>', unsafe_allow_html=True)
c5, c6, c7 = st.columns(3)
with c5:
    lookahead = st.slider("측정 캔들 수 (기준 이후 N봉)", 1, 60, 10)
with c6:
    threshold_pct = st.slider("성공/실패 기준 값(%)", 0.1, 5.0, 1.0, step=0.1)
    hit_basis = st.selectbox("성공 판정 기준", ["종가 기준", "고가 기준(스침 인정)", "종가 또는 고가"], index=0)
with c7:
    rsi_mode = st.selectbox("RSI 조건", ["없음", "현재(과매도/과매수 중 하나)", "과매도 기준", "과매수 기준"], index=0)
    rsi_low = st.slider("과매도 RSI 기준", 0, 100, 30, step=1)
    rsi_high = st.slider("과매수 RSI 기준", 0, 100, 70, step=1)

c8, c9, c10 = st.columns(3)
with c8:
    bb_cond = st.selectbox("볼린저밴드 조건", ["없음", "상한선", "중앙선", "하한선"], index=0)
with c9:
    bb_window = st.number_input("BB 기간", min_value=5, max_value=100, value=30, step=1)
with c10:
    bb_dev = st.number_input("BB 승수", min_value=1.0, max_value=4.0, value=2.0, step=0.1)

# -----------------------------
# 실행
# -----------------------------
try:
    if start_date > end_date:
        st.error("시작 날짜가 종료 날짜보다 이후입니다.")
        st.stop()

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())

    # -----------------------------
    # 매수가 입력 + 최적화뷰 버튼 (기본 설정 UI 크기와 동일)
    # -----------------------------
    c11, c12 = st.columns([3,1])
    with c11:
        buy_price_text = st.text_input("💰 매수가 입력", value="0")
        try:
            buy_price = int(buy_price_text.replace(",", ""))
        except ValueError:
            buy_price = 0
        buy_price_text = f"{buy_price:,}"
    with c12:
        label = "↩ 되돌아가기" if st.session_state.opt_view else "📈 최적화뷰"
        if st.button(label, key="btn_opt_view_top"):
            st.session_state.opt_view = not st.session_state.opt_view

    # -----------------------------
    # 차트 데이터 및 수익률
    # -----------------------------
    df = pd.DataFrame({
        "time": pd.date_range(start=start_dt, end=end_dt, freq="min"),
        "open": np.random.rand(100)*100,
        "high": np.random.rand(100)*100,
        "low": np.random.rand(100)*100,
        "close": np.random.rand(100)*100
    })
    if buy_price > 0:
        df["수익률(%)"] = (df["close"]/buy_price - 1) * 100
    else:
        df["수익률(%)"] = np.nan

    fig = make_subplots(rows=1, cols=1)
    if buy_price > 0:
        hovertext = []
        for p in df["수익률(%)"].fillna(0):
            color = "red" if p > 0 else "blue"
            hovertext.append(f"<span style='color:{color}'>수익률: {p:.2f}%</span>")
    else:
        hovertext = ["수익률: 0.00%" for _ in df["time"]]

    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="가격", increasing=dict(line=dict(color="red")), decreasing=dict(line=dict(color="blue")),
        hovertext=hovertext, hoverinfo="text"
    ))

    if buy_price > 0:
        colors = ["red" if p > 0 else "blue" for p in df["수익률(%)"].fillna(0)]
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["close"], mode="markers",
            marker=dict(opacity=0, color=colors),
            showlegend=False,
            hovertext=[f"<span style='color:{c}'>수익률: {p:.2f}%</span>" for p, c in zip(df["수익률(%)"].fillna(0), colors)],
            hoverinfo="text", name="PnL Hover"
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
