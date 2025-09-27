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
</style>
""", unsafe_allow_html=True)

st.title("📊 코인 시뮬레이션")

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
                rows.append((f"{it.get('korean_name','')} ({mk[4:]}) — {mk}", mk))
        return sorted(rows, key=lambda x: (x[1] != "KRW-BTC", x[1]))
    except Exception:
        return [("비트코인 (BTC) — KRW-BTC", "KRW-BTC")]

MARKET_LIST = get_upbit_krw_markets()
default_idx = next((i for i, (_, code) in enumerate(MARKET_LIST) if code=="KRW-BTC"), 0)

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
    "일봉": ("days", 24*60),
}

# -----------------------------
# ① 기본 설정 (기존 UI/UX 유지)
# -----------------------------
st.markdown('<div class="section-title">① 기본 설정</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    market_label, market_code = st.selectbox("종목 선택", MARKET_LIST, index=default_idx, format_func=lambda x: x[0])
with c2:
    tf_label = st.selectbox("봉 종류 선택", list(TF_MAP.keys()), index=2)
with c3:
    start_date = st.date_input("시작 날짜", value=datetime.now().date() - timedelta(days=1))
with c4:
    end_date = st.date_input("종료 날짜", value=datetime.now().date())

interval_key, minutes_per_bar = TF_MAP[tf_label]

# -----------------------------
# ② 조건 설정 (원래 코드 유지)
# -----------------------------
st.markdown('<div class="section-title">② 조건 설정</div>', unsafe_allow_html=True)

lookahead = st.slider("측정 캔들 수 (기준 이후 N봉)", 1, 60, 10)
threshold_pct = st.slider("성공/실패 기준 값(%)", 0.1, 5.0, 1.0, step=0.1)
hit_basis = st.selectbox("성공 판정 기준", ["종가 기준", "고가 기준(스침 인정)", "종가 또는 고가"], index=0)
rsi_mode = st.selectbox("RSI 조건", ["없음", "현재(과매도/과매수 중 하나)", "과매도 기준", "과매수 기준"], index=0)
rsi_low = st.slider("과매도 RSI 기준", 0, 100, 30, step=1)
rsi_high = st.slider("과매수 RSI 기준", 0, 100, 70, step=1)
bb_cond = st.selectbox("볼린저밴드 조건", ["없음", "상한선", "중앙선", "하한선"], index=0)
bb_window = st.number_input("BB 기간", min_value=5, max_value=100, value=30, step=1)
bb_dev = st.number_input("BB 승수", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
bottom_mode = st.checkbox("🟢 바닥탐지(실시간) 모드", value=False)
cci_window = st.number_input("CCI 기간", min_value=5, max_value=100, value=14, step=1)
sec_cond = st.selectbox("2차 조건 선택", ["없음", "양봉 2개 연속 상승", "BB 기반 첫 양봉 50% 진입", "매물대 터치 후 반등(위→아래→반등)"], index=0)
supply_filter = None
dup_mode = st.radio("신호 중복 처리", ["중복 포함 (연속 신호 모두)", "중복 제거 (연속 동일 결과 1개)"], horizontal=True)

st.markdown("---")

# -----------------------------
# 실행
# -----------------------------
try:
    if start_date > end_date:
        st.error("시작 날짜가 종료 날짜보다 이후입니다.")
        st.stop()

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt   = datetime.combine(end_date, datetime.max.time())
    warmup_bars = max(13, bb_window, int(cci_window)) * 5

    # 데이터 수집
    df_raw = fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars)
    if df_raw.empty:
        st.error("데이터가 없습니다.")
        st.stop()

    df_ind = add_indicators(df_raw, bb_window, bb_dev, cci_window)
    df = df_ind[(df_ind["time"] >= start_dt) & (df_ind["time"] <= end_dt)].reset_index(drop=True)

    # ✅ 최대 수집 가능 일수 계산
    if "minutes/" in interval_key:
        unit = int(interval_key.split("/")[1])
        max_days = (12000 * unit) / 1440
    else:
        max_days = 12000  # 일봉은 12000일 가능

    # ✅ 실제 수집 범위 확인 & 안내
    if not df.empty:
        actual_start, actual_end = df["time"].min(), df["time"].max()
        if actual_start > start_dt or actual_end < end_dt:
            st.warning(
                f"⚠ 선택한 기간({start_dt.date()} ~ {end_dt.date()}) 전체 데이터를 가져오지 못했습니다.\n"
                f"- 봉 단위: {tf_label}, 이론상 최대 수집 가능 일수 ≈ {int(max_days)}일\n"
                f"- 실제 수집 범위: {actual_start.date()} ~ {actual_end.date()}"
            )

    # -----------------------------
    # ③ 요약 & 차트 (원래 코드 유지)
    # -----------------------------
    st.markdown('<div class="section-title">③ 요약 & 차트</div>', unsafe_allow_html=True)
    if df.empty:
        st.info("데이터가 없습니다.")
    else:
        chart_box = st.container()
        with chart_box:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                row_heights=[0.7, 0.3], specs=[[{"secondary_y": False}], [{"secondary_y": False}]])
            fig.add_trace(go.Candlestick(x=df["time"], open=df["open"], high=df["high"],
                                         low=df["low"], close=df["close"], name="Candlestick"), row=1, col=1)
            st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # ④ 신호 결과 (최신 순, 원래 코드 유지)
    # -----------------------------
    st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)
    if df.empty:
        st.info("신호 결과가 없습니다.")
    else:
        result_df = simulate(df, lookahead, threshold_pct, hit_basis, rsi_mode, rsi_low, rsi_high,
                             bb_cond, bb_window, bb_dev, bottom_mode, cci_window,
                             sec_cond, supply_filter, dup_mode)
        if result_df.empty:
            st.info("신호 결과가 없습니다.")
        else:
            st.dataframe(result_df)

except Exception as e:
    st.error(f"오류: {e}")
