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
from typing import Optional, Set

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

# 타이틀
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

# -----------------------------
# 상단: 신호 중복 처리
# -----------------------------
dup_mode = st.radio(
    "신호 중복 처리",
    ["중복 포함 (연속 신호 모두)", "중복 제거 (연속 동일 결과 1개)"],
    horizontal=True,
)

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

# ✅ 차트 컨테이너
chart_box = st.container()

# -----------------------------
# ② 조건 설정 (생략 - 원본 동일)
# -----------------------------
# ... 조건 설정 코드 동일 (RSI, BB, 바닥탐지, 2차 조건 등) ...

# -----------------------------
# 데이터 수집/지표/시뮬레이션 함수
# -----------------------------
# ... fetch_upbit_paged, add_indicators, build_supply_levels_3m_daily, simulate (원본 유지) ...

# -----------------------------
# 실행
# -----------------------------
try:
    if start_date > end_date:
        st.error("시작 날짜가 종료 날짜보다 이후입니다.")
        st.stop()

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())
    warmup_bars = 60

    df_raw = fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars)
    if df_raw.empty:
        st.error("데이터가 없습니다.")
        st.stop()

    df_ind = add_indicators(df_raw, 30, 2.0, 14)
    df = df_ind[(df_ind["time"] >= start_dt) & (df_ind["time"] <= end_dt)].reset_index(drop=True)

    # -----------------------------
    # 수익률 customdata 준비
    # -----------------------------
    buy_price = st.session_state.get("buy_price", 0)
    if buy_price > 0:
        pnl_vals = (df["close"] / buy_price - 1) * 100
    else:
        pnl_vals = pd.Series([0] * len(df))
    pnl_vals = pnl_vals.round(1)
    pnl_colors = np.where(pnl_vals >= 0, "red", "blue")
    pnl_cd = np.stack([pnl_vals, pnl_colors], axis=-1)

    # -----------------------------
    # 차트
    # -----------------------------
    fig = make_subplots(rows=1, cols=1)

    # 캔들스틱
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="가격",
        increasing=dict(line=dict(color="red", width=1.1)),
        decreasing=dict(line=dict(color="blue", width=1.1)),
        customdata=pnl_cd,
        hovertemplate="<span style='color:%{customdata[1]};'>수익률: %{customdata[0]:.1f}%</span><extra></extra>"
    ))

    # BB 선들
    for col, name, color in [("BB_up", "BB 상단", "#FFB703"),
                             ("BB_low", "BB 하단", "#219EBC"),
                             ("BB_mid", "BB 중앙", "#8D99AE")]:
        fig.add_trace(go.Scatter(
            x=df["time"], y=df[col], mode="lines",
            line=dict(color=color, width=1.2 if "중앙" not in name else 1.0, dash="dot" if "중앙" in name else "solid"),
            name=name,
            customdata=pnl_cd,
            hovertemplate="<span style='color:%{customdata[1]};'>수익률: %{customdata[0]:.1f}%</span><extra></extra>"
        ))

    # 빈영역 Hover용 투명 trace
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["close"], mode="lines",
        line=dict(color="rgba(0,0,0,0)", width=1e-3),
        showlegend=False,
        customdata=pnl_cd,
        hovertemplate="<span style='color:%{customdata[1]};'>수익률: %{customdata[0]:.1f}%</span><extra></extra>"
    ))

    # -----------------------------
    # 차트 출력
    # -----------------------------
    with chart_box:
        top_l, top_r = st.columns([4, 1])
        with top_l:
            buy_price = st.number_input(
                "💰 매수가 입력",
                min_value=0,
                value=st.session_state.get("buy_price", 0),
                step=1000,
                format="%d",
                key="buy_price_num"
            )
            st.session_state.buy_price = buy_price
            st.session_state.buy_price_text = f"{buy_price:,}" if buy_price > 0 else "0"
            st.markdown("<style>div[data-testid='stNumberInput'] {width:220px !important;}</style>", unsafe_allow_html=True)
        with top_r:
            st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)
            label = "↩ 되돌아가기" if st.session_state.opt_view else "📈 최적화뷰"
            if st.button(label, key="btn_opt_view_top"):
                st.session_state.opt_view = not st.session_state.opt_view

        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"오류: {e}")
