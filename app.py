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

dup_mode = st.radio("신호 중복 처리",
    ["중복 포함 (연속 신호 모두)", "중복 제거 (연속 동일 결과 1개)"], horizontal=True)

# -----------------------------
# 기본 설정
# -----------------------------
c1, c2, c3 = st.columns(3)
with c1:
    market_label, market_code = st.selectbox("종목 선택", MARKET_LIST, index=default_idx, format_func=lambda x: x[0])
with c2:
    tf_label = st.selectbox("봉 종류 선택", list(TF_MAP.keys()), index=2)
with c3:
    KST = timezone("Asia/Seoul")
    today_kst = datetime.now(KST).date()
    default_start = today_kst - timedelta(days=1)
    start_date = st.date_input("시작 날짜", value=default_start)
    end_date = st.date_input("종료 날짜", value=today_kst)

interval_key, minutes_per_bar = TF_MAP[tf_label]
chart_box = st.container()

# -----------------------------
# 조건 설정 (기존 위치 유지)
# -----------------------------
c4, c5, c6 = st.columns(3)
with c4:
    lookahead = st.slider("측정 캔들 수 (기준 이후 N봉)", 1, 60, 10)
with c5:
    threshold_pct = st.slider("성공/실패 기준 값(%)", 0.1, 5.0, 1.0, step=0.1)
    hit_basis = st.selectbox("성공 판정 기준",
        ["종가 기준", "고가 기준(스침 인정)", "종가 또는 고가"], index=0)
with c6:
    rsi_mode = st.selectbox("RSI 조건",
        ["없음", "현재(과매도/과매수 중 하나)", "과매도 기준", "과매수 기준"], index=0)
    rsi_low = st.slider("과매도 RSI 기준", 0, 100, 30, step=1)
    rsi_high = st.slider("과매수 RSI 기준", 0, 100, 70, step=1)

c7, c8, c9 = st.columns(3)
with c7:
    bb_cond = st.selectbox("볼린저밴드 조건", ["없음", "상한선", "중앙선", "하한선"], index=0)
with c8:
    bb_window = st.number_input("BB 기간", min_value=5, max_value=100, value=30, step=1)
with c9:
    bb_dev = st.number_input("BB 승수", min_value=1.0, max_value=4.0, value=2.0, step=0.1)

sec_cond = st.selectbox("2차 조건 선택", ["없음", "양봉 2개 연속 상승", "BB 기반 첫 양봉 50% 진입"], index=0)
st.session_state["bb_cond"] = bb_cond

# -----------------------------
# 실행
# -----------------------------
try:
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())

    # 예시 데이터
    df = pd.DataFrame({
        "time": pd.date_range(start_dt, end_dt, freq="5min"),
        "open": np.random.randint(280000, 290000, 100),
        "high": np.random.randint(280000, 290000, 100),
        "low": np.random.randint(280000, 290000, 100),
        "close": np.random.randint(280000, 290000, 100),
    })
    df_plot = df.copy()

    # -----------------------------
    # 매수가 입력 + 최적화뷰 버튼 (차트 상단 우측으로 이동)
    # -----------------------------
    top_l, top_r = st.columns([4, 2])
    with top_l:
        buy_price = st.number_input("💰 매수가 입력", min_value=0, value=0, step=1, format="%d")
    with top_r:
        if "opt_view" not in st.session_state:
            st.session_state.opt_view = False
        if st.button("↩ 되돌아가기" if st.session_state.opt_view else "📈 최적화뷰"):
            st.session_state.opt_view = not st.session_state.opt_view

    # -----------------------------
    # 차트
    # -----------------------------
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(
        x=df_plot["time"], open=df_plot["open"], high=df_plot["high"],
        low=df_plot["low"], close=df_plot["close"],
        name="가격", increasing=dict(line=dict(color="red")),
        decreasing=dict(line=dict(color="blue"))
    ))

    # 수익률 계산 및 표시
    if buy_price > 0:
        cur_price = df_plot["close"].iloc[-1]
        pnl = (cur_price / buy_price - 1) * 100
        color = "red" if pnl >= 0 else "blue"
        st.markdown(f"<span style='color:{color}; font-weight:600'>수익률: {pnl:.1f}%</span>", unsafe_allow_html=True)

        # 빈 영역 hover trace (PnL만)
        fig.add_trace(go.Scatter(
            x=df_plot["time"], y=df_plot["close"],
            mode="lines", line=dict(color="rgba(0,0,0,0)", width=1e-3),
            customdata=np.expand_dims((df_plot["close"] / buy_price - 1) * 100, axis=-1),
            hovertemplate="매수가 대비 수익률: %{customdata[0]:.1f}%<extra></extra>",
            showlegend=False
        ))

    # 최적화뷰 적용 (즉시 반영)
    if st.session_state.opt_view and len(df_plot) > 0:
        window_n = max(int(len(df_plot) * 0.15), 200)
        start_idx = max(len(df_plot) - window_n, 0)
        x_start, x_end = df_plot.iloc[start_idx]["time"], df_plot.iloc[-1]["time"]
        fig.update_xaxes(range=[x_start, x_end], fixedrange=False)

    fig.update_layout(
        dragmode="pan",
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=30, r=30, t=30, b=40),
        yaxis=dict(title="가격"),
        hovermode="closest",
        uirevision="chart-static"
    )
    chart_box.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})

except Exception as e:
    st.error(f"오류: {e}")
