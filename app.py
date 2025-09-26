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

# -----------------------------
# 타이틀
# -----------------------------
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
        return rows
    except Exception:
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

# -----------------------------
# ① 기본 설정
# -----------------------------
st.markdown('<div class="section-title">① 기본 설정</div>', unsafe_allow_html=True)
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
st.markdown("---")
chart_box = st.container()

# -----------------------------
# ② 조건 설정
# -----------------------------
st.markdown('<div class="section-title">② 조건 설정</div>', unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
with c4:
    lookahead = st.slider("측정 캔들 수 (기준 이후 N봉)", 1, 60, 10)
with c5:
    threshold_pct = st.slider("성공/실패 기준 값(%)", 0.1, 5.0, 1.0, step=0.1)
    hit_basis = st.selectbox("성공 판정 기준", ["종가 기준", "고가 기준(스침 인정)", "종가 또는 고가"], index=0)
with c6:
    r1, r2, r3 = st.columns(3)
    with r1:
        rsi_mode = st.selectbox("RSI 조건", ["없음", "현재(과매도/과매수 중 하나)", "과매도 기준", "과매수 기준"], index=0)
    with r2:
        rsi_low = st.slider("과매도 RSI 기준", 0, 100, 30, step=1)
    with r3:
        rsi_high = st.slider("과매수 RSI 기준", 0, 100, 70, step=1)

c7, c8, c9 = st.columns(3)
with c7:
    bb_cond = st.selectbox("볼린저밴드 조건", ["없음", "상한선", "중앙선", "하한선"], index=0)
with c8:
    bb_window = st.number_input("BB 기간", min_value=5, max_value=100, value=30, step=1)
with c9:
    bb_dev = st.number_input("BB 승수", min_value=1.0, max_value=4.0, value=2.0, step=0.1)

c10, c11, c12 = st.columns(3)
with c10:
    bottom_mode = st.checkbox("🟢 바닥탐지 모드", value=False)
with c11:
    cci_window = st.number_input("CCI 기간", min_value=5, max_value=100, value=14, step=1)
with c12:
    cci_signal = st.number_input("CCI 신호선 기간", min_value=1, max_value=50, value=9, step=1)

c13, c14, c15 = st.columns(3)
with c13:
    show_cci_chart = st.checkbox("CCI 보조차트 표시", value=True)
with c14:
    div_mode = st.checkbox("다이버전스 탐지", value=False)
with c15:
    pass

sec_cond = st.selectbox("2차 조건 선택", ["없음", "양봉 2개 연속 상승", "BB 기반 첫 양봉 50% 진입"], index=0)
st.markdown("---")

# -----------------------------
# 데이터 수집 및 지표
# -----------------------------
_session = requests.Session()
_retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
_session.mount("https://", HTTPAdapter(max_retries=_retries))

def fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars=0):
    if warmup_bars > 0:
        start_cutoff = start_dt - timedelta(minutes=warmup_bars * minutes_per_bar)
    else:
        start_cutoff = start_dt
    if "minutes/" in interval_key:
        unit = interval_key.split("/")[1]
        url = f"https://api.upbit.com/v1/candles/minutes/{unit}"
    else:
        url = "https://api.upbit.com/v1/candles/days"
    all_data, to_time = [], None
    try:
        for _ in range(60):
            params = {"market": market_code, "count": 200}
            if to_time: params["to"] = to_time.strftime("%Y-%m-%d %H:%M:%S")
            r = _session.get(url, params=params, headers={"Accept": "application/json"}, timeout=10)
            r.raise_for_status()
            batch = r.json()
            if not batch: break
            all_data.extend(batch)
            last_ts = pd.to_datetime(batch[-1]["candle_date_time_kst"])
            if last_ts <= start_cutoff: break
            to_time = last_ts - timedelta(seconds=1)
    except Exception:
        return pd.DataFrame()
    if not all_data: return pd.DataFrame()
    df = pd.DataFrame(all_data).rename(columns={
        "candle_date_time_kst":"time","opening_price":"open","high_price":"high",
        "low_price":"low","trade_price":"close","candle_acc_trade_volume":"volume"
    })
    df["time"] = pd.to_datetime(df["time"])
    df = df[["time","open","high","low","close","volume"]].sort_values("time").reset_index(drop=True)
    return df[(df["time"]>=start_cutoff)&(df["time"]<=end_dt)]

def add_indicators(df, bb_window, bb_dev, cci_window, cci_signal):
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(out["close"], window=bb_window, window_dev=bb_dev)
    out["BB_up"], out["BB_low"], out["BB_mid"] = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_mavg()
    cci = ta.trend.CCIIndicator(out["high"], out["low"], out["close"], window=cci_window, constant=0.015)
    out["CCI"] = cci.cci()
    out["CCI_signal"] = out["CCI"].rolling(cci_signal).mean()
    return out

# -----------------------------
# 실행
# -----------------------------
try:
    if start_date > end_date: st.error("시작 날짜가 종료 날짜보다 이후입니다."); st.stop()
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())
    warmup_bars = max(13, bb_window, cci_window) * 5
    df_raw = fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars)
    if df_raw.empty: st.error("데이터 없음"); st.stop()
    df = add_indicators(df_raw, bb_window, bb_dev, cci_window, cci_signal)
    df = df[(df["time"]>=start_dt)&(df["time"]<=end_dt)].reset_index(drop=True)

    # -----------------------------
    # ③ 요약 & 차트
    # -----------------------------
    st.markdown('<div class="section-title">③ 요약 & 차트</div>', unsafe_allow_html=True)
    fig = make_subplots(rows=2 if show_cci_chart else 1, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        row_heights=[0.7,0.3] if show_cci_chart else [1.0])
    fig.add_trace(go.Candlestick(x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                 name="가격", increasing=dict(line=dict(color="red")), decreasing=dict(line=dict(color="blue"))), row=1,col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_up"], mode="lines", line=dict(color="orange"), name="BB 상단"), row=1,col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_low"], mode="lines", line=dict(color="blue"), name="BB 하단"), row=1,col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_mid"], mode="lines", line=dict(color="gray", dash="dot"), name="BB 중앙"), row=1,col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["RSI13"], mode="lines", line=dict(color="green"), name="RSI(13)", yaxis="y2"), row=1,col=1)

    if show_cci_chart:
        fig.add_trace(go.Scatter(x=df["time"], y=df["CCI"], mode="lines", line=dict(color="purple"), name="CCI"), row=2,col=1)
        fig.add_trace(go.Scatter(x=df["time"], y=df["CCI_signal"], mode="lines", line=dict(color="red", dash="dot"), name="CCI Signal"), row=2,col=1)
        fig.add_shape(type="line", x0=df["time"].iloc[0], x1=df["time"].iloc[-1], y0=-100, y1=-100, line=dict(color="blue", dash="dash"), row=2,col=1)
        fig.add_shape(type="line", x0=df["time"].iloc[0], x1=df["time"].iloc[-1], y0=100, y1=100, line=dict(color="red", dash="dash"), row=2,col=1)

    if div_mode and len(df)>10:
        for i in range(5,len(df)):
            if df["close"].iloc[i]<df["close"].iloc[i-5] and df["RSI13"].iloc[i]>df["RSI13"].iloc[i-5]:
                fig.add_trace(go.Scatter(x=[df["time"].iloc[i]], y=[df["low"].iloc[i]], mode="markers",
                                         marker=dict(size=12, color="lime", symbol="triangle-up"), name="강세 다이버전스"), row=1,col=1)

    fig.update_layout(height=700, xaxis_rangeslider_visible=False, legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # ④ 신호 결과
    # -----------------------------
    st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)
    st.info("※ 확장 옵션은 시각적 탐지용. 시뮬레이션 결과는 기존 신호 조건 기반입니다.")

except Exception as e:
    st.error(f"오류: {e}")
