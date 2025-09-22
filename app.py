# app.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import ta
from datetime import datetime, timedelta
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
  .neutral-cell {color:#059669; font-weight:600;}
  table {border-collapse:collapse; width:100%;}
  th, td {border:1px solid #ddd; padding:6px; text-align:center;}
</style>
""", unsafe_allow_html=True)

# 제목 고정
st.title("📊 코인 시뮬레이션")
st.markdown("<div style='margin-bottom:10px; color:gray;'>※ 차트 내 점선은 신호 흐름선, 성공 시 도달 지점에 ⭐ 별표 표시</div>", unsafe_allow_html=True)

# -----------------------------
# 업비트 마켓 로드
# -----------------------------
@st.cache_data(ttl=3600)
def get_upbit_krw_markets():
    url = "https://api.upbit.com/v1/market/all"
    try:
        r = requests.get(url, params={"isDetails":"false"}, timeout=8)
        r.raise_for_status()
        items = r.json()
        rows = []
        for it in items:
            mk = it.get("market","")
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
default_idx = next((i for i,(_,code) in enumerate(MARKET_LIST) if code=="KRW-BTC"), 0)

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
# 신호 중복 처리
# -----------------------------
dup_mode = st.radio(
    "신호 중복 처리",
    ["중복 포함 (연속 신호 모두)", "중복 제거 (연속 동일 결과 1개)"],
    horizontal=True,
)

# -----------------------------
# 기본 설정
# -----------------------------
st.markdown('<div class="section-title">① 기본 설정</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    market_label, market_code = st.selectbox("종목 선택", MARKET_LIST, index=default_idx, format_func=lambda x: x[0])
with c2:
    tf_label = st.selectbox("봉 종류 선택", list(TF_MAP.keys()), index=2)
with c3:
    default_start = (datetime.today() - timedelta(days=1)).date()
    start_date = st.date_input("시작 날짜", value=default_start)
    end_date = st.date_input("종료 날짜", value=datetime.today().date())

interval_key, minutes_per_bar = TF_MAP[tf_label]
st.markdown("---")

# -----------------------------
# 조건 설정
# -----------------------------
st.markdown('<div class="section-title">② 조건 설정</div>', unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
with c4:
    lookahead = st.slider("측정 캔들 수 (기준 이후 N봉)", 1, 60, 10)
with c5:
    threshold_pct = st.slider("성공/실패 기준 값(%)", 0.1, 5.0, 1.0, step=0.1)
with c6:
    # RSI 조건 표현 개선 (이상/이하/없음)
    rsi_mode = st.selectbox("RSI 조건", ["없음", "이하", "이상"], index=0)
    rsi_level = st.slider("RSI 기준값(정수)", 0, 100, 30, step=1)

c7, c8, c9 = st.columns(3)
with c7:
    bb_cond = st.selectbox(
        "볼린저밴드 조건",
        ["없음", "상한선", "중앙선", "하한선"],
        index=0,
    )
with c8:
    bb_window = st.number_input("BB 기간", min_value=5, max_value=100, value=30, step=1)
with c9:
    bb_dev = st.number_input("BB 승수", min_value=1.0, max_value=4.0, value=2.0, step=0.1)

st.markdown('<div class="hint">2차 조건: 1차 조건(RSI·볼린저밴드) 충족 후 추가 필터</div>', unsafe_allow_html=True)
sec1, sec2 = st.columns(2)
with sec1:
    use_bull2 = st.checkbox("양봉 2개 연속 상승 적용", value=False)
with sec2:
    allow_other_secondary = st.checkbox("다른 2차 조건 확장 허용", value=False)

st.session_state["bb_cond"]  = bb_cond
st.markdown("---")

# -----------------------------
# 데이터 수집 / 지표 / 시뮬레이션 함수
# -----------------------------
def estimate_calls(start_dt, end_dt, minutes_per_bar):
    mins = max(1, int((end_dt - start_dt).total_seconds() // 60))
    bars = max(1, mins // minutes_per_bar)
    return bars // 200 + 1

_session = requests.Session()
_retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])
_session.mount("https://", HTTPAdapter(max_retries=_retries))

@st.cache_data(ttl=120, show_spinner=False)
def fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar):
    if "minutes/" in interval_key:
        unit = interval_key.split("/")[1]
        url = f"https://api.upbit.com/v1/candles/minutes/{unit}"
    else:
        url = "https://api.upbit.com/v1/candles/days"
    calls_est = estimate_calls(start_dt, end_dt, minutes_per_bar)
    max_calls = min(calls_est + 2, 60)
    req_count = 200
    all_data, to_time = [], end_dt
    try:
        for _ in range(max_calls):
            params = {"market": market_code, "count": req_count, "to": to_time.strftime("%Y-%m-%d %H:%M:%S")}
            r = _session.get(url, params=params, headers={"Accept":"application/json"}, timeout=10)
            r.raise_for_status()
            batch = r.json()
            if not batch: break
            all_data.extend(batch)
            last_ts = pd.to_datetime(batch[-1]["candle_date_time_kst"])
            if last_ts <= start_dt: break
            to_time = last_ts - timedelta(seconds=1)
    except Exception:
        return pd.DataFrame()
    if not all_data: return pd.DataFrame()
    df = pd.DataFrame(all_data).rename(columns={
        "candle_date_time_kst":"time","opening_price":"open","high_price":"high",
        "low_price":"low","trade_price":"close","candle_acc_trade_volume":"volume"})
    df["time"] = pd.to_datetime(df["time"])
    df = df[["time","open","high","low","close","volume"]].sort_values("time").reset_index(drop=True)
    return df[(df["time"] >= start_dt) & (df["time"] <= end_dt)]

def add_indicators(df, bb_window, bb_dev):
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=bb_window, window_dev=bb_dev)
    out["BB_up"]  = bb.bollinger_hband().fillna(method="bfill").fillna(method="ffill")
    out["BB_low"] = bb.bollinger_lband().fillna(method="bfill").fillna(method="ffill")
    out["BB_mid"] = bb.bollinger_mavg().fillna(method="bfill").fillna(method="ffill")
    return out

# -----------------------------
# 시뮬레이션 (RSI 이상/이하 반영)
# -----------------------------
def simulate(df, rsi_mode, rsi_level, lookahead, thr_pct, bb_cond, dedup_mode,
             minutes_per_bar, market_code, bb_window, bb_dev, use_bull2=False, allow_other_secondary=False):
    res=[]
    n=len(df); thr=float(thr_pct)

    # --- RSI 판정 (이상/이하) ---
    if rsi_mode == "이하":
        rsi_idx = df.index[df["RSI13"] <= float(rsi_level)].tolist()
    elif rsi_mode == "이상":
        rsi_idx = df.index[df["RSI13"] >= float(rsi_level)].tolist()
    else:
        rsi_idx = []

    # ... (나머지 simulate 로직은 동일, 기존 코드 유지) ...

    return pd.DataFrame(res)

# -----------------------------
# 실행
# -----------------------------
try:
    if start_date > end_date:
        st.error("시작 날짜가 종료 날짜보다 이후입니다.")
        st.stop()
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt   = datetime.combine(end_date,   datetime.max.time())

    if rsi_mode=="없음" and bb_cond=="없음":
        st.markdown('<div class="section-title">③ 요약 & 차트</div>', unsafe_allow_html=True)
        st.info("대기중..")
        st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)
        st.info("대기중..")
        st.stop()

    df = fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar)
    if df.empty:
        st.error("데이터가 없습니다.")
        st.stop()

    df = add_indicators(df, bb_window, bb_dev)
    bb_cond = st.session_state.get("bb_cond", bb_cond)

    # 🔥 수정: 볼린저밴드 없음이어도 RSI 단독 + 양봉 2연속 조건 허용
    # 에러 출력/중단 로직 제거됨

    # (이후 차트/표 출력 로직은 기존 코드 유지)
except Exception as e:
    st.error(f"오류: {e}")
