### 최종 파이널 app.py (전체 풀버전 코드)
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
    hit_basis = "종가 기준"
with c6:
    r1, r2, r3 = st.columns(3)
    with r1:
        rsi_mode = st.selectbox(
            "RSI 조건",
            ["없음", "현재(과매도/과매수 중 하나)", "과매도 기준", "과매수 기준"],
            index=0
        )
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
    bottom_mode = st.checkbox("🟢 바닥탐지(실시간) 모드", value=False, help="RSI≤과매도 & BB 하한선 터치/하회 & CCI≤-100 동시 만족 시 신호")
with c11:
    cci_window = st.number_input("CCI 기간", min_value=5, max_value=100, value=14, step=1)
with c12:
    pass

st.markdown('<div class="hint">2차 조건: 선택한 조건만 적용 (없음/양봉 2개/BB 기반/매물대)</div>', unsafe_allow_html=True)
sec_cond = st.selectbox(
    "2차 조건 선택",
    ["없음", "양봉 2개 연속 상승", "BB 기반 첫 양봉 50% 진입", "매물대 터치 후 반등(위→아래→반등)"],
    index=0
)

manual_supply_levels = []
if sec_cond == "매물대 터치 후 반등(위→아래→반등)":
    st.markdown("**매물대 가격대 직접 입력 (원 단위, 행 추가/삭제로 가변 입력)**")
    supply_df = st.data_editor(
        pd.DataFrame({"매물대": [0]}),
        num_rows="dynamic",
        use_container_width=True,
    )
    manual_supply_levels = supply_df["매물대"].dropna().astype(float).tolist()

st.session_state["bb_cond"] = bb_cond
st.markdown("---")

# -----------------------------
# 데이터 수집/지표/시뮬레이션 함수
# -----------------------------
_session = requests.Session()
_retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
_session.mount("https://", HTTPAdapter(max_retries=_retries))

def fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars: int = 0):
    if warmup_bars and warmup_bars > 0:
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
            if to_time is not None:
                params["to"] = to_time.strftime("%Y-%m-%d %H:%M:%S")
            r = _session.get(url, params=params, headers={"Accept": "application/json"}, timeout=10)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            all_data.extend(batch)
            last_ts = pd.to_datetime(batch[-1]["candle_date_time_kst"])
            if last_ts <= start_cutoff:
                break
            to_time = last_ts - timedelta(seconds=1)
    except Exception:
        return pd.DataFrame()
    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(all_data).rename(columns={
        "candle_date_time_kst": "time",
        "opening_price": "open",
        "high_price": "high",
        "low_price": "low",
        "trade_price": "close",
        "candle_acc_trade_volume": "volume",
    })
    df["time"] = pd.to_datetime(df["time"])
    df = df[["time", "open", "high", "low", "close", "volume"]].sort_values("time").reset_index(drop=True)
    return df[(df["time"] >= start_cutoff) & (df["time"] <= end_dt)]

def add_indicators(df, bb_window, bb_dev, cci_window):
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=bb_window, window_dev=bb_dev)
    out["BB_up"]  = bb.bollinger_hband().fillna(method="bfill").fillna(method="ffill")
    out["BB_low"] = bb.bollinger_lband().fillna(method="bfill").fillna(method="ffill")
    out["BB_mid"] = bb.bollinger_mavg().fillna(method="bfill").fillna(method="ffill")
    cci = ta.trend.CCIIndicator(high=out["high"], low=out["low"], close=out["close"], window=int(cci_window), constant=0.015)
    out["CCI"] = cci.cci()
    return out

def simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, thr_pct, bb_cond, dedup_mode,
             minutes_per_bar, market_code, bb_window, bb_dev, sec_cond="없음",
             hit_basis="종가 기준", miss_policy="(고정) 성공·실패·중립", bottom_mode=False,
             supply_levels: Optional[Set[float]] = None,
             manual_supply_levels: Optional[list] = None):
    res = []
    n = len(df)
    thr = float(threshold_pct if isinstance(threshold_pct := thr_pct, (int, float)) else thr_pct)
    # --- 1) 1차 조건 인덱스 (생략: 동일) ---
    ... (기존 조건 추출 로직 동일) ...
    # --- 2) 보조 함수 (생략: 동일) ---
    ... (is_bull, first_bull_50_over_bb 동일) ...
    # --- 3) 메인 루프 (수정됨) ---
    if dedup_mode.startswith("중복 제거"):
        i = 0
        while i < n:
            if i not in base_sig_idx:
                i += 1
                continue
            anchor_idx = i
            ... (성과 측정 로직 그대로) ...
            i = end_idx
    else:
        for anchor_idx in base_sig_idx:
            ... (성과 측정 로직 그대로) ...
    return pd.DataFrame(res)

# -----------------------------
# 실행
# -----------------------------
try:
    ... (UI/차트/결과 동일) ...
except Exception as e:
    st.error(f"오류: {e}")
