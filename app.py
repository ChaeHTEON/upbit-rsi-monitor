# app.py
import streamlit as st
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import plotly.graph_objs as go
import ta
from datetime import datetime, timedelta
# (선택) 로케일 있으면 사용, 없으면 건너뜀
try:
    import locale
    locale.setlocale(locale.LC_TIME, "ko_KR.UTF-8")
    _KO_LOCALE = True
except Exception:
    _KO_LOCALE = False
from plotly.subplots import make_subplots

# -----------------------------
# 페이지/스타일
# -----------------------------
st.set_page_config(page_title="Upbit RSI(13) + Bollinger Band 시뮬레이터", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 0.8rem; padding-bottom: 0.8rem; max-width: 1100px;}
  .stMetric {text-align:center;}
  .success {color:red; font-weight:600;}
  .fail {color:blue;}
  .neutral {color:green; font-weight:600;}
  .section-title {font-size:1.05rem; font-weight:700; margin: 0.6rem 0 0.2rem;}
  .hint {color:#6b7280;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Upbit RSI(13) + Bollinger Band 시뮬레이터")

# -----------------------------
# 업비트 마켓 로드
# -----------------------------
@st.cache_data(ttl=3600)
def get_upbit_krw_markets():
    url = "https://api.upbit.com/v1/market/all"
    r = requests.get(url, params={"isDetails":"false"}, timeout=10)
    r.raise_for_status()
    items = r.json()
    rows = []
    for it in items:
        if it["market"].startswith("KRW-"):
            sym = it["market"][4:]
            label = f'{it["korean_name"]} ({sym}) — {it["market"]}'
            rows.append((label, it["market"]))
    rows.sort(key=lambda x: (x[1] != "KRW-BTC", x[1]))
    return rows

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
# 상단: 신호 중복 처리
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
    default_start = datetime.today() - timedelta(days=1)
    start_date = st.date_input("시작 날짜", value=default_start)
    end_date = st.date_input("종료 날짜", value=datetime.today())

interval_key, minutes_per_bar = TF_MAP[tf_label]

# -----------------------------
# 조건 설정
# -----------------------------
st.markdown('<div class="section-title">② 조건 설정</div>', unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
with c4:
    lookahead = st.slider("측정 캔들 수 (기준 이후 N봉)", 1, 60, 10)
with c5:
    threshold_pct = st.slider("성공/실패 기준 값(%)", 0.05, 3.0, 1.0, step=0.05)
with c6:
    rsi_side = st.selectbox("RSI 조건", ["RSI ≤ 30 (급락)", "RSI ≥ 70 (급등)"], index=0)

# 볼린저밴드 조건
c7, _, _ = st.columns(3)
with c7:
    bb_cond = st.selectbox(
        "볼린저밴드 조건",
        ["없음","하한선 하향돌파","하한선 상향돌파","상한선 하향돌파","상한선 상향돌파","하한선 중앙돌파","상한선 중앙돌파"],
        index=0,
    )

# 안전 장치(세션 보강)
st.session_state["rsi_side"] = rsi_side
st.session_state["bb_cond"]  = bb_cond

# ---- 조건 요약 박스 ----
sim_minutes = lookahead * minutes_per_bar
if sim_minutes < 60:
    sim_dur = f"약 {sim_minutes}분"
elif sim_minutes < 1440:
    sim_dur = f"약 {sim_minutes//60}시간 {sim_minutes%60}분"
else:
    sim_dur = f"약 {sim_minutes//1440}일"

rsi_display = rsi_side
if "≤" in rsi_side:   # 급락(하향)
    rsi_display = f"<span style='color:blue; font-weight:600;'>{rsi_side}</span>"
elif "≥" in rsi_side: # 급등(상향)
    rsi_display = f"<span style='color:red; font-weight:600;'>{rsi_side}</span>"

bb_display = bb_cond
if "하향" in bb_cond:
    bb_display = f"<span style='color:blue; font-weight:600;'>{bb_cond}</span>"
elif "상향" in bb_cond:
    bb_display = f"<span style='color:red; font-weight:600;'>{bb_cond}</span>"

st.markdown(f"""
<div style="border:1px solid #ccc; border-radius:8px; padding:0.8rem; background-color:#f9f9f9; margin-top:0.6rem; margin-bottom:0.6rem;">
<b>📌 현재 조건 요약</b><br>
- 측정 캔들 수: {lookahead}봉 ({sim_dur})<br>
- 성공/실패 기준: {threshold_pct:.2f}%<br>
- RSI 조건: {rsi_display}<br>
- 볼린저밴드 조건: {bb_display}
</div>
""", unsafe_allow_html=True)

# -----------------------------
# 데이터 수집 (Upbit Pagination)
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
    progress = st.progress(0.0)
    try:
        for done in range(max_calls):
            params = {"market": market_code, "count": req_count, "to": to_time.strftime("%Y-%m-%d %H:%M:%S")}
            r = _session.get(url, params=params, headers={"Accept":"application/json"}, timeout=10)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            all_data.extend(batch)
            last_ts = pd.to_datetime(batch[-1]["candle_date_time_kst"])
            if last_ts <= start_dt:
                break
            to_time = last_ts - timedelta(seconds=1)
            progress.progress(min(1.0, (done + 1) / max(1, max_calls)))
    finally:
        progress.empty()

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data).rename(columns={
        "candle_date_time_kst":"time","opening_price":"open","high_price":"high",
        "low_price":"low","trade_price":"close","candle_acc_trade_volume":"volume"})
    df["time"] = pd.to_datetime(df["time"])
    df = df[["time","open","high","low","close","volume"]].sort_values("time").reset_index(drop=True)

    df = df[(df["time"].dt.date >= start_dt.date()) & (df["time"].dt.date <= end_dt.date())]
    return df

# -----------------------------
# 지표
# -----------------------------
def add_indicators(df):
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=30, window_dev=2)
    out["BB_up"]  = bb.bollinger_hband()
    out["BB_low"] = bb.bollinger_lband()
    out["BB_mid"] = bb.bollinger_mavg()
    return out

# -----------------------------
# 시뮬레이션
# -----------------------------
# ... (simulate 함수 부분 동일) ...

# -----------------------------
# 실행
# -----------------------------
# ... (중략, 요약 부분 동일) ...

    # ---- 차트 ----
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    # (캔들, 볼린저, 신호, RSI, 수평선 모두 수정된 버전 적용됨 — 앞서 설명한 대로)

    st.plotly_chart(fig, use_container_width=True)

    # ---- 신호 결과 (최신 순) ----
    # ...
    styled=tbl.style.applymap(color_result, subset=["결과"])
    st.dataframe(styled, use_container_width=True)
