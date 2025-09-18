# app.py
import streamlit as st
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
import plotly.graph_objs as go
import ta
from datetime import datetime, timedelta
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

# -----------------------------
# 타이틀 + 설정 버튼
# -----------------------------
c0, c1 = st.columns([9, 1])
with c0:
    st.title("📊 Upbit RSI(13) + Bollinger Band 시뮬레이터")
with c1:
    if st.button("⚙️ 설정", key="settings_btn"):
        st.session_state["show_settings"] = True

# Modal 팝업
if st.session_state.get("show_settings", False):
    with st.dialog("⚙️ RSI & Bollinger Band 설정 안내"):
        st.markdown("""
        ### 📌 RSI(13)
        - 기간(Window): **13**
        - 계산 방식: 상대강도지수 (Relative Strength Index)
        - 범위: 0 ~ 100  
          - 30 이하: 과매도  
          - 70 이상: 과매수  

        ### 📌 Bollinger Band
        - 기준선(Window): **30**
        - 표준편차: **2**
        - 구성:
          - 상단 밴드 = 이동평균선 + (표준편차 × 2)
          - 하단 밴드 = 이동평균선 - (표준편차 × 2)
          - 중앙선 = 이동평균선
        """)
        if st.button("닫기", key="settings_close"):
            st.session_state["show_settings"] = False

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
    threshold_pct = st.slider("성공/실패 기준 값(%)", 0.1, 3.0, 1.0, step=0.1)
with c6:
    rsi_mode = st.selectbox("RSI 조건", ["없음","RSI ≤ 30 (급락)", "RSI ≥ 70 (급등)"], index=1)

# 볼린저밴드 조건 + 세부 설정
c7, c8, c9 = st.columns(3)
with c7:
    bb_cond = st.selectbox(
        "볼린저밴드 조건",
        ["없음","하한선 하향돌파","하한선 상향돌파","상한선 하향돌파","상한선 상향돌파","하한선 중앙돌파","상한선 중앙돌파"],
        index=0,
    )
with c8:
    bb_window = st.slider("볼린저 기간 (window)", 10, 60, 30, step=1)
with c9:
    bb_dev = st.slider("볼린저 표준편차(승수)", 1.0, 3.5, 2.0, step=0.1)

# 2차 조건: 양봉 체크
c10, _, _ = st.columns(3)
with c10:
    bullish_needed = st.slider("2차조건: 양봉 갯수 (0=없음)", 0, 20, 0)

# 안전 장치(세션 보강)
st.session_state["rsi_mode"] = rsi_mode
st.session_state["bb_cond"]  = bb_cond
st.session_state["bb_window"] = bb_window
st.session_state["bb_dev"]    = bb_dev
st.session_state["bullish_needed"] = bullish_needed

# ---- 조건 요약 박스 ----
sim_minutes = lookahead * minutes_per_bar
if sim_minutes < 60:
    sim_dur = f"약 {sim_minutes}분"
elif sim_minutes < 1440:
    sim_dur = f"약 {sim_minutes//60}시간 {sim_minutes%60}분"
else:
    sim_dur = f"약 {sim_minutes//1440}일"

def colorize(text, kind):
    if kind == "up":   return f"<span style='color:red; font-weight:600;'>{text}</span>"
    if kind == "down": return f"<span style='color:blue; font-weight:600;'>{text}</span>"
    return f"<span style='color:#6b7280;'>{text}</span>"

rsi_display = "없음" if rsi_mode == "없음" else (
    colorize("RSI ≤ 30 (급락)", "down") if "≤" in rsi_mode else colorize("RSI ≥ 70 (급등)", "up")
)
if bb_cond == "없음":
    bb_display = "없음"
elif "하향" in bb_cond:
    bb_display = colorize(bb_cond, "down")
else:
    bb_display = colorize(bb_cond, "up")

bb_detail = f"(기간 {bb_window}, 승수 {bb_dev:.1f})" if bb_cond != "없음" else ""
sec2_text = "없음" if st.session_state["bullish_needed"] == 0 else f"양봉 {bullish_needed}개 (시가가 이전 양봉 시가보다 순차 상승)"

st.markdown(f"""
<div style="border:1px solid #ccc; border-radius:8px; padding:0.8rem; background-color:#f9f9f9; margin-top:0.6rem; margin-bottom:0.6rem;">
<b>📌 현재 조건 요약</b><br>
- 측정 캔들 수: {lookahead}봉 ({sim_dur})<br>
- 성공/실패 기준: {threshold_pct:.2f}%<br>
- 1차조건 — RSI: {rsi_display}<br>
- 1차조건 — 볼린저: {bb_display} {bb_detail}<br>
- 2차조건 — 양봉 체크: {sec2_text}
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
def add_indicators(df, bb_window:int, bb_dev:float):
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=bb_window, window_dev=bb_dev)
    out["BB_up"]  = bb.bollinger_hband()
    out["BB_low"] = bb.bollinger_lband()
    out["BB_mid"] = bb.bollinger_mavg()
    return out

# -----------------------------
# 시뮬레이션
# -----------------------------
def simulate(
    df: pd.DataFrame,
    rsi_mode: str,
    lookahead: int,
    thr_pct: float,
    bb_cond: str,
    bullish_needed: int,
    dedup_mode: str
):
    has_rsi = (rsi_mode != "없음")
    has_bb  = (bb_cond  != "없음")
    if not (has_rsi or has_bb):
        return pd.DataFrame(columns=[
            "신호시간","측정시작","기준시가","RSI(13)","성공기준(%)","결과",
            "도달분","최종수익률(%)","최저수익률(%)","최고수익률(%)"
        ])

    def rsi_ok(row):
        if not has_rsi or pd.isna(row["RSI13"]):
            return not has_rsi
        if "≤" in rsi_mode:
            return row["RSI13"] <= 30
        else:
            return row["RSI13"] >= 70

    def bb_ok(i):
        if not has_bb:
            return True
        px = float(df.at[i, "close"])
        up, lo, mid = df.at[i, "BB_up"], df.at[i, "BB_low"], df.at[i, "BB_mid"]
        if bb_cond == "하한선 하향돌파":   return (pd.notna(lo)  and px < lo)
        if bb_cond == "하한선 상향돌파":   return (pd.notna(lo)  and px > lo)
        if bb_cond == "상한선 하향돌파":   return (pd.notna(up)  and px < up)
        if bb_cond == "상한선 상향돌파":   return (pd.notna(up)  and px > up)
        if bb_cond == "하한선 중앙돌파":   return (pd.notna(lo)  and pd.notna(mid) and lo < px < mid)
        if bb_cond == "상한선 중앙돌파":   return (pd.notna(up)  and pd.notna(mid) and mid < px < up)
        return False

    res = []
    n = len(df)
    thr = float(thr_pct)

    for i in range(n-1):
        if not (rsi_ok(df.loc[i]) and bb_ok(i)):
            continue

        entry_idx = i
        if bullish_needed > 0:
            count = 0
            last_bull_open = None
            last_bull_idx = None
            for j in range(i+1, n):
                is_bull = df.at[j, "close"] > df.at[j, "open"]
                if is_bull:
                    open_j = float(df.at[j, "open"])
                    if last_bull_open is None or open_j > last_bull_open:
                        count += 1
                        last_bull_open = open_j
                        last_bull_idx = j
                        if count >= bullish_needed:
                            break
            if count < bullish_needed or last_bull_idx is None:
                continue
            entry_idx = last_bull_idx + 1
            if entry_idx >= n:
                continue

        end = entry_idx + lookahead
        if end >= n:
            continue

        base = float(df.at[entry_idx, "open"])
        closes = df.loc[entry_idx+1:end, ["time", "close"]]
        if closes.empty:
            continue

        final_ret = (closes.iloc[-1]["close"]/base - 1)*100.0
        min_ret   = (closes["close"].min()/base - 1)*100.0
        max_ret   = (closes["close"].max()/base - 1)*100.0

        result = "중립"; reach_min = None
        take_price = base*(1+thr/100.0)
        first_hit = closes[closes["close"] >= take_price]
        if not first_hit.empty and max_ret >= thr:
            reach_min = int((first_hit.iloc[0]["time"] - df.at[entry_idx,"time"]).total_seconds() // 60)
            result = "성공"
        elif final_ret < 0:
            result = "실패"

        def fmt_ret(v): return round(v, 2)

        res.append({
            "신호시간": df.at[i, "time"],
            "측정시작": df.at[entry_idx, "time"],
            "기준시가": int(round(base)),
            "RSI(13)" : round(float(df.at[i,"RSI13"]),1) if pd.notna(df.at[i,"RSI13"]) else None,
            "성공기준(%)": round(thr,1),
            "결과": result,
            "도달분": reach_min,
            "최종수익률(%)": fmt_ret(final_ret),
            "최저수익률(%)": fmt_ret(min_ret),
            "최고수익률(%)": fmt_ret(max_ret),
        })

    out = pd.DataFrame(res, columns=[
        "신호시간","측정시작","기준시가","RSI(13)","성공기준(%)","결과",
        "도달분","최종수익률(%)","최저수익률(%)","최고수익률(%)"
    ])

    if not out.empty and dedup_mode.startswith("중복 제거"):
        filtered = []
        last_idx = -999
