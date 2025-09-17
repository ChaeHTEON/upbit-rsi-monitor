
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
import ta
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

# -----------------------------
# 페이지 스타일
# -----------------------------
st.set_page_config(page_title="Upbit RSI(13) + Bollinger Band 시뮬레이터", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 0.8rem; padding-bottom: 0.8rem; max-width: 1100px;}
  .stMetric {text-align:center;}
  .success {color:red; font-weight:600;}
  .fail {color:blue; font-weight:600;}
  .neutral {color:green; font-weight:600;}
  .section-title {font-size:1.05rem; font-weight:700; margin: 0.6rem 0 0.2rem;}
  .hint {color:#6b7280;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Upbit RSI(13) + Bollinger Band 시뮬레이터")

# -----------------------------
# 업비트 마켓 리스트
# -----------------------------
@st.cache_data(ttl=3600)
def get_upbit_krw_markets():
    url = "https://api.upbit.com/v1/market/all"
    r = requests.get(url, params={"isDetails": "false"})
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
default_idx = 0
for i, (_, code) in enumerate(MARKET_LIST):
    if code == "KRW-BTC":
        default_idx = i
        break

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
    market_label, market_code = st.selectbox(
        "종목 선택", MARKET_LIST, index=default_idx, format_func=lambda x: x[0]
    )
with c2:
    tf_label = st.selectbox("봉 종류 선택", list(TF_MAP.keys()), index=2)
with c3:
    default_start = datetime.today() - timedelta(days=1)
    start_date = st.date_input("시작 날짜", value=default_start)
    end_date = st.date_input("종료 날짜", value=datetime.today())

# -----------------------------
# 조건 설정
# -----------------------------
st.markdown('<div class="section-title">② 조건 설정</div>', unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
with c4:
    lookahead = st.slider("측정 캔들 수 (기준 이후 N봉)", 1, 60, 10)
with c5:
    threshold_pct = st.slider("성공/실패 기준 값(%)", 0.1, 3.0, 1.0, step=0.1)
    st.caption(f"현재 설정: **{threshold_pct:.1f}%**")
with c6:
    rsi_side = st.selectbox("RSI 조건", ["RSI ≤ 30 (급락)", "RSI ≥ 70 (급등)"], index=0)

c7 = st.container()
with c7:
    bb_cond = st.selectbox(
        "볼린저밴드 조건",
        [
            "없음",
            "하한선 하향돌파",
            "하한선 상향돌파",
            "상한선 하향돌파",
            "상한선 상향돌파",
            "하한선 중앙돌파",
            "상한선 중앙돌파",
        ],
        index=0,
    )

interval_key, minutes_per_bar = TF_MAP[tf_label]
total_minutes = lookahead * minutes_per_bar
st.caption(f"측정 범위: **{lookahead} ({total_minutes}분)**  · 봉 종류: **{tf_label}**")

# -----------------------------
# 데이터 수집
# -----------------------------
def estimate_calls(start_dt: datetime, end_dt: datetime, minutes_per_bar: int) -> int:
    mins = max(1, int((end_dt - start_dt).total_seconds() // 60))
    bars = max(1, mins // minutes_per_bar)
    calls = bars // 200 + 1
    return min(calls, 5000)

def fetch_upbit_paged(market_code: str, interval_key: str,
                      start_dt: datetime, end_dt: datetime,
                      minutes_per_bar: int) -> pd.DataFrame:
    if "minutes/" in interval_key:
        unit = interval_key.split("/")[1]
        url = f"https://api.upbit.com/v1/candles/minutes/{unit}"
    else:
        url = f"https://api.upbit.com/v1/candles/{interval_key}"
    all_data = []
    to_time = end_dt
    calls_est = estimate_calls(start_dt, end_dt, minutes_per_bar)
    progress = st.progress(0.0)
    done = 0
    while True:
        params = {"market": market_code, "count": 200, "to": to_time.strftime("%Y-%m-%d %H:%M:%S")}
        r = requests.get(url, params=params, headers={"Accept": "application/json"})
        if r.status_code != 200:
            raise RuntimeError(f"Upbit API 오류: {r.text}")
        batch = r.json()
        if not batch:
            break
        all_data.extend(batch)
        last_ts = pd.to_datetime(batch[-1]["candle_date_time_kst"])
        if last_ts <= start_dt:
            break
        to_time = last_ts - timedelta(seconds=1)
        done += 1
        progress.progress(min(1.0, done / max(1, calls_est)))
        if done > 5000:
            break
    progress.empty()
    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(all_data)
    df = df.rename(columns={
        "candle_date_time_kst": "time",
        "opening_price": "open",
        "high_price": "high",
        "low_price": "low",
        "trade_price": "close",
        "candle_acc_trade_volume": "volume"
    })
    df["time"] = pd.to_datetime(df["time"])
    df = df[["time", "open", "high", "low", "close", "volume"]].sort_values("time").reset_index(drop=True)
    return df[df["time"].between(start_dt, end_dt)]

# -----------------------------
# 지표 추가
# -----------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=30, window_dev=2)
    out["BB_up"]  = bb.bollinger_hband()
    out["BB_low"] = bb.bollinger_lband()
    out["BB_mid"] = bb.bollinger_mavg()
    return out

# -----------------------------
# ✅ 시뮬레이션 (정리된 버전)
# -----------------------------
def simulate(df: pd.DataFrame, rsi_side: str, lookahead: int, thr_pct: float,
             bb_cond: str, dedup_mode: str) -> pd.DataFrame:
    res = []
    n = len(df)
    thr = float(thr_pct)

    # 1) 신호 후보: RSI 조건을 만족하는 모든 봉
    if "≤" in rsi_side:
        sig_idx = df.index[(df["RSI13"].notna()) & (df["RSI13"] <= 30)].tolist()
    else:
        sig_idx = df.index[(df["RSI13"].notna()) & (df["RSI13"] >= 70)].tolist()

    for i in sig_idx:
        end = i + lookahead
        if end >= n:
            continue

        # 2) 볼린저 조건
        if bb_cond != "없음":
            px = float(df.at[i, "close"])
            up  = float(df.at[i, "BB_up"])  if pd.notna(df.at[i, "BB_up"])  else None
            lo  = float(df.at[i, "BB_low"]) if pd.notna(df.at[i, "BB_low"]) else None
            mid = float(df.at[i, "BB_mid"]) if pd.notna(df.at[i, "BB_mid"]) else None
            ok = True
            if   bb_cond == "하한선 하향돌파": ok = (lo is not None) and (px < lo)
            elif bb_cond == "하한선 상향돌파": ok = (lo is not None) and (px > lo)
            elif bb_cond == "상한선 하향돌파": ok = (up is not None) and (px < up)
            elif bb_cond == "상한선 상향돌파": ok = (up is not None) and (px > up)
            elif bb_cond == "하한선 중앙돌파": ok = (mid is not None) and (lo is not None) and (px > lo) and (px < mid)
            elif bb_cond == "상한선 중앙돌파": ok = (mid is not None) and (up is not None) and (px < up) and (px > mid)
            if not ok:
                continue

        # 3) 기준가 = 저가(low)
        base_price = float(df.at[i, "low"])

        # 4) 이후 N봉 종가 시퀀스
        closes = df.loc[i+1:end, "close"]
        if closes.empty:
            continue

        # 5) 수익률 계산
        final_ret = (closes.iloc[-1] / base_price - 1.0) * 100.0
        min_ret   = (closes.min() / base_price - 1.0) * 100.0
        max_ret   = (closes.max() / base_price - 1.0) * 100.0

        # 6) 판정
        if final_ret <= -thr:
            result = "실패"
        elif final_ret >= thr:
            result = "성공"
        elif final_ret > 0:
            result = "중립"
        else:
            result = "실패"

        res.append({
            "신호시간": df.at[i, "time"],
            "기준시가": int(round(base_price)),
            "RSI(13)": round(float(df.at[i, "RSI13"]), 1) if pd.notna(df.at[i, "RSI13"]) else None,
            "성공기준(%)": round(thr, 1),
            "결과": result,
            "최종수익률(%)": round(final_ret, 1),
            "최저수익률(%)": round(min_ret, 1),
            "최고수익률(%)": round(max_ret, 1),
        })

    out = pd.DataFrame(res)

    # 7) dedup_mode가 "중복 제거"일 때만 적용
    if not out.empty and dedup_mode.startswith("중복 제거"):
        out = out.loc[out["결과"].shift() != out["결과"]]

    return out
