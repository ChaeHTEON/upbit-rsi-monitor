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
import numpy as np
from pytz import timezone  # ✅ 한국시간 반영

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

st.title("📊 코인 시뮬레이터")
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
# 기본 설정
# -----------------------------
st.markdown('<div class="section-title">① 기본 설정</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    market_label, market_code = st.selectbox("종목 선택", MARKET_LIST, index=default_idx, format_func=lambda x: x[0])
with c2:
    tf_label = st.selectbox("봉 종류 선택", list(TF_MAP.keys()), index=2)
with c3:
    KST = timezone("Asia/Seoul")  # ✅ 한국시간 적용
    today_kst = datetime.now(KST).date()
    default_start = today_kst - timedelta(days=1)  # 시작: 어제, 종료: 오늘
    start_date = st.date_input("시작 날짜", value=default_start)
    end_date = st.date_input("종료 날짜", value=today_kst)

interval_key, minutes_per_bar = TF_MAP[tf_label]
st.markdown("---")

# -----------------------------
# 데이터/지표 준비 함수
# -----------------------------
def estimate_calls(start_dt, end_dt, minutes_per_bar):
    mins = max(1, int((end_dt - start_dt).total_seconds() // 60))
    bars = max(1, mins // minutes_per_bar)
    return bars // 200 + 1

_session = requests.Session()
_retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
_session.mount("https://", HTTPAdapter(max_retries=_retries))

@st.cache_data(ttl=120, show_spinner=False)
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

    calls_est = estimate_calls(start_cutoff, end_dt, minutes_per_bar)
    max_calls = min(calls_est + 2, 60)
    req_count = 200
    all_data = []
    to_time = None

    try:
        for _ in range(max_calls):
            params = {"market": market_code, "count": req_count}
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
        "candle_date_time_kst": "time", "opening_price": "open", "high_price": "high",
        "low_price": "low", "trade_price": "close", "candle_acc_trade_volume": "volume"})
    df["time"] = pd.to_datetime(df["time"])
    df = df[["time", "open", "high", "low", "close", "volume"]].sort_values("time").reset_index(drop=True)
    return df[(df["time"] >= start_cutoff) & (df["time"] <= end_dt)]

def add_indicators(df, bb_window, bb_dev):
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=bb_window, window_dev=bb_dev)
    out["BB_up"] = bb.bollinger_hband().fillna(method="bfill").fillna(method="ffill")
    out["BB_low"] = bb.bollinger_lband().fillna(method="bfill").fillna(method="ffill")
    out["BB_mid"] = bb.bollinger_mavg().fillna(method="bfill").fillna(method="ffill")
    return out

# -----------------------------
# 시뮬레이션 (성공=종가 기준 고정, 미도달 처리 고정)
# -----------------------------
def simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, thr_pct,
             bb_cond, dedup_mode, minutes_per_bar, market_code, bb_window, bb_dev, sec_cond="없음"):
    res = []
    n = len(df)
    thr = float(thr_pct)

    # 1차 조건
    if rsi_mode == "없음":
        rsi_idx = []
    elif rsi_mode == "현재(과매도/과매수 중 하나)":
        rsi_idx = sorted(set(df.index[df["RSI13"] <= float(rsi_low)].tolist()) |
                         set(df.index[df["RSI13"] >= float(rsi_high)].tolist()))
    elif rsi_mode == "과매도 기준":
        rsi_idx = df.index[df["RSI13"] <= float(rsi_low)].tolist()
    else:
        rsi_idx = df.index[df["RSI13"] >= float(rsi_high)].tolist()

    def bb_ok(i):
        c = float(df.at[i, "close"])
        up, lo, mid = df.at[i, "BB_up"], df.at[i, "BB_low"], df.at[i, "BB_mid"]
        if bb_cond == "상한선": return pd.notna(up) and (c > float(up))
        if bb_cond == "하한선": return pd.notna(lo) and (c <= float(lo))
        if bb_cond == "중앙선": return pd.notna(mid) and (c >= float(mid))
        return False

    bb_idx = [i for i in df.index if bb_cond != "없음" and bb_ok(i)]
    if rsi_mode != "없음" and bb_cond != "없음":
        base_sig_idx = sorted(set(rsi_idx) & set(bb_idx))
    elif rsi_mode != "없음": base_sig_idx = rsi_idx
    elif bb_cond != "없음": base_sig_idx = bb_idx
    else: base_sig_idx = list(range(n)) if sec_cond != "없음" else []

    def is_bull(idx): return float(df.at[idx, "close"]) > float(df.at[idx, "open"])

    i = 0
    while i < n:
        if i not in base_sig_idx:
            i += 1
            continue

        anchor_idx = i
        signal_time = df.at[i, "time"]
        base_price = float(df.at[i, "close"])
        end_idx = anchor_idx + lookahead
        if end_idx >= n:
            i += 1
            continue

        win_slice = df.iloc[anchor_idx + 1:end_idx + 1]
        end_time = df.at[end_idx, "time"]
        end_close = float(df.at[end_idx, "close"])
        final_ret = (end_close / base_price - 1) * 100

        # 목표 도달 체크 (종가 기준 고정)
        target = base_price * (1.0 + thr / 100.0)
        result, reach_min, hit_idx = "중립", None, None
        for j in range(anchor_idx + 1, end_idx + 1):
            if float(df.at[j, "close"]) >= target:
                hit_idx = j
                break

        if hit_idx is not None:
            end_time = df.at[hit_idx, "time"]
            end_close = target
            final_ret = thr
            result = "성공"
        else:
            result = "실패" if final_ret < 0 else "중립"

        res.append({
            "신호시간": signal_time,
            "종료시간": end_time,
            "기준시가": int(round(base_price)),
            "종료가": end_close,
            "RSI(13)": round(float(df.at[anchor_idx, "RSI13"]), 1) if pd.notna(df.at[anchor_idx, "RSI13"]) else None,
            "BB값": None,
            "성공기준(%)": round(thr, 1),
            "결과": result,
            "최종수익률(%)": round(final_ret, 2),
        })

        i = end_idx if dedup_mode.startswith("중복 제거") else i + 1

    return pd.DataFrame(res)

# -----------------------------
# 실행
# -----------------------------
try:
    if start_date > end_date:
        st.error("시작 날짜가 종료 날짜보다 이후입니다.")
        st.stop()

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())
    warmup_bars = 100

    df_raw = fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars=warmup_bars)
    if df_raw.empty:
        st.error("데이터가 없습니다.")
        st.stop()

    df = add_indicators(df_raw, 30, 2)
    df = df[(df["time"] >= start_dt) & (df["time"] <= end_dt)].reset_index(drop=True)

    # -----------------------------
    # 차트 먼저 출력
    # -----------------------------
    st.markdown('<div class="section-title">② 차트</div>', unsafe_allow_html=True)
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="가격", increasing_line_color="red", decreasing_line_color="blue", line=dict(width=1.1)
    ))
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # 조건 설정 & 결과
    # -----------------------------
    st.markdown('<div class="section-title">③ 조건 설정 & 결과</div>', unsafe_allow_html=True)
    # (조건 설정 및 테이블 출력 블록은 기존 그대로 유지)
    # ...

except Exception as e:
    st.error(f"오류: {e}")
