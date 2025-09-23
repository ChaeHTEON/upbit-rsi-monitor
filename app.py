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
  .neutral-cell {color:#059669; font-weight:600;}
  table {border-collapse:collapse; width:100%;}
  th, td {border:1px solid #ddd; padding:6px; text-align:center;}
</style>
""", unsafe_allow_html=True)

st.title("📊 코인 시뮬레이션")
st.markdown("<div style='margin-bottom:10px; color:gray;'>※ 차트 내 점선은 신호 흐름선, 성공 시 도달 지점에 ⭐ 별표 표시</div>", unsafe_allow_html=True)

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
    default_start = today_kst - timedelta(days=1)
    start_date = st.date_input("시작 날짜", value=default_start)
    end_date = st.date_input("종료 날짜", value=today_kst)

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
    r1, r2, r3 = st.columns(3)
    with r1:
        rsi_mode = st.selectbox(
            "RSI 조건",
            ["없음", "현재(과매도/과매수 중 하나)", "과매도 기준", "과매수 기준"],
            index=0,
            help="현재: RSI≤과매도 또는 RSI≥과매수 중 하나라도 충족"
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

st.markdown('<div class="hint">2차 조건: 선택한 조건만 적용 (없음/양봉 2개/BB 기반)</div>', unsafe_allow_html=True)
sec_cond = st.selectbox("2차 조건 선택", ["없음", "양봉 2개 연속 상승", "BB 기반 첫 양봉 50% 진입"], index=0)
st.session_state["bb_cond"] = bb_cond
st.markdown("---")

# -----------------------------
# 데이터 수집
# -----------------------------
def estimate_calls(start_dt, end_dt, minutes_per_bar):
    mins = max(1, int((end_dt - start_dt).total_seconds() // 60))
    bars = max(1, mins // minutes_per_bar)
    return bars // 200 + 1

_session = requests.Session()
_retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
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
    all_data = []
    to_time = None  # ✅ 첫 호출은 to 없이

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
            if last_ts <= start_dt:
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
    return df[(df["time"] >= start_dt) & (df["time"] <= end_dt)]

# -----------------------------
# 지표
# -----------------------------
def add_indicators(df, bb_window, bb_dev):
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=bb_window, window_dev=bb_dev)
    out["BB_up"] = bb.bollinger_hband().fillna(method="bfill").fillna(method="ffill")
    out["BB_low"] = bb.bollinger_lband().fillna(method="bfill").fillna(method="ffill")
    out["BB_mid"] = bb.bollinger_mavg().fillna(method="bfill").fillna(method="ffill")
    return out

# -----------------------------
# 시뮬레이션
# -----------------------------
def simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, thr_pct, bb_cond, dedup_mode,
             minutes_per_bar, market_code, bb_window, bb_dev, sec_cond="없음"):
    res = []
    n = len(df)
    thr = float(thr_pct)

    # -------- RSI 인덱스 --------
    if rsi_mode == "없음":
        rsi_idx = []
    elif rsi_mode == "현재(과매도/과매수 중 하나)":
        rsi_idx = sorted(set(df.index[df["RSI13"] <= float(rsi_low)].tolist()) |
                         set(df.index[df["RSI13"] >= float(rsi_high)].tolist()))
    elif rsi_mode == "과매도 기준":
        rsi_idx = df.index[df["RSI13"] <= float(rsi_low)].tolist()
    else:  # 과매수 기준
        rsi_idx = df.index[df["RSI13"] >= float(rsi_high)].tolist()

    # -------- BB 인덱스 --------
    def bb_ok(i):
        close_i = float(df.at[i, "close"])
        up, lo, mid = df.at[i, "BB_up"], df.at[i, "BB_low"], df.at[i, "BB_mid"]
        if bb_cond == "상한선":
            return pd.notna(up) and (close_i > float(up))
        if bb_cond == "하한선":
            return pd.notna(lo) and (close_i <= float(lo))  # 스침 인정
        if bb_cond == "중앙선":
            if pd.isna(mid) or pd.isna(up) or pd.isna(lo):
                return False
            band_w = max(1e-9, float(up) - float(lo))
            near_eps = 0.1 * band_w
            return (close_i >= float(mid)) or (abs(close_i - float(mid)) <= near_eps)
        return False

    bb_idx = [i for i in df.index if bb_cond != "없음" and bb_ok(i)]

    # -------- 1차 결합 --------
    if rsi_mode != "없음" and bb_cond != "없음":
        base_sig_idx = sorted(set(rsi_idx) & set(bb_idx))
    elif rsi_mode != "없음":
        base_sig_idx = rsi_idx
    elif bb_cond != "없음":
        base_sig_idx = bb_idx
    else:
        base_sig_idx = list(range(n)) if sec_cond != "없음" else []

    # -------- 보조 함수 --------
    def is_bull(idx):
        return float(df.at[idx, "close"]) > float(df.at[idx, "open"])

    def b1_pass(j):
        if not is_bull(j):
            return False
        if bb_cond == "상한선":
            ref = float(df.at[j, "BB_up"])
        elif bb_cond == "중앙선":
            ref = float(df.at[j, "BB_mid"])
        elif bb_cond == "하한선":
            ref = float(df.at[j, "BB_low"])
        else:
            return False
        if pd.isna(ref):
            return False
        o, c = float(df.at[j, "open"]), float(df.at[j, "close"])
        return (c >= o + 0.5 * (ref - o)) if (o < ref) else (c >= ref)

    # -------- 메인 루프 --------
    i = 0
    while i < n:
        if i not in base_sig_idx:
            i += 1
            continue

        # --- Anchor 기본값 (첫 신호봉 종가) ---
        anchor_idx = i
        signal_time = df.at[i, "time"]
        base_price = float(df.at[i, "close"])

        # --- 2차 조건 ---
        if sec_cond == "양봉 2개 연속 상승":
            if i + 2 < n:
                c0, o0 = float(df.at[i + 1, "close"]), float(df.at[i + 1, "open"])
                c1, o1 = float(df.at[i + 2, "close"]), float(df.at[i + 2, "open"])
                if not ((c0 > o0) and (c1 > o1) and (c1 > c0)):
                    i += 1
                    continue
            else:
                i += 1
                continue

        elif sec_cond == "BB 기반 첫 양봉 50% 진입":
            # B1 찾기
            B1_idx, B1_close = None, None
            for j in range(i + 1, n):
                if b1_pass(j):
                    v = df.at[j, "close"]
                    if pd.notna(v):
                        B1_idx, B1_close = j, float(v)
                        break
            if B1_idx is None:
                i += 1
                continue

            # B2, B3 찾기 (연속 아님)
            bull_cnt, B3_idx = 0, None
            for j in range(B1_idx + 1, min(B1_idx + lookahead, n)):
                if is_bull(j):
                    bull_cnt += 1
                    if bull_cnt == 2:
                        B3_idx = j
                        break
            if B3_idx is None:
                i += 1
                continue

            # T 찾기 (B3 이후, B1_close 이상)
            T_idx = None
            for j in range(B3_idx + 1, n):
                cj = df.at[j, "close"]
                if pd.notna(cj) and float(cj) >= B1_close:
                    T_idx = j
                    break
            if T_idx is None:
                i += 1
                continue

            # ✅ Anchor를 T_idx로 교체
            anchor_idx = T_idx
            signal_time = df.at[T_idx, "time"]
            base_price = float(df.at[T_idx, "close"])

        # --- 성과 측정 ---
        end_idx = anchor_idx + lookahead
        if end_idx >= n:
            i += 1
            continue

        # ✅ 종료 캔들을 포함해서 정확히 N개 확보
        window = df.iloc[anchor_idx + 1 : end_idx + 1]

        end_time = df.at[end_idx, "time"]
        end_close = float(df.at[end_idx, "close"])
        final_ret = (end_close / base_price - 1) * 100

        min_ret = (window["close"].min() / base_price - 1) * 100 if not window.empty else 0.0
        max_ret = (window["close"].max() / base_price - 1) * 100 if not window.empty else 0.0

        result, reach_min = "중립", None
        target_price = base_price * (1 + thr / 100)
        hit_rows = window[window["close"] >= target_price]
        if not hit_rows.empty:
            hit_time = hit_rows.iloc[0]["time"]
            if pd.notna(hit_time) and pd.notna(signal_time):
                reach_min = int((hit_time - signal_time).total_seconds() // 60)
            end_time, end_close = hit_time, target_price
            final_ret, result = thr, "성공"
        else:
            if final_ret <= -thr:
                result = "실패"

        bb_value = None
        if bb_cond == "상한선": bb_value = df.at[i, "BB_up"]
        elif bb_cond == "중앙선": bb_value = df.at[i, "BB_mid"]
        elif bb_cond == "하한선": bb_value = df.at[i, "BB_low"]

        res.append({
            "신호시간": signal_time,
            "종료시간": end_time,
            "기준시가": int(round(base_price)),
            "종료가": end_close,
            "RSI(13)": round(float(df.at[i, "RSI13"]), 1) if pd.notna(df.at[i, "RSI13"]) else None,
            "BB값": round(float(bb_value), 1) if bb_value is not None and pd.notna(bb_value) else None,
            "성공기준(%)": round(thr, 1),
            "결과": result,
            "도달분": reach_min,
            "최종수익률(%)": round(final_ret, 2),
            "최저수익률(%)": round(min_ret, 2),
            "최고수익률(%)": round(max_ret, 2)
        })

        i = end_idx if dedup_mode.startswith("중복 제거") else i + 1

    return pd.DataFrame(res)

# -----------------------------
# 실행
# -----------------------------
try:
    if start_date > end_date:
        st.error("시작 날짜가 종료 날짜보다 이후입니다."); st.stop()

    start_dt=datetime.combine(start_date, datetime.min.time())
    end_dt=datetime.combine(end_date, datetime.max.time())
    df=fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar)
    if df.empty: st.error("데이터가 없습니다."); st.stop()
    df=add_indicators(df, bb_window, bb_dev)
    bb_cond=st.session_state.get("bb_cond",bb_cond)

    res_all=simulate(df,rsi_mode,rsi_low,rsi_high,lookahead,threshold_pct,bb_cond,
                     "중복 포함 (연속 신호 모두)",minutes_per_bar,market_code,bb_window,bb_dev,sec_cond)
    res_dedup=simulate(df,rsi_mode,rsi_low,rsi_high,lookahead,threshold_pct,bb_cond,
                       "중복 제거 (연속 동일 결과 1개)",minutes_per_bar,market_code,bb_window,bb_dev,sec_cond)
    res=res_all if dup_mode.startswith("중복 포함") else res_dedup

    st.dataframe(res)
except Exception as e:
    st.error(f"오류: {e}")
