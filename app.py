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
from typing import Optional, Set, List, Dict, Tuple

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

# (선택사항) 신호 중복 처리 옵션이 초기코드에 있었다면 그대로 유지
dup_mode = st.radio(
    "신호 중복 처리",
    ["중복 포함 (연속 신호 모두)", "중복 제거 (연속 동일 결과 1개)"],
    horizontal=True,
)

# -----------------------------
# ① 기본 설정 (날짜+시간 입력 & 보정)
# -----------------------------
st.markdown('<div class="section-title">① 기본 설정</div>', unsafe_allow_html=True)

KST = timezone("Asia/Seoul")
now_kst = datetime.now(KST)               # tz-aware
now = now_kst.replace(tzinfo=None)        # tz-naive(KST 값)로 통일

default_start_dt = now - timedelta(hours=24)
default_end_dt = now

c1, c2, c3, c4 = st.columns(4)
with c1:
    market_label, market_code = st.selectbox("종목 선택", MARKET_LIST, index=default_idx, format_func=lambda x: x[0])
with c2:
    tf_label = st.selectbox("봉 종류 선택", list(TF_MAP.keys()), index=2)
with c3:
    start_date = st.date_input("시작 날짜", value=default_start_dt.date())
    start_time = st.time_input("시작 시간", value=default_start_dt.time())
with c4:
    end_date = st.date_input("종료 날짜", value=default_end_dt.date())
    end_time = st.time_input("종료 시간", value=default_end_dt.time())

interval_key, minutes_per_bar = TF_MAP[tf_label]

# 시작/종료 datetime 결합 (naive)
start_dt = datetime.combine(start_date, start_time)
end_dt   = datetime.combine(end_date, end_time)

# 종료 보정
today = now.date()
if interval_key == "days" and end_date >= today:
    st.info("일봉은 당일 데이터가 제공되지 않습니다. 전일까지로 보정합니다.")
    end_dt = datetime.combine(today - timedelta(days=1), datetime.max.time())
elif end_dt > now:
    end_dt = now

# 경고 자리(기본 설정 아래 고정)
warn_box = st.empty()
st.markdown("---")

# -----------------------------
# ② 조건 설정 (초기 UI 유지)
# -----------------------------
st.markdown('<div class="section-title">② 조건 설정</div>', unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
with c4:
    lookahead = st.slider("측정 캔들 수 (기준 이후 N봉)", 1, 60, 10)
with c5:
    threshold_pct = st.slider("성공/실패 기준 값(%)", 0.1, 5.0, 1.0, step=0.1)
    hit_basis = st.selectbox(
        "성공 판정 기준",
        ["종가 기준", "고가 기준(스침 인정)", "종가 또는 고가"],
        index=0
    )
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
    bottom_mode = st.checkbox(
        "🟢 바닥탐지(실시간) 모드",
        value=False,
        help="RSI≤과매도 & BB 하한선 터치/하회 & CCI≤-100 동시 만족 시 신호"
    )
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
supply_filter = None
if sec_cond == "매물대 터치 후 반등(위→아래→반등)":
    supply_filter = st.selectbox(
        "매물대 종류",
        ["모두 포함", "양봉 매물대만", "음봉 매물대만"],
        index=0
    )

st.session_state["bb_cond"] = bb_cond
st.markdown("---")

# -----------------------------
# 데이터 수집/지표/시뮬레이션 함수
# -----------------------------
_session = requests.Session()
_retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
_session.mount("https://", HTTPAdapter(max_retries=_retries))

def fetch_upbit_paged(market_code: str, interval_key: str,
                      start_dt: datetime, end_dt: datetime,
                      minutes_per_bar: int, warmup_bars: int = 0) -> pd.DataFrame:
    """Upbit 캔들 페이징 수집 (워밍업 포함). 모든 시각은 KST 기준 naive로 처리."""
    start_cutoff = start_dt - timedelta(minutes=max(0, warmup_bars) * minutes_per_bar)

    if "minutes/" in interval_key:
        unit = interval_key.split("/")[1]
        url = f"https://api.upbit.com/v1/candles/minutes/{unit}"
    else:
        url = "https://api.upbit.com/v1/candles/days"

    all_rows: List[Dict] = []
    to_time = end_dt
    try:
        for _ in range(60):  # 최대 12,000봉
            params = {"market": market_code, "count": 200, "to": to_time.strftime("%Y-%m-%d %H:%M:%S")}
            r = _session.get(url, params=params, headers={"Accept": "application/json"}, timeout=10)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            all_rows.extend(batch)
            last_ts = pd.to_datetime(batch[-1]["candle_date_time_kst"])
            if last_ts <= start_cutoff:
                break
            to_time = last_ts - timedelta(seconds=1)
    except Exception:
        return pd.DataFrame()

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows).rename(columns={
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

def add_indicators(df: pd.DataFrame, bb_window: int, bb_dev: float, cci_window: int) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    # RSI(13)
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=out["close"], window=int(bb_window), window_dev=float(bb_dev))
    out["BB_up"]  = bb.bollinger_hband()
    out["BB_low"] = bb.bollinger_lband()
    out["BB_mid"] = bb.bollinger_mavg()
    # CCI
    cci = ta.trend.CCIIndicator(high=out["high"], low=out["low"], close=out["close"], window=int(cci_window), constant=0.015)
    out["CCI"] = cci.cci()
    return out

def _passes_primary_conditions(row_prev: pd.Series, row: pd.Series,
                               rsi_mode: str, rsi_low: int, rsi_high: int,
                               bb_cond: str, bottom_mode: bool) -> bool:
    ok = True
    # RSI
    if rsi_mode == "과매도 기준":
        ok &= (row["RSI13"] <= rsi_low)
    elif rsi_mode == "과매수 기준":
        ok &= (row["RSI13"] >= rsi_high)
    elif rsi_mode == "현재(과매도/과매수 중 하나)":
        ok &= (row["RSI13"] <= rsi_low) or (row["RSI13"] >= rsi_high)
    # BB
    if bb_cond == "상한선":
        ok &= (row["close"] >= row["BB_up"])
    elif bb_cond == "중앙선":
        ok &= (row["close"] >= row["BB_mid"])
    elif bb_cond == "하한선":
        ok &= (row["close"] <= row["BB_low"])
    # 바닥탐지 (실시간)
    if bottom_mode:
        ok &= (row["RSI13"] <= rsi_low) and (row["close"] <= row["BB_low"]) and (row["CCI"] <= -100)
    return bool(ok)

def _passes_secondary_condition(df: pd.DataFrame, idx: int, sec_cond: str) -> bool:
    # 2차 조건 (선택한 조건만 적용)
    if sec_cond == "없음":
        return True
    if sec_cond == "양봉 2개 연속 상승":
        if idx < 2: return False
        c1 = df.iloc[idx-1]
        c2 = df.iloc[idx]
        return (c1["close"] > c1["open"]) and (c2["close"] > c2["open"])
    if sec_cond == "BB 기반 첫 양봉 50% 진입":
        # 직전 캔들이 BB 하단 근처에서 음봉 -> 현재 양봉이며, 몸통 50% 상회
        if idx < 1: return False
        prev = df.iloc[idx-1]
        cur  = df.iloc[idx]
        if cur["close"] <= cur["open"]: return False
        body = cur["close"] - cur["open"]
        midpoint = cur["open"] + body * 0.5
        return (prev["close"] <= prev["BB_low"]) and (cur["close"] >= midpoint)
    if sec_cond == "매물대 터치 후 반등(위→아래→반등)":
        # 간이판정: 직전 하락 후 현재 양봉
        if idx < 2: return False
        p2 = df.iloc[idx-2]; p1 = df.iloc[idx-1]; cur = df.iloc[idx]
        return (p2["close"] > p1["close"]) and (cur["close"] > cur["open"])
    return True

def simulate(df: pd.DataFrame,
             rsi_mode: str, rsi_low: int, rsi_high: int,
             lookahead: int, threshold_pct: float, hit_basis: str,
             bb_cond: str, bottom_mode: bool, sec_cond: str,
             dedup_mode: str) -> pd.DataFrame:
    """
    신호 발생시점의 종가를 기준으로 목표가를 계산하고,
    이후 N봉 내에 도달하면 성공, 아니면 실패(미도달)로 판정.
    - hit_basis: '종가 기준' | '고가 기준(스침 인정)' | '종가 또는 고가'
    - dedup_mode: '중복 포함...' | '중복 제거...'
    """
    if df.empty:
        return pd.DataFrame(columns=["time","price","target","hit_idx","hit_price","result","note"])

    rows = []
    last_kept_side = None  # 중복 제거용
    n = len(df)

    for i in range(1, n):
        row_prev = df.iloc[i-1]
        row = df.iloc[i]

        # 1) 1차 조건
        if not _passes_primary_conditions(row_prev, row, rsi_mode, rsi_low, rsi_high, bb_cond, bottom_mode):
            continue

        side = "LONG"  # 현재 로직은 롱만 가정(예전과 동일 컨셉)
        if dedup_mode.startswith("중복 제거") and last_kept_side == side:
            continue

        # 2) 2차 조건
        if not _passes_secondary_condition(df, i, sec_cond):
            continue

        entry_price = float(row["close"])
        target_price = entry_price * (1.0 + threshold_pct/100.0)

        # 3) 룩어헤드 성공 판정
        future_end = min(n - 1, i + lookahead)
        hit_idx = None
        hit_price = None

        for j in range(i+1, future_end+1):
            f = df.iloc[j]
            if hit_basis == "종가 기준":
                if f["close"] >= target_price:
                    hit_idx = j - i
                    hit_price = float(f["close"])
                    break
            elif hit_basis == "고가 기준(스침 인정)":
                if f["high"] >= target_price:
                    hit_idx = j - i
                    hit_price = float(max(f["high"], target_price))
                    break
            else:  # 종가 또는 고가
                if (f["close"] >= target_price) or (f["high"] >= target_price):
                    hit_idx = j - i
                    hit_price = float(max(f["close"], f["high"]))
                    break

        if hit_idx is not None:
            result = "성공"
            note = f"{hit_idx}번째 캔들 도달"
        else:
            result = "실패"
            note = "미도달"

        rows.append({
            "time": row["time"],
            "price": entry_price,
            "target": float(target_price),
            "hit_idx": hit_idx if hit_idx is not None else "",
            "hit_price": hit_price if hit_price is not None else "",
            "result": result,
            "note": note
        })
        last_kept_side = side

    return pd.DataFrame(rows)

# -----------------------------
# 실행
# -----------------------------
try:
    # 날짜 검증
    if start_dt > end_dt:
        st.error("시작 시간이 종료 시간보다 이후입니다.")
        st.stop()

    # 안전 가드(혹시 변수 누락 시)
    if "bb_window" not in locals(): bb_window = 30
    if "bb_dev" not in locals(): bb_dev = 2.0
    if "cci_window" not in locals(): cci_window = 14
    if "bb_cond" not in locals(): bb_cond = "없음"

    # 워밍업 바 (지표 안정화용)
    warmup_bars = max(13, int(bb_window), int(cci_window)) * 5

    # 데이터 수집
    df_raw = fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars)
    if df_raw.empty:
        st.error("데이터가 없습니다.")
        st.stop()

    # 지표 + 최종 구간 필터
    df_ind = add_indicators(df_raw, int(bb_window), float(bb_dev), int(cci_window))
    df = df_ind[(df_ind["time"] >= start_dt) & (df_ind["time"] <= end_dt)].reset_index(drop=True)

    # 실제 수집 범위 안내
    if not df.empty:
        actual_start, actual_end = df["time"].min(), df["time"].max()
        if actual_start > start_dt or actual_end < end_dt:
            warn_box.warning(
                f"⚠ 선택한 기간({start_dt} ~ {end_dt}) 전체 데이터를 가져오지 못했습니다.\n"
                f"- 실제 수집 범위: {actual_start} ~ {actual_end}"
            )

    # -----------------------------
    # ③ 요약 & 차트
    # -----------------------------
    st.markdown('<div class="section-title">③ 요약 & 차트</div>', unsafe_allow_html=True)
    if df.empty:
        st.info("표시할 차트 데이터가 없습니다.")
    else:
        st.markdown(
            f"- 표본 캔들 수: **{len(df)}**개  |  "
            f"표시 구간: **{df['time'].min()} ~ {df['time'].max()}**  |  "
            f"봉: **{tf_label}**",
            unsafe_allow_html=True
        )

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            row_heights=[0.72, 0.28], specs=[[{"secondary_y": False}], [{"secondary_y": False}]])
        # 캔들
        fig.add_trace(go.Candlestick(
            x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"
        ), row=1, col=1)
        # BB
        fig.add_trace(go.Scatter(x=df["time"], y=df["BB_up"],  mode="lines", name="BB Upper"),  row=1, col=1)
        fig.add_trace(go.Scatter(x=df["time"], y=df["BB_mid"], mode="lines", name="BB Middle"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["time"], y=df["BB_low"],  mode="lines", name="BB Lower"),  row=1, col=1)
        # RSI
        fig.add_trace(go.Scatter(x=df["time"], y=df["RSI13"], mode="lines", name="RSI(13)"), row=2, col=1)

        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI(13)", row=2, col=1, range=[0, 100])
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_rangeslider_visible=False,
                          uirevision="chart-static")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # -----------------------------
    # ④ 신호 결과 (최신 순)
    # -----------------------------
    st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)
    res = simulate(df, rsi_mode, int(rsi_low), int(rsi_high),
                   int(lookahead), float(threshold_pct), hit_basis,
                   bb_cond, bool(bottom_mode), sec_cond, dup_mode)
    if res.empty:
        st.info("신호 없음")
    else:
        res_sorted = res.sort_values("time", ascending=False).reset_index(drop=True)
        st.dataframe(res_sorted, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"오류: {e}")
