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
import os, base64, shutil

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
    options=["중복 제거 (연속 동일 결과 1개)", "중복 포함 (연속 신호 모두)"],
    index=0,
    horizontal=True
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

# ✅ 차트 컨테이너
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
    bottom_mode = st.checkbox("🟢 바닥탐지(실시간) 모드", value=False)
with c11:
    cci_window = st.number_input("CCI 기간", min_value=5, max_value=100, value=14, step=1)
with c12:
    pass

st.markdown('<div class="hint">2차 조건: 선택한 조건만 적용</div>', unsafe_allow_html=True)
sec_cond = st.selectbox(
    "2차 조건 선택",
    ["없음", "양봉 2개 연속 상승", "양봉 2개 (범위 내)", "BB 기반 첫 양봉 50% 진입", "매물대 터치 후 반등(위→아래→반등)"],
    index=0
)

# -----------------------------
# GitHub 저장 연동 (매물대)
# -----------------------------
CSV_FILE = os.path.join(os.path.dirname(__file__), "supply_levels.csv")
if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=["market", "level"]).to_csv(CSV_FILE, index=False)

def load_supply_levels(market_code):
    df = pd.read_csv(CSV_FILE)
    df_market = df[df["market"] == market_code]
    return df_market["level"].tolist()

def save_supply_levels(market_code, levels):
    df = pd.read_csv(CSV_FILE)
    df = df[df["market"] != market_code]
    new_df = pd.DataFrame({"market": [market_code]*len(levels), "level": levels})
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

# -----------------------------
# 데이터 수집/지표/시뮬레이션 함수
# -----------------------------
_session = requests.Session()
_retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
_session.mount("https://", HTTPAdapter(max_retries=_retries))

def fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars: int = 0):
    import shutil
    if warmup_bars and warmup_bars > 0:
        start_cutoff = start_dt - timedelta(minutes=warmup_bars * minutes_per_bar)
    else:
        start_cutoff = start_dt
    if "minutes/" in interval_key:
        unit = interval_key.split("/")[1]
        url = f"https://api.upbit.com/v1/candles/minutes/{unit}"
        tf_key = f"{unit}min"
    else:
        url = "https://api.upbit.com/v1/candles/days"
        tf_key = "day"
    data_dir = os.path.join(os.path.dirname(__file__), "data_cache")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, f"{market_code}_{tf_key}.csv")
    if os.path.exists(csv_path):
        df_cache = pd.read_csv(csv_path, parse_dates=["time"])
    else:
        df_cache = pd.DataFrame(columns=["time","open","high","low","close","volume"])
    if not df_cache.empty:
        first_cached = df_cache["time"].min()
        last_cached  = df_cache["time"].max()
        if first_cached <= start_cutoff and last_cached >= end_dt:
            return df_cache[(df_cache["time"] >= start_cutoff) & (df_cache["time"] <= end_dt)].reset_index(drop=True)
    df_all = df_cache.copy()
    def _fetch(limit_time):
        out, to_time = [], limit_time
        for _ in range(500):
            params = {"market": market_code, "count": 200}
            if to_time is not None:
                params["to"] = to_time.strftime("%Y-%m-%d %H:%M:%S")
            r = _session.get(url, params=params, headers={"Accept": "application/json"}, timeout=10)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            out.extend(batch)
            last_ts = pd.to_datetime(batch[-1]["candle_date_time_kst"])
            if last_ts <= start_cutoff:
                break
            to_time = last_ts - timedelta(seconds=1)
        return out
    try:
        new_data = _fetch(end_dt)
        if new_data:
            df_new = pd.DataFrame(new_data).rename(columns={
                "candle_date_time_kst": "time",
                "opening_price": "open",
                "high_price": "high",
                "low_price": "low",
                "trade_price": "close",
                "candle_acc_trade_volume": "volume",
            })
            df_new["time"] = pd.to_datetime(df_new["time"])
            df_new = df_new[["time","open","high","low","close","volume"]]
            df_all = pd.concat([df_all, df_new], ignore_index=True)
    except Exception:
        pass
    if not df_all.empty:
        df_all = df_all.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
        tmp_path = csv_path + ".tmp"
        df_all.to_csv(tmp_path, index=False)
        shutil.move(tmp_path, csv_path)
    return df_all[(df_all["time"] >= start_cutoff) & (df_all["time"] <= end_dt)].reset_index(drop=True)

def add_indicators(df, bb_window, bb_dev, cci_window):
    df = df.copy()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(df["close"], window=bb_window, window_dev=bb_dev)
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    cci = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=cci_window)
    df["cci"] = cci.cci()
    return df

# -----------------------------
# 실행
# -----------------------------
try:
    if start_date > end_date:
        st.error("시작 날짜가 종료 날짜보다 이후입니다.")
        st.stop()

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())
    warmup_bars = max(13, bb_window, int(cci_window)) * 5

    df_raw = fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars)
    if df_raw.empty:
        st.error("데이터가 없습니다.")
        st.stop()

    df_ind = add_indicators(df_raw, bb_window, bb_dev, cci_window)
    df = df_ind[(df_ind["time"] >= start_dt) & (df_ind["time"] <= end_dt)].reset_index(drop=True)

    # =========================
    # ③ 요약 & 차트
    # =========================
    if df.empty:
        st.warning("선택한 기간에 데이터가 없습니다.")
    else:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.72, 0.28])
        fig.add_trace(go.Candlestick(
            x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name="가격"
        ), row=1, col=1)
        if "bb_mid" in df.columns:
            fig.add_trace(go.Scatter(x=df["time"], y=df["bb_mid"], mode="lines", name="BB 중앙선"), row=1, col=1)
        if "bb_high" in df.columns:
            fig.add_trace(go.Scatter(x=df["time"], y=df["bb_high"], mode="lines", name="BB 상한선"), row=1, col=1)
        if "bb_low" in df.columns:
            fig.add_trace(go.Scatter(x=df["time"], y=df["bb_low"], mode="lines", name="BB 하한선"), row=1, col=1)
        if "rsi" in df.columns:
            fig.add_trace(go.Scatter(x=df["time"], y=df["rsi"], mode="lines", name="RSI(13)"), row=2, col=1)
            fig.add_hline(y=rsi_low, line_dash="dot", row=2, col=1)
            fig.add_hline(y=rsi_high, line_dash="dot", row=2, col=1)
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False, uirevision="keep")
        with chart_box:
            st.plotly_chart(fig, use_container_width=True, theme=None)

    # =========================
    # ④ 신호 결과
    # =========================
    signals = []
    if not df.empty:
        closes = df["close"].to_numpy()
        times = df["time"].to_numpy()
        for i in range(len(df) - lookahead):
            base = closes[i]
            hi = closes[i+1:i+lookahead+1].max()
            lo = closes[i+1:i+lookahead+1].min()
            up = (hi - base) / base * 100
            dn = (base - lo) / base * 100
            if up >= threshold_pct:
                result = "성공"
            elif dn >= threshold_pct:
                result = "실패"
            else:
                result = "중립"
            signals.append({"시간": times[i], "가격": base, "결과": result})
    if signals:
        succ = sum(1 for s in signals if s["결과"] == "성공")
        fail = sum(1 for s in signals if s["결과"] == "실패")
        neu = sum(1 for s in signals if s["결과"] == "중립")
        st.markdown('<div class="section-title">③ 요약</div>', unsafe_allow_html=True)
        st.write(pd.DataFrame([{"성공": succ, "중립": neu, "실패": fail, "총 신호": len(signals)}]))
        st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(signals).iloc[::-1].reset_index(drop=True), use_container_width=True, hide_index=True)
    else:
        st.info("신호 없음")

except Exception as e:
    st.error(f"오류: {e}")
