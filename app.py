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

# -----------------------------
# 세션 상태 초기화
# -----------------------------
if "opt_view" not in st.session_state:
    st.session_state.opt_view = False
if "buy_price" not in st.session_state:
    st.session_state.buy_price = 0
if "buy_price_text" not in st.session_state:
    st.session_state.buy_price_text = "0"

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
# ② 조건 설정 (간소화)
# -----------------------------
c4, c5, c6 = st.columns(3)
with c4:
    lookahead = st.slider("측정 캔들 수 (기준 이후 N봉)", 1, 60, 10)
with c5:
    threshold_pct = st.slider("성공/실패 기준 값(%)", 0.1, 5.0, 1.0, step=0.1)
    hit_basis = st.selectbox("성공 판정 기준", ["종가 기준", "고가 기준(스침 인정)", "종가 또는 고가"], index=0)
with c6:
    rsi_low = st.slider("과매도 RSI 기준", 0, 100, 30, step=1)
    rsi_high = st.slider("과매수 RSI 기준", 0, 100, 70, step=1)

bb_window = 30
bb_dev = 2.0
cci_window = 14

# -----------------------------
# 데이터 수집/지표 함수
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
    for _ in range(60):
        params = {"market": market_code, "count": 200}
        if to_time is not None:
            params["to"] = to_time.strftime("%Y-%m-%d %H:%M:%S")
        r = _session.get(url, params=params, headers={"Accept": "application/json"}, timeout=10)
        if r.status_code != 200: break
        batch = r.json()
        if not batch: break
        all_data.extend(batch)
        last_ts = pd.to_datetime(batch[-1]["candle_date_time_kst"])
        if last_ts <= start_cutoff: break
        to_time = last_ts - timedelta(seconds=1)
    if not all_data: return pd.DataFrame()
    df = pd.DataFrame(all_data).rename(columns={
        "candle_date_time_kst": "time",
        "opening_price": "open",
        "high_price": "high",
        "low_price": "low",
        "trade_price": "close",
        "candle_acc_trade_volume": "volume",
    })
    df["time"] = pd.to_datetime(df["time"])
    return df[["time","open","high","low","close","volume"]].sort_values("time").reset_index(drop=True)

def add_indicators(df, bb_window, bb_dev, cci_window):
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=bb_window, window_dev=bb_dev)
    out["BB_up"]  = bb.bollinger_hband()
    out["BB_low"] = bb.bollinger_lband()
    out["BB_mid"] = bb.bollinger_mavg()
    return out

# -----------------------------
# 실행
# -----------------------------
try:
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())
    df_raw = fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar, 60)
    if df_raw.empty:
        st.error("데이터가 없습니다.")
        st.stop()

    df = add_indicators(df_raw, bb_window, bb_dev, cci_window)

    # 수익률 customdata
    buy_price = st.session_state.get("buy_price", 0)
    pnl_vals = (df["close"]/buy_price -1)*100 if buy_price>0 else pd.Series([0]*len(df))
    pnl_vals = pnl_vals.round(1).to_numpy()
    pnl_colors = np.where(pnl_vals>0,"red",np.where(pnl_vals<0,"blue","gray"))
    pnl_cd = np.column_stack([pnl_vals, pnl_colors])

    # 차트
    fig = make_subplots(rows=1, cols=1)

    # Candlestick (hovertext)
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="가격",
        increasing=dict(line=dict(color="red", width=1.1)),
        decreasing=dict(line=dict(color="blue", width=1.1)),
        customdata=pnl_cd,
        hovertext=[f"<span style='color:{c[1]};'>수익률: {float(c[0]):.1f}%</span>" for c in pnl_cd],
        hoverinfo="text"
    ))

    # 빈영역 Hover
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["close"], mode="lines",
        line=dict(color="rgba(0,0,0,0)", width=1e-3),
        showlegend=False,
        customdata=pnl_cd,
        hovertemplate="<span style='color:%{customdata[1]};'>수익률: %{customdata[0]:.1f}%</span><extra></extra>"
    ))

    # UI + 차트 출력
    with chart_box:
        top_l, top_r = st.columns([4,1])
        with top_l:
            buy_price = st.number_input("💰 매수가 입력", min_value=0,
                                        value=st.session_state.get("buy_price",0),
                                        step=1000, format="%d", key="buy_price_num")
            st.session_state.buy_price = buy_price
            st.session_state.buy_price_text = f"{buy_price:,}" if buy_price>0 else "0"
            st.markdown("<style>div[data-testid='stNumberInput'] {width:220px !important;}</style>", unsafe_allow_html=True)
        with top_r:
            st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)
            label = "↩ 되돌아가기" if st.session_state.opt_view else "📈 최적화뷰"
            if st.button(label, key="btn_opt_view_top"):
                st.session_state.opt_view = not st.session_state.opt_view
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"오류: {e}")
