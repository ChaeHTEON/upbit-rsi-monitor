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
from pytz import timezone

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

# -----------------------------
# 상단: 신호 중복 처리
# -----------------------------
dup_mode = st.radio("신호 중복 처리", ["중복 포함 (연속 신호 모두)", "중복 제거 (연속 동일 결과 1개)"], horizontal=True)

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
    KST = timezone("Asia/Seoul")
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
    hit_basis = st.selectbox("성공 판정 기준", ["종가 기준","고가 기준(스침 인정)","종가 또는 고가"], index=0)
with c6:
    r1, r2, r3 = st.columns(3)
    with r1: rsi_mode = st.selectbox("RSI 조건", ["없음","현재(과매도/과매수 중 하나)","과매도 기준","과매수 기준"], index=0)
    with r2: rsi_low = st.slider("과매도 RSI 기준", 0, 100, 30, step=1)
    with r3: rsi_high = st.slider("과매수 RSI 기준", 0, 100, 70, step=1)

c7, c8, c9 = st.columns(3)
with c7: bb_cond = st.selectbox("볼린저밴드 조건", ["없음","상한선","중앙선","하한선"], index=0)
with c8: bb_window = st.number_input("BB 기간", min_value=5, max_value=100, value=30, step=1)
with c9: bb_dev = st.number_input("BB 승수", min_value=1.0, max_value=4.0, value=2.0, step=0.1)

miss_policy = st.selectbox("미도달 처리", ["실패(권장)","중립(미도달=항상 중립)","중립(예전: -thr 이하면 실패)"], index=0)
sec_cond = st.selectbox("2차 조건 선택", ["없음","양봉 2개 연속 상승","BB 기반 첫 양봉 50% 진입"], index=0)
st.session_state["bb_cond"] = bb_cond
st.markdown("---")

# -----------------------------
# 데이터 수집
# -----------------------------
_session = requests.Session()
_retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
_session.mount("https://", HTTPAdapter(max_retries=_retries))

@st.cache_data(ttl=120, show_spinner=False)
def fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars: int = 0):
    start_cutoff = start_dt - timedelta(minutes=max(0, warmup_bars) * minutes_per_bar)
    if "minutes/" in interval_key:
        unit = interval_key.split("/")[1]
        url = f"https://api.upbit.com/v1/candles/minutes/{unit}"
    else:
        url = "https://api.upbit.com/v1/candles/days"
    all_rows, to_time = [], None
    try:
        for _ in range(60):
            params = {"market": market_code, "count": 200}
            if to_time is not None:
                params["to"] = to_time.strftime("%Y-%m-%d %H:%M:%S")
            r = _session.get(url, params=params, headers={"Accept": "application/json"}, timeout=10)
            r.raise_for_status()
            batch = r.json()
            if not batch: break
            all_rows.extend(batch)
            last_kst = pd.to_datetime(batch[-1]["candle_date_time_kst"])
            if last_kst <= start_cutoff: break
            to_time = last_kst - timedelta(seconds=1)
    except Exception:
        return pd.DataFrame()
    if not all_rows: return pd.DataFrame()
    df = pd.DataFrame(all_rows).rename(columns={
        "candle_date_time_kst":"time","opening_price":"open","high_price":"high",
        "low_price":"low","trade_price":"close","candle_acc_trade_volume":"volume"})
    df["time"] = pd.to_datetime(df["time"])
    return df[["time","open","high","low","close","volume"]].sort_values("time").reset_index(drop=True)

# -----------------------------
# 지표
# -----------------------------
def add_indicators(df, bb_window, bb_dev):
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=int(bb_window), window_dev=float(bb_dev))
    out["BB_up"]  = bb.bollinger_hband()
    out["BB_low"] = bb.bollinger_lband()
    out["BB_mid"] = bb.bollinger_mavg()
    for c in ["RSI13","BB_up","BB_low","BB_mid"]:
        out[c] = out[c].fillna(method="bfill").fillna(method="ffill")
    return out

# -----------------------------
# 시뮬레이션 (간소화: 핵심 로직만 유지)
# -----------------------------
def simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, thr_pct, bb_cond, dedup_mode,
             minutes_per_bar, market_code, bb_window, bb_dev, sec_cond="없음",
             hit_basis="종가 기준", miss_policy="실패(권장)"):
    res = []
    n, thr = len(df), float(thr_pct)
    for i in range(n - lookahead):
        base_price = float(df.at[i,"close"]); signal_time = df.at[i,"time"]
        end_idx = i + lookahead; end_time = df.at[end_idx,"time"]
        end_price = float(df.at[end_idx,"close"])
        final_ret = (end_price/base_price - 1) * 100
        target = base_price * (1+thr/100)
        result = "실패" if final_ret < 0 else "성공" if end_price>=target else "중립"
        res.append({"신호시간":signal_time,"종료시간":end_time,"기준시가":base_price,"종료가":end_price,"결과":result})
    return pd.DataFrame(res)

# -----------------------------
# 실행
# -----------------------------
try:
    if start_date > end_date:
        st.error("시작 날짜가 종료 날짜보다 이후입니다."); st.stop()
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt   = datetime.combine(end_date, datetime.max.time())
    warmup_bars = max(13, bb_window) * 5

    df_raw = fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars)
    if df_raw.empty: st.error("데이터가 없습니다."); st.stop()
    df = add_indicators(df_raw, bb_window, bb_dev)

    # 🔄 차트 컨트롤
    if "last_refresh" not in st.session_state: st.session_state["last_refresh"] = datetime.now()
    st.markdown("### 🔄 차트 컨트롤")
    cc1, cc2 = st.columns([1,2])
    with cc1:
        if st.button("🔄 새로고침"):
            now = datetime.now()
            if (now-st.session_state["last_refresh"]).total_seconds()>=3:
                st.session_state["last_refresh"]=now; st.experimental_rerun()
            else: st.warning("새로고침은 3초 간격으로만 가능합니다.")
    with cc2:
        sel_idx = next((i for i,(_,code) in enumerate(MARKET_LIST) if code==market_code),0)
        market_label2, market_code2 = st.selectbox("차트 근처 종목 선택", MARKET_LIST, index=sel_idx, format_func=lambda x:x[0])
        if market_code2 != market_code: market_code=market_code2; st.experimental_rerun()

    # 차트
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                 name="가격", increasing_line_color="red", decreasing_line_color="blue"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_up"],  mode="lines", name="BB 상단"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_low"], mode="lines", name="BB 하단"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_mid"], mode="lines", name="BB 중앙"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["RSI13"], mode="lines", name="RSI(13)", yaxis="y2"))

    fig.update_layout(title=f"{market_label.split(' — ')[0]} · {tf_label} · RSI(13)+BB 시뮬레이션",
        dragmode="zoom", xaxis_rangeslider_visible=False, height=600,
        yaxis=dict(title="가격"), yaxis2=dict(overlaying="y", side="right", range=[0,100], title="RSI(13)"),
        uirevision="chart-view")
    st.plotly_chart(fig, width="stretch", config={"scrollZoom":True,"doubleClick":"reset"})

    # 신호 결과
    st.markdown("### ④ 신호 결과 (최신 순)")
    res = simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, threshold_pct, bb_cond, dup_mode,
                   minutes_per_bar, market_code, bb_window, bb_dev, sec_cond, hit_basis, miss_policy)
    if res.empty: st.info("조건을 만족하는 신호가 없습니다.")
    else: st.dataframe(res.sort_values("신호시간",ascending=False), width="stretch")

except Exception as e:
    st.error(f"오류: {e}")
