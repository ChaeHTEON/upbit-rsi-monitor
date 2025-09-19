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
</style>
""", unsafe_allow_html=True)

st.title("📊 Upbit RSI(13) + Bollinger Band 시뮬레이터")

# -----------------------------
# 업비트 마켓 로드 (네트워크 폴백 포함)
# -----------------------------
@st.cache_data(ttl=3600)
def get_upbit_krw_markets():
    url = "https://api.upbit.com/v1/market/all"
    try:
        r = requests.get(url, params={"isDetails":"false"}, timeout=8)
        r.raise_for_status()
        items = r.json()
        rows = []
        for it in items:
            mk = it.get("market","")
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
# (자리 유지) 신호 중복 처리 라디오
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
    rsi_side = st.selectbox("RSI 조건", ["없음", "RSI ≤ 30 (급락)", "RSI ≥ 70 (급등)"], index=0)

c7, c8, c9 = st.columns(3)
with c7:
    bb_cond = st.selectbox("볼린저밴드 조건", ["없음", "상한선", "중앙선", "하한선"], index=0)
with c8:
    bb_window = st.number_input("BB 기간", min_value=5, max_value=100, value=30, step=1)
with c9:
    bb_dev = st.number_input("BB 승수", min_value=1.0, max_value=4.0, value=2.0, step=0.1)

# -----------------------------
# 데이터 수집
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
    df = df[(df["time"] >= start_dt) & (df["time"] <= end_dt)]
    return df

# -----------------------------
# 지표
# -----------------------------
def add_indicators(df, bb_window, bb_dev):
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=bb_window, window_dev=bb_dev)
    out["BB_up"]  = bb.bollinger_hband().fillna(method="bfill").fillna(method="ffill")
    out["BB_low"] = bb.bollinger_lband().fillna(method="bfill").fillna(method="ffill")
    out["BB_mid"] = bb.bollinger_mavg().fillna(method="bfill").fillna(method="ffill")
    return out

# -----------------------------
# 순수 NumPy Holt 선형(이중 지수평활)
# -----------------------------
def holt_linear_in_out(y: np.ndarray, steps: int, alpha: float = 0.6, beta: float = 0.3):
    """과거 one-step 예측(pred_in) + 미래 k-step 예측(pred_out)"""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        if n == 0:
            return np.array([]), np.array([])
        if n == 1:
            return np.repeat(y[0], n), np.repeat(y[0], steps)
        slope = y[1] - y[0]
        pred_in = np.array([y[0], y[0] + slope])
        pred_out = np.array([y[-1] + (i+1)*slope for i in range(steps)])
        return pred_in, pred_out

    l = y[0]; b = y[1] - y[0]
    pred_in = np.zeros(n)
    pred_in[0] = y[0]
    for t in range(1, n):
        yhat_t = l + b            # t시점 one-step 예측
        pred_in[t] = yhat_t
        l_new = alpha * y[t] + (1 - alpha) * (l + b)
        b_new = beta * (l_new - l) + (1 - beta) * b
        l, b = l_new, b_new

    pred_out = np.array([l + (i+1)*b for i in range(steps)], dtype=float)
    return pred_in, pred_out

def holt_autotune(y: np.ndarray):
    alphas = [0.2, 0.4, 0.6, 0.8]
    betas  = [0.1, 0.3, 0.5]
    best = (0.6, 0.3); best_rmse = float("inf")
    if len(y) < 5:
        return best
    for a in alphas:
        for b in betas:
            pred_in, _ = holt_linear_in_out(y, steps=1, alpha=a, beta=b)
            actual = y[1:]; pred = pred_in[1:]
            rmse = np.sqrt(np.mean((actual - pred)**2))
            if rmse < best_rmse:
                best_rmse = rmse; best = (a, b)
    return best

def forecast_curve(df, minutes_per_bar):
    if df.empty:
        return pd.DataFrame(columns=["time","curve"]), np.array([]), np.array([])
    y = df["close"].astype(float).values
    steps = 1 if minutes_per_bar >= 1440 else max(1, 1440 // minutes_per_bar)
    a, b = holt_autotune(y)
    pred_in, pred_out = holt_linear_in_out(y, steps=steps, alpha=a, beta=b)

    past_times = df["time"].values
    if minutes_per_bar >= 1440:
        future_times = [df["time"].iloc[-1] + timedelta(days=i) for i in range(1, steps+1)]
    else:
        future_times = [df["time"].iloc[-1] + timedelta(minutes=minutes_per_bar*i) for i in range(1, steps+1)]

    times = list(past_times) + future_times
    curve = list(pred_in) + list(pred_out)
    return pd.DataFrame({"time": times, "curve": curve}), pred_in, pred_out

# -----------------------------
# 실행
# -----------------------------
try:
    if start_date > end_date:
        st.error("시작 날짜가 종료 날짜보다 이후입니다.")
        st.stop()

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt   = datetime.combine(end_date,   datetime.max.time())

    df = fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar)
    if df.empty:
        st.error("데이터가 없습니다."); st.stop()

    df = add_indicators(df, bb_window, bb_dev)

    # 예측 표시 및 적중률 허용오차
    show_forecast = st.checkbox("예측 추세선 표시 (1일치)", value=True)
    tol_pct = st.slider("예측 허용오차(%) — 실제 종가가 예측선 ±오차 이내면 적중", 0.5, 5.0, 2.0, 0.5)

    # -----------------------------
    # 차트
    # -----------------------------
    fig=make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="가격", increasing_line_color="red", decreasing_line_color="blue", line=dict(width=1.1)
    ))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_up"], mode="lines",
                             line=dict(color="#FFB703", width=1.4), name="BB 상단"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_low"], mode="lines",
                             line=dict(color="#219EBC", width=1.4), name="BB 하단"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_mid"], mode="lines",
                             line=dict(color="#8D99AE", width=1.1, dash="dot"), name="BB 중앙"))

    acc_stats = None
    if show_forecast:
        fc_df, pred_in, pred_out = forecast_curve(df, minutes_per_bar)
        if not fc_df.empty:
            fig.add_trace(go.Scatter(
                x=fc_df["time"], y=fc_df["curve"], mode="lines",
                line=dict(color="red", width=2), name="추세선(과거+예측)"
            ))

            # ----- 적중률 통계(과거 one-step 예측 vs 실제) -----
            y = df["close"].astype(float).values
            if len(y) >= 3:
                actual = y[1:]        # t 시점 실제
                pred   = pred_in[1:]  # t 시점 one-step 예측(직전상태)
                # 허용오차 적중률
                hit = np.abs(actual - pred) / np.maximum(1e-12, np.abs(actual)) * 100.0 <= tol_pct
                hit_rate = float(hit.mean()*100.0)

                # 방향 적중률 (이전 실제 대비 방향)
                d_actual = actual - y[:-1]
                d_pred   = pred   - y[:-1]
                dir_match = np.sign(d_actual) == np.sign(d_pred)
                dir_acc = float(dir_match.mean()*100.0)

                # RMSE% / MAPE%
                rmse_pct = float(np.sqrt(np.mean(((actual - pred)/np.maximum(1e-12, np.abs(actual)))**2))*100.0)
                mape_pct = float(np.mean(np.abs((actual - pred)/np.maximum(1e-12, np.abs(actual))))*100.0)

                acc_stats = {"N": len(actual),
                             "hit_rate": hit_rate,
                             "dir_acc": dir_acc,
                             "rmse_pct": rmse_pct,
                             "mape_pct": mape_pct}

    fig.update_layout(
        title=f"{market_label.split(' — ')[0]} · {tf_label} · RSI(13) + BB 시뮬레이션",
        dragmode="zoom", xaxis_rangeslider_visible=False, height=600, autosize=False,
        legend_orientation="h", legend_y=1.05,
        margin=dict(l=60,r=40,t=60,b=40),
        yaxis=dict(title="가격"),
        yaxis2=dict(overlaying="y", side="right", showgrid=False, title="RSI(13)", range=[0,100])
    )
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "doubleClick": "reset"})

    # -----------------------------
    # 적중률 요약 메트릭
    # -----------------------------
    if acc_stats is not None:
        st.markdown('<div class="section-title">③ 예측 적중률 통계 (과거 구간)</div>', unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("표본 N", f"{acc_stats['N']}")
        m2.metric(f"허용오차 적중률 (±{tol_pct:.1f}%)", f"{acc_stats['hit_rate']:.1f}%")
        m3.metric("방향 적중률", f"{acc_stats['dir_acc']:.1f}%")
        m4.metric("RMSE%", f"{acc_stats['rmse_pct']:.2f}%")
        m5.metric("MAPE%", f"{acc_stats['mape_pct']:.2f}%")

except Exception as e:
    st.error(f"오류: {e}")
