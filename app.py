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
    threshold_pct = st.slider("성공/실패 기준 값(%)", 0.1, 3.0, 1.0, step=0.1)
with c6:
    rsi_side = st.selectbox(
        "RSI 조건",
        ["없음", "RSI ≤ 30 (급락)", "RSI ≥ 70 (급등)"],
        index=0
    )

c7, c8, c9 = st.columns(3)
with c7:
    bb_cond = st.selectbox(
        "볼린저밴드 조건",
        ["없음","하한선 하향돌파","하한선 상향돌파","상한선 하향돌파","상한선 상향돌파","하한선 중앙돌파","상한선 중앙돌파"],
        index=0,
    )
with c8:
    bb_window = st.number_input("BB 기간", min_value=5, max_value=100, value=30, step=1)
with c9:
    bb_dev = st.number_input("BB 승수", min_value=1.0, max_value=4.0, value=2.0, step=0.1)

st.session_state["rsi_side"] = rsi_side
st.session_state["bb_cond"]  = bb_cond

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
    while True:
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
        if len(all_data) >= req_count*max_calls:
            break

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
def add_indicators(df, bb_window, bb_dev):
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=bb_window, window_dev=bb_dev)
    # NaN 처리 보정
    out["BB_up"]  = bb.bollinger_hband().fillna(method="ffill").fillna(method="bfill")
    out["BB_low"] = bb.bollinger_lband().fillna(method="ffill").fillna(method="bfill")
    out["BB_mid"] = bb.bollinger_mavg().fillna(method="ffill").fillna(method="bfill")
    return out

# -----------------------------
# 시뮬레이션
# -----------------------------
def simulate(df, rsi_side, lookahead, thr_pct, bb_cond, dedup_mode,
             minutes_per_bar, market_code, bb_window, bb_dev):
    res=[]
    n=len(df); thr=float(thr_pct)

    if rsi_side == "없음":
        sig_idx = df.index.tolist()
    elif rsi_side == "RSI ≤ 30 (급락)":
        sig_idx = df.index[(df["RSI13"].shift(1) > 30) & (df["RSI13"] <= 30)].tolist()
    elif rsi_side == "RSI ≥ 70 (급등)":
        sig_idx = df.index[(df["RSI13"].shift(1) < 70) & (df["RSI13"] >= 70)].tolist()
    else:
        sig_idx = []
    for i in sig_idx:
        end=i+lookahead
        if end>=n: continue

        if bb_cond != "없음":
            hi = float(df.at[i, "high"])
            lo_px = float(df.at[i, "low"])
            up, lo, mid = df.at[i, "BB_up"], df.at[i, "BB_low"], df.at[i, "BB_mid"]
            ok = False

            if bb_cond == "하한선 하향돌파":
                ok = pd.notna(lo) and (lo_px <= lo <= hi)
            elif bb_cond == "하한선 상향돌파":
                ok = pd.notna(lo) and (lo_px <= lo <= hi)
            elif bb_cond == "상한선 하향돌파":
                ok = pd.notna(up) and (lo_px <= up <= hi)
            elif bb_cond == "상한선 상향돌파":
                ok = pd.notna(up) and (lo_px <= up <= hi)
            elif bb_cond == "하한선 중앙돌파":
                ok = pd.notna(lo) and pd.notna(mid) and (lo_px <= lo <= hi and lo_px <= mid <= hi)
            elif bb_cond == "상한선 중앙돌파":
                ok = pd.notna(up) and pd.notna(mid) and (lo_px <= up <= hi and lo_px <= mid <= hi)

            # 하위봉 판정
            if not ok and minutes_per_bar > 1:
                lower_unit = 1
                lower_key = f"minutes/{lower_unit}"
                start_i = df.at[i,"time"]
                end_i   = df.at[i,"time"] + timedelta(minutes=minutes_per_bar)
                sub_df = fetch_upbit_paged(market_code, lower_key, start_i, end_i, lower_unit)
                if not sub_df.empty:
                    sub_df = add_indicators(sub_df, bb_window, bb_dev)
                    for j in sub_df.index:
                        bb_low_val = sub_df.at[j,"BB_low"]
                        bb_up_val  = sub_df.at[j,"BB_up"]
                        bb_mid_val = sub_df.at[j,"BB_mid"]
                        sub_low    = sub_df.at[j,"low"]
                        sub_high   = sub_df.at[j,"high"]
                        if bb_cond in ("하한선 하향돌파","하한선 상향돌파"):
                            if pd.notna(bb_low_val) and (sub_low <= bb_low_val <= sub_high):
                                ok = True; break
                        elif bb_cond in ("상한선 하향돌파","상한선 상향돌파"):
                            if pd.notna(bb_up_val) and (sub_low <= bb_up_val <= sub_high):
                                ok = True; break
                        elif bb_cond == "하한선 중앙돌파":
                            if pd.notna(bb_low_val) and pd.notna(bb_mid_val) and \
                               (sub_low <= bb_low_val <= sub_high and sub_low <= bb_mid_val <= sub_high):
                                ok = True; break
                        elif bb_cond == "상한선 중앙돌파":
                            if pd.notna(bb_up_val) and pd.notna(bb_mid_val) and \
                               (sub_low <= bb_up_val <= sub_high and sub_low <= bb_mid_val <= sub_high):
                                ok = True; break
            if not ok:
                continue

        # --- 성과 계산 (기존 로직 그대로) ---
        base = float(df.at[i, "close"])
        closes = df.loc[i+1:end, ["time","close"]]
        if closes.empty: continue
        final_ret = (closes.iloc[-1]["close"]/base - 1)*100
        min_ret   = (closes["close"].min()/base - 1)*100
        max_ret   = (closes["close"].max()/base - 1)*100

        result="중립"; reach_min=None
        if max_ret >= thr:
            first_hit = closes[closes["close"] >= base*(1+thr/100)]
            if not first_hit.empty:
                reach_min = int((first_hit.iloc[0]["time"] - df.at[i,"time"]).total_seconds() // 60)
            result="성공"
        elif final_ret < 0:
            result="실패"

        res.append({
            "신호시간": df.at[i,"time"], "기준시가": int(round(base)),
            "RSI(13)": round(float(df.at[i,"RSI13"]),1) if pd.notna(df.at[i,"RSI13"]) else None,
            "성공기준(%)": round(thr,1), "결과": result, "도달분": reach_min,
            "최종수익률(%)": round(final_ret,2), "최저수익률(%)": round(min_ret,2), "최고수익률(%)": round(max_ret,2)
        })
    return pd.DataFrame(res)

# -----------------------------
# 메인 실행
# -----------------------------
if st.button("시뮬레이션 실행"):
    df = fetch_upbit_paged(market_code, interval_key, datetime.combine(start_date, datetime.min.time()), datetime.combine(end_date, datetime.max.time()), minutes_per_bar)
    if df.empty:
        st.warning("데이터 없음")
    else:
        df = add_indicators(df, bb_window, bb_dev)
        res_all = simulate(df, rsi_side, lookahead, threshold_pct, bb_cond, dup_mode, minutes_per_bar, market_code, bb_window, bb_dev)
        st.dataframe(res_all)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="캔들"))
        fig.add_trace(go.Scatter(x=df["time"], y=df["BB_up"], mode="lines", name="BB 상단", line=dict(color="#FFB703"), connectgaps=True))
        fig.add_trace(go.Scatter(x=df["time"], y=df["BB_low"], mode="lines", name="BB 하단", line=dict(color="#219EBC"), connectgaps=True))
        fig.add_trace(go.Scatter(x=df["time"], y=df["BB_mid"], mode="lines", name="BB 중앙", line=dict(color="#8D99AE", dash="dot"), connectgaps=True))
        st.plotly_chart(fig, use_container_width=True)


except Exception as e:
    st.error(f"오류: {e}")













