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
# 시뮬레이션
# -----------------------------
def simulate(df, rsi_side, lookahead, thr_pct, bb_cond, dedup_mode,
             minutes_per_bar, market_code, bb_window, bb_dev):

    res=[]
    n=len(df); thr=float(thr_pct)

    def bb_ok(i: int) -> bool:
        if bb_cond == "없음": return True
        hi = float(df.at[i, "high"])
        lo_px = float(df.at[i, "low"])
        cl = float(df.at[i, "close"])
        up, lo, mid = df.at[i, "BB_up"], df.at[i, "BB_low"], df.at[i, "BB_mid"]

        if bb_cond == "하한선 하향돌파":
            return pd.notna(lo) and (lo_px <= lo or cl <= lo)
        if bb_cond == "하한선 상향돌파":
            prev_cl = float(df.at[i-1,"close"]) if i > 0 else None
            return pd.notna(lo) and ((prev_cl is not None and prev_cl < lo <= cl) or (cl >= lo and lo_px <= lo))
        if bb_cond == "상한선 하향돌파":
            prev_cl = float(df.at[i-1,"close"]) if i > 0 else None
            return pd.notna(up) and ((prev_cl is not None and prev_cl > up >= cl) or (hi >= up and cl <= up))
        if bb_cond == "상한선 상향돌파":
            return pd.notna(up) and (cl >= up or hi >= up)
        if bb_cond == "하한선 중앙돌파":
            prev_cl = float(df.at[i-1,"close"]) if i > 0 else None
            return pd.notna(mid) and ((prev_cl is not None and prev_cl < mid <= cl) or (cl >= mid and lo_px <= mid))
        if bb_cond == "상한선 중앙돌파":
            prev_cl = float(df.at[i-1,"close"]) if i > 0 else None
            return pd.notna(mid) and ((prev_cl is not None and prev_cl > mid >= cl) or (hi >= mid and cl <= mid))
        return False

    rsi_idx = []
    if rsi_side == "RSI ≤ 30 (급락)":
        rsi_idx = df.index[(df["RSI13"] <= 30) | ((df["RSI13"].shift(1) > 30) & (df["RSI13"] <= 30))].tolist()
    elif rsi_side == "RSI ≥ 70 (급등)":
        rsi_idx = df.index[(df["RSI13"] >= 70) | ((df["RSI13"].shift(1) < 70) & (df["RSI13"] >= 70))].tolist()

    bb_idx = []
    if bb_cond != "없음":
        for i in df.index:
            try:
                if bb_ok(i): bb_idx.append(i)
            except Exception: continue

    if rsi_side != "없음" and bb_cond != "없음": sig_idx = sorted(set(rsi_idx) | set(bb_idx))
    elif rsi_side != "없음": sig_idx = rsi_idx
    elif bb_cond != "없음": sig_idx = bb_idx
    else: sig_idx = []

    for i in sig_idx:
        end=i+lookahead
        if end>=n: continue

        # ✅ 기준가: 시가와 저가의 중간
        base = (float(df.at[i,"open"]) + float(df.at[i,"low"])) / 2.0
        closes=df.loc[i+1:end,["time","close"]]
        if closes.empty: continue

        # ✅ 특정 시간대 디버깅
        if df.at[i,"time"].strftime("%Y-%m-%d %H:%M") == "2025-09-18 04:00":
            st.write({
                "time": df.at[i,"time"],
                "open": float(df.at[i,"open"]),
                "low": float(df.at[i,"low"]),
                "close": float(df.at[i,"close"]),
                "BB_low": float(df.at[i,"BB_low"]),
                "BB_mid": float(df.at[i,"BB_mid"]),
                "BB_up": float(df.at[i,"BB_up"]),
                "RSI13": float(df.at[i,"RSI13"]) if pd.notna(df.at[i,"RSI13"]) else None,
                "base": base
            })

        final_ret=(closes.iloc[-1]["close"]/base-1)*100.0
        min_ret=(closes["close"].min()/base-1)*100.0
        max_ret=(closes["close"].max()/base-1)*100.0

        result="중립"; reach_min=None
        if max_ret >= thr:
            first_hit = closes[closes["close"] >= base*(1+thr/100)]
            if not first_hit.empty:
                reach_min = int((first_hit.iloc[0]["time"] - df.at[i,"time"]).total_seconds() // 60)
            result = "성공"
        elif final_ret < 0:
            result = "실패"

        def fmt_ret(val): return round(val, 2)

        res.append({
            "신호시간": df.at[i,"time"],
            "기준시가": int(round(base)),
            "RSI(13)": round(float(df.at[i,"RSI13"]),1) if pd.notna(df.at[i,"RSI13"]) else None,
            "성공기준(%)": round(thr,1),
            "결과": result,
            "도달분": reach_min,
            "최종수익률(%)": fmt_ret(final_ret),
            "최저수익률(%)": fmt_ret(min_ret),
            "최고수익률(%)": fmt_ret(max_ret),
        })

    out=pd.DataFrame(res, columns=["신호시간","기준시가","RSI(13)","성공기준(%)","결과","도달분","최종수익률(%)","최저수익률(%)","최고수익률(%)"])

    if not out.empty and dedup_mode.startswith("중복 제거"):
        out["분"] = pd.to_datetime(out["신호시간"]).dt.strftime("%Y-%m-%d %H:%M")
        out = out.drop_duplicates(subset=["분"], keep="first").drop(columns=["분"])

    return out

# -----------------------------
# 실행
# -----------------------------
try:
    if start_date > end_date:
        st.error("시작 날짜가 종료 날짜보다 이후입니다."); st.stop()

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt   = datetime.combine(end_date,   datetime.max.time())

    df = fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar)
    if df.empty: st.error("데이터가 없습니다."); st.stop()

    if rsi_side == "없음" and bb_cond == "없음":
        st.markdown('<div class="section-title">③ 요약 & 차트</div>', unsafe_allow_html=True)
        st.info("대기중..")
        st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)
        st.info("대기중..")
        st.stop()

    df = add_indicators(df, bb_window, bb_dev)
    rsi_side = st.session_state.get("rsi_side", rsi_side)
    bb_cond  = st.session_state.get("bb_cond", bb_cond)

    res_all   = simulate(df, rsi_side, lookahead, threshold_pct, bb_cond,
                         "중복 포함 (연속 신호 모두)", minutes_per_bar, market_code, bb_window, bb_dev)
    res_dedup = simulate(df, rsi_side, lookahead, threshold_pct, bb_cond,
                         "중복 제거 (연속 동일 결과 1개)", minutes_per_bar, market_code, bb_window, bb_dev)

    st.markdown('<div class="section-title">③ 요약 & 차트</div>', unsafe_allow_html=True)

    # 👉 여기서 차트 + 테이블 출력 코드 동일하게 유지 (생략)
    # ... (기존 그래프 및 DataFrame 출력 부분 붙여넣기) ...

except Exception as e:
    st.error(f"오류: {e}")
