import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
import ta
from datetime import datetime

# -----------------------------
# 페이지 설정 & 스타일
# -----------------------------
st.set_page_config(page_title="Upbit RSI(13) + Bollinger Band 시뮬레이터", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 0.5rem; padding-bottom: 0.5rem; max-width: 1200px;}
  h2 {margin-top: 1rem;}
  table td {text-align: center;}
  .success {color: red; font-weight: bold;}
  .fail {color: blue; font-weight: bold;}
  .neutral {color: green; font-weight: bold;}
  .bb-up {color: red; font-weight: bold;}
  .bb-dn {color: blue; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 옵션
# -----------------------------
st.title("📊 Upbit RSI(13) + Bollinger Band 시뮬레이터")

# 업비트 전체 종목 가져오기
@st.cache_data(ttl=3600)
def fetch_markets():
    url = "https://api.upbit.com/v1/market/all"
    r = requests.get(url, headers={"Accept": "application/json"})
    data = r.json()
    return {item["korean_name"] + f" ({item['market']})": item["market"] for item in data if item["market"].startswith("KRW-")}

MARKETS = fetch_markets()

TF_MAP = {
    "1분": "minutes/1", "3분": "minutes/3", "5분": "minutes/5",
    "10분": "minutes/10", "15분": "minutes/15", "30분": "minutes/30",
    "60분": "minutes/60", "4시간": "minutes/240", "일봉": "days"
}

# ① 기본 설정
st.header("① 기본 설정")
c1, c2 = st.columns(2)
with c1:
    market_label = st.selectbox("종목 선택", list(MARKETS.keys()))
with c2:
    tf_label = st.selectbox("봉 종류 선택", list(TF_MAP.keys()), index=2)

c3, c4 = st.columns(2)
with c3:
    count = st.slider("캔들 개수", 100, 400, 200, step=20)
with c4:
    lookahead = st.slider("측정 캔들 수 (기준 이후 N봉)", 1, 60, 20)

c5, c6 = st.columns(2)
with c5:
    threshold_pct = st.slider("성공/실패 기준 값(%)", 0.1, 3.0, 1.0, step=0.1)
with c6:
    dup_option = st.radio("신호 중복 처리", ["중복 포함", "중복 제외"], horizontal=True)

# ② 조건 설정
st.header("② 조건 설정")
c7, c8 = st.columns(2)
with c7:
    rsi_side = st.selectbox("RSI 조건", ["RSI ≤ 30 (급락)", "RSI ≥ 70 (급등)"])
with c8:
    bb_cond = st.selectbox(
        "볼린저밴드 조건",
        ["없음", "상향돌파 (Upper ↑)", "하향돌파 (Upper ↓)", "상향돌파 (Lower ↑)", "하향돌파 (Lower ↓)"]
    )

# -----------------------------
# 데이터 수집
# -----------------------------
def fetch_upbit(market_code: str, tf_label: str, count: int) -> pd.DataFrame:
    interval = TF_MAP[tf_label]
    if "minutes/" in interval:
        unit = interval.split("/")[1]
        url = f"https://api.upbit.com/v1/candles/minutes/{unit}"
    else:
        url = f"https://api.upbit.com/v1/candles/{interval}"
    params = {"market": market_code, "count": count}
    r = requests.get(url, params=params, headers={"Accept": "application/json"}, timeout=10)
    df = pd.DataFrame(r.json())
    df = df.rename(columns={
        "candle_date_time_kst": "time", "opening_price": "open",
        "high_price": "high", "low_price": "low",
        "trade_price": "close", "candle_acc_trade_volume": "volume"
    })
    df["time"] = pd.to_datetime(df["time"])
    return df[["time", "open", "high", "low", "close", "volume"]].sort_values("time").reset_index(drop=True)

# -----------------------------
# 지표 계산
# -----------------------------
def add_indicators(df):
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=30, window_dev=2)
    out["BB_up"] = bb.bollinger_hband()
    out["BB_dn"] = bb.bollinger_lband()
    return out

# -----------------------------
# 시뮬레이션
# -----------------------------
def simulate(df, rsi_side, bb_cond, lookahead, thr_pct, dup_option):
    signals, used_idx = [], set()
    n = len(df)
    thr = thr_pct / 100.0
    sig_idx = []
    if "≤" in rsi_side:
        sig_idx = df.index[df["RSI13"] <= 30].tolist()
    else:
        sig_idx = df.index[df["RSI13"] >= 70].tolist()

    for i in sig_idx:
        if dup_option == "중복 제외" and i in used_idx:
            continue
        end = i + lookahead
        if end >= n: continue
        base_open = float(df.at[i, "open"])
        final_close = float(df.at[end, "close"])
        final_ret = (final_close / base_open - 1.0) * 100.0

        # 결과 분류
        if final_ret <= -thr_pct:
            result = "실패"
        elif final_ret >= thr_pct:
            result = "성공"
        else:
            result = "중립"

        # 볼린저 조건 필터링
        bb_pass = True
        if "Upper" in bb_cond:
            if "상향" in bb_cond: bb_pass = df.at[i, "close"] > df.at[i, "BB_up"]
            if "하향" in bb_cond: bb_pass = df.at[i, "close"] < df.at[i, "BB_up"]
        elif "Lower" in bb_cond:
            if "상향" in bb_cond: bb_pass = df.at[i, "close"] > df.at[i, "BB_dn"]
            if "하향" in bb_cond: bb_pass = df.at[i, "close"] < df.at[i, "BB_dn"]
        if not bb_pass:
            continue

        signals.append({
            "신호시간": df.at[i, "time"],
            "기준시가": f"{int(base_open):,}",
            "RSI(13)": round(df.at[i, "RSI13"], 1),
            "성공기준(%)": f"{thr_pct:.1f}%",
            "결과": result,
            "최종수익률(%)": f"{final_ret:.1f}%",
        })
        used_idx.add(i)
    return pd.DataFrame(signals)

# -----------------------------
# 실행
# -----------------------------
try:
    market_code = MARKETS[market_label]
    df = fetch_upbit(market_code, tf_label, count)
    df = add_indicators(df)
    res = simulate(df, rsi_side, bb_cond, lookahead, threshold_pct, dup_option)

    # ③ 기준 요약
    st.header("③ 기준 요약")
    total = len(res)
    wins = (res["결과"] == "성공").sum()
    fails = (res["결과"] == "실패").sum()
    neuts = (res["결과"] == "중립").sum()
    winrate = (wins + neuts) / total * 100 if total else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("신호 수", total)
    m2.metric("성공", wins)
    m3.metric("실패", fails)
    m4.metric("중립", neuts)
    m5.metric("승률", f"{winrate:.1f}%")

    # ④ 차트
    st.header("④ 차트 및 결과")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="가격"
    ))
    if not res.empty:
        for label, color, symbol in [("성공", "red", "triangle-up"),
                                     ("실패", "blue", "triangle-down"),
                                     ("중립", "green", "circle")]:
            sub = res[res["결과"] == label]
            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub["신호시간"], y=[float(s.replace(",", "")) for s in sub["기준시가"]],
                    mode="markers", name=label,
                    marker=dict(size=9, color=color, symbol=symbol, line=dict(width=1, color="black"))
                ))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_up"], line=dict(color="red", dash="dot"), name="BB Upper"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_dn"], line=dict(color="blue", dash="dot"), name="BB Lower"))
    st.plotly_chart(fig, use_container_width=True)

    # 결과 표
    if not res.empty:
        res_styled = res.copy()
        res_styled["결과"] = res_styled["결과"].map(
            lambda x: f'<span class="{ "success" if x=="성공" else "fail" if x=="실패" else "neutral"}">{x}</span>'
        )
        st.write(res_styled.to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.info("조건에 맞는 신호가 없습니다.")

except Exception as e:
    st.error(f"오류 발생: {e}")
