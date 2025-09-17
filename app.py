import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
import ta

# -----------------------------
# 페이지 스타일
# -----------------------------
st.set_page_config(page_title="Upbit RSI+BB 시뮬레이터", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 0.8rem; padding-bottom: 0.8rem; max-width: 1100px;}
  @media (max-width: 600px) {
    h1, h2, h3 {font-size: 1.1rem;}
    .stMetric {text-align:center;}
  }
</style>
""", unsafe_allow_html=True)

st.title("📊 Upbit RSI(13) + Bollinger Band 시뮬레이터")

# -----------------------------
# 옵션
# -----------------------------
MARKETS = {
    "비트 (BTC)": "KRW-BTC",
    "리플 (XRP)": "KRW-XRP",
    "도지 (DOGE)": "KRW-DOGE",
    "이더리움 (ETH)": "KRW-ETH",
    "솔라나 (SOL)": "KRW-SOL",
}
TF_MAP = {
    "1분": ("minutes/1", 1, "분"),
    "3분": ("minutes/3", 3, "분"),
    "5분": ("minutes/5", 5, "분"),
    "10분": ("minutes/10", 10, "분"),
    "15분": ("minutes/15", 15, "분"),
    "30분": ("minutes/30", 30, "분"),
    "60분": ("minutes/60", 60, "분"),
    "4시간": ("minutes/240", 240, "분"),
    "일봉": ("days", 1440, "일"),
}

c1, c2 = st.columns(2)
with c1:
    market_label = st.selectbox("종목 선택", list(MARKETS.keys()), index=0)
with c2:
    tf_label = st.selectbox("봉 종류 선택", list(TF_MAP.keys()), index=0)

c3, c4 = st.columns(2)
with c3:
    count = st.slider("캔들 개수", 100, 400, 200, step=20)
with c4:
    lookahead = st.slider("측정 캔들 수 (기준 이후 N봉)", 1, 60, 10)

c5, c6 = st.columns(2)
with c5:
    threshold_pct = st.slider("성공/실패 기준 값(%)", 0.1, 3.0, 1.0, step=0.1)
with c6:
    rsi_side = st.selectbox("RSI 조건", ["RSI ≤ 30 (급락)", "RSI ≥ 70 (급등)"], index=0)

bb_cond = st.selectbox("볼린저밴드 조건", [
    "없음",
    "하한선 상향돌파",
    "하한선 하향돌파",
    "상한선 상향돌파",
    "상한선 하향돌파"
], index=0)

# -----------------------------
# 데이터 수집
# -----------------------------
def fetch_upbit(market_code: str, tf_label: str, count: int) -> pd.DataFrame:
    interval, _, _ = TF_MAP[tf_label]
    if "minutes/" in interval:
        unit = interval.split("/")[1]
        url = f"https://api.upbit.com/v1/candles/minutes/{unit}"
    else:
        url = f"https://api.upbit.com/v1/candles/{interval}"
    params = {"market": market_code, "count": count}
    r = requests.get(url, params=params, headers={"Accept": "application/json"}, timeout=10)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df = df.rename(columns={
        "candle_date_time_kst": "time",
        "opening_price": "open",
        "high_price": "high",
        "low_price": "low",
        "trade_price": "close",
        "candle_acc_trade_volume": "volume"
    })
    df["time"] = pd.to_datetime(df["time"])
    df = df[["time", "open", "high", "low", "close", "volume"]].sort_values("time").reset_index(drop=True)
    return df

# -----------------------------
# RSI & Bollinger
# -----------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=30, window_dev=2)
    out["BB_upper"] = bb.bollinger_hband()
    out["BB_lower"] = bb.bollinger_lband()
    return out

# -----------------------------
# 시뮬레이션
# -----------------------------
def simulate(df: pd.DataFrame, rsi_side: str, lookahead: int, thr_pct: float, bb_cond: str):
    out = []
    n = len(df)
    thr = thr_pct / 100.0

    if "≤" in rsi_side:
        sig_idx = df.index[df["RSI13"] <= 30].tolist()
    else:
        sig_idx = df.index[df["RSI13"] >= 70].tolist()

    for i in sig_idx:
        end = i + lookahead
        if end >= n:
            continue

        # 볼린저밴드 조건 체크
        valid = True
        if bb_cond != "없음":
            prev_close = df.at[i-1, "close"] if i > 0 else df.at[i, "close"]
            curr_close = df.at[i, "close"]
            upper, lower = df.at[i, "BB_upper"], df.at[i, "BB_lower"]

            if bb_cond == "하한선 상향돌파":
                valid = prev_close < lower and curr_close > lower
            elif bb_cond == "하한선 하향돌파":
                valid = prev_close > lower and curr_close < lower
            elif bb_cond == "상한선 상향돌파":
                valid = prev_close < upper and curr_close > upper
            elif bb_cond == "상한선 하향돌파":
                valid = prev_close > upper and curr_close < upper

        if not valid:
            continue

        base_open = float(df.at[i, "open"])
        win = df.loc[i+1:end, :]
        win_high, win_low = float(win["high"].max()), float(win["low"].min())

        target_up, target_dn = base_open * (1 + thr), base_open * (1 - thr)
        hit_up, hit_dn = win_high >= target_up, win_low <= target_dn

        if hit_up and not hit_dn:
            result = "성공"
        elif hit_dn and not hit_up:
            result = "실패"
        elif hit_up and hit_dn:
            result = "중립"
        else:
            final_close = float(df.at[end, "close"])
            result = "실패" if (final_close - base_open) < 0 else "중립"

        final_close = float(df.at[end, "close"])
        final_ret = (final_close / base_open - 1) * 100
        max_runup, max_drawdn = (win_high / base_open - 1) * 100, (win_low / base_open - 1) * 100

        out.append({
            "신호시간": df.at[i, "time"],
            "기준시가": round(base_open, 6),
            "RSI(13)": round(float(df.at[i, "RSI13"]), 2),
            "성공기준(%)": thr_pct,
            "결과": result,
            "최종수익률(%)": f"{final_ret:.1f}%",
            "최대상승(%)": f"{max_runup:.1f}%",
            "최대하락(%)": f"{max_drawdn:.1f}%"
        })

    return pd.DataFrame(out)

# -----------------------------
# 실행
# -----------------------------
try:
    market_code = MARKETS[market_label]
    df = fetch_upbit(market_code, tf_label, count)
    df = add_indicators(df)
    res = simulate(df, rsi_side, lookahead, threshold_pct, bb_cond)

    # 메트릭
    total = len(res)
    wins = (res["결과"] == "성공").sum()
    fails = (res["결과"] == "실패").sum()
    neuts = (res["결과"] == "중립").sum()
    winrate = (wins / total * 100) if total else 0.0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("신호 수", total)
    m2.metric("성공", wins)
    m3.metric("실패", fails)
    m4.metric("중립", neuts)
    m5.metric("승률", f"{winrate:.1f}%")

    # 차트
    fig = go.Figure()

    # 가격 + 볼린저밴드
    fig.add_trace(go.Candlestick(x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="가격"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_upper"], mode="lines", line=dict(color="red", width=1), name="BB Upper"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_lower"], mode="lines", line=dict(color="blue", width=1), name="BB Lower"))

    # RSI
    fig.add_trace(go.Scatter(x=df["time"], y=df["RSI13"], mode="lines", line=dict(color="purple"), name="RSI(13)", yaxis="y2"))

    fig.update_layout(
        title=f"{market_label} · {tf_label} · RSI+BB 시뮬레이션",
        xaxis=dict(domain=[0,1]),
        yaxis=dict(title="가격"),
        yaxis2=dict(title="RSI", overlaying="y", side="right", range=[0,100]),
        xaxis_rangeslider_visible=False,
        height=700
    )

    # 신호 마커
    if total > 0:
        sig_times = res["신호시간"].tolist()
        df_sig = df[df["time"].isin(sig_times)]
        for t, r in zip(sig_times, res["결과"]):
            price = float(df[df["time"] == t]["open"])
            color = "green" if r == "성공" else "red" if r == "실패" else "orange"
            symbol = "triangle-up" if r == "성공" else "triangle-down" if r == "실패" else "circle"
            fig.add_trace(go.Scatter(
                x=[t], y=[price], mode="markers",
                marker=dict(size=10, color=color, symbol=symbol, line=dict(width=1, color="black")),
                name=f"신호({r})"
            ))

    st.plotly_chart(fig, use_container_width=True)

    # 결과 표
    st.subheader("신호 결과 (최신 순)")
    if total > 0:
        styled = res.style.applymap(lambda v: "color:red;" if v=="성공" else "color:blue;" if v=="실패" else "color:green;", subset=["결과"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("조건을 만족하는 신호가 없습니다.")

except Exception as e:
    st.error(f"오류: {e}")
