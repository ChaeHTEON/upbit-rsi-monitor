import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
import ta  # pandas_ta 대신 ta 사용 (Cloud 호환성 좋음)

# -----------------------------
# 페이지/스타일: 모바일 가독성
# -----------------------------
st.set_page_config(page_title="Upbit RSI(13) 시뮬레이터", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 0.8rem; padding-bottom: 0.8rem; max-width: 1080px;}
  @media (max-width: 600px) {
    h1, h2, h3 {font-size: 1.1rem;}
    .stMetric {text-align:center;}
  }
</style>
""", unsafe_allow_html=True)

st.title("📈 Upbit RSI(13) 시뮬레이터")

# -----------------------------
# 옵션 (요구사항 반영)
# -----------------------------
MARKETS = {
    "비트 (BTC)": "KRW-BTC",
    "리플 (XRP)": "KRW-XRP",
    "도지 (DOGE)": "KRW-DOGE",
    "이더리움 (ETH)": "KRW-ETH",
    "솔라나 (SOL)": "KRW-SOL",
}
TF_MAP = {
    "1분": "minutes/1",
    "3분": "minutes/3",
    "5분": "minutes/5",
    "10분": "minutes/10",
    "15분": "minutes/15",
    "30분": "minutes/30",
    "60분": "minutes/60",
    "4시간": "minutes/240",
    "일봉": "days",
}

c1, c2 = st.columns(2)
with c1:
    market_label = st.selectbox("종목 선택", list(MARKETS.keys()), index=0)
with c2:
    tf_label = st.selectbox("봉 종류 선택", list(TF_MAP.keys()), index=0)

c3, c4 = st.columns(2)
with c3:
    count = st.slider("캔들 개수", 80, 200, 180, step=10)
with c4:
    lookahead = st.slider("측정 캔들 수 (기준 이후 N봉)", 1, 60, 10)

c5, c6 = st.columns(2)
with c5:
    threshold_pct = st.slider(
        "성공/실패 기준 값(%)",
        min_value=0.1,
        max_value=3.0,
        value=1.0,
        step=0.1
    )
with c6:
    rsi_side = st.selectbox("RSI 조건", ["RSI ≤ 30 (급락30)", "RSI ≥ 70 (급등70)"], index=0)

# 안내
st.caption(
    "- 기준 캔들: RSI(13) 조건(급락30 또는 급등70)을 만족한 시점의 **시가**를 기준가격으로 사용합니다.\n"
    f"- 이후 N봉 내에 **+{threshold_pct:.1f}% 이상 고가 도달 → 성공**, **-{threshold_pct:.1f}% 이하 저가 도달 → 실패**, 그 외는 **중립**으로 분류합니다.\n"
    "- 단, 핵심 조건이 모두 미충족 시 최종 수익률 < 0 → 실패, ≥ 0 → 중립으로 판정합니다.\n"
    "- 추가로 기준 시가 대비 **최대상승(%) / 최대하락(%)**과, (i+N)번째 **종가 기준 최종수익률(%)**을 제공합니다."
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
    if r.status_code != 200:
        raise RuntimeError(f"Upbit API 오류: {r.text}")
    data = r.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(data["error"]["message"])

    df = pd.DataFrame(data)
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
# RSI(13) 계산
# -----------------------------
def add_rsi(df: pd.DataFrame, window: int = 13) -> pd.DataFrame:
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=window).rsi()
    return out

# -----------------------------
# 시뮬레이션
# -----------------------------
def simulate(df: pd.DataFrame, side: str, lookahead: int, thr_pct: float) -> pd.DataFrame:
    out = []
    n = len(df)
    thr = thr_pct / 100.0

    if "≤" in side:
        sig_idx = df.index[df["RSI13"] <= 30].tolist()
    else:
        sig_idx = df.index[df["RSI13"] >= 70].tolist()

    for i in sig_idx:
        end = i + lookahead
        if end >= n:
            continue

        base_open = float(df.at[i, "open"])
        win = df.loc[i+1:end, :]
        win_high = float(win["high"].max())
        win_low  = float(win["low"].min())

        target_up = base_open * (1 + thr)
        target_dn = base_open * (1 - thr)

        hit_up = (win_high >= target_up)
        hit_dn = (win_low  <= target_dn)

        final_close = float(df.at[end, "close"])
        final_ret = (final_close / base_open - 1.0) * 100.0
        max_runup  = (win_high / base_open - 1.0) * 100.0
        max_drawdn = (win_low  / base_open - 1.0) * 100.0

        # 성공/실패/중립 판정
        if hit_up and not hit_dn:
            result = "성공"
        elif hit_dn and not hit_up:
            result = "실패"
        elif hit_up and hit_dn:
            result = "중립"
        else:
            # 핵심 조건 불충족 → 최종 수익률 기준
            if final_ret < 0:
                result = "실패"
            else:
                result = "중립"

        out.append({
            "신호시간": df.at[i, "time"],
            "종료시간": df.at[end, "time"],
            "기준시가": round(base_open, 8),
            "RSI(13)": round(float(df.at[i, "RSI13"]), 2) if pd.notna(df.at[i, "RSI13"]) else None,
            "측정캔들수": lookahead,
            "성공기준(%)": thr_pct,
            "결과": result,
            "최종수익률(%)": round(final_ret, 1),
            "최대상승(%)": round(max_runup, 1),
            "최대하락(%)": round(max_drawdn, 1),
        })

    return pd.DataFrame(out)

# -----------------------------
# 실행
# -----------------------------
try:
    market_code = MARKETS[market_label]
    df = fetch_upbit(market_code, tf_label, count)
    df = add_rsi(df, window=13)
    res = simulate(df, rsi_side, lookahead, threshold_pct)

    # 요약 메트릭
    total = int(res.shape[0])
    wins  = int((res["결과"] == "성공").sum()) if total else 0
    fails = int((res["결과"] == "실패").sum()) if total else 0
    neuts = int((res["결과"] == "중립").sum()) if total else 0
    winrate = (wins / total * 100.0) if total else 0.0
    avg_final = float(res["최종수익률(%)"].mean()) if total else 0.0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("신호 수", f"{total}")
    m2.metric("성공", f"{wins}")
    m3.metric("실패", f"{fails}")
    m4.metric("중립", f"{neuts}")
    m5.metric("승률", f"{winrate:.2f}%")
    st.caption(f"참고: (i+{lookahead}) 종가 기준 평균 수익률 = {avg_final:.1f}%")

    # 가격 차트 + 신호 마커
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="가격"
    ))

    if total > 0:
        merged = pd.merge(
            res[["신호시간", "결과"]],
            df[["time", "open"]],
            left_on="신호시간", right_on="time", how="left"
        )
        for label, color, symbol in [("성공", "red", "triangle-up"),
                                     ("실패", "blue", "triangle-down"),
                                     ("중립", "green", "circle")]:
            sub = merged[merged["결과"] == label]
            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub["time"], y=sub["open"], mode="markers",
                    name=f"신호 ({label})",
                    marker=dict(size=9, color=color, symbol=symbol, line=dict(width=1, color="black")),
                    hovertemplate="신호시간=%{x}<br>기준시가=%{y}<extra></extra>"
                ))

    fig.update_layout(
        title=f"{market_label} · {tf_label} · RSI(13) 시뮬레이션",
        xaxis_title="시간", yaxis_title="가격",
        xaxis_rangeslider_visible=False, height=600
    )
    st.plotly_chart(fig, use_container_width=True)

    # RSI 차트 항상 표시
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df["time"], y=df["RSI13"], mode="lines", name="RSI(13)"))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="blue")
    fig_rsi.update_layout(height=280, xaxis_title="시간", yaxis_title="RSI(13)")
    st.plotly_chart(fig_rsi, use_container_width=True)

    # 결과 표
    st.subheader("신호 결과 (최신 순)")
    if total > 0:
        table = res.sort_values("신호시간", ascending=False).reset_index(drop=True).copy()
        pct_cols = ["최종수익률(%)", "최대상승(%)", "최대하락(%)"]

        def color_result(series):
            return [
                "color: red" if v == "성공" else
                "color: blue" if v == "실패" else
                "color: green"
                for v in series
            ]

        styled = (
            table.style
            .format({c: "{:.1f}%".format for c in pct_cols})
            .map(color_result, subset=["결과"])
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("현재 조건을 만족하는 신호가 없습니다. 옵션을 조절해 보세요.")

    # 수동 새로고침 버튼
    if st.button("🔄 새로고침"):
        st.rerun()

except Exception as e:
    st.error(f"오류: {e}")
