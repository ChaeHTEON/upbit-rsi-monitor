import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
import ta
from datetime import datetime, timedelta

# -----------------------------
# 페이지 설정
# -----------------------------
st.set_page_config(page_title="Upbit RSI(13)+BB 시뮬레이터", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 0.8rem; padding-bottom: 0.8rem; max-width: 1200px;}
  .section-title {font-weight: bold; font-size: 1.2rem; margin-top: 1rem;}
  .bb-up {color: red;}
  .bb-down {color: blue;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Upbit RSI(13) + Bollinger Band 시뮬레이터")

# -----------------------------
# 유틸
# -----------------------------
def fetch_upbit(market_code: str, tf: str, start: datetime, end: datetime) -> pd.DataFrame:
    """지정된 기간의 캔들 데이터를 분할 로딩"""
    delta = (end - start).days
    all_data = []
    url_base = "https://api.upbit.com/v1/candles/"
    if "minutes/" in tf:
        unit = tf.split("/")[1]
        url = f"{url_base}minutes/{unit}"
    else:
        url = f"{url_base}{tf}"

    # 최대 200개 캔들씩만 조회 가능하므로 분할 요청
    total_chunks = max(1, delta // 2)
    cancel_btn = st.button("⏹ 로딩 취소", key="cancel")
    progress = st.progress(0, text="데이터 로딩 중...")

    cur_end = end
    for i in range(total_chunks):
        if cancel_btn:
            st.warning("로딩 취소됨")
            break
        params = {"market": market_code, "to": cur_end.strftime("%Y-%m-%d %H:%M:%S"), "count": 200}
        r = requests.get(url, params=params, headers={"Accept": "application/json"})
        if r.status_code != 200:
            raise RuntimeError(f"Upbit API 오류: {r.text}")
        data = r.json()
        if not data:
            break
        all_data.extend(data)
        cur_end = datetime.strptime(data[-1]["candle_date_time_kst"], "%Y-%m-%dT%H:%M:%S") - timedelta(minutes=1)
        progress.progress((i+1)/total_chunks, text=f"{i+1}/{total_chunks} 로딩 중...")

    progress.empty()

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
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

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=30, window_dev=2)
    out["BB_up"] = bb.bollinger_hband()
    out["BB_dn"] = bb.bollinger_lband()
    return out

def simulate(df, rsi_cond, bb_cond, lookahead, thr_pct, dedup=False):
    out = []
    thr = thr_pct / 100
    n = len(df)

    sig_idx = []
    if rsi_cond == "RSI ≤ 30":
        sig_idx = df.index[df["RSI13"] <= 30].tolist()
    elif rsi_cond == "RSI ≥ 70":
        sig_idx = df.index[df["RSI13"] >= 70].tolist()

    if bb_cond == "BB 상향돌파":
        bb_idx = df.index[df["close"] > df["BB_up"]].tolist()
        sig_idx = list(set(sig_idx) & set(bb_idx))
    elif bb_cond == "BB 하향돌파":
        bb_idx = df.index[df["close"] < df["BB_dn"]].tolist()
        sig_idx = list(set(sig_idx) & set(bb_idx))

    last_time = None
    for i in sig_idx:
        if dedup and last_time and (df.at[i, "time"] - last_time).seconds < 60:
            continue
        end = i + lookahead
        if end >= n:
            continue
        base_open = df.at[i, "open"]
        final_close = df.at[end, "close"]
        ret = (final_close/base_open - 1)*100

        if ret >= thr_pct:
            result = "성공"
        elif ret < 0:
            result = "실패"
        else:
            result = "중립"

        out.append({
            "신호시간": df.at[i, "time"],
            "기준시가": f"{int(base_open):,}",
            "RSI(13)": round(df.at[i, "RSI13"], 1),
            "성공기준(%)": f"{thr_pct:.1f}%",
            "결과": result,
            "최종수익률(%)": f"{ret:.1f}%",
        })
        last_time = df.at[i, "time"]

    return pd.DataFrame(out)

# -----------------------------
# UI
# -----------------------------
st.markdown('<p class="section-title">① 기본 설정</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("시작일", datetime.now()-timedelta(days=7))
with col2:
    end_date = st.date_input("종료일", datetime.now())

market = st.text_input("종목 코드 (예: KRW-BTC)", "KRW-BTC")
tf = st.selectbox("봉 종류", ["minutes/1", "minutes/5", "minutes/15", "minutes/60", "days"])

st.markdown('<p class="section-title">② 조건 설정</p>', unsafe_allow_html=True)

lookahead = st.slider("측정 캔들 수", 1, 60, 10)
thr_pct = st.slider("성공/실패 기준값 (%)", 0.1, 3.0, 1.0, step=0.1)

rsi_cond = st.selectbox("RSI 조건", ["RSI ≤ 30", "RSI ≥ 70"])
bb_cond = st.selectbox("볼린저밴드 조건", ["없음", "BB 상향돌파", "BB 하향돌파"])

dedup = st.checkbox("중복 신호 제외", value=False)

# -----------------------------
# 실행
# -----------------------------
try:
    df = fetch_upbit(market, tf, datetime.combine(start_date, datetime.min.time()), datetime.combine(end_date, datetime.max.time()))
    if df.empty:
        st.warning("데이터 없음")
    else:
        df = add_indicators(df)
        res = simulate(df, rsi_cond, bb_cond, lookahead, thr_pct, dedup)

        st.markdown('<p class="section-title">③ 기준 요약</p>', unsafe_allow_html=True)
        st.write(f"- RSI 조건: {rsi_cond}, BB 조건: {bb_cond}, 기준: {thr_pct:.1f}%")
        st.write(f"- 측정: {lookahead}봉")

        st.markdown('<p class="section-title">④ 시뮬레이션 결과</p>', unsafe_allow_html=True)

        total = len(res)
        wins = len(res[res["결과"]=="성공"])
        fails = len(res[res["결과"]=="실패"])
        neuts = len(res[res["결과"]=="중립"])
        winrate = (wins+neuts)/total*100 if total else 0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("신호 수", total)
        m2.metric("성공", wins)
        m3.metric("실패", fails)
        m4.metric("중립", neuts)
        m5.metric("승률", f"{winrate:.1f}%")

        # 차트
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df["time"], open=df["open"], high=df["high"],
                                     low=df["low"], close=df["close"], name="가격"))

        fig.add_trace(go.Scatter(x=df["time"], y=df["RSI13"], mode="lines", name="RSI(13)", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=df["time"], y=df["BB_up"], mode="lines", name="BB 상단", line=dict(color="red", dash="dot")))
        fig.add_trace(go.Scatter(x=df["time"], y=df["BB_dn"], mode="lines", name="BB 하단", line=dict(color="blue", dash="dot")))

        for label, color, symbol in [("성공","red","triangle-up"),("실패","blue","triangle-down"),("중립","green","circle")]:
            sub = res[res["결과"]==label]
            if not sub.empty:
                fig.add_trace(go.Scatter(x=sub["신호시간"], y=[float(s.replace(",","")) for s in sub["기준시가"]],
                                         mode="markers", name=f"신호({label})",
                                         marker=dict(size=9,color=color,symbol=symbol,line=dict(width=1,color="black"))))

        fig.update_layout(xaxis_rangeslider_visible=False, height=700)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(res, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"오류: {e}")
