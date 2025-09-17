import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
import numpy as np
import ta

# -----------------------------
# UI 기본 설정 (모바일 친화)
# -----------------------------
st.set_page_config(page_title="Upbit RSI 시뮬레이터", layout="wide")
st.markdown(
    """
    <style>
      /* 모바일 가독성 향상 */
      .block-container {padding-top: 0.8rem; padding-bottom: 0.8rem; max-width: 1100px;}
      .stMetric {text-align:center;}
      @media (max-width: 600px) {
        h1, h2, h3 {font-size: 1.1rem;}
        .stSlider > div > div {padding: 0.15rem 0;}
      }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📈 Upbit RSI(13) 시뮬레이터")

# -----------------------------
# 옵션 영역
# -----------------------------
colA, colB = st.columns(2)
with colA:
    market = st.text_input("종목 선택 (예: KRW-BTC, KRW-ETH)", "KRW-BTC")
with colB:
    tf_label = st.selectbox(
        "봉 종류 선택",
        ["1분", "3분", "5분", "10분", "15분", "30분", "60분", "4시간", "일봉"],
        index=0
    )

col1, col2 = st.columns(2)
with col1:
    count = st.slider("캔들 개수 (최대 200)", 80, 200, 180, step=10)
with col2:
    lookahead = st.slider("측정 캔들 수 (기준 이후 N봉)", 1, 60, 10)

col3, col4 = st.columns(2)
with col3:
    threshold_pct = st.slider("성공/실패 기준 값(%)", 1, 100, 3)
with col4:
    condition_side = st.selectbox("RSI 조건", ["RSI ≤ 30 (급락30)", "RSI ≥ 70 (급등70)"], index=0)

st.caption(
    "- 기준 캔들은 RSI(13) 조건을 만족한 캔들입니다. "
    "기준 캔들 **시가**를 기준 가격으로 삼아, 이후 N봉 구간에서 "
    f"**+{threshold_pct}% 이상 시 고점 도달 → 성공**, **-{threshold_pct}% 이하 시 저점 도달 → 실패**, "
    "둘 다 없으면 중립으로 분류합니다."
)

# -----------------------------
# 보조 함수
# -----------------------------
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

def fetch_upbit(market: str, tf_label: str, count: int) -> pd.DataFrame:
    """업비트 캔들 데이터 수집 (최대 200개)"""
    interval = TF_MAP[tf_label]
    if "minutes/" in interval:
        unit = interval.split("/")[1]
        url = f"https://api.upbit.com/v1/candles/minutes/{unit}"
    else:
        url = f"https://api.upbit.com/v1/candles/{interval}"

    params = {"market": market, "count": count}
    res = requests.get(url, params=params, headers={"Accept": "application/json"})
    if res.status_code != 200:
        raise RuntimeError(f"Upbit API 오류: {res.text}")
    data = res.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(data["error"]["message"])

    df = pd.DataFrame(data)
    df.rename(
        columns={
            "candle_date_time_kst": "time",
            "opening_price": "open",
            "high_price": "high",
            "low_price": "low",
            "trade_price": "close",
            "candle_acc_trade_volume": "volume",
        },
        inplace=True,
    )
    # 업비트는 최신→과거 순으로 반환하므로 시간 오름차순으로 정렬
    df["time"] = pd.to_datetime(df["time"])
    df = df[["time", "open", "high", "low", "close", "volume"]].sort_values("time").reset_index(drop=True)
    return df

def compute_rsi(df: pd.DataFrame, window: int = 13) -> pd.DataFrame:
    rsi = ta.momentum.RSIIndicator(close=df["close"], window=window).rsi()
    df = df.copy()
    df["RSI13"] = rsi
    return df

def simulate_signals(df: pd.DataFrame, side: str, lookahead: int, thr_pct: float):
    """
    시뮬레이션 로직:
    - 신호: RSI(13) 조건 만족한 캔들 i
    - 기준가: i번째 캔들의 '시가'
    - 윈도우: (i+1) ~ (i+lookahead)
    - 규칙: 윈도우 내 고가가 기준가*(1+thr) 이상 도달 → 성공
            윈도우 내 저가가 기준가*(1-thr) 이하 도달 → 실패
            둘 다 미발생 → 중립
    - 추가로 i+lookahead 종가 기준의 최종 수익률도 계산(참고)
    """
    df = df.copy()
    n = len(df)
    outcomes = []  # dict list

    # 신호 인덱스 후보
    if "≤" in side:  # RSI ≤ 30
        idx = df.index[df["RSI13"] <= 30].tolist()
    else:  # RSI ≥ 70
        idx = df.index[df["RSI13"] >= 70].tolist()

    thr = thr_pct / 100.0

    for i in idx:
        end = i + lookahead
        if end >= n:  # 충분한 미래 캔들이 없으면 스킵
            continue

        base_open = float(df.at[i, "open"])
        win_high = float(df.loc[i+1:end, "high"].max())
        win_low  = float(df.loc[i+1:end, "low"].min())
        target_up   = base_open * (1.0 + thr)
        target_down = base_open * (1.0 - thr)

        # 도달 여부 판단 (성공/실패 동시발생 가능성 → 최초 도달 가정 어려워서 우선순위: 성공 우선/실패 우선 중 택1)
        # 실전에서는 '도달 순서'가 중요하지만, 단일 캔들 고/저가만으로 순서를 알 수 없어 규칙을 명시해야 함.
        # 여기서는 '성공 우선' 또는 '실패 우선' 토글을 둘 수도 있지만, 기본은 '둘 중 먼저 도달했다고 가정 불가 → 둘 다 충족 시 중립'으로 둠.
        hit_up = (win_high >= target_up)
        hit_dn = (win_low  <= target_down)

        if hit_up and not hit_dn:
            outcome = "성공"
        elif hit_dn and not hit_up:
            outcome = "실패"
        elif hit_up and hit_dn:
            # 도달 순서를 단일 캔들 데이터로는 특정할 수 없으므로 중립 처리(또는 '양쪽 도달'로 분리 가능)
            outcome = "중립"
        else:
            outcome = "중립"

        # 최종 측정 시점(i+lookahead)의 종가 대비 수익률(참고 지표)
        final_close = float(df.at[end, "close"])
        final_ret = (final_close / base_open - 1.0) * 100.0

        # 기준 캔들에서 측정 구간 내 최고 상승률/최저 하락률 (참고 지표)
        max_runup  = (win_high / base_open - 1.0) * 100.0
        max_drawdn = (win_low  / base_open - 1.0) * 100.0  # 음수일 것

        outcomes.append({
            "신호시간": df.at[i, "time"],
            "기준시가": base_open,
            "RSI(13)": float(df.at[i, "RSI13"]),
            "측정캔들수": lookahead,
            "성공기준(%)": thr_pct,
            "결과": outcome,
            "최종수익률(%)": round(final_ret, 3),
            "최대상승(%)": round(max_runup, 3),
            "최대하락(%)": round(max_drawdn, 3),
            "종료시간": df.at[end, "time"],
        })

    return pd.DataFrame(outcomes)

# -----------------------------
# 실행
# -----------------------------
try:
    df = fetch_upbit(market, tf_label, count)
    df = compute_rsi(df, window=13)

    # 시뮬레이션
    result = simulate_signals(df, condition_side, lookahead, threshold_pct)

    # 요약 메트릭
    total = int(result.shape[0])
    wins = int((result["결과"] == "성공").sum())
    loses = int((result["결과"] == "실패").sum())
    neutr = int((result["결과"] == "중립").sum())
    winrate = (wins / total * 100.0) if total > 0 else 0.0
    avg_final = float(result["최종수익률(%)"].mean()) if total > 0 else 0.0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("신호 수", f"{total}")
    m2.metric("성공", f"{wins}")
    m3.metric("실패", f"{loses}")
    m4.metric("중립", f"{neutr}")
    m5.metric("승률", f"{winrate:0.2f}%")

    st.caption(f"참고: 측정 시점(i+{lookahead}) 종가 기준 평균 수익률 = {avg_final:0.3f}%")

    # 차트 (가격 + 신호 마커)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="가격"
    ))

    # 신호 위치 마킹: 성공/실패/중립별 색상
    if total > 0:
        merged = pd.merge(
            result[["신호시간", "결과"]],
            df[["time", "open"]],
            left_on="신호시간", right_on="time", how="left"
        )
        for label, color, symbol in [("성공", "green", "triangle-up"),
                                     ("실패", "red", "triangle-down"),
                                     ("중립", "orange", "circle")]:
            sub = merged[merged["결과"] == label]
            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub["time"], y=sub["open"],
                    mode="markers",
                    name=f"신호 ({label})",
                    marker=dict(size=9, color=color, symbol=symbol, line=dict(width=1, color="black")),
                    hovertemplate="신호시간=%{x}<br>기준시가=%{y}<extra></extra>"
                ))

    fig.update_layout(
        title=f"{market} · {tf_label} · RSI(13) 조건 시뮬레이션",
        xaxis_title="시간", yaxis_title="가격",
        xaxis_rangeslider_visible=False, height=600
    )
    st.plotly_chart(fig, use_container_width=True)

    # RSI 라인 (보조)
    with st.expander("RSI(13) 보조지표 보기"):
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df["time"], y=df["RSI13"], mode="lines", name="RSI(13)"))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="blue")
        fig_rsi.update_layout(height=280, xaxis_title="시간", yaxis_title="RSI(13)")
        st.plotly_chart(fig_rsi, use_container_width=True)

    # 결과 테이블
    st.subheader("신호별 결과")
    if total > 0:
        st.dataframe(
            result.sort_values("신호시간", ascending=False).reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("현재 조건을 만족하는 신호가 없습니다. 캔들 개수/측정 캔들 수/기준값(%)을 조정해 보세요.")

except Exception as e:
    st.error(f"오류: {e}")
