import streamlit as st
import pandas as pd
import numpy as np
import requests
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
  .fail {color:blue; font-weight:600;}
  .neutral {color:green; font-weight:600;}
  .section-title {font-size:1.05rem; font-weight:700; margin: 0.6rem 0 0.2rem;}
  .hint {color:#6b7280;}
  .small-note {font-size:0.9rem; color:#6b7280;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Upbit RSI(13) + Bollinger Band 시뮬레이터")

# -----------------------------
# 업비트 마켓 로드 (KRW-만)
# -----------------------------
@st.cache_data(ttl=3600)
def get_upbit_krw_markets():
    url = "https://api.upbit.com/v1/market/all"
    r = requests.get(url, params={"isDetails": "false"})
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
default_idx = 0
for i, (_, code) in enumerate(MARKET_LIST):
    if code == "KRW-BTC":
        default_idx = i
        break

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
# 최상단 카테고리: 신호 중복 처리
# -----------------------------
dup_mode = st.radio(
    "신호 중복 처리",
    ["중복 포함 (연속 신호 모두)", "중복 제거 (연속 동일 결과 1개)"],
    horizontal=True,
)

# -----------------------------
# 섹션: 기본 설정
# -----------------------------
st.markdown('<div class="section-title">① 기본 설정</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    market_label, market_code = st.selectbox(
        "종목 선택",
        MARKET_LIST,
        index=default_idx,
        format_func=lambda x: x[0]
    )
with c2:
    tf_label = st.selectbox("봉 종류 선택", list(TF_MAP.keys()), index=2)
with c3:
    default_start = datetime.today() - timedelta(days=1)
    start_date = st.date_input("시작 날짜", value=default_start)
    end_date = st.date_input("종료 날짜", value=datetime.today())

# -----------------------------
# 섹션: 조건 설정
# -----------------------------
st.markdown('<div class="section-title">② 조건 설정</div>', unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
with c4:
    lookahead = st.slider("측정 캔들 수 (기준 이후 N봉)", 1, 60, 10)
with c5:
    threshold_pct = st.slider("성공/실패 기준 값(%)", 0.1, 3.0, 1.0, step=0.1)
with c6:
    rsi_side = st.selectbox("RSI 조건", ["RSI ≤ 30 (급락)", "RSI ≥ 70 (급등)"], index=0)

# 볼린저밴드 조건 UI 폭 정렬
c7, _, _ = st.columns(3)
with c7:
    bb_cond = st.selectbox(
        "볼린저밴드 조건",
        [
            "없음",
            "하한선 하향돌파",
            "하한선 상향돌파",
            "상한선 하향돌파",
            "상한선 상향돌파",
            "하한선 중앙돌파",
            "상한선 중앙돌파",
        ],
        index=0,
    )

interval_key, minutes_per_bar = TF_MAP[tf_label]
total_minutes = lookahead * minutes_per_bar
st.caption(f"측정 범위: **{lookahead} ({total_minutes}분)**  · 봉 종류: **{tf_label}**")

# -----------------------------
# 데이터 수집
# -----------------------------
def estimate_calls(start_dt: datetime, end_dt: datetime, minutes_per_bar: int) -> int:
    mins = max(1, int((end_dt - start_dt).total_seconds() // 60))
    bars = max(1, mins // minutes_per_bar)
    calls = bars // 200 + 1
    return min(calls, 5000)

def fetch_upbit_paged(market_code: str, interval_key: str, start_dt: datetime, end_dt: datetime,
                      minutes_per_bar: int) -> pd.DataFrame:
    if "minutes/" in interval_key:
        unit = interval_key.split("/")[1]
        url = f"https://api.upbit.com/v1/candles/minutes/{unit}"
    else:
        url = f"https://api.upbit.com/v1/candles/{interval_key}"
    all_data = []
    to_time = end_dt
    calls_est = estimate_calls(start_dt, end_dt, minutes_per_bar)
    progress = st.progress(0.0)
    done = 0
    while True:
        params = {"market": market_code, "count": 200, "to": to_time.strftime("%Y-%m-%d %H:%M:%S")}
        r = requests.get(url, params=params, headers={"Accept": "application/json"})
        if r.status_code != 200:
            raise RuntimeError(f"Upbit API 오류: {r.text}")
        batch = r.json()
        if not batch:
            break
        all_data.extend(batch)
        last_ts = pd.to_datetime(batch[-1]["candle_date_time_kst"])
        if last_ts <= start_dt:
            break
        to_time = last_ts - timedelta(seconds=1)
        done += 1
        progress.progress(min(1.0, done / max(1, calls_est)))
        if done > 5000:
            break
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
    return df[df["time"].between(start_dt, end_dt)]

# -----------------------------
# 지표 추가
# -----------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=30, window_dev=2)
    out["BB_up"] = bb.bollinger_hband()
    out["BB_low"] = bb.bollinger_lband()
    out["BB_mid"] = bb.bollinger_mavg()
    return out

# -----------------------------
# 시뮬레이션 (고속/최적화)
# -----------------------------
def simulate(df: pd.DataFrame, rsi_side: str, lookahead: int, thr_pct: float,
             bb_cond: str, dedup_mode: str) -> pd.DataFrame:
    res = []
    thr = float(thr_pct)
    n = len(df)

    rsi = df["RSI13"].to_numpy()
    closes = df["close"].to_numpy()
    lows = df["low"].to_numpy()
    times = df["time"].to_numpy()
    bb_up = df["BB_up"].to_numpy()
    bb_low = df["BB_low"].to_numpy()
    bb_mid = df["BB_mid"].to_numpy()

    if "≤" in rsi_side:
        sig_idx = np.where(~np.isnan(rsi) & (rsi <= 30))[0]
    else:
        sig_idx = np.where(~np.isnan(rsi) & (rsi >= 70))[0]

    for i in sig_idx:
        end = i + lookahead
        if end >= n:
            continue

        if bb_cond != "없음":
            px = closes[i]
            up, lo, mid = bb_up[i], bb_low[i], bb_mid[i]
            ok = True
            if bb_cond == "하한선 하향돌파": ok = (not np.isnan(lo)) and (px < lo)
            elif bb_cond == "하한선 상향돌파": ok = (not np.isnan(lo)) and (px > lo)
            elif bb_cond == "상한선 하향돌파": ok = (not np.isnan(up)) and (px < up)
            elif bb_cond == "상한선 상향돌파": ok = (not np.isnan(up)) and (px > up)
            elif bb_cond == "하한선 중앙돌파": ok = (not np.isnan(lo) and not np.isnan(mid)) and (lo < px < mid)
            elif bb_cond == "상한선 중앙돌파": ok = (not np.isnan(up) and not np.isnan(mid)) and (mid < px < up)
            if not ok:
                continue

        base_price = lows[i]
        future_closes = closes[i + 1:end + 1]
        future_times = times[i + 1:end + 1]
        if future_closes.size == 0:
            continue

        target_up = base_price * (1 + thr / 100)
        target_down = base_price * (1 - thr / 100)

        hit_up_idx = np.where(future_closes >= target_up)[0]
        hit_down_idx = np.where(future_closes <= target_down)[0]

        result = "중립"
        reach_time = None

        if hit_up_idx.size > 0 and hit_down_idx.size > 0:
            if hit_up_idx[0] < hit_down_idx[0]:
                result = "성공"
                reach_time = pd.to_datetime(future_times[hit_up_idx[0]]).strftime("%H:%M")
            else:
                result = "실패"
        elif hit_up_idx.size > 0:
            result = "성공"
            reach_time = pd.to_datetime(future_times[hit_up_idx[0]]).strftime("%H:%M")
        elif hit_down_idx.size > 0:
            result = "실패"
        else:
            final_price = future_closes[-1]
            result = "중립" if final_price > base_price else "실패"

        final_ret = (future_closes[-1] / base_price - 1.0) * 100.0
        min_ret = (future_closes.min() / base_price - 1.0) * 100.0
        max_ret = (future_closes.max() / base_price - 1.0) * 100.0

        res.append({
            "신호시간": times[i],
            "기준시가": int(round(base_price)),
            "RSI(13)": round(rsi[i], 1) if not np.isnan(rsi[i]) else None,
            "성공기준(%)": round(thr, 1),
            "결과": result,
            "도달시간": reach_time if result == "성공" else None,
            "최종수익률(%)": round(final_ret, 1),
            "최저수익률(%)": round(min_ret, 1),
            "최고수익률(%)": round(max_ret, 1),
        })

    out = pd.DataFrame(res)
    if not out.empty and dedup_mode.startswith("중복 제거"):
        out = out.loc[out["결과"].shift() != out["결과"]]
    return out

# -----------------------------
# 실행
# -----------------------------
try:
    if start_date > end_date:
        st.error("시작 날짜가 종료 날짜보다 이후입니다.")
        st.stop()

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())
    df = fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar)
    if df.empty:
        st.error("데이터가 없습니다.")
        st.stop()

    df = add_indicators(df)

    res_all = simulate(df, rsi_side, lookahead, threshold_pct, bb_cond, "중복 포함 (연속 신호 모두)")
    res_dedup = simulate(df, rsi_side, lookahead, threshold_pct, bb_cond, "중복 제거 (연속 동일 결과 1개)")

    # -----------------------------
    # 요약 & 차트
    # -----------------------------
    st.markdown('<div class="section-title">③ 요약 & 차트</div>', unsafe_allow_html=True)

    def _summarize(df_in: pd.DataFrame):
        if df_in is None or df_in.empty or "결과" not in df_in.columns:
            return 0, 0, 0, 0, 0.0, 0.0, 0.0
        total = len(df_in)
        succ = int((df_in["결과"] == "성공").sum())
        fail = int((df_in["결과"] == "실패").sum())
        neu = int((df_in["결과"] == "중립").sum())
        win = (succ / total * 100.0) if total > 0 else 0.0
        range_sum = float((df_in["최고수익률(%)"] - df_in["최저수익률(%)"]).sum()) if total > 0 else 0.0
        final_succ = float(df_in.loc[df_in["결과"] == "성공", "최종수익률(%)"].sum()) if total > 0 else 0.0
        final_fail = float(df_in.loc[df_in["결과"] == "실패", "최종수익률(%)"].sum()) if total > 0 else 0.0
        return total, succ, fail, neu, win, range_sum, final_succ, final_fail

    for label, data in [("중복 포함 (연속 신호 모두)", res_all),
                        ("중복 제거 (연속 동일 결과 1개)", res_dedup)]:
        total, succ, fail, neu, win, range_sum, final_succ, final_fail = _summarize(data)
        st.markdown(f"**{label}**")
        if total == 0:
            st.info("해당 조건에 맞는 신호가 없습니다.")
            st.markdown("---")
            continue
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("신호 수", f"{total}")
        c2.metric("성공", f"{succ}")
        c3.metric("실패", f"{fail}")
        c4.metric("중립", f"{neu}")
        c5.metric("승률", f"{win:.1f}%")
        c6.metric("총 변동폭 합(%)", f"{range_sum:.1f}%")
        final_sum = final_succ + final_fail
        c7.metric("최종수익률 합계", f"{final_sum:+.1f}%")
        st.markdown("---")

    res = res_all if dup_mode.startswith("중복 포함") else res_dedup

    # -----------------------------
    # 차트
    # -----------------------------
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="가격",
        increasing_line_color="#E63946", decreasing_line_color="#457B9D",
        line=dict(width=1.2)
    ))

    # BB 점선 처리
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_up"], mode="lines",
                             line=dict(color="#FFB703", width=1.2, dash="dot"),
                             name="BB 상단"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_low"], mode="lines",
                             line=dict(color="#219EBC", width=1.2, dash="dot"),
                             name="BB 하단"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_mid"], mode="lines",
                             line=dict(color="#8D99AE", width=1.2, dash="dot"),
                             name="BB 중앙"))

    # 신호 마커
    if not res.empty:
        for _, row in res.iterrows():
            color = {"성공": "red", "실패": "blue", "중립": "green"}[row["결과"]]
            fig.add_trace(go.Scatter(
                x=[row["신호시간"]], y=[row["기준시가"]],
                mode="markers", marker=dict(color=color, size=9, symbol="circle"),
                name=f"신호({row['결과']})"
            ))

    # RSI 보라색, 얇게
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["RSI13"], mode="lines",
        line=dict(color="purple", width=0.8), opacity=0.85,
        name="RSI(13)", yaxis="y2"
    ))

    # RSI 기준선
    fig.add_hline(y=70, line_dash="dash", line_color="#E63946", line_width=1.0,
                  annotation_text="RSI 70", annotation_position="top left", yref="y2")
    fig.add_hline(y=30, line_dash="dash", line_color="#457B9D", line_width=1.0,
                  annotation_text="RSI 30", annotation_position="bottom left", yref="y2")

    fig.update_layout(
        title=f"{market_label.split(' — ')[0]} · {tf_label} · RSI(13) + BB 시뮬레이션",
        xaxis_rangeslider_visible=False,
        height=620,
        autosize=False,
        legend_orientation="h", legend_y=1.05,
        margin=dict(l=60, r=40, t=60, b=40),
        yaxis=dict(title="가격"),
        yaxis2=dict(overlaying="y", side="right", showgrid=False, title="RSI(13)", range=[0, 100])
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # 신호 결과 표
    # -----------------------------
    st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)
    if not res.empty:
        tbl = res.sort_values("신호시간", ascending=False).reset_index(drop=True).copy()

        tbl["기준시가"] = tbl["기준시가"].map(lambda v: f"{int(v):,}")
        if "RSI(13)" in tbl.columns:
            tbl["RSI(13)"] = tbl["RSI(13)"].map(lambda v: f"{float(v):.1f}" if pd.notna(v) else "")
        if "성공기준(%)" in tbl.columns:
            tbl["성공기준(%)"] = tbl["성공기준(%)"].map(lambda v: f"{float(v):.1f}%")
        if "최종수익률(%)" in tbl.columns:
            tbl["최종수익률(%)"] = tbl["최종수익률(%)"].map(lambda v: f"{float(v):.1f}%")
        if "최저수익률(%)" in tbl.columns:
            tbl["최저수익률(%)"] = tbl["최저수익률(%)"].map(lambda v: f"{float(v):.1f}%")
        if "최고수익률(%)" in tbl.columns:
            tbl["최고수익률(%)"] = tbl["최고수익률(%)"].map(lambda v: f"{float(v):.1f}%")

        if "도달시간" in tbl.columns:
            tbl["도달시간"] = tbl["도달시간"].fillna("-").astype(str)

        def color_result(val):
            if val == "성공":
                return "color:red; font-weight:600;"
            if val == "실패":
                return "color:blue; font-weight:600;"
            return "color:green; font-weight:600;"

        styled = tbl.style.applymap(color_result, subset=["결과"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("조건을 만족하는 신호가 없습니다.")

except Exception as e:
    st.error(f"오류: {e}")
