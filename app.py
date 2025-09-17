import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
import ta
from datetime import datetime, timedelta

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
</style>
""", unsafe_allow_html=True)

st.title("📊 Upbit RSI(13) + Bollinger Band 시뮬레이터")

# -----------------------------
# 업비트 마켓 로드 (KRW-만)
# -----------------------------
@st.cache_data(ttl=3600)
def get_upbit_krw_markets():
    url = "https://api.upbit.com/v1/market/all"
    r = requests.get(url, params={"isDetails":"false"})
    r.raise_for_status()
    items = r.json()
    rows = []
    for it in items:
        if it["market"].startswith("KRW-"):
            sym = it["market"][4:]
            label = f'{it["korean_name"]} ({sym}) — {it["market"]}'
            rows.append((label, it["market"]))
    # BTC가 기본으로 위쪽에 오도록 정렬 tweak
    rows.sort(key=lambda x: (x[1] != "KRW-BTC", x[1]))
    return rows  # list of (label, market_code)

MARKET_LIST = get_upbit_krw_markets()
default_idx = 0
for i,(_, code) in enumerate(MARKET_LIST):
    if code == "KRW-BTC":
        default_idx = i; break

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
        format_func=lambda x: x[0]  # label만 보이게
    )
with c2:
    tf_label = st.selectbox("봉 종류 선택", list(TF_MAP.keys()), index=2)
with c3:
    default_start = datetime.today() - timedelta(days=7)
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
    st.caption(f"현재 설정: **{threshold_pct:.1f}%**")
with c6:
    rsi_side = st.selectbox("RSI 조건", ["RSI ≤ 30 (급락)", "RSI ≥ 70 (급등)"], index=0)

c7, c8 = st.columns(2)
with c7:
    bb_cond = st.selectbox(
        "볼린저밴드 조건",
        ["없음", "하한선 하향돌파", "하한선 상향돌파", "상한선 하향돌파", "상한선 상향돌파"],
        index=0
    )

# 측정 N → 총합 시간 표기
interval_key, minutes_per_bar = TF_MAP[tf_label]
total_minutes = lookahead * minutes_per_bar
st.caption(f"측정 범위: **{lookahead} ({total_minutes}분)**  · 봉 종류: **{tf_label}**")

# BB 선택 강조 요약 라인
if "상향" in bb_cond:
    bb_note = f'<span class="success">볼린저밴드 {bb_cond}</span>'
elif "하향" in bb_cond:
    bb_note = f'<span class="fail">볼린저밴드 {bb_cond}</span>'
else:
    bb_note = '<span class="neutral">볼린저밴드 조건 없음</span>'
st.markdown(
    f'현재 조건 요약: RSI = **{rsi_side}**, {bb_note}, 성공/실패 기준 = **{threshold_pct:.1f}%**',
    unsafe_allow_html=True
)

st.caption("※ 판정은 **최종(N번째 종가) 기준**입니다. (성공: 기준 초과, 실패: 기준 이하, 중립: 사이값)")

# -----------------------------
# 데이터 수집: 200봉 단위 자동 페이징 + 프로그레스
# -----------------------------
def estimate_calls(start_dt: datetime, end_dt: datetime, minutes_per_bar: int) -> int:
    # 대략적인 캔들 수 추정 → 200봉 단위 호출 횟수 추정
    mins = max(1, int((end_dt - start_dt).total_seconds() // 60))
    bars = max(1, mins // minutes_per_bar)
    calls = bars // 200 + 1
    return min(calls, 5000)  # 안전상한

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
        params = {
            "market": market_code,
            "count": 200,
            "to": to_time.strftime("%Y-%m-%d %H:%M:%S")
        }
        r = requests.get(url, params=params, headers={"Accept":"application/json"})
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

        # 진행상태 업데이트
        done += 1
        progress.progress(min(1.0, done / max(1, calls_est)))

        # 안전 차단
        if done > 5000:
            break

    progress.empty()

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df = df.rename(columns={
        "candle_date_time_kst":"time",
        "opening_price":"open",
        "high_price":"high",
        "low_price":"low",
        "trade_price":"close",
        "candle_acc_trade_volume":"volume"
    })
    df["time"] = pd.to_datetime(df["time"])
    df = df[["time","open","high","low","close","volume"]].sort_values("time").reset_index(drop=True)
    return df[df["time"].between(start_dt, end_dt)]

# -----------------------------
# 지표 추가 (RSI, BB)
# -----------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=30, window_dev=2)
    out["BB_up"] = bb.bollinger_hband()
    out["BB_low"] = bb.bollinger_lband()
    return out

# -----------------------------
# 시뮬레이션 (최종 종가 기준 판정)
# -----------------------------
def simulate(df: pd.DataFrame, rsi_side: str, lookahead: int, thr_pct: float, bb_cond: str,
             dedup_mode: str) -> pd.DataFrame:
    res = []
    n = len(df)
    thr = thr_pct

    # RSI 신호 인덱스
    if "≤" in rsi_side:
        sig_idx = df.index[df["RSI13"] <= 30].tolist()
    else:
        sig_idx = df.index[df["RSI13"] >= 70].tolist()

    for i in sig_idx:
        end = i + lookahead
        if end >= n:
            continue

        # BB 조건 체크
        if bb_cond != "없음":
            px = float(df.at[i, "close"])
            up = float(df.at[i, "BB_up"]) if pd.notna(df.at[i, "BB_up"]) else None
            lo = float(df.at[i, "BB_low"]) if pd.notna(df.at[i, "BB_low"]) else None

            ok = True
            if bb_cond == "하한선 하향돌파":
                ok = (lo is not None) and (px < lo)
            elif bb_cond == "하한선 상향돌파":
                ok = (lo is not None) and (px > lo)
            elif bb_cond == "상한선 하향돌파":
                ok = (up is not None) and (px < up)
            elif bb_cond == "상한선 상향돌파":
                ok = (up is not None) and (px > up)
            if not ok:
                continue

        base_open = float(df.at[i, "open"])
        final_close = float(df.at[end, "close"])
        final_ret = (final_close / base_open - 1.0) * 100.0

        if final_ret <= -thr:
            result = "실패"
        elif final_ret >= thr:
            result = "성공"
        else:
            result = "중립"

        res.append({
            "신호시간": df.at[i, "time"],
            "기준시가": int(round(base_open)),
            "RSI(13)": round(float(df.at[i, "RSI13"]), 1) if pd.notna(df.at[i, "RSI13"]) else None,
            "성공기준(%)": round(thr, 1),
            "결과": result,
            "최종수익률(%)": round(final_ret, 1),
        })

    out = pd.DataFrame(res)
    if not out.empty and "중복 제거" in dedup_mode:
        # 연속 동일 결과 제거
        out = out.loc[out["결과"].shift() != out["결과"]]
    return out

# -----------------------------
# 실행
# -----------------------------
try:
    if start_date > end_date:
        st.error("시작 날짜가 종료 날짜보다 이후입니다. 다시 선택해 주세요.")
        st.stop()

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())

    df = fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar)
    if df.empty:
        st.error("데이터가 없습니다. 기간을 변경해 보세요.")
        st.stop()

    df = add_indicators(df)
    res = simulate(df, rsi_side, lookahead, threshold_pct, bb_cond, dup_mode)

    # -----------------------------
    # 섹션: 요약 & 차트
    # -----------------------------
    st.markdown('<div class="section-title">③ 요약 & 차트</div>', unsafe_allow_html=True)

    total = len(res)
    wins  = int((res["결과"] == "성공").sum()) if total else 0
    fails = int((res["결과"] == "실패").sum()) if total else 0
    neuts = int((res["결과"] == "중립").sum()) if total else 0
    winrate = ((wins + neuts) / total * 100.0) if total else 0.0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("신호 수", f"{total}")
    m2.metric("성공", f"{wins}")
    m3.metric("실패", f"{fails}")
    m4.metric("중립", f"{neuts}")
    m5.metric("승률", f"{winrate:.1f}%")

    # 가격 차트
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="가격"
    ))

    if total > 0:
        for label, color, symbol in [("성공", "red", "triangle-up"),
                                     ("실패", "blue", "triangle-down"),
                                     ("중립", "green", "circle")]:
            sub = res[res["결과"] == label]
            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub["신호시간"], y=sub["기준시가"], mode="markers",
                    name=f"신호 ({label})",
                    marker=dict(size=9, color=color, symbol=symbol, line=dict(width=1, color="black")),
                    hovertemplate="신호시간=%{x}<br>기준시가=%{y:,}<extra></extra>"
                ))

    fig.update_layout(
        title=f"{market_label.split(' — ')[0]} · {tf_label} · RSI(13) + BB 시뮬레이션",
        xaxis_title="시간", yaxis_title="가격",
        xaxis_rangeslider_visible=False, height=540,
        legend_orientation="h", legend_y=-0.15
    )
    st.plotly_chart(fig, use_container_width=True)

    # RSI 차트(동일 x축 연동)
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df["time"], y=df["RSI13"], mode="lines", name="RSI(13)"))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="blue")
    fig_rsi.update_layout(height=220, xaxis_title="시간", yaxis_title="RSI(13)")
    fig_rsi.update_xaxes(matches="x")
    st.plotly_chart(fig_rsi, use_container_width=True)

    # -----------------------------
    # 섹션: 신호 결과 표
    # -----------------------------
    st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)

    if total > 0:
        tbl = res.sort_values("신호시간", ascending=False).reset_index(drop=True).copy()
        tbl["기준시가"] = tbl["기준시가"].map(lambda v: f"{int(v):,}")
        tbl["RSI(13)"] = tbl["RSI(13)"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "")
        tbl["성공기준(%)"] = tbl["성공기준(%)"].map(lambda v: f"{v:.1f}%")
        tbl["최종수익률(%)"] = tbl["최종수익률(%)"].map(lambda v: f"{v:.1f}%")

        def color_result(val):
            if val == "성공":
                return 'color:red; font-weight:600;'
            if val == "실패":
                return 'color:blue; font-weight:600;'
            return 'color:green; font-weight:600;'

        styled = (tbl.style
                  .applymap(color_result, subset=["결과"])
                  )
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("조건을 만족하는 신호가 없습니다.")


except Exception as e:
    st.error(f"오류: {e}")

