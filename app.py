import streamlit as st
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
import plotly.graph_objs as go
from plotly.subplots import make_subplots
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

st.caption("※ 판정은 최종(N번째 종가) 기준입니다.")

# -----------------------------
# 데이터 수집 (페이징 안정화)
# -----------------------------
def estimate_calls(start_dt: datetime, end_dt: datetime, minutes_per_bar: int) -> int:
    mins = max(1, int((end_dt - start_dt).total_seconds() // 60))
    bars = max(1, mins // minutes_per_bar)
    calls = bars // 200 + 1
    return min(calls, 5000)

# 재사용 세션 + 재시도
_session = requests.Session()
_retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
_session.mount("https://", HTTPAdapter(max_retries=_retries))

@st.cache_data(ttl=120, show_spinner=False)
def fetch_upbit_paged(market_code: str, interval_key: str, start_dt: datetime, end_dt: datetime,
                      minutes_per_bar: int) -> pd.DataFrame:
    # 엔드포인트 구성
    if "minutes/" in interval_key:
        unit = interval_key.split("/")[1]
        url = f"https://api.upbit.com/v1/candles/minutes/{unit}"
    else:
        url = f"https://api.upbit.com/v1/candles/{interval_key}"

    calls_est = estimate_calls(start_dt, end_dt, minutes_per_bar)
    max_calls = min(calls_est + 2, 60)  # 호출 상한

    all_data = []
    to_time = end_dt
    progress = st.progress(0.0)

    try:
        for done in range(max_calls):
            params = {
                "market": market_code,
                "count": 200,
                "to": to_time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            r = _session.get(url, params=params, headers={"Accept": "application/json"}, timeout=10)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break

            all_data.extend(batch)

            # 최신→과거 정렬, 마지막 원소가 가장 오래된 봉
            last_ts = pd.to_datetime(batch[-1]["candle_date_time_kst"])
            if last_ts <= start_dt:
                break

            # 다음 페이징 기준시간
            to_time = last_ts - timedelta(seconds=1)

            # 진행률
            progress.progress(min(1.0, (done + 1) / max(1, calls_est)))
    finally:
        progress.empty()

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data).rename(columns={
        "candle_date_time_kst": "time",
        "opening_price": "open",
        "high_price": "high",
        "low_price": "low",
        "trade_price": "close",
        "candle_acc_trade_volume": "volume",
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
    out["BB_up"]  = bb.bollinger_hband()
    out["BB_low"] = bb.bollinger_lband()
    out["BB_mid"] = bb.bollinger_mavg()   # 중앙선
    return out

# -----------------------------
# 시뮬레이션
# -----------------------------
def simulate(df: pd.DataFrame, rsi_side: str, lookahead: int, thr_pct: float,
             bb_cond: str, dedup_mode: str) -> pd.DataFrame:
    res = []
    n = len(df)
    thr = float(thr_pct)

    # RSI 조건 인덱스
    if "≤" in rsi_side:
        sig_idx = df.index[(df["RSI13"].notna()) & (df["RSI13"] <= 30)].tolist()
    else:
        sig_idx = df.index[(df["RSI13"].notna()) & (df["RSI13"] >= 70)].tolist()

    for i in sig_idx:
        end = i + lookahead
        if end >= n:
            continue

        # 볼린저 조건
        if bb_cond != "없음":
            px = float(df.at[i, "close"])
            up  = float(df.at[i, "BB_up"])  if pd.notna(df.at[i, "BB_up"])  else None
            lo  = float(df.at[i, "BB_low"]) if pd.notna(df.at[i, "BB_low"]) else None
            mid = float(df.at[i, "BB_mid"]) if pd.notna(df.at[i, "BB_mid"]) else None
            ok = True
            if   bb_cond == "하한선 하향돌파": ok = (lo is not None) and (px < lo)
            elif bb_cond == "하한선 상향돌파": ok = (lo is not None) and (px > lo)
            elif bb_cond == "상한선 하향돌파": ok = (up is not None) and (px < up)
            elif bb_cond == "상한선 상향돌파": ok = (up is not None) and (px > up)
            elif bb_cond == "하한선 중앙돌파": ok = (mid is not None) and (lo is not None) and (px > lo) and (px < mid)
            elif bb_cond == "상한선 중앙돌파": ok = (mid is not None) and (up is not None) and (px < up) and (px > mid)
            if not ok:
                continue

        base_price = float(df.at[i, "low"])
        closes = df.loc[i+1:end, ["time", "close"]]
        if closes.empty:
            continue

        target_up = base_price * (1 + thr / 100)
        target_down = base_price * (1 - thr / 100)

        # 도달 여부
        hit_up = closes[closes["close"] >= target_up]
        hit_down = closes[closes["close"] <= target_down]

        result = "중립"
        reach_time = None  # 성공일 경우 HH:MM 표기

        if not hit_up.empty and not hit_down.empty:
            if hit_up.iloc[0]["time"] < hit_down.iloc[0]["time"]:
                result = "성공"
                reach_time = hit_up.iloc[0]["time"].strftime("%H:%M")
            else:
                result = "실패"
        elif not hit_up.empty:
            result = "성공"
            reach_time = hit_up.iloc[0]["time"].strftime("%H:%M")
        elif not hit_down.empty:
            result = "실패"
        else:
            final_price = closes.iloc[-1]["close"]
            if final_price > base_price:
                result = "중립"
            else:
                result = "실패"

        # 수익률 계산
        final_ret = (closes.iloc[-1]["close"] / base_price - 1.0) * 100.0
        min_ret   = (closes["close"].min() / base_price - 1.0) * 100.0
        max_ret   = (closes["close"].max() / base_price - 1.0) * 100.0

        res.append({
            "신호시간": df.at[i, "time"],
            "기준시가": int(round(base_price)),
            "RSI(13)": round(float(df.at[i, "RSI13"]), 1) if pd.notna(df.at[i, "RSI13"]) else None,
            "성공기준(%)": round(thr, 1),
            "결과": result,
            "도달시간": reach_time,  # 성공일 경우 HH:MM
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

    # (선택) 분봉 조회 범위 안전가드: 지나치게 넓은 분봉 범위 제한
    if "분" in tf_label:
        max_days_for_minute = 7  # 필요 시 조정
        if (end_dt - start_dt).days > max_days_for_minute:
            st.warning(f"분봉 조회 범위가 넓어 자동으로 최근 {max_days_for_minute}일로 제한합니다.")
            start_dt = end_dt - timedelta(days=max_days_for_minute)

    df = fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar)
    if df.empty:
        st.error("데이터가 없습니다.")
        st.stop()

    df = add_indicators(df)

    # -----------------------------
    # 요약 & 차트
    # -----------------------------
    st.markdown('<div class="section-title">③ 요약 & 차트</div>', unsafe_allow_html=True)

    def _summarize(_df: pd.DataFrame):
        total = len(_df)
        succ = int((_df["결과"] == "성공").sum())
        fail = int((_df["결과"] == "실패").sum())
        neu  = int((_df["결과"] == "중립").sum())
        win  = (succ / total * 100.0) if total > 0 else 0.0
        range_sum = float((_df["최고수익률(%)"] - _df["최저수익률(%)"]).sum()) if total > 0 else 0.0
        final_succ = float(_df.loc[_df["결과"] == "성공", "최종수익률(%)"].sum()) if total > 0 else 0.0
        final_fail = float(_df.loc[_df["결과"] == "실패", "최종수익률(%)"].sum()) if total > 0 else 0.0
        return total, succ, fail, neu, win, range_sum, final_succ, final_fail

    # 두 모드만 계산 (중복 호출 제거)
    res_all   = simulate(df, rsi_side, lookahead, threshold_pct, bb_cond, "중복 포함 (연속 신호 모두)")
    res_dedup = simulate(df, rsi_side, lookahead, threshold_pct, bb_cond, "중복 제거 (연속 동일 결과 1개)")

    for label, data in [("중복 포함 (연속 신호 모두)", res_all), ("중복 제거 (연속 동일 결과 1개)", res_dedup)]:
        total, succ, fail, neu, win, range_sum, final_succ, final_fail = _summarize(data)
        st.markdown(f"**{label}**")
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("신호 수", f"{total}")
        c2.metric("성공", f"{succ}")
        c3.metric("실패", f"{fail}")
        c4.metric("중립", f"{neu}")
        c5.metric("승률", f"{win:.1f}%")
        c6.metric("총 변동폭 합(%)", f"{range_sum:.1f}%")
        final_sum = final_succ + final_fail
        c7.metric("최종수익률 합계", f"{final_sum:.1f}%")
        st.markdown("---")

    # 선택 모드 적용
    res = res_all if dup_mode.startswith("중복 포함") else res_dedup
    has_signal = len(res) > 0

    # -----------------------------
    # 가격 + RSI 함께 표시
    # -----------------------------
    fig = make_subplots(rows=1, cols=1)

    # 캔들 (상승=레드, 하락=블루)
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="가격",
        increasing_line_color="#E63946", decreasing_line_color="#457B9D",
        line=dict(width=1.2)
    ))

    # 볼린저밴드 (상/중/하)
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["BB_up"], mode="lines",
        line=dict(color="#FFB703", width=1.5),
        name="BB 상단"
    ))
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["BB_low"], mode="lines",
        line=dict(color="#219EBC", width=1.5),
        name="BB 하단"
    ))
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["BB_mid"], mode="lines",
        line=dict(color="#8D99AE", width=1.2, dash="dot"),
        name="BB 중앙"
    ))

    # 신호 마커
    if has_signal:
        for _label, _color in [("성공","#06D6A0"), ("실패","#EF476F"), ("중립","#FFD166")]:
            sub = res[res["결과"] == _label]
            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub["신호시간"], y=sub["기준시가"], mode="markers",
                    name=f"신호 ({_label})",
                    marker=dict(size=10, color=_color, symbol="circle",
                                line=dict(width=1, color="black"))
                ))

    # RSI → 보조 y축
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["RSI13"], mode="lines",
        line=dict(color="#2A9D8F", width=2), opacity=0.85,
        name="RSI(13)", yaxis="y2"
    ))

    # RSI 기준선
    fig.add_hline(y=70, line_dash="dash", line_color="#E63946",
                  line_width=1.2, annotation_text="RSI 70",
                  annotation_position="top left", yref="y2")
    fig.add_hline(y=30, line_dash="dash", line_color="#457B9D",
                  line_width=1.2, annotation_text="RSI 30",
                  annotation_position="bottom left", yref="y2")

    fig.update_layout(
        title=f"{market_label.split(' — ')[0]} · {tf_label} · RSI(13) + BB 시뮬레이션",
        xaxis_rangeslider_visible=False,
        height=600,
        autosize=False,
        legend_orientation="h", legend_y=1.05,
        margin=dict(l=60, r=40, t=60, b=40),
        yaxis=dict(title="가격"),
        yaxis2=dict(overlaying="y", side="right", showgrid=False, title="RSI(13)", range=[0,100])
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # 신호 결과 표
    # -----------------------------
    st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)
    if has_signal:
        tbl = res.sort_values("신호시간", ascending=False).reset_index(drop=True).copy()

        # 표시 형식
        tbl["기준시가"] = tbl["기준시가"].map(lambda v: f"{int(v):,}")
        if "RSI(13)" in tbl.columns:
            tbl["RSI(13)"] = tbl["RSI(13)"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "")
        if "성공기준(%)" in tbl.columns:
            tbl["성공기준(%)"] = tbl["성공기준(%)"].map(lambda v: f"{v:.1f}%")
        if "최종수익률(%)" in tbl.columns:
            tbl["최종수익률(%)"] = tbl["최종수익률(%)"].map(lambda v: f"{v:.1f}%")
        if "최저수익률(%)" in tbl.columns:
            tbl["최저수익률(%)"] = tbl["최저수익률(%)"].map(lambda v: f"{v:.1f}%")
        if "최고수익률(%)" in tbl.columns:
            tbl["최고수익률(%)"] = tbl["최고수익률(%)"].map(lambda v: f"{v:.1f}%")

        # 도달시간 (HH:MM, 없으면 "-")
        if "도달시간" in tbl.columns:
            tbl["도달시간"] = tbl["도달시간"].fillna("-").astype(str)

        # 결과 색상 강조
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
