# -*- coding: utf-8 -*-
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

# ➜ 볼린저밴드 조건 단순화 (상/중앙/하)  + “걸침(고가~저가가 해당선 포함)” 판정
c7, c8, c9 = st.columns(3)
with c7:
    bb_cond = st.selectbox(
        "볼린저밴드 조건",
        ["없음", "상한선", "중앙선", "하한선"],
        index=0,
    )
with c8:
    bb_window = st.number_input("BB 기간", min_value=5, max_value=100, value=30, step=1)
with c9:
    bb_dev = st.number_input("BB 승수", min_value=1.0, max_value=4.0, value=2.0, step=0.1)

st.session_state["rsi_side"] = rsi_side
st.session_state["bb_cond"]  = bb_cond

# ---- 조건 요약 박스 ----
sim_minutes = lookahead * minutes_per_bar
if sim_minutes < 60:
    sim_dur = f"약 {sim_minutes}분"
elif sim_minutes < 1440:
    sim_dur = f"약 {sim_minutes//60}시간 {sim_minutes%60}분"
else:
    sim_dur = f"약 {sim_minutes//1440}일"

rsi_display = rsi_side
if "≤" in rsi_side:
    rsi_display = f"<span style='color:blue; font-weight:600;'>{rsi_side}</span>"
elif "≥" in rsi_side:
    rsi_display = f"<span style='color:red; font-weight:600;'>{rsi_side}</span>"

bb_display = bb_cond
if bb_cond == "상한선":
    bb_display = f"<span style='color:red; font-weight:600;'>{bb_cond}</span>"
elif bb_cond == "하한선":
    bb_display = f"<span style='color:blue; font-weight:600;'>{bb_cond}</span>"
elif bb_cond == "중앙선":
    bb_display = f"<span style='color:green; font-weight:600;'>{bb_cond}</span>"

st.markdown(f"""
<div style="border:1px solid #ccc; border-radius:8px; padding:0.8rem; background-color:#f9f9f9; margin-top:0.6rem; margin-bottom:0.6rem;">
<b>📌 현재 조건 요약</b><br>
- 측정 캔들 수: {lookahead}봉 ({sim_dur})<br>
- 성공/실패 기준: {threshold_pct:.2f}%<br>
- RSI 조건: {rsi_display}<br>
- 볼린저밴드 조건: {bb_display}
</div>
""", unsafe_allow_html=True)

# -----------------------------
# 데이터 수집 (Upbit Pagination)
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
    progress = st.progress(0.0)
    try:
        for done in range(max_calls):
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
            progress.progress(min(1.0, (done + 1) / max(1, max_calls)))
    finally:
        progress.empty()

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data).rename(columns={
        "candle_date_time_kst":"time","opening_price":"open","high_price":"high",
        "low_price":"low","trade_price":"close","candle_acc_trade_volume":"volume"})
    df["time"] = pd.to_datetime(df["time"])
    df = df[["time","open","high","low","close","volume"]].sort_values("time").reset_index(drop=True)
    # 날짜 필터 (UI와 동일)
    df = df[(df["time"] >= start_dt) & (df["time"] <= end_dt)]
    return df

# -----------------------------
# 지표
# -----------------------------
def add_indicators(df, bb_window, bb_dev):
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=bb_window, window_dev=bb_dev)
    # 차트 끊김 방지
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

    # (A) 볼린저 조건: “걸침” 판정 (해당 라인값이 그 봉의 [저가, 고가] 사이면 신호)
    def bb_ok(i: int) -> bool:
        if bb_cond == "없음":
            return True
        hi = float(df.at[i, "high"])
        lo_px = float(df.at[i, "low"])
        up, lo, mid = df.at[i, "BB_up"], df.at[i, "BB_low"], df.at[i, "BB_mid"]
        if bb_cond == "상한선":
            return pd.notna(up) and (lo_px <= up <= hi)
        if bb_cond == "중앙선":
            return pd.notna(mid) and (lo_px <= mid <= hi)
        if bb_cond == "하한선":
            return pd.notna(lo) and (lo_px <= lo <= hi)
        return False

    # (B) RSI 후보 (임계값 충족이면 신호)
    rsi_idx = []
    if rsi_side == "RSI ≤ 30 (급락)":
        rsi_idx = df.index[df["RSI13"] <= 30].tolist()
    elif rsi_side == "RSI ≥ 70 (급등)":
        rsi_idx = df.index[df["RSI13"] >= 70].tolist()

    # (C) BB 후보
    bb_idx = []
    if bb_cond != "없음":
        for i in df.index:
            try:
                if bb_ok(i): bb_idx.append(i)
            except Exception:
                continue

    # (D) 최종 후보 (둘 다 선택 시 AND, 하나만 선택 시 해당 조건만)
    if rsi_side != "없음" and bb_cond != "없음":
        sig_idx = sorted(set(rsi_idx) & set(bb_idx))
    elif rsi_side != "없음":
        sig_idx = rsi_idx
    elif bb_cond != "없음":
        sig_idx = bb_idx
    else:
        sig_idx = []

    # (E) 결과 계산
    for i in sig_idx:
        end = i + lookahead
        if end >= n:
            continue

        # 기준가: (시가 + 저가) / 2
        base = (float(df.at[i,"open"]) + float(df.at[i,"low"])) / 2.0

        closes = df.loc[i+1:end, ["time","close"]]
        if closes.empty:
            continue

        final_ret = (closes.iloc[-1]["close"]/base - 1)*100.0
        min_ret   = (closes["close"].min()/base - 1)*100.0
        max_ret   = (closes["close"].max()/base - 1)*100.0

        result="중립"; reach_min=None
        if max_ret >= thr:
            first_hit = closes[closes["close"] >= base*(1+thr/100)]
            if not first_hit.empty:
                reach_min = int((first_hit.iloc[0]["time"] - df.at[i,"time"]).total_seconds() // 60)
            result = "성공"
        elif final_ret < 0:
            result = "실패"

        res.append({
            "신호시간": df.at[i,"time"],
            "기준시가": int(round(base)),
            "RSI(13)": round(float(df.at[i,"RSI13"]),1) if pd.notna(df.at[i,"RSI13"]) else None,
            "성공기준(%)": round(thr,1),
            "결과": result,
            "도달분": reach_min,
            "최종수익률(%)": round(final_ret, 2),
            "최저수익률(%)": round(min_ret, 2),
            "최고수익률(%)": round(max_ret, 2),
        })

    out = pd.DataFrame(res, columns=["신호시간","기준시가","RSI(13)","성공기준(%)","결과","도달분","최종수익률(%)","최저수익률(%)","최고수익률(%)"])

    # (F) 중복 제거 옵션
    if not out.empty and dedup_mode.startswith("중복 제거"):
        out["분"] = pd.to_datetime(out["신호시간"]).dt.strftime("%Y-%m-%d %H:%M")
        out = out.drop_duplicates(subset=["분"], keep="first").drop(columns=["분"])
        filtered = []
        last_time = pd.Timestamp("1970-01-01")
        delta = pd.Timedelta(minutes=lookahead * minutes_per_bar)
        for _, row in out.sort_values("신호시간").iterrows():
            if row["신호시간"] >= last_time + delta:
                filtered.append(row)
                last_time = row["신호시간"]
        out = pd.DataFrame(filtered) if filtered else pd.DataFrame(columns=out.columns)

    return out

# -----------------------------
# 실행
# -----------------------------
try:
    if start_date > end_date:
        st.error("시작 날짜가 종료 날짜보다 이후입니다.")
        st.stop()

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt   = datetime.combine(end_date,   datetime.max.time())

    df = fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar)
    if df.empty:
        st.error("데이터가 없습니다.")
        st.stop()

    if rsi_side == "없음" and bb_cond == "없음":
        st.markdown('<div class="section-title">③ 요약 & 차트</div>', unsafe_allow_html=True)
        st.info("대기중..")
        st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)
        st.info("대기중..")
        st.stop()

    df = add_indicators(df, bb_window, bb_dev)
    rsi_side = st.session_state.get("rsi_side", rsi_side)
    bb_cond  = st.session_state.get("bb_cond", bb_cond)

    # 두 버전(중복 포함/제거) 모두 계산
    res_all   = simulate(df, rsi_side, lookahead, threshold_pct, bb_cond,
                         "중복 포함 (연속 신호 모두)", minutes_per_bar, market_code, bb_window, bb_dev)
    res_dedup = simulate(df, rsi_side, lookahead, threshold_pct, bb_cond,
                         "중복 제거 (연속 동일 결과 1개)", minutes_per_bar, market_code, bb_window, bb_dev)

    st.markdown('<div class="section-title">③ 요약 & 차트</div>', unsafe_allow_html=True)

    def _summarize(df_in: pd.DataFrame):
        if df_in is None or df_in.empty:
            return 0,0,0,0,0.0,0.0
        total=len(df_in)
        succ=int((df_in["결과"]=="성공").sum())
        fail=int((df_in["결과"]=="실패").sum())
        neu =int((df_in["결과"]=="중립").sum())
        win=succ/total*100.0 if total else 0.0
        total_final=float(df_in["최종수익률(%)"].sum())
        return total,succ,fail,neu,win,total_final

    for label, data in [("중복 포함 (연속 신호 모두)",res_all), ("중복 제거 (연속 동일 결과 1개)",res_dedup)]:
        total,succ,fail,neu,win,total_final=_summarize(data)
        st.markdown(f"**{label}**")
        c1,c2,c3,c4,c5,c6=st.columns(6)
        c1.metric("신호 수",f"{total}")
        c2.metric("성공",f"{succ}")
        c3.metric("실패",f"{fail}")
        c4.metric("중립",f"{neu}")
        c5.metric("승률",f"{win:.1f}%")
        col = "red" if total_final > 0 else "blue" if total_final < 0 else "black"
        c6.markdown(
            f"<div style='font-weight:600;'>최종수익률 합계: "
            f"<span style='color:{col}; font-size:1.1rem'>{total_final:.1f}%</span></div>",
            unsafe_allow_html=True
        )
        st.markdown("---")

    # 현재 화면 선택 적용
    res = res_all if dup_mode.startswith("중복 포함") else res_dedup

    # ---- 차트 ----
    fig=make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="가격", increasing_line_color="red", decreasing_line_color="blue",
        line=dict(width=1.1)
    ))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_up"], mode="lines",
                             line=dict(color="#FFB703", width=1.4), name="BB 상단", connectgaps=True))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_low"], mode="lines",
                             line=dict(color="#219EBC", width=1.4), name="BB 하단", connectgaps=True))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_mid"], mode="lines",
                             line=dict(color="#8D99AE", width=1.1, dash="dot"), name="BB 중앙", connectgaps=True))

    # 신호 마커/흐름선
    if res is not None and not res.empty:
        legend_once = { "신호_성공": False, "신호_실패": False, "신호_중립": False,
                        "목표도달": False, "선_성공": False, "선_실패": False, "선_중립": False }
        for _label, _color in [("성공","red"), ("실패","blue"), ("중립","#FFD166")]:
            sub = res[res["결과"] == _label]
            if sub.empty: continue
            fig.add_trace(go.Scatter(
                x=sub["신호시간"], y=sub["기준시가"], mode="markers",
                name=f"신호 ({_label})",
                marker=dict(size=9, color=_color, symbol="circle", line=dict(width=1, color="black")),
                legendgroup=f"신호_{_label}", showlegend=not legend_once[f"신호_{_label}"]
            ))
            legend_once[f"신호_{_label}"] = True
            for _, row in sub.iterrows():
                if _label == "성공" and pd.notna(row["도달분"]):
                    signal_time = row["신호시간"]; signal_price = row["기준시가"]
                    target_time = row["신호시간"] + pd.Timedelta(minutes=int(row["도달분"]))
                    target_price = row["기준시가"] * (1 + row["성공기준(%)"]/100)
                    fig.add_trace(go.Scatter(
                        x=[target_time], y=[target_price], mode="markers", name="목표 도달",
                        marker=dict(size=11, color="red", symbol="star", line=dict(width=1, color="black")),
                        legendgroup="목표도달", showlegend=not legend_once["목표도달"]
                    ))
                    legend_once["목표도달"] = True
                    fig.add_trace(go.Scatter(
                        x=[signal_time, target_time], y=[signal_price, target_price], mode="lines",
                        line=dict(color="red", width=2.2, dash="dot"), name="흐름선(성공)",
                        legendgroup="선_성공", showlegend=not legend_once["선_성공"]
                    ))
                    legend_once["선_성공"] = True
                elif _label in ["실패", "중립"]:
                    signal_time = row["신호시간"]
                    start_price = row["기준시가"]
                    end_time = row["신호시간"] + pd.Timedelta(minutes=lookahead * minutes_per_bar)
                    end_price = row["기준시가"] * (1 + row["최종수익률(%)"]/100)
                    key = "선_실패" if _label == "실패" else "선_중립"
                    fig.add_trace(go.Scatter(
                        x=[signal_time, end_time], y=[start_price, end_price], mode="lines",
                        line=dict(color=_color, width=1, dash="dot"), name=f"흐름선({_label})",
                        opacity=0.55, legendgroup=key, showlegend=not legend_once[key]
                    ))
                    legend_once[key] = True

    # RSI 보조축
    fig.add_trace(go.Scatter(x=df["time"], y=df["RSI13"], mode="lines",
                             line=dict(color="rgba(42,157,143,0.3)", width=6),
                             opacity=0.6, name="RSI Glow", yaxis="y2", showlegend=False))
    fig.add_trace(go.Scatter(x=df["time"], y=df["RSI13"], mode="lines",
                             line=dict(color="#2A9D8F", width=2.4, dash="dot"),
                             opacity=1, name="RSI(13)", yaxis="y2"))
    fig.add_hline(y=70, line_dash="dash", line_color="#E63946", line_width=1.1,
                  annotation_text="RSI 70", annotation_position="top left", yref="y2")
    fig.add_hline(y=30, line_dash="dash", line_color="#457B9D", line_width=1.1,
                  annotation_text="RSI 30", annotation_position="bottom left", yref="y2")

    fig.update_layout(title=f"{market_label.split(' — ')[0]} · {tf_label} · RSI(13) + BB 시뮬레이션",
                      xaxis_rangeslider_visible=False, height=600, autosize=False,
                      legend_orientation="h", legend_y=1.05,
                      margin=dict(l=60, r=40, t=60, b=40),
                      yaxis=dict(title="가격"),
                      yaxis2=dict(overlaying="y", side="right", showgrid=False, title="RSI(13)", range=[0,100]))
    st.plotly_chart(fig, use_container_width=True)

    # ---- 표 (예전 스타일) ----
    st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)
    if res is not None and not res.empty:
        tbl = res.sort_values("신호시간", ascending=False).reset_index(drop=True).copy()
        tbl["신호시간"] = pd.to_datetime(tbl["신호시간"]).dt.strftime("%Y-%m-%d %H:%M")
        tbl["기준시가"] = tbl["기준시가"].map(lambda v: f"{int(v):,}")
        if "RSI(13)" in tbl:
            tbl["RSI(13)"] = tbl["RSI(13)"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "")
        for col in ["성공기준(%)","최종수익률(%)","최저수익률(%)","최고수익률(%)"]:
            if col in tbl:
                tbl[col] = tbl[col].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "")
        # 도달시간 HH:MM
        def fmt_hhmm(m):
            if pd.isna(m): return "None"
            m = int(m); h,mm = divmod(m,60)
            return f"{h:02d}:{mm:02d}"
        tbl["도달시간"] = res["도달분"].map(fmt_hhmm) if "도달분" in res else "None"
        if "도달분" in tbl:  # 사용자는 표에서 분은 숨김
            tbl = tbl.drop(columns=["도달분"])
        # 컬럼 순서 보정
        cols = ["신호시간","기준시가","RSI(13)","성공기준(%)","결과","도달시간","최종수익률(%)","최저수익률(%)","최고수익률(%)"]
        tbl = tbl[[c for c in cols if c in tbl.columns]]
        st.dataframe(tbl, use_container_width=True, hide_index=True)
    else:
        st.info("조건을 만족하는 신호가 없습니다.")
except Exception as e:
    st.error(f"오류: {e}")

