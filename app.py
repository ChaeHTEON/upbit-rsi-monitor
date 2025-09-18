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
# ① 기본 설정
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
# ② 조건 설정
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

# 볼린저밴드 조건 + 설정
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

# 안전 장치(세션 보강)
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
if "하향" in bb_cond:
    bb_display = f"<span style='color:blue; font-weight:600;'>{bb_cond}</span>"
elif "상향" in bb_cond:
    bb_display = f"<span style='color:red; font-weight:600;'>{bb_cond}</span>"

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
# 데이터 수집
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

    # 날짜 필터 후 인덱스 리셋
    df = df[(df["time"].dt.date >= start_dt.date()) & (df["time"].dt.date <= end_dt.date())].reset_index(drop=True)
    return df

# -----------------------------
# 지표
# -----------------------------
def add_indicators(df, bb_window, bb_dev):
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=bb_window, window_dev=bb_dev)
    out["BB_up"]  = bb.bollinger_hband()
    out["BB_low"] = bb.bollinger_lband()
    out["BB_mid"] = bb.bollinger_mavg()
    return out

# -----------------------------
# 시뮬레이션
# -----------------------------
def simulate(df, rsi_side, lookahead, thr_pct, bb_cond, dedup_mode):
    res=[]
    n=len(df); thr=float(thr_pct)

    candidates = []
    if rsi_side != "없음":
        if "≤" in rsi_side:
            candidates = df.index[(df["RSI13"].notna()) & (df["RSI13"] <= 30)].tolist()
        elif "≥" in rsi_side:
            candidates = df.index[(df["RSI13"].notna()) & (df["RSI13"] >= 70)].tolist()

    if rsi_side == "없음" and bb_cond != "없음":
        candidates = list(range(n))

    if rsi_side == "없음" and bb_cond == "없음":
        return pd.DataFrame(columns=[
            "신호시간","기준시가","RSI(13)","성공기준(%)","결과","도달분",
            "최종수익률(%)","최저수익률(%)","최고수익률(%)"
        ])

    for i in candidates:
        if i+lookahead >= n: 
            continue

        row = df.iloc[i]
        px, up, lo, mid = float(row["close"]), row["BB_up"], row["BB_low"], row["BB_mid"]

        if bb_cond!="없음":
            ok=True
            if bb_cond=="하한선 하향돌파": ok=pd.notna(lo) and px<lo
            elif bb_cond=="하한선 상향돌파": ok=pd.notna(lo) and px>lo
            elif bb_cond=="상한선 하향돌파": ok=pd.notna(up) and px<up
            elif bb_cond=="상한선 상향돌파": ok=pd.notna(up) and px>up
            elif bb_cond=="하한선 중앙돌파": ok=pd.notna(lo) and pd.notna(mid) and lo<px<mid
            elif bb_cond=="상한선 중앙돌파": ok=pd.notna(up) and pd.notna(mid) and mid<px<up
            if not ok: 
                continue

        base=float(row["open"])
        closes=df.iloc[i+1:i+lookahead+1][["time","close"]]
        if closes.empty: continue

        final_ret=(closes.iloc[-1]["close"]/base-1)*100.0
        min_ret=(closes["close"].min()/base-1)*100.0
        max_ret=(closes["close"].max()/base-1)*100.0

        result="중립"; reach_min=None
        if max_ret >= thr:
            first_hit = closes[closes["close"] >= base*(1+thr/100)]
            if not first_hit.empty:
                reach_min = int((first_hit.iloc[0]["time"] - row["time"]).total_seconds() // 60)
            result = "성공"
        elif final_ret < 0:
            result = "실패"

        res.append({
            "신호시간": row["time"],
            "기준시가": int(round(base)),
            "RSI(13)": round(float(row["RSI13"]),1) if pd.notna(row["RSI13"]) else None,
            "성공기준(%)": round(thr,1),
            "결과": result,
            "도달분": reach_min,
            "최종수익률(%)": round(final_ret,2),
            "최저수익률(%)": round(min_ret,2),
            "최고수익률(%)": round(max_ret,2),
        })

    out=pd.DataFrame(res, columns=["신호시간","기준시가","RSI(13)","성공기준(%)","결과","도달분","최종수익률(%)","최저수익률(%)","최고수익률(%)"])

    if not out.empty:
        out["분"] = pd.to_datetime(out["신호시간"]).dt.strftime("%Y-%m-%d %H:%M")
        out = out.drop_duplicates(subset=["분"], keep="first").drop(columns=["분"])

    if not out.empty and dedup_mode.startswith("중복 제거"):
        filtered = []
        last_idx = -9999
        for idx, row in out.reset_index().iterrows():
            if row["index"] >= last_idx + lookahead:
                filtered.append(row)
                last_idx = row["index"]
        out = pd.DataFrame(filtered).drop(columns=["index"]) if filtered else pd.DataFrame()
    return out

# -----------------------------
# 실행
# -----------------------------
try:
    if start_date>end_date:
        st.error("시작 날짜가 종료 날짜보다 이후입니다."); st.stop()

    start_dt=datetime.combine(start_date, datetime.min.time())
    end_dt  =datetime.combine(end_date,   datetime.max.time())

    df=fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar)
    if df.empty: st.error("데이터가 없습니다."); st.stop()

    if rsi_side == "없음" and bb_cond == "없음":
        st.markdown('<div class="section-title">③ 요약 & 차트</div>', unsafe_allow_html=True)
        st.info("대기중..")
        st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)
        st.info("대기중..")
        st.stop()

    df=add_indicators(df, bb_window, bb_dev)

    rsi_side = st.session_state.get("rsi_side", rsi_side)
    bb_cond  = st.session_state.get("bb_cond", bb_cond)

    res_all  = simulate(df, rsi_side, lookahead, threshold_pct, bb_cond, "중복 포함 (연속 신호 모두)")
    res_dedup= simulate(df, rsi_side, lookahead, threshold_pct, bb_cond, "중복 제거 (연속 동일 결과 1개)")

    # ---- 요약 & 차트 ----
    st.markdown('<div class="section-title">③ 요약 & 차트</div>', unsafe_allow_html=True)

    def _summarize(df_in):
        if df_in is None or df_in.empty: return 0,0,0,0,0.0,0.0,0.0,0.0
        total=len(df_in)
        succ=int((df_in["결과"]=="성공").sum())
        fail=int((df_in["결과"]=="실패").sum())
        neu =int((df_in["결과"]=="중립").sum())
        win=succ/total*100.0
        range_sum=float((df_in["최고수익률(%)"]-df_in["최저수익률(%)"]).sum())
        final_succ=float(df_in.loc[df_in["결과"]=="성공","최종수익률(%)"].sum())
        final_fail=float(df_in.loc[df_in["결과"]=="실패","최종수익률(%)"].sum())
        return total,succ,fail,neu,win,range_sum,final_succ,final_fail

    for label,data in [("중복 포함 (연속 신호 모두)",res_all), ("중복 제거 (연속 동일 결과 1개)",res_dedup)]:
        total,succ,fail,neu,win,range_sum,final_succ,final_fail=_summarize(data)
        st.markdown(f"**{label}**")
        c1,c2,c3,c4,c5,c6,c7=st.columns(7)
        c1.metric("신호 수",f"{total}")
        c2.metric("성공",f"{succ}")
        c3.metric("실패",f"{fail}")
        c4.metric("중립",f"{neu}")
        c5.metric("승률(%)",f"{win:.2f}")
        c6.metric("합계범위",f"{range_sum:.2f}")
        c7.metric("순이익",f"{final_succ+final_fail:.2f}")

    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.06,row_heights=[0.7,0.3])
    fig.add_trace(go.Candlestick(x=df["time"],open=df["open"],high=df["high"],low=df["low"],close=df["close"],
        increasing_line_color='red',decreasing_line_color='blue',name="Price"),row=1,col=1)
    fig.add_trace(go.Scatter(x=df["time"],y=df["BB_up"],mode="lines",name="BB 상단"),row=1,col=1)
    fig.add_trace(go.Scatter(x=df["time"],y=df["BB_mid"],mode="lines",name="BB 중앙"),row=1,col=1)
    fig.add_trace(go.Scatter(x=df["time"],y=df["BB_low"],mode="lines",name="BB 하단"),row=1,col=1)
    fig.add_trace(go.Scatter(x=df["time"],y=df["RSI13"],mode="lines",name="RSI(13)"),row=2,col=1)
    fig.add_hline(y=70,line_dash="dash",row=2,col=1)
    fig.add_hline(y=30,line_dash="dash",row=2,col=1)
    fig.update_layout(height=600,margin=dict(t=10,b=10,l=10,r=10),xaxis_rangeslider_visible=False)
    st.plotly_chart(fig,use_container_width=True)

    # ---- 결과 ----
    st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)
    def _render_table(df_in, caption):
        if df_in is None or df_in.empty: st.info(f"{caption}: 결과 없음"); return
        view=df_in.sort_values("신호시간",ascending=False).copy()
        for col in ["최종수익률(%)","최저수익률(%)","최고수익률(%)","성공기준(%)"]:
            view[col]=view[col].map(lambda v: None if pd.isna(v) else round(float(v),2))
        st.dataframe(view,use_container_width=True,hide_index=True)
    _render_table(res_all,"중복 포함")
    _render_table(res_dedup,"중복 제거")

except Exception as e:
    st.error(f"오류 발생: {e}")
