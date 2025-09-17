import streamlit as st
import pandas as pd
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
</style>
""", unsafe_allow_html=True)

st.title("📊 Upbit RSI(13) + Bollinger Band 시뮬레이터")

# -----------------------------
# 업비트 마켓 로드
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
    rows.sort(key=lambda x: (x[1] != "KRW-BTC", x[1]))
    return rows

MARKET_LIST = get_upbit_krw_markets()
default_idx = next(i for i,(_, code) in enumerate(MARKET_LIST) if code == "KRW-BTC")

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
# 기본 설정
# -----------------------------
st.markdown('<div class="section-title">① 기본 설정</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    # ✅ combo_box → 수동 입력 허용
    market_label, market_code = st.combo_box(
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
# 조건 설정
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
    bb_cond = st.selectbox("볼린저밴드 조건",
        ["없음", "하한선 하향돌파", "하한선 상향돌파",
         "상한선 하향돌파", "상한선 상향돌파",
         "하한선 중앙돌파", "상한선 중앙돌파"], index=0)
with c8:
    max_bars = st.slider("표시할 최대 봉 개수 (UI 전용)", 50, 200, 100)

interval_key, minutes_per_bar = TF_MAP[tf_label]

# -----------------------------
# 데이터 수집
# -----------------------------
def fetch_upbit_paged(market_code, interval_key, start_dt, end_dt):
    if "minutes/" in interval_key:
        unit = interval_key.split("/")[1]
        url = f"https://api.upbit.com/v1/candles/minutes/{unit}"
    else:
        url = f"https://api.upbit.com/v1/candles/{interval_key}"
    all_data, to_time = [], end_dt
    while True:
        params = {"market": market_code, "count": 200, "to": to_time.strftime("%Y-%m-%d %H:%M:%S")}
        r = requests.get(url, params=params)
        if r.status_code != 200: break
        batch = r.json()
        if not batch: break
        all_data.extend(batch)
        last_ts = pd.to_datetime(batch[-1]["candle_date_time_kst"])
        if last_ts <= start_dt: break
        to_time = last_ts - timedelta(seconds=1)
        if len(all_data) > 50000: break
    if not all_data: return pd.DataFrame()
    df = pd.DataFrame(all_data)
    df = df.rename(columns={
        "candle_date_time_kst":"time","opening_price":"open","high_price":"high",
        "low_price":"low","trade_price":"close","candle_acc_trade_volume":"volume"})
    df["time"] = pd.to_datetime(df["time"])
    return df[["time","open","high","low","close","volume"]].sort_values("time").reset_index(drop=True)

# -----------------------------
# 지표 추가
# -----------------------------
def add_indicators(df):
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=30, window_dev=2)
    out["BB_up"], out["BB_low"], out["BB_mid"] = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_mavg()
    return out

# -----------------------------
# 시뮬레이션
# -----------------------------
def simulate(df, rsi_side, lookahead, thr_pct, bb_cond):
    res, n = [], len(df)
    sig_idx = df.index[df["RSI13"] <= 30].tolist() if "≤" in rsi_side else df.index[df["RSI13"] >= 70].tolist()
    for i in sig_idx:
        end = i + lookahead
        if end >= n: continue
        px, up, lo, mid = df.at[i,"close"], df.at[i,"BB_up"], df.at[i,"BB_low"], df.at[i,"BB_mid"]
        if bb_cond != "없음":
            if bb_cond=="하한선 하향돌파" and not (px < lo): continue
            if bb_cond=="하한선 상향돌파" and not (px > lo): continue
            if bb_cond=="상한선 하향돌파" and not (px < up): continue
            if bb_cond=="상한선 상향돌파" and not (px > up): continue
            if bb_cond=="하한선 중앙돌파" and not (px > mid): continue
            if bb_cond=="상한선 중앙돌파" and not (px < mid): continue
        base_open, final_close = df.at[i,"open"], df.at[end,"close"]
        future = df.iloc[i+1:end+1]["close"]
        final_ret = (final_close/base_open-1)*100
        min_ret = ((future.min()/base_open)-1)*100
        max_ret = ((future.max()/base_open)-1)*100
        # 기본 판정 (중립 포함)
        if final_ret <= -thr_pct: result="실패"
        elif final_ret >= thr_pct: result="성공"
        elif final_ret > 0: result="중립"
        else: result="실패"
        res.append({"신호시간":df.at[i,"time"],"기준시가":int(round(base_open)),
                    "RSI(13)":round(df.at[i,"RSI13"],1),"성공기준(%)":round(thr_pct,1),
                    "결과":result,"최종수익률(%)":round(final_ret,1),
                    "최저수익률(%)":round(min_ret,1),"최고수익률(%)":round(max_ret,1)})
    return pd.DataFrame(res)

# -----------------------------
# 실행
# -----------------------------
try:
    start_dt, end_dt = datetime.combine(start_date, datetime.min.time()), datetime.combine(end_date, datetime.max.time())
    df = fetch_upbit_paged(market_code, interval_key, start_dt, end_dt)
    if df.empty: st.error("데이터 없음"); st.stop()
    df = add_indicators(df)
    res = simulate(df, rsi_side, lookahead, threshold_pct, bb_cond)

    # ③ 요약 & 차트 (중립 따로)
    st.markdown('<div class="section-title">③ 요약 & 차트</div>', unsafe_allow_html=True)
    total=len(res); wins=(res["결과"]=="성공").sum(); fails=(res["결과"]=="실패").sum(); neuts=(res["결과"]=="중립").sum()
    winrate=((wins+neuts)/total*100) if total else 0
    m1,m2,m3,m4,m5=st.columns(5)
    m1.metric("신호 수",f"{total}"); m2.metric("성공",f"{wins}"); m3.metric("실패",f"{fails}")
    m4.metric("중립",f"{neuts}"); m5.metric("승률",f"{winrate:.1f}%")

    # 차트
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="가격", increasing_line_color="red", decreasing_line_color="blue", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_up"], mode="lines", line=dict(color="orange", width=1.5, dash="dot"), name="BB 상단"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_low"], mode="lines", line=dict(color="purple", width=1.5, dash="dot"), name="BB 하단"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_mid"], mode="lines", line=dict(color="gray", width=1.2, dash="dot"), name="BB 중앙"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["RSI13"], mode="lines", line=dict(color="green", width=2), name="RSI(13)", yaxis="y2"))
    fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1.5, annotation_text="RSI 70", yref="y2")
    fig.add_hline(y=30, line_dash="dash", line_color="blue", line_width=1.5, annotation_text="RSI 30", yref="y2")
    fig.update_layout(xaxis_rangeslider_visible=False,height=700,
        legend_orientation="h",legend_y=-0.25,
        yaxis=dict(title="가격"),
        yaxis2=dict(overlaying="y", side="right", showgrid=False,title="RSI(13)",range=[0,100]))
    st.plotly_chart(fig, use_container_width=True)

    # ④ 신호 결과 (최신 순) → 중립을 별도 재판정
    st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)
    if total>0:
        tbl=res.copy()
        # ✅ 중립을 성공/실패로 재판정
        def adjust_result(row):
            if row["결과"]=="중립":
                thr= row["성공기준(%)"]
                return "성공" if row["최종수익률(%)"]>=thr*0.6 else "실패"
            return row["결과"]
        tbl["최종판정"]=tbl.apply(adjust_result,axis=1)
        tbl=tbl.sort_values("신호시간",ascending=False).reset_index(drop=True)
        tbl["기준시가"]=tbl["기준시가"].map(lambda v:f"{int(v):,}")
        for col in ["RSI(13)","성공기준(%)","최종수익률(%)","최저수익률(%)","최고수익률(%)"]:
            tbl[col]=tbl[col].map(lambda v:f"{v:.1f}%" if pd.notna(v) else "")
        def color_result(val):
            if val=="성공": return 'color:red; font-weight:600;'
            if val=="실패": return 'color:blue; font-weight:600;'
            return 'color:green; font-weight:600;'
        styled=(tbl.style.applymap(color_result,subset=["최종판정"]))
        st.dataframe(styled,use_container_width=True,hide_index=True)
    else:
        st.info("조건을 만족하는 신호가 없습니다.")

except Exception as e:
    st.error(f"오류: {e}")
