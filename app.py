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
    rsi_side = st.selectbox("RSI 조건", ["RSI ≤ 30 (급락)", "RSI ≥ 70 (급등)"], index=0)

c7, _, _ = st.columns(3)
with c7:
    bb_cond = st.selectbox(
        "볼린저밴드 조건",
        ["없음","하한선 하향돌파","하한선 상향돌파","상한선 하향돌파","상한선 상향돌파","하한선 중앙돌파","상한선 중앙돌파"],
        index=0,
    )

interval_key, minutes_per_bar = TF_MAP[tf_label]

# -----------------------------
# 데이터 수집 (세션+재시도+상한)
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
        url = f"https://api.upbit.com/v1/candles/{interval_key}"

    calls_est = estimate_calls(start_dt, end_dt, minutes_per_bar)
    max_calls = min(calls_est+2, 60)

    all_data, to_time = [], end_dt
    progress = st.progress(0.0)
    try:
        for done in range(max_calls):
            params = {"market": market_code, "count": 200, "to": to_time.strftime("%Y-%m-%d %H:%M:%S")}
            r = _session.get(url, params=params, headers={"Accept":"application/json"}, timeout=10)
            r.raise_for_status()
            batch = r.json()
            if not batch: break
            all_data.extend(batch)
            last_ts = pd.to_datetime(batch[-1]["candle_date_time_kst"])
            if last_ts <= start_dt: break
            to_time = last_ts - timedelta(seconds=1)
            progress.progress(min(1.0,(done+1)/max(1,calls_est)))
    finally:
        progress.empty()

    if not all_data: return pd.DataFrame()
    df = pd.DataFrame(all_data).rename(columns={
        "candle_date_time_kst":"time","opening_price":"open","high_price":"high",
        "low_price":"low","trade_price":"close","candle_acc_trade_volume":"volume"})
    df["time"] = pd.to_datetime(df["time"])
    return df[["time","open","high","low","close","volume"]].sort_values("time").reset_index(drop=True)

# -----------------------------
# 지표
# -----------------------------
def add_indicators(df):
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=30, window_dev=2)
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
    if "≤" in rsi_side:
        sig_idx=df.index[(df["RSI13"].notna())&(df["RSI13"]<=30)].tolist()
    else:
        sig_idx=df.index[(df["RSI13"].notna())&(df["RSI13"]>=70)].tolist()
    for i in sig_idx:
        end=i+lookahead
        if end>=n: continue
        if bb_cond!="없음":
            px=float(df.at[i,"close"]); up,lo,mid=df.at[i,"BB_up"],df.at[i,"BB_low"],df.at[i,"BB_mid"]
            ok=True
            if bb_cond=="하한선 하향돌파": ok=pd.notna(lo) and px<lo
            elif bb_cond=="하한선 상향돌파": ok=pd.notna(lo) and px>lo
            elif bb_cond=="상한선 하향돌파": ok=pd.notna(up) and px<up
            elif bb_cond=="상한선 상향돌파": ok=pd.notna(up) and px>up
            elif bb_cond=="하한선 중앙돌파": ok=pd.notna(lo) and pd.notna(mid) and lo<px<mid
            elif bb_cond=="상한선 중앙돌파": ok=pd.notna(up) and pd.notna(mid) and mid<px<up
            if not ok: continue
        base_price=float(df.at[i,"low"]); closes=df.loc[i+1:end,["time","close"]]
        if closes.empty: continue
        target_up, target_down=base_price*(1+thr/100), base_price*(1-thr/100)
        hit_up, hit_down=closes[closes["close"]>=target_up], closes[closes["close"]<=target_down]
        result="중립"; reach_minutes=None
        if not hit_up.empty and not hit_down.empty:
            if hit_up.iloc[0]["time"] < hit_down.iloc[0]["time"]:
                reach_minutes = int((hit_up.iloc[0]["time"]-df.at[i,"time"]).total_seconds()//60)
                result="성공"
            else:
                result="실패"
        elif not hit_up.empty:
            reach_minutes = int((hit_up.iloc[0]["time"]-df.at[i,"time"]).total_seconds()//60)
            result="성공"
        elif not hit_down.empty:
            result="실패"
        else:
            final_price=closes.iloc[-1]["close"]
            result="중립" if final_price>base_price else "실패"
        final_ret=(closes.iloc[-1]["close"]/base_price-1)*100.0
        min_ret=(closes["close"].min()/base_price-1)*100.0
        max_ret=(closes["close"].max()/base_price-1)*100.0
        res.append({
            "신호시간": df.at[i,"time"],
            "기준시가": int(round(base_price)),
            "RSI(13)": round(float(df.at[i,"RSI13"]),1) if pd.notna(df.at[i,"RSI13"]) else None,
            "성공기준(%)": round(thr,1),
            "결과": result,
            "도달분": reach_minutes,
            "최종수익률(%)": round(final_ret,1),
            "최저수익률(%)": round(min_ret,1),
            "최고수익률(%)": round(max_ret,1)
        })
    cols=["신호시간","기준시가","RSI(13)","성공기준(%)","결과","도달분","최종수익률(%)","최저수익률(%)","최고수익률(%)"]
    out=pd.DataFrame(res,columns=cols)
    if not out.empty and dedup_mode.startswith("중복 제거"):
        out=out.loc[out["결과"].shift()!=out["결과"]]
    return out

# -----------------------------
# 실행
# -----------------------------
try:
    if start_date>end_date:
        st.error("시작 날짜가 종료 날짜보다 이후입니다.")
        st.stop()
    start_dt=datetime.combine(start_date,datetime.min.time())
    end_dt=datetime.combine(end_date,datetime.max.time())
    df=fetch_upbit_paged(market_code,interval_key,start_dt,end_dt,minutes_per_bar)
    if df.empty:
        st.error("데이터가 없습니다.")
        st.stop()
    df=add_indicators(df)
    res_all=simulate(df,rsi_side,lookahead,threshold_pct,bb_cond,"중복 포함 (연속 신호 모두)")
    res_dedup=simulate(df,rsi_side,lookahead,threshold_pct,bb_cond,"중복 제거 (연속 동일 결과 1개)")

    st.markdown('<div class="section-title">③ 요약 & 차트</div>',unsafe_allow_html=True)
    def _summarize(df_in):
        if df_in is None or df_in.empty:
            return 0,0,0,0,0.0,0.0,0.0,0.0
        total=len(df_in)
        succ=int((df_in["결과"]=="성공").sum())
        fail=int((df_in["결과"]=="실패").sum())
        neu=int((df_in["결과"]=="중립").sum())
        win=succ/total*100.0
        range_sum=float((df_in["최고수익률(%)"]-df_in["최저수익률(%)"]).sum())
        final_succ=float(df_in.loc[df_in["결과"]=="성공","최종수익률(%)"].sum())
        final_fail=float(df_in.loc[df_in["결과"]=="실패","최종수익률(%)"].sum())
        return total,succ,fail,neu,win,range_sum,final_succ,final_fail

    for label,data in [("중복 포함 (연속 신호 모두)",res_all),("중복 제거 (연속 동일 결과 1개)",res_dedup)]:
        total,succ,fail,neu,win,range_sum,final_succ,final_fail=_summarize(data)
        st.markdown(f"**{label}**")
        c1,c2,c3,c4,c5,c6,c7=st.columns(7)
        c1.metric("신호 수",f"{total}")
        c2.metric("성공",f"{succ}")
        c3.metric("실패",f"{fail}")
        c4.metric("중립",f"{neu}")
        c5.metric("승률",f"{win:.1f}%")
        c6.metric("총 변동폭 합(%)",f"{range_sum:.1f}%")
        c7.metric("최종수익률 합계",f"{(final_succ+final_fail):.1f}%")
        st.markdown("---")

    res=res_all if dup_mode.startswith("중복 포함") else res_dedup

    # 차트
    fig=make_subplots(rows=1,cols=1)
    fig.add_trace(go.Candlestick(
        x=df["time"],open=df["open"],high=df["high"],low=df["low"],close=df["close"],name="가격",
        increasing_line_color="#E63946",decreasing_line_color="#457B9D",line=dict(width=1.2)
    ))
    fig.add_trace(go.Scatter(x=df["time"],y=df["BB_up"],mode="lines",line=dict(color="#FFB703",width=1.5),name="BB 상단"))
    fig.add_trace(go.Scatter(x=df["time"],y=df["BB_low"],mode="lines",line=dict(color="#219EBC",width=1.5),name="BB 하단"))
    fig.add_trace(go.Scatter(x=df["time"],y=df["BB_mid"],mode="lines",line=dict(color="#8D99AE",width=1.2,dash="dot"),name="BB 중앙"))

    if not res.empty:
        for _label,_color in [("성공","#06D6A0"),("실패","#EF476F"),("중립","#FFD166")]:
            sub=res[res["결과"]==_label]
            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub["신호시간"], y=sub["기준시가"],
                    mode="markers", name=f"신호 ({_label})",
                    marker=dict(size=10,color=_color,symbol="circle",line=dict(width=1,color="black"))
                ))

    # === RSI(13) 네온 + 점선 효과 적용 ===
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["RSI13"], mode="lines",
        line=dict(color="rgba(42,157,143,0.3)", width=6),
        opacity=0.6, name="RSI Glow", yaxis="y2", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["RSI13"], mode="lines",
        line=dict(color="#2A9D8F", width=2.5, dash="dot"),
        opacity=1, name="RSI(13)", yaxis="y2"
    ))

    fig.add_hline(y=70,line_dash="dash",line_color="#E63946",line_width=1.2,
                  annotation_text="RSI 70",annotation_position="top left",yref="y2")
    fig.add_hline(y=30,line_dash="dash",line_color="#457B9D",line_width=1.2,
                  annotation_text="RSI 30",annotation_position="bottom left",yref="y2")

    fig.update_layout(
        title=f"{market_label.split(' — ')[0]} · {tf_label} · RSI(13) + BB 시뮬레이션",
        xaxis_rangeslider_visible=False,height=600,autosize=False,
        legend_orientation="h",legend_y=1.05,
        margin=dict(l=60,r=40,t=60,b=40),
        yaxis=dict(title="가격"),
        yaxis2=dict(overlaying="y",side="right",showgrid=False,title="RSI(13)",range=[0,100])
    )
    st.plotly_chart(fig,use_container_width=True)

    # -----------------------------
    # ④ 신호 결과 (최신 순)
    # -----------------------------
    st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>',unsafe_allow_html=True)
    if not res.empty:
        tbl=res.sort_values("신호시간",ascending=False).reset_index(drop=True).copy()
        tbl["신호시간"]=pd.to_datetime(tbl["신호시간"]).dt.strftime("%Y-%m-%d %H:%M")
        tbl["기준시가"]=tbl["기준시가"].map(lambda v:f"{int(v):,}")
        if "RSI(13)" in tbl: tbl["RSI(13)"]=tbl["RSI(13)"].map(lambda v:f"{v:.1f}" if pd.notna(v) else "")
        for col in ["성공기준(%)","최종수익률(%)","최저수익률(%)","최고수익률(%)"]:
            if col in tbl: tbl[col]=tbl[col].map(lambda v:f"{v:.1f}%" if pd.notna(v) else "")
        def fmt_hhmm(m):
            if pd.isna(m): return "-"
            m=int(m); h,mm=divmod(m,60)
            return f"{h:02d}:{mm:02d}"
        tbl["도달시간"]=tbl["도달분"].map(fmt_hhmm) if "도달분" in tbl else "-"
        if "도달분" in tbl: tbl=tbl.drop(columns=["도달분"])
        def color_result(val):
            if val=="성공": return "color:red; font-weight:600; background-color:#FFFACD;"
            if val=="실패": return "color:blue;"
            return "color:green; font-weight:600;"
        styled=tbl.style.applymap(color_result,subset=["결과"])
        st.dataframe(styled,use_container_width=True,hide_index=True)
    else:
        st.info("조건을 만족하는 신호가 없습니다.")

except Exception as e:
    st.error(f"오류: {e}")
