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
    rsi_side = st.selectbox("RSI 조건", ["RSI ≤ 30 (급락)", "RSI ≥ 70 (급등)"], index=0)

# 볼린저밴드 조건
c7, _, _ = st.columns(3)
with c7:
    bb_cond = st.selectbox(
        "볼린저밴드 조건",
        ["없음","하한선 하향돌파","하한선 상향돌파","상한선 하향돌파","상한선 상향돌파","하한선 중앙돌파","상한선 중앙돌파"],
        index=0,
    )

# 안전 장치(세션 보강: 위젯 재실행 시 NameError 방지)
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
# 데이터 수집 (생략: 원본 코드 유지)
# -----------------------------
# ... [중간 부분 동일, simulate 함수까지 기존 코드 유지] ...

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

    df=add_indicators(df)

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
        c1.metric("신호 수",f"{total}"); c2.metric("성공",f"{succ}"); c3.metric("실패",f"{fail}")
        c4.metric("중립",f"{neu}"); c5.metric("승률",f"{win:.1f}%"); c6.metric("총 변동폭 합(%)",f"{range_sum:.1f}%")

        total_final = final_succ + final_fail
        color = "red" if total_final > 0 else "blue" if total_final < 0 else "black"
        c7.markdown(f"<div style='font-weight:600; color:{color};'>최종수익률 합계: {total_final:.1f}%</div>", unsafe_allow_html=True)
        st.markdown("---")

    res = res_all if dup_mode.startswith("중복 포함") else res_dedup

    # ---- 차트 ----
    fig=make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                 name="가격", increasing_line_color="red", decreasing_line_color="blue",
                                 line=dict(width=1.2)))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_up"],  mode="lines", line=dict(color="#FFB703", width=1.5), name="BB 상단"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_low"], mode="lines", line=dict(color="#219EBC", width=1.5), name="BB 하단"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_mid"], mode="lines", line=dict(color="#8D99AE", width=1.2, dash="dot"), name="BB 중앙"))

    if not res.empty:
        for _label,_color in [("성공","red"),("실패","blue"),("중립","#FFD166")]:
            sub=res[res["결과"]==_label]
            if not sub.empty:
                fig.add_trace(go.Scatter(x=sub["신호시간"], y=sub["기준시가"], mode="markers",
                                         name=f"신호 ({_label})",
                                         marker=dict(size=10, color=_color, symbol="circle", line=dict(width=1, color="black"))))

    # RSI(13) 네온 + 점선
    fig.add_trace(go.Scatter(x=df["time"], y=df["RSI13"], mode="lines",
                             line=dict(color="rgba(42,157,143,0.3)", width=6),
                             opacity=0.6, name="RSI Glow", yaxis="y2", showlegend=False))
    fig.add_trace(go.Scatter(x=df["time"], y=df["RSI13"], mode="lines",
                             line=dict(color="#2A9D8F", width=2.5, dash="dot"),
                             opacity=1, name="RSI(13)", yaxis="y2"))

    fig.add_hline(y=70, line_dash="dash", line_color="#E63946", line_width=1.2,
                  annotation_text="RSI 70", annotation_position="top left", yref="y2")
    fig.add_hline(y=30, line_dash="dash", line_color="#457B9D", line_width=1.2,
                  annotation_text="RSI 30", annotation_position="bottom left", yref="y2")

    fig.update_layout(title=f"{market_label.split(' — ')[0]} · {tf_label} · RSI(13) + BB 시뮬레이션",
                      xaxis_rangeslider_visible=False, height=600, autosize=False,
                      legend_orientation="h", legend_y=1.05,
                      margin=dict(l=60, r=40, t=60, b=40),
                      yaxis=dict(title="가격"),
                      yaxis2=dict(overlaying="y", side="right", showgrid=False, title="RSI(13)", range=[0,100]))
    st.plotly_chart(fig, use_container_width=True)

    # ---- 신호 결과 (최신 순) ----
    st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)
    if not res.empty:
        tbl=res.sort_values("신호시간", ascending=False).reset_index(drop=True).copy()
        tbl["신호시간"]=pd.to_datetime(tbl["신호시간"]).dt.strftime("%Y-%m-%d %H:%M")  # 초 제거
        tbl["기준시가"]=tbl["기준시가"].map(lambda v: f"{int(v):,}")
        if "RSI(13)" in tbl: tbl["RSI(13)"]=tbl["RSI(13)"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "")
        for col in ["성공기준(%)","최종수익률(%)","최저수익률(%)","최고수익률(%)"]:
            if col in tbl: tbl[col]=tbl[col].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "")

        def fmt_hhmm(m):
            if pd.isna(m): return "-"
            m=int(m); h,mm=divmod(m,60)
            return f"{h:02d}:{mm:02d}"

        tbl["도달시간"]=tbl["도달분"].map(fmt_hhmm) if "도달분" in tbl else "-"
        if "도달분" in tbl: tbl=tbl.drop(columns=["도달분"])

        def color_result(v):
            if v=="성공": return "color:red; font-weight:600; background-color:#FFFACD;"
            if v=="실패": return "color:blue;"
            return "color:green; font-weight:600;"

        styled=tbl.style.applymap(color_result, subset=["결과"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("조건을 만족하는 신호가 없습니다.")

except Exception as e:
    st.error(f"오류: {e}")
