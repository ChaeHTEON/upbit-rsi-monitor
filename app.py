# app.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import ta
from datetime import datetime, timedelta
import numpy as np
from pytz import timezone

# -----------------------------
# 페이지/스타일
# -----------------------------
st.set_page_config(page_title="Upbit RSI(13) + Bollinger Band 시뮬레이터", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 0.8rem; padding-bottom: 0.8rem; max-width: 1150px;}
  .stMetric {text-align:center;}
  .section-title {font-size:1.05rem; font-weight:700; margin: 0.6rem 0 0.2rem;}
  .hint {color:#6b7280;}
  .success-cell {background-color:#FFF59D; color:#E53935; font-weight:600;}
  .fail-cell {color:#1E40AF; font-weight:600;}
  .neutral-cell {color:#FF9800; font-weight:600;}
  table {border-collapse:collapse; width:100%;}
  th, td {border:1px solid #ddd; padding:6px; text-align:center;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Upbit RSI(13) + Bollinger Band 시뮬레이터")
st.markdown("<div style='margin-bottom:10px; color:gray;'>※ 차트 내 점선은 신호 흐름선, 성공 시 도달 지점에 ⭐ 별표 표시</div>", unsafe_allow_html=True)

# -----------------------------
# 업비트 마켓 로드
# -----------------------------
@st.cache_data(ttl=3600)
def get_upbit_krw_markets():
    url = "https://api.upbit.com/v1/market/all"
    try:
        r = requests.get(url, params={"isDetails": "false"}, timeout=8)
        r.raise_for_status()
        items = r.json()
        rows = []
        for it in items:
            mk = it.get("market", "")
            if mk.startswith("KRW-"):
                sym = mk[4:]
                label = f'{it.get("korean_name","")} ({sym}) — {mk}'
                rows.append((label, mk))
        rows.sort(key=lambda x: (x[1] != "KRW-BTC", x[1]))
        return rows
    except Exception:
        return [("비트코인 (BTC) — KRW-BTC", "KRW-BTC")]

MARKET_LIST = get_upbit_krw_markets()
default_idx = next((i for i, (_, code) in enumerate(MARKET_LIST) if code == "KRW-BTC"), 0)

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
# 신호 중복 처리
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
    KST = timezone("Asia/Seoul")
    today_kst = datetime.now(KST).date()
    default_start = today_kst - timedelta(days=1)
    start_date = st.date_input("시작 날짜", value=default_start)
    end_date = st.date_input("종료 날짜", value=today_kst)

interval_key, minutes_per_bar = TF_MAP[tf_label]
st.markdown("---")

# -----------------------------
# 조건 설정
# -----------------------------
st.markdown('<div class="section-title">② 조건 설정</div>', unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
with c4:
    lookahead = st.slider("측정 캔들 수 (기준 이후 N봉)", 1, 60, 10)
with c5:
    threshold_pct = st.slider("성공/실패 기준 값(%)", 0.1, 5.0, 1.0, step=0.1)
with c6:
    r1, r2, r3 = st.columns(3)
    with r1:
        rsi_mode = st.selectbox("RSI 조건", ["없음", "현재(과매도/과매수 중 하나)", "과매도 기준", "과매수 기준"], index=0)
    with r2:
        rsi_low = st.slider("과매도 RSI 기준", 0, 100, 30, step=1)
    with r3:
        rsi_high = st.slider("과매수 RSI 기준", 0, 100, 70, step=1)

c7, c8, c9 = st.columns(3)
with c7:
    bb_cond = st.selectbox("볼린저밴드 조건", ["없음", "상한선", "중앙선", "하한선"], index=0)
with c8:
    bb_window = st.number_input("BB 기간", min_value=5, max_value=100, value=30, step=1)
with c9:
    bb_dev = st.number_input("BB 승수", min_value=1.0, max_value=4.0, value=2.0, step=0.1)

st.markdown('<div class="hint">2차 조건: 선택한 조건만 적용 (없음/양봉 2개/BB 기반)</div>', unsafe_allow_html=True)
sec_cond = st.selectbox("2차 조건 선택", ["없음", "양봉 2개 연속 상승", "BB 기반 첫 양봉 진입"], index=0)
bb_strength = st.slider("BB 양봉 진입 강도 (%)", 10, 90, 50, step=5)
st.markdown("---")

# -----------------------------
# 데이터 수집
# -----------------------------
_session = requests.Session()
_retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
_session.mount("https://", HTTPAdapter(max_retries=_retries))

@st.cache_data(ttl=120, show_spinner=False)
def fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar):
    if "minutes/" in interval_key:
        unit = interval_key.split("/")[1]
        url = f"https://api.upbit.com/v1/candles/minutes/{unit}"
    else:
        url = "https://api.upbit.com/v1/candles/days"
    all_data, to_time = [], end_dt
    try:
        for _ in range(60):
            params = {"market": market_code, "count": 200, "to": to_time.strftime("%Y-%m-%d %H:%M:%S")}
            r = _session.get(url, params=params, headers={"Accept": "application/json"}, timeout=10)
            r.raise_for_status()
            batch = r.json()
            if not batch: break
            all_data.extend(batch)
            last_ts = pd.to_datetime(batch[-1]["candle_date_time_kst"])
            if last_ts <= start_dt: break
            to_time = last_ts - timedelta(seconds=1)
    except Exception:
        return pd.DataFrame()
    if not all_data: return pd.DataFrame()
    df = pd.DataFrame(all_data).rename(columns={
        "candle_date_time_kst": "time", "opening_price": "open", "high_price": "high",
        "low_price": "low", "trade_price": "close", "candle_acc_trade_volume": "volume"})
    df["time"] = pd.to_datetime(df["time"])
    df = df[["time", "open", "high", "low", "close", "volume"]].sort_values("time").reset_index(drop=True)
    return df[(df["time"] >= start_dt) & (df["time"] <= end_dt)]

# -----------------------------
# 지표
# -----------------------------
def add_indicators(df, bb_window, bb_dev):
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=bb_window, window_dev=bb_dev)
    out["BB_up"] = bb.bollinger_hband().fillna(method="bfill").fillna(method="ffill")
    out["BB_low"] = bb.bollinger_lband().fillna(method="bfill").fillna(method="ffill")
    out["BB_mid"] = bb.bollinger_mavg().fillna(method="bfill").fillna(method="ffill")
    return out

# -----------------------------
# 시뮬레이션
# -----------------------------
def simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, thr_pct, bb_cond, dedup_mode, sec_cond="없음", bb_strength=50):
    res, n = [], len(df)
    thr = float(thr_pct)

    def is_bull(k): return float(df.at[k, "close"]) > float(df.at[k, "open"])
    def b1_pass(j):
        if not is_bull(j): return False
        if bb_cond == "상한선": ref = float(df.at[j, "BB_up"])
        elif bb_cond == "중앙선": ref = float(df.at[j, "BB_mid"])
        elif bb_cond == "하한선": ref = float(df.at[j, "BB_low"])
        else: return False
        if pd.isna(ref): return False
        o, c = float(df.at[j, "open"]), float(df.at[j, "close"])
        return (c >= o + (bb_strength/100)*(ref - o)) if (o < ref) else (c >= ref)

    i = 0
    while i < n:
        signal_time, base_price = df.at[i, "time"], float(df.at[i, "close"])
        anchor_idx = i

        if sec_cond == "BB 기반 첫 양봉 진입":
            B1_idx, B1_close = None, None
            for j in range(i+1, n):
                if b1_pass(j): B1_idx, B1_close = j, float(df.at[j, "close"]); break
            if B1_idx is None: i+=1; continue
            bull_cnt, B3_idx = 0, None
            for j in range(B1_idx+1, n):
                if is_bull(j): bull_cnt+=1
                if bull_cnt==2: B3_idx=j; break
            if B3_idx is None: i+=1; continue
            T_idx = None
            for j in range(B3_idx+1, n):
                if float(df.at[j,"close"])>=B1_close: T_idx=j; break
            if T_idx is None: i+=1; continue
            anchor_idx, signal_time, base_price = T_idx, df.at[T_idx,"time"], float(df.at[T_idx,"close"])

        end_idx = anchor_idx + lookahead
        if end_idx >= n: break
        window = df.iloc[anchor_idx+1:end_idx+1]
        end_time, end_close = df.at[end_idx,"time"], float(df.at[end_idx,"close"])
        final_ret = (end_close/base_price-1)*100
        min_ret = (window["close"].min()/base_price-1)*100 if not window.empty else 0
        max_ret = (window["close"].max()/base_price-1)*100 if not window.empty else 0

        result, reach_min = "중립", None
        target_price = base_price*(1+thr/100)
        hit_rows = window[window["close"]>=target_price]
        if not hit_rows.empty:
            hit_time = hit_rows.iloc[0]["time"]
            reach_min = int((hit_time-signal_time).total_seconds()//60)
            end_time, end_close, final_ret, result = hit_time, target_price, thr, "성공"
        elif final_ret <= -thr: result="실패"

        res.append({"신호시간":signal_time,"종료시간":end_time,"기준시가":int(base_price),"종료가":end_close,
                    "성공기준(%)":thr,"결과":result,"도달분":reach_min,"최종수익률(%)":round(final_ret,2),
                    "최저수익률(%)":round(min_ret,2),"최고수익률(%)":round(max_ret,2)})
        i = end_idx+1 if dedup_mode.startswith("중복 제거") else i+1
    return pd.DataFrame(res)

# -----------------------------
# 실행
# -----------------------------
try:
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())
    df = fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar)
    if df.empty:
        st.error("데이터가 없습니다. (조회된 캔들이 없음)")
        # 빈 차트 기본 뼈대
        fig = make_subplots(rows=1, cols=1)
        fig.update_layout(
            title=f"{market_label.split(' — ')[0]} · {tf_label} · RSI(13)+BB 시뮬",
            dragmode="zoom", xaxis_rangeslider_visible=False, height=600
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        df = add_indicators(df, bb_window, bb_dev)
        res = simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, threshold_pct, bb_cond, dup_mode, sec_cond, bb_strength)

        # 요약
        total, succ, fail, neu = len(res), (res["결과"]=="성공").sum(), (res["결과"]=="실패").sum(), (res["결과"]=="중립").sum()
        win_rate = succ/total*100 if total>0 else 0
        st.markdown('<div class="section-title">③ 요약</div>', unsafe_allow_html=True)
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("총 신호",total); m2.metric("성공",succ); m3.metric("실패",fail); m4.metric("중립",neu); m5.metric("승률",f"{win_rate:.1f}%")

        # 차트
        st.markdown('<div class="section-title">④ 차트</div>', unsafe_allow_html=True)
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Candlestick(x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                     name="가격", increasing_line_color="red", decreasing_line_color="blue", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=df["time"], y=df["BB_up"], mode="lines", line=dict(color="orange",width=1), name="BB 상단"))
        fig.add_trace(go.Scatter(x=df["time"], y=df["BB_low"], mode="lines", line=dict(color="skyblue",width=1), name="BB 하단"))
        fig.add_trace(go.Scatter(x=df["time"], y=df["BB_mid"], mode="lines", line=dict(color="gray",dash="dot",width=1), name="BB 중앙"))

        if not res.empty:
            for _,row in res.iterrows():
                color = "red" if row["결과"]=="성공" else "blue" if row["결과"]=="실패" else "orange"
                fig.add_trace(go.Scatter(x=[row["신호시간"]], y=[row["기준시가"]], mode="markers",
                                         marker=dict(size=9,color=color,symbol="circle",line=dict(width=1,color="black")),
                                         name=f"신호({row['결과']})"))

        fig.update_layout(title=f"{market_label.split(' — ')[0]} · {tf_label} · RSI(13)+BB 시뮬",
                          dragmode="zoom",xaxis_rangeslider_visible=False,height=600)
        st.plotly_chart(fig,use_container_width=True)

        # 결과 테이블
        st.markdown('<div class="section-title">⑤ 결과 테이블</div>', unsafe_allow_html=True)
        if not res.empty:
            tbl = res.copy()
            tbl["신호시간"] = pd.to_datetime(tbl["신호시간"]).dt.strftime("%Y-%m-%d %H:%M")
            tbl["종료시간"] = pd.to_datetime(tbl["종료시간"]).dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(tbl,use_container_width=True)
        else:
            st.info("조건을 만족하는 신호가 없습니다. (데이터는 불러왔지만 조건 불충족)")

except Exception as e:
    st.error(f"예상치 못한 오류 발생: {str(e)}")
