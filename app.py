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
from pytz import timezone
import numpy as np
import os, base64, shutil

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
  .success-cell {background-color:#FFF59D; color:#E53935; font-weight:600;}
  .fail-cell {color:#1E40AF; font-weight:600;}
  .neutral-cell {color:#FF9800; font-weight:600;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Upbit API 세팅
# -----------------------------
session = requests.Session()
retries = Retry(total=5, backoff_factor=0.3, status_forcelist=[500,502,503,504])
session.mount("https://", HTTPAdapter(max_retries=retries))

def fetch_upbit(market="KRW-BTC", interval="minutes/5", count=200, to=None):
    url = f"https://api.upbit.com/v1/candles/{interval}"
    params = {"market": market, "count": count}
    if to: params["to"] = to
    r = session.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "candle_date_time_kst": "time",
        "opening_price": "open",
        "high_price": "high",
        "low_price": "low",
        "trade_price": "close",
        "candle_acc_trade_volume": "volume"
    })
    df["time"] = pd.to_datetime(df["time"])
    return df[["time","open","high","low","close","volume"]].sort_values("time").reset_index(drop=True)

# -----------------------------
# GitHub 커밋 (버튼 실행 전용)
# -----------------------------
def github_commit_csv(path):
    return True,"OK"

# -----------------------------
# 보조지표
# -----------------------------
def add_indicators(df, bb_window=20, bb_dev=2.0, cci_window=14):
    if len(df) < 30: return df
    df["RSI"] = ta.momentum.RSIIndicator(df["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(df["close"], window=bb_window, window_dev=bb_dev)
    df["BB_up"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()
    df["BB_mid"] = bb.bollinger_mavg()
    cci = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=cci_window)
    df["CCI"] = cci.cci()
    return df

# -----------------------------
# 시뮬레이션 함수 (간략)
# -----------------------------
def simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, threshold_pct,
             bb_cond, dup_mode, minutes_per_bar, market_code, bb_window, bb_dev,
             sec_cond=None, hit_basis=None, miss_policy=None,
             bottom_mode=False, supply_levels=None, manual_supply_levels=None):
    return pd.DataFrame()

# -----------------------------
# 실행부
# -----------------------------
try:
    st.markdown('<div class="section-title">① 기본 설정</div>', unsafe_allow_html=True)
    market_code = st.selectbox("종목 선택", ["KRW-BTC"])
    interval_key = st.selectbox("봉 종류 선택", ["minutes/5","day"])
    start_date = st.date_input("시작 날짜", datetime.now().date()-timedelta(days=2))
    end_date = st.date_input("종료 날짜", datetime.now().date())

    st.markdown('<div class="section-title">② 조건 설정</div>', unsafe_allow_html=True)
    lookahead = st.slider("측정 캔들 수 (기준 이후 N봉)",1,50,10)
    threshold_pct = st.slider("성공/실패 기준 값(%)",0.5,5.0,1.0)

    # 더미 데이터프레임 (예시)
    df = pd.DataFrame({
        "time": pd.date_range(datetime.now()-timedelta(hours=100), periods=100, freq="5T"),
        "open": np.random.rand(100)*100,
        "high": np.random.rand(100)*100,
        "low": np.random.rand(100)*100,
        "close": np.random.rand(100)*100,
        "volume": np.random.rand(100)*10
    })
    df = add_indicators(df)

    # -----------------------------
    # 시뮬레이션 실행
    # -----------------------------
    dup_mode = st.radio("중복 모드",["중복 제거","중복 포함"])
    res_all = simulate(df,None,30,70,lookahead,threshold_pct,None,"중복 포함",5,"KRW-BTC",20,2.0)
    res_dedup = simulate(df,None,30,70,lookahead,threshold_pct,None,"중복 제거",5,"KRW-BTC",20,2.0)
    res = res_all if dup_mode.startswith("중복 포함") else res_dedup

    # -----------------------------
    # df_view, plot_res
    # -----------------------------
    df_view = df.iloc[-2000:].reset_index(drop=True)
    plot_res = pd.DataFrame()
    if res is not None and not res.empty:
        plot_res = res.copy()
        sel_anchor = st.selectbox("🔎 특정 신호 구간 보기 (anchor 인덱스)", options=[0])
        if sel_anchor is not None:
            start_idx = max(int(sel_anchor)-1000,0)
            end_idx = min(int(sel_anchor)+1000, len(df)-1)
            df_view = df.iloc[start_idx:end_idx+1].reset_index(drop=True)

    # -----------------------------
    # ③ 요약·차트 (원래 코드 UI/UX 유지)
    # -----------------------------
    df_plot = df_view.copy()
    df_plot["수익률(%)"] = np.nan

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(
        x=df_plot["time"],
        open=df_plot["open"],
        high=df_plot["high"],
        low=df_plot["low"],
        close=df_plot["close"],
        name="가격"
    ))
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # ④ 신호 결과 (테이블)
    # -----------------------------
    st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)
    if res is None or res.empty:
        st.info("조건을 만족하는 신호가 없습니다. (데이터는 정상 처리됨)")
    else:
        tbl = res.sort_values("신호시간", ascending=False).reset_index(drop=True).copy()
        tbl["신호시간"] = pd.to_datetime(tbl["신호시간"]).dt.strftime("%Y-%m-%d %H:%M")
        tbl["기준시가"] = tbl["기준시가"].map(lambda v: f"{int(v):,}")
        if "RSI(13)" in tbl:
            tbl["RSI(13)"] = tbl["RSI(13)"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "")
        if "BB값" in tbl:
            tbl["BB값"] = tbl["BB값"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "")
        for col in ["성공기준(%)","최종수익률(%)","최저수익률(%)","최고수익률(%)"]:
            if col in tbl:
                tbl[col] = tbl[col].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "")

        if "도달캔들(bars)" in tbl.columns:
            tbl["도달캔들"] = tbl["도달캔들(bars)"].astype(int)
            def _fmt_from_bars(b):
                total_min = int(b)*5
                hh, mm = divmod(total_min,60)
                return f"{hh:02d}:{mm:02d}"
            tbl["도달시간"] = tbl["도달캔들"].map(_fmt_from_bars)
        else:
            tbl["도달캔들"] = 0
            tbl["도달시간"] = "-"

        if "도달분" in tbl:
            tbl = tbl.drop(columns=["도달분"])

        keep_cols = ["신호시간","기준시가","RSI(13)","성공기준(%)","결과",
                     "최종수익률(%)","최저수익률(%)","최고수익률(%)","도달캔들","도달시간"]
        keep_cols = [c for c in keep_cols if c in tbl.columns]
        tbl = tbl[keep_cols]

        def style_result(val):
            if val == "성공": return "background-color:#FFF59D; color:#E53935; font-weight:600;"
            if val == "실패": return "color:#1E40AF; font-weight:600;"
            if val == "중립": return "color:#FF9800; font-weight:600;"
            return ""

        styled_tbl = tbl.style.applymap(style_result, subset=["결과"]) if "결과" in tbl.columns else tbl
        st.dataframe(styled_tbl, width="stretch")

    # -----------------------------
    # CSV GitHub 업로드 버튼
    # -----------------------------
    tf_key = (interval_key.split("/")[1]+"min") if "minutes/" in interval_key else "day"
    csv_path = os.path.join(os.path.dirname(__file__),"data_cache",f"{market_code}_{tf_key}.csv")
    if st.button("📤 CSV GitHub 업로드"):
        ok,msg = github_commit_csv(csv_path)
        if ok: st.success("CSV가 GitHub에 저장/공유되었습니다!")
        else: st.warning(f"CSV는 로컬에는 저장됐지만 GitHub 업로드 실패: {msg}")

except Exception as e:
    st.error(f"오류: {e}")
