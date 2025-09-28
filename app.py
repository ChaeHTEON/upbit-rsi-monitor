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
  table {border-collapse:collapse; width:100%;}
  th, td {border:1px solid #ddd; padding:6px; text-align:center;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# (상단 UI, 설정, GitHub 저장 연동 등은 네 최신본과 동일 — 그대로 유지)
# -----------------------------
# ... [중략: 네 최신 app (3).py UI/UX, 설정, 조건, 지표 부분 그대로]

# -----------------------------
# 데이터 수집/지표/시뮬레이션 함수
# -----------------------------
_session = requests.Session()
_retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
_session.mount("https://", HTTPAdapter(max_retries=_retries))

def fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars: int = 0):
    """Upbit 캔들 수집 최적화
    - CSV가 요청 범위를 전부 커버하면 API 호출 스킵
    - 부족한 앞/뒤 구간만 최소한 API 보충
    - 반환은 항상 오름차순으로 정렬된 [start_cutoff ~ end_dt] 범위
    """
    import shutil

    # 시작 컷오프 (워밍업 적용)
    if warmup_bars and warmup_bars > 0:
        start_cutoff = start_dt - timedelta(minutes=warmup_bars * minutes_per_bar)
    else:
        start_cutoff = start_dt

    # 엔드포인트
    if "minutes/" in interval_key:
        unit = interval_key.split("/")[1]
        url = f"https://api.upbit.com/v1/candles/minutes/{unit}"
        tf_key = f"{unit}min"
    else:
        url = "https://api.upbit.com/v1/candles/days"
        tf_key = "day"

    # CSV 로드
    data_dir = os.path.join(os.path.dirname(__file__), "data_cache")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, f"{market_code}_{tf_key}.csv")

    if os.path.exists(csv_path):
        df_cache = pd.read_csv(csv_path, parse_dates=["time"])
    else:
        df_cache = pd.DataFrame(columns=["time","open","high","low","close","volume"])

    # CSV가 요청 구간 전체를 커버 → API 호출 스킵
    if not df_cache.empty:
        first_cached = df_cache["time"].min()
        last_cached  = df_cache["time"].max()
        if first_cached <= start_cutoff and last_cached >= end_dt:
            return df_cache[(df_cache["time"] >= start_cutoff) & (df_cache["time"] <= end_dt)].sort_values("time").reset_index(drop=True)

    df_all = df_cache.copy()

    def _pull_until(limit_to):
        out, to_time = [], limit_to
        for _ in range(500):
            params = {"market": market_code, "count": 200}
            if to_time is not None:
                params["to"] = to_time.strftime("%Y-%m-%d %H:%M:%S")
            r = _session.get(url, params=params, headers={"Accept": "application/json"}, timeout=10)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            out.extend(batch)
            last_kst = pd.to_datetime(batch[-1]["candle_date_time_kst"])
            if last_kst <= start_cutoff:
                break
            to_time = last_kst - timedelta(seconds=1)
        if not out:
            return pd.DataFrame(columns=["time","open","high","low","close","volume"])
        df_new = pd.DataFrame(out).rename(columns={
            "candle_date_time_kst": "time",
            "opening_price": "open",
            "high_price": "high",
            "low_price": "low",
            "trade_price": "close",
            "candle_acc_trade_volume": "volume",
        })
        df_new["time"] = pd.to_datetime(df_new["time"])
        return df_new[["time","open","high","low","close","volume"]]

    need_front = df_cache.empty or (start_cutoff < (df_cache["time"].min() if not df_cache.empty else end_dt))
    need_back  = df_cache.empty or (end_dt      > (df_cache["time"].max() if not df_cache.empty else start_cutoff))

    if need_front:
        try:
            df_front = _pull_until(None)
            if not df_front.empty:
                df_all = pd.concat([df_all, df_front], ignore_index=True)
        except Exception:
            pass

    if need_back:
        try:
            df_back = _pull_until(end_dt)
            if not df_back.empty:
                df_all = pd.concat([df_all, df_back], ignore_index=True)
        except Exception:
            pass

    if not df_all.empty:
        df_all = df_all.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
        tmp_path = csv_path + ".tmp"
        df_all.to_csv(tmp_path, index=False)
        shutil.move(tmp_path, csv_path)

    return df_all[(df_all["time"] >= start_cutoff) & (df_all["time"] <= end_dt)].sort_values("time").reset_index(drop=True)

def add_indicators(df, bb_window, bb_dev, cci_window):
    df = df.copy()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(df["close"], window=bb_window, window_dev=bb_dev)
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    cci = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=cci_window)
    df["cci"] = cci.cci()
    return df

# -----------------------------
# 실행 (네 최신본과 동일하게 try~except까지 그대로)
# -----------------------------
try:
    # ... (네 최신 app (3).py 실행부 전체 유지)
    pass
except Exception as e:
    st.error(f"오류: {e}")
