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
import os

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
  .neutral-cell {color:#374151; font-weight:600;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 세션 상태 초기화
# -----------------------------
if "opt_view" not in st.session_state:
    st.session_state.opt_view = False
if "supply_levels" not in st.session_state:
    st.session_state.supply_levels = {}

# -----------------------------
# HTTP 세션 (재시도 설정)
# -----------------------------
_session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
_session.mount("https://", HTTPAdapter(max_retries=retries))

# -----------------------------
# 업비트 캔들 데이터 (CSV 캐시 + 과거/최신 양방향 보충)
# -----------------------------
def fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars: int = 0):
    """Upbit 캔들 페이징 수집: CSV 캐시 + 과거/최신 양방향 보충."""
    if warmup_bars and warmup_bars > 0:
        start_cutoff = start_dt - timedelta(minutes=warmup_bars * minutes_per_bar)
    else:
        start_cutoff = start_dt

    if "minutes/" in interval_key:
        unit = interval_key.split("/")[1]
        url = f"https://api.upbit.com/v1/candles/minutes/{unit}"
        tf_key = f"{unit}min"
    else:
        url = "https://api.upbit.com/v1/candles/days"
        tf_key = "day"

    # CSV 경로
    data_dir = os.path.join(os.path.dirname(__file__), "data_cache")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, f"{market_code}_{tf_key}.csv")

    # CSV 로드
    if os.path.exists(csv_path):
        df_cache = pd.read_csv(csv_path, parse_dates=["time"])
    else:
        df_cache = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    last_cached_time  = df_cache["time"].max() if not df_cache.empty else None
    first_cached_time = df_cache["time"].min() if not df_cache.empty else None

    all_batches = []
    try:
        # ✅ 과거 보충: 요청 시작일이 CSV보다 과거면 앞 구간 수집
        if first_cached_time is None or start_cutoff < first_cached_time:
            to_time = first_cached_time if first_cached_time is not None else None
            for _ in range(800):  # 들여쓰기 규칙: try 이후 +4, 내부 블록 +4
                params = {"market": market_code, "count": 200}
                if to_time is not None:
                    params["to"] = to_time.strftime("%Y-%m-%d %H:%M:%S")
                r = _session.get(url, params=params, headers={"Accept": "application/json"}, timeout=10)
                r.raise_for_status()
                batch = r.json()
                if not batch:
                    break
                all_batches.extend(batch)
                last_ts = pd.to_datetime(batch[-1]["candle_date_time_kst"])
                if last_ts <= start_cutoff:
                    break
                to_time = last_ts - timedelta(seconds=1)

        # ✅ 최신 보충: 요청 종료일이 CSV보다 최신이면 뒤 구간 수집
        if last_cached_time is None or end_dt > last_cached_time:
            to_time = None
            fetch_start = start_cutoff if last_cached_time is None else last_cached_time + timedelta(seconds=1)
            for _ in range(800):
                params = {"market": market_code, "count": 200}
                if to_time is not None:
                    params["to"] = to_time.strftime("%Y-%m-%d %H:%M:%S")
                r = _session.get(url, params=params, headers={"Accept": "application/json"}, timeout=10)
                r.raise_for_status()
                batch = r.json()
                if not batch:
                    break
                all_batches.extend(batch)
                last_ts = pd.to_datetime(batch[-1]["candle_date_time_kst"])
                if last_ts <= fetch_start:
                    break
                to_time = last_ts - timedelta(seconds=1)

    except Exception:
        # 네트워크 오류 등: 캐시 범위 내에서라도 결과 반환
        return df_cache[(df_cache["time"] >= start_cutoff) & (df_cache["time"] <= end_dt)]

    # ✅ 병합 및 저장
    if all_batches:
        df_new = pd.DataFrame(all_batches).rename(columns={
            "candle_date_time_kst": "time",
            "opening_price": "open",
            "high_price": "high",
            "low_price": "low",
            "trade_price": "close",
            "candle_acc_trade_volume": "volume",
        })
        df_new["time"] = pd.to_datetime(df_new["time"])
        df_new = df_new[["time", "open", "high", "low", "close", "volume"]]

        df_all = pd.concat([df_cache, df_new], ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

        df_all.to_csv(csv_path, index=False)
    else:
        df_all = df_cache

    # ✅ 요청 구간만 필터링하여 반환
    return df_all[(df_all["time"] >= start_cutoff) & (df_all["time"] <= end_dt)]

# -----------------------------
# 이하 기존 코드 (UI/UX, 요약, 차트, 신호결과) 절대 변경 금지
# -----------------------------
try:
    # ... (사용자 최신 app.py 나머지 전체 실행 코드 유지)
    pass
except Exception as e:
    st.error(f"오류 발생: {e}")
