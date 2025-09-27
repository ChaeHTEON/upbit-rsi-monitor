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

st.title("📊 코인 시뮬레이션")

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
                rows.append((f"{it.get('korean_name','')} ({mk[4:]}) — {mk}", mk))
        return sorted(rows, key=lambda x: (x[1] != "KRW-BTC", x[1]))
    except Exception:
        return [("비트코인 (BTC) — KRW-BTC", "KRW-BTC")]

MARKET_LIST = get_upbit_krw_markets()
default_idx = next((i for i, (_, code) in enumerate(MARKET_LIST) if code=="KRW-BTC"), 0)

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
# ① 기본 설정
# -----------------------------
st.markdown('<div class="section-title">① 기본 설정</div>', unsafe_allow_html=True)

KST = timezone("Asia/Seoul")
now_kst = datetime.now(KST)
now = now_kst.replace(tzinfo=None)  # tz-naive (KST 기준 값)

default_start_dt = now - timedelta(hours=24)
default_end_dt = now

c1, c2, c3, c4 = st.columns(4)
with c1:
    market_label, market_code = st.selectbox("종목 선택", MARKET_LIST, index=default_idx, format_func=lambda x: x[0])
with c2:
    tf_label = st.selectbox("봉 종류 선택", list(TF_MAP.keys()), index=2)
with c3:
    start_date = st.date_input("시작 날짜", value=default_start_dt.date())
    start_time = st.time_input("시작 시간", value=default_start_dt.time())
with c4:
    end_date = st.date_input("종료 날짜", value=default_end_dt.date())
    end_time = st.time_input("종료 시간", value=default_end_dt.time())

interval_key, minutes_per_bar = TF_MAP[tf_label]

# ✅ 시작/종료 datetime 결합
start_dt = datetime.combine(start_date, start_time)
end_dt   = datetime.combine(end_date, end_time)

today = now.date()
# ✅ 종료 보정
if interval_key == "days" and end_date >= today:
    st.info("일봉은 당일 데이터가 제공되지 않습니다. 전일까지로 보정합니다.")
    end_dt = datetime.combine(today - timedelta(days=1), datetime.max.time())
elif end_dt > now:
    end_dt = now

# 경고 컨테이너
warn_box = st.empty()
st.markdown("---")

# -----------------------------
# 이후 (조건 설정, 데이터 수집, 차트, 신호결과 등)
# -----------------------------
# ⚠️ 이 아래는 기존 초기 코드의 UI/UX 및 시뮬레이션 로직을 그대로 사용
#     (RSI, BB, CCI, simulate(), chart_box, signal 결과 출력 등 원형 그대로 유지)
# -----------------------------
try:
    # 여기서부터는 기존 코드 로직 그대로 이어짐
    # 예: df_raw = fetch_upbit_paged(...), df = add_indicators(...)
    #     warn_box.warning(...) ← 범위 불일치시
    #     차트 출력, 신호결과 출력 등
    pass
except Exception as e:
    st.error(f"오류: {e}")
