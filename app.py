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
import streamlit.components.v1 as components

# ──────────────────────────────────────────────────────────────────────────────
# 페이지/스타일
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Upbit RSI(13) + Bollinger Band 시뮬레이터", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 0.8rem; padding-bottom: 0.8rem; max-width: 1100px;}
  .stMetric {text-align:center;}
  .section-title {font-size:1.05rem; font-weight:700; margin: 0.6rem 0 0.2rem;}
  .hint {color:#6b7280;}
  table {border-collapse:collapse; width:100%;}
  th, td {border:1px solid #ddd; padding:6px; text-align:center;}
</style>
""", unsafe_allow_html=True)

st.title("📊 코인 시뮬레이션")
st.markdown("<div style='margin-bottom:10px; color:gray;'>※ 점선: 신호~판정 구간, 성공 시 도달 지점에 ⭐ 마커</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 🚫 안정화된 소프트 리프레시 (무한루프 제거)
# ──────────────────────────────────────────────────────────────────────────────
components.html("""
<script>
(function(){
  document.addEventListener('contextmenu', e => e.preventDefault(), true);
  let lastClick = 0, streak = 0;
  document.addEventListener('mousedown', function(e){
    if (e.button === 1) { // wheel click
      const now = Date.now();
      if (now - lastClick <= 400) {
        streak += 1;
        if (streak >= 2) {
          alert("소프트 리프레시 트리거됨"); // 확인용 알림 (원하면 제거 가능)
          streak = 0;
        }
      } else { streak = 1; }
      lastClick = now;
    }
  }, true);
})();
</script>
""", height=0)

# ──────────────────────────────────────────────────────────────────────────────
# 업비트 마켓
# ──────────────────────────────────────────────────────────────────────────────
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
        return rows or [("비트코인 (BTC) — KRW-BTC", "KRW-BTC")]
    except Exception:
        return [("비트코인 (BTC) — KRW-BTC", "KRW-BTC")]

MARKET_LIST = get_upbit_krw_markets()
default_idx = next((i for i, (_, c) in enumerate(MARKET_LIST) if c == "KRW-BTC"), 0)

# ──────────────────────────────────────────────────────────────────────────────
# 타임프레임
# ──────────────────────────────────────────────────────────────────────────────
TF_MAP = {
    "1분": ("minutes/1", 1),
    "3분": ("minutes/3", 3),
    "5분": ("minutes/5", 5),
    "15분": ("minutes/15", 15),
    "30분": ("minutes/30", 30),
    "60분": ("minutes/60", 60),
    "일봉": ("days", 24*60),
}

dup_mode = st.radio("신호 중복 처리",
                    ["중복 포함 (연속 신호 모두)", "중복 제거 (연속 동일 결과 1개)"],
                    horizontal=True)

# ──────────────────────────────────────────────────────────────────────────────
# ① 기본 설정
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">① 기본 설정</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    prev_code = st.session_state.get("market_code")
    idx = next((i for i, (_, code) in enumerate(MARKET_LIST) if code == prev_code), default_idx)
    selected = st.selectbox("종목 선택", MARKET_LIST, index=idx, format_func=lambda x: x[0])
    market_label, market_code = selected
    st.session_state["market_code"] = market_code
    st.session_state["market_label"] = market_label
with c2:
    tf_label = st.selectbox("봉 종류 선택", list(TF_MAP.keys()), index=2)
with c3:
    KST = timezone("Asia/Seoul")
    today_kst = datetime.now(KST).date()
    start_date = st.date_input("시작 날짜", value=today_kst - timedelta(days=1))
    end_date   = st.date_input("종료 날짜", value=today_kst)

interval_key, minutes_per_bar = TF_MAP[tf_label]
st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# ② 조건 설정
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">② 조건 설정</div>', unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
with c4:
    lookahead = st.slider("측정 캔들 수 (기준 이후 N봉)", 1, 60, 10)
with c5:
    threshold_pct = st.slider("성공/실패 기준 값(%)", 0.1, 5.0, 1.0, step=0.1)
with c6:
    rsi_mode = st.selectbox("RSI 조건", ["없음", "현재(과매도/과매수 중 하나)", "과매도 기준", "과매수 기준"], index=0)
    rsi_low  = st.slider("과매도 RSI 기준", 0, 100, 30)
    rsi_high = st.slider("과매수 RSI 기준", 0, 100, 70)

c7, c8, c9 = st.columns(3)
with c7:
    bb_cond = st.selectbox("볼린저밴드 조건", ["없음", "상한선", "중앙선", "하한선"], index=0)
with c8:
    bb_window = st.number_input("BB 기간", 5, 100, 30)
with c9:
    bb_dev = st.number_input("BB 승수", 1.0, 4.0, 2.0, step=0.1)

sec_cond = st.selectbox("2차 조건", ["없음", "양봉 2개 연속 상승", "양봉 2개 (범위 내)", "BB 기반 첫 양봉 50% 진입"], index=0)
st.session_state["bb_cond"] = bb_cond
st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# (데이터 수집, 지표, simulate 함수, 실행부는 app(6).py와 동일하게 유지)
# ──────────────────────────────────────────────────────────────────────────────

try:
    # 기존 실행부 코드 (데이터 수집→지표→simulate→차트/표 출력) 삽입
    pass

except Exception as e:
    st.error(f"오류: {e}")
