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
# 마우스 휠 버튼 더블 클릭 → 소프트 리프레시 (중복 제거 안정화)
# ──────────────────────────────────────────────────────────────────────────────
refresh_token = components.html("""
<script>
(function(){
  document.addEventListener('contextmenu', e => e.preventDefault(), true);

  let counter = 0;
  let lastClick = 0;
  let streak = 0;

  function triggerRefresh(e){
      if (window.Streamlit && window.Streamlit.setComponentValue) {
          window.Streamlit.setComponentValue(++counter);
      }
      if (e) e.preventDefault();
  }

  document.addEventListener('mousedown', function(e){
      if (e.button === 1) {  // 1 = wheel click
          const now = Date.now();
          if (now - lastClick <= 400) {
              streak += 1;
              if (streak >= 2) {
                  streak = 0;
                  triggerRefresh(e);
              }
          } else {
              streak = 1;
          }
          lastClick = now;
      }
  }, true);

  if (window.Streamlit && window.Streamlit.setFrameHeight) {
      window.Streamlit.setFrameHeight(0);
  }
})();
</script>
""", height=0)

# 소프트 리프레시 이벤트 처리 (무한루프 방지)
if refresh_token:
    if not st.session_state.get("soft_refresh_triggered", False):
        st.session_state["soft_refresh_triggered"] = True
        st.cache_data.clear()
        st.experimental_rerun()
else:
    st.session_state["soft_refresh_triggered"] = False

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
        if rows:
            return rows
    except Exception:
        pass
    return [("비트코인 (BTC) — KRW-BTC", "KRW-BTC")]

MARKET_LIST = get_upbit_krw_markets()
default_idx = next((i for i, (_, code) in enumerate(MARKET_LIST) if code == "KRW-BTC"), 0)

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
    "일봉": ("days", 24 * 60),
}

dup_mode = st.radio(
    "신호 중복 처리",
    ["중복 포함 (연속 신호 모두)", "중복 제거 (연속 동일 결과 1개)"],
    horizontal=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# ① 기본 설정
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">① 기본 설정</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    prev_code = st.session_state.get("market_code", None)
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
    default_start = today_kst - timedelta(days=1)
    start_date = st.date_input("시작 날짜", value=default_start)
    end_date = st.date_input("종료 날짜", value=today_kst)

interval_key, minutes_per_bar = TF_MAP[tf_label]
st.markdown("---")

# (중략) — 지표 계산, 시뮬레이션 함수, 실행부 로직은 기존과 동일
# 실행부에서는 반드시 selected_code = st.session_state.get("market_code", market_code) 사용

# ──────────────────────────────────────────────────────────────────────────────
# ⑥ 실행
# ──────────────────────────────────────────────────────────────────────────────
try:
    # ... (데이터 수집, 시뮬레이션 실행, 차트 그리기, 결과 표 출력 등 기존과 동일)
    pass

except Exception as e:
    st.error(f"오류: {e}")
