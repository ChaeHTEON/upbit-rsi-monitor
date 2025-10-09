# app.py
# -*- coding: utf-8 -*-
import os
# Streamlit 감시 한도 초과 방지
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["WATCHDOG_DISABLE_FILE_SYSTEM_EVENTS"] = "true"

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import ta, threading, time

# ===============================================
# ✅ 카카오 Webhook 전송 함수
# ===============================================
def send_kakao_alert(msg: str):
    try:
        url = st.secrets.get("KAKAO_WEBHOOK_URL", None)
        if not url:
            st.warning("⚠️ Webhook URL이 설정되어 있지 않습니다.")
            return
        payload = {"userRequest": {"utterance": msg}}
        headers = {"Content-Type": "application/json"}
        requests.post(url, json=payload, headers=headers, timeout=5)
    except Exception as e:
        st.error(f"❌ 카카오 알림 오류: {e}")

# ===============================================
# ✅ 페이지 기본 설정
# ===============================================
st.set_page_config(page_title="Upbit RSI(13)+Bollinger 시뮬레이터", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top:0.8rem; max-width:1100px;}
  .section-title {font-weight:700; margin:0.6rem 0 0.2rem;}
  .success-cell {color:#E53935;font-weight:600;}
  .fail-cell {color:#1E40AF;font-weight:600;}
  .neutral-cell {color:#FF9800;font-weight:600;}
  th,td {border:1px solid #ddd;padding:6px;text-align:center;}
</style>
""", unsafe_allow_html=True)
st.title("📊 코인 시뮬레이터")
st.markdown("<div style='color:gray;'>※ 점선: 신호~판정 구간, 성공 시 ⭐ 마커 표시</div>", unsafe_allow_html=True)

# ===============================================
# ✅ ① 기본 설정
# ===============================================
st.markdown('<div class="section-title">① 기본 설정</div>', unsafe_allow_html=True)
MARKET_LIST = [("비트코인", "KRW-BTC"), ("이더리움", "KRW-ETH"), ("리플", "KRW-XRP"), ("도지코인", "KRW-DOGE")]
TF_MAP = {"1분": ("minutes/1", 1), "3분": ("minutes/3", 3), "5분": ("minutes/5", 5),
          "15분": ("minutes/15", 15), "30분": ("minutes/30", 30), "60분": ("minutes/60", 60), "일봉": ("days", 24*60)}
col1, col2 = st.columns(2)
with col1:
    market_label, market_code = st.selectbox("종목 선택", MARKET_LIST, format_func=lambda x: x[0])
with col2:
    tf_label = st.selectbox("봉 종류 선택", list(TF_MAP.keys()), index=2)

# ===============================================
# ✅ ② 조건 설정
# ===============================================
st.markdown('<div class="section-title">② 조건 설정</div>', unsafe_allow_html=True)
rsi_low = st.number_input("RSI 과매도 기준", 10, 50, 30)
rsi_high = st.number_input("RSI 과매수 기준", 50, 90, 70)
bb_window = st.number_input("볼린저 기간", 10, 50, 20)
bb_dev = st.number_input("표준편차", 1.0, 4.0, 2.0)
cci_window = st.number_input("CCI 기간", 5, 50, 20)
cci_signal = st.number_input("CCI 시그널 기간", 3, 20, 5)
lookahead = st.number_input("판정봉 수 (lookahead)", 1, 50, 10)
target_thr = st.number_input("목표 수익률(%)", 0.5, 10.0, 1.0)
winrate_thr = st.number_input("승률 기준(%)", 50.0, 100.0, 60.0)

# ===============================================
# ✅ ③ 요약 & 차트
# ===============================================
st.markdown('<div class="section-title">③ 요약 & 차트</div>', unsafe_allow_html=True)
st.info(f"선택: {market_label} / {tf_label} | RSI({rsi_low}~{rsi_high}) | Bollinger ±{bb_dev}")

# (차트 예시)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
x = np.arange(50)
y = np.sin(x/3)+np.random.randn(50)*0.2
fig.add_trace(go.Candlestick(open=y, high=y+0.2, low=y-0.2, close=y, name="차트"), row=1, col=1)
fig.add_trace(go.Scatter(y=np.random.rand(50)*100, mode="lines", name="RSI(13)"), row=2, col=1)
fig.update_layout(height=500, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# ===============================================
# ✅ ④ 신호 결과 (최신 순)
# ===============================================
st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)
sample = pd.DataFrame({
    "날짜": pd.date_range("2025-10-01", periods=5),
    "결과": ["성공", "실패", "중립", "성공", "성공"],
    "수익률(%)": [1.2, -0.4, 0.0, 0.8, 1.1]
})
st.dataframe(sample, use_container_width=True)

# ===============================================
# ✅ ⑤ 실시간 감시
# ===============================================
st.markdown('<div class="section-title">⑤ 실시간 감시</div>', unsafe_allow_html=True)

def render_realtime_monitor():
    if "alerts" not in st.session_state:
        st.session_state["alerts"] = []
    if "last_alert_time" not in st.session_state:
        st.session_state["last_alert_time"] = {}
    if "watch_stop" not in st.session_state:
        st.session_state["watch_stop"] = threading.Event()
    if "watch_thread" not in st.session_state:
        st.session_state["watch_thread"] = None

    TF_MAP_LOCAL = {
        "1분": ("minutes/1", 1), "3분": ("minutes/3", 3),
        "5분": ("minutes/5", 5), "15분": ("minutes/15", 15),
        "30분": ("minutes/30", 30), "60분": ("minutes/60", 60), "일봉": ("days", 24*60)
    }

    def _add_alert(msg: str):
        if msg not in st.session_state["alerts"]:
            st.session_state["alerts"].append(msg)

    def _periodic_multi_check(stop_event: threading.Event):
        while not stop_event.is_set():
            now = datetime.now()
            syms = st.session_state.get("watch_symbols", [])
            tfs = st.session_state.get("watch_timeframes", [])
            for symbol in (syms or ["KRW-BTC"]):
                for tf in (tfs or ["5분"]):
                    try:
                        msg = f"🚨 [{symbol}] {tf} 실시간 감시 테스트 신호 ({now:%H:%M})"
                        key = f"{symbol}_{tf}"
                        last_time = st.session_state["last_alert_time"].get(key)
                        allow = True
                        if last_time is not None:
                            allow = (now - last_time).seconds >= 600
                        if allow:
                            _add_alert(msg)
                            send_kakao_alert(msg)
                            st.session_state["last_alert_time"][key] = now
                    except Exception as e:
                        print(f"[WARN] realtime check failed: {e}")
                        continue
            for _ in range(60):
                if stop_event.is_set():
                    break
                time.sleep(1)

    th = st.session_state.get("watch_thread")
    if th is None or not th.is_alive():
        st.session_state["watch_stop"].clear()
        th = threading.Thread(target=_periodic_multi_check, args=(st.session_state["watch_stop"],), daemon=True)
        th.start()
        st.session_state["watch_thread"] = th

    st.markdown("### 🚨 실시간 알람 목록")
    if st.session_state["alerts"]:
        for i, alert in enumerate(st.session_state["alerts"]):
            st.warning(f"{i+1}. {alert}")
    else:
        st.info("현재까지 감지된 실시간 알람이 없습니다.")

render_realtime_monitor()

# ===============================================
# ✅ 예외 처리
# ===============================================
try:
    pass
except Exception as e:
    st.error(f"오류 발생: {e}")
