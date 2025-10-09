# app.py
# -*- coding: utf-8 -*-
import os
# Streamlit 감시 한도 초과 방지
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["WATCHDOG_DISABLE_FILE_SYSTEM_EVENTS"] = "true"

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
import threading, time

# ===============================================
# ✅ 공통 함수 정의
# ===============================================
def send_kakao_alert(msg: str):
    """카카오 Webhook(site)으로 메시지 전송"""
    try:
        url = st.secrets["KAKAO_WEBHOOK_URL"]
        payload = {"userRequest": {"utterance": msg}}
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers, timeout=5)
        if response.status_code == 200:
            st.success("✅ 메시지 전송 성공!")
        else:
            st.warning(f"⚠️ 전송 실패 (응답 코드: {response.status_code})")
    except Exception as e:
        st.error(f"❌ 전송 중 오류 발생: {e}")

# ===============================================
# ✅ 페이지 기본 설정
# ===============================================
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

st.title("📊 코인 시뮬레이션")
st.markdown("<div style='margin-bottom:10px; color:gray;'>※ 차트 점선: 신호~판정 구간, 성공 시 도달 지점에 ⭐ 마커</div>", unsafe_allow_html=True)

# ===============================================
# ✅ 기본 설정
# ===============================================
st.markdown('<div class="section-title">① 기본 설정</div>', unsafe_allow_html=True)

MARKET_LIST = [("비트코인", "KRW-BTC"), ("이더리움", "KRW-ETH"), ("리플", "KRW-XRP"), ("도지코인", "KRW-DOGE")]
TF_MAP = {"1분": "minutes/1", "3분": "minutes/3", "5분": "minutes/5", "15분": "minutes/15", "30분": "minutes/30", "60분": "minutes/60", "일봉": "days"}

col1, col2 = st.columns(2)
with col1:
    market_label, market_code = st.selectbox("종목 선택", MARKET_LIST, index=0, format_func=lambda x: x[0])
with col2:
    tf_label = st.selectbox("봉 종류 선택", list(TF_MAP.keys()), index=2)

# ===============================================
# ✅ 실시간 감시 렌더 함수 정의
# ===============================================
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
        "1분": ("minutes/1", 1),
        "3분": ("minutes/3", 3),
        "5분": ("minutes/5", 5),
        "15분": ("minutes/15", 15),
        "30분": ("minutes/30", 30),
        "60분": ("minutes/60", 60),
        "일봉": ("days", 24*60),
    }

    def _add_alert(msg: str):
        if msg not in st.session_state["alerts"]:
            st.session_state["alerts"].append(msg)

    def _periodic_multi_check(stop_event: threading.Event):
        from datetime import datetime, timedelta
        while not stop_event.is_set():
            now = datetime.now()
            syms = st.session_state.get("watch_symbols", [])
            tfs  = st.session_state.get("watch_timeframes", [])
            for symbol in (syms or ["KRW-BTC"]):
                for tf_label in (tfs or ["5분"]):
                    try:
                        msg = f"🚨 [{symbol}] {tf_label} 실시간 감시 테스트 신호"
                        key = f"{symbol}_{tf_label}"
                        last_time = st.session_state["last_alert_time"].get(key)
                        allow = True
                        if last_time is not None:
                            allow = (now - last_time).seconds >= 600
                        if allow:
                            _add_alert(msg)
                            send_kakao_alert(msg)
                            st.session_state["last_alert_time"][key] = now
                    except Exception as e:
                        print(f"[WARN] realtime check failed for {symbol} {tf_label}: {e}")
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

    alive = st.session_state["watch_thread"].is_alive() if st.session_state["watch_thread"] else False
    if alive:
        st.info("✅ 실시간 감시 스레드가 실행 중입니다.")
    else:
        st.warning("⚠️ 실시간 감시 스레드가 정지 상태입니다. (자동 재시작됨)")

    st.markdown("### 🚨 실시간 알람 목록")
    if st.session_state["alerts"]:
        for i, alert in enumerate(st.session_state["alerts"]):
            st.warning(f"{i+1}. {alert}")
    else:
        st.info("현재까지 감지된 실시간 알람이 없습니다.")

# ===============================================
# ✅ 신호 결과 출력 (기존 유지)
# ===============================================
st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)
st.write("여기에 기존 신호 결과 출력이 표시됩니다. (UI 고정 영역)")

# ===============================================
# ✅ 실시간 감시 (⑤)
# ===============================================
st.markdown('<div class="section-title">⑤ 실시간 감시</div>', unsafe_allow_html=True)
render_realtime_monitor()

# ===============================================
# ✅ 예외 처리
# ===============================================
try:
    pass
except Exception as e:
    st.error(f"오류 발생: {e}")
