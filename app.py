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
    if start_date > end_date:
        st.error("시작 날짜가 종료 날짜보다 이후입니다.")
        st.stop()

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt   = datetime.combine(end_date, datetime.max.time())

    warmup_bars = max(13, bb_window) * 5
    selected_code = st.session_state.get("market_code", market_code)

    df_raw = fetch_upbit_paged(selected_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars=warmup_bars)
    if df_raw.empty:
        st.error(f"{selected_code} 데이터가 없습니다.")
        st.stop()

    df_ind = add_indicators(df_raw, bb_window, bb_dev)
    df = df_ind[(df_ind["time"] >= start_dt) & (df_ind["time"] <= end_dt)].reset_index(drop=True)
    bb_cond = st.session_state.get("bb_cond", bb_cond)

    _bar_diff = df["time"].diff().dropna()
    bar_min = int(round(_bar_diff.median().total_seconds() / 60)) if not _bar_diff.empty else minutes_per_bar
    if bar_min <= 0:
        bar_min = minutes_per_bar

    total_min = lookahead * bar_min
    hh, mm = divmod(int(total_min), 60)
    look_str = f"{lookahead}봉 / {hh:02d}:{mm:02d}"

    # --- 요약 ---
    st.markdown('<div class="section-title">③ 요약 & 차트</div>', unsafe_allow_html=True)
    st.info(
        "설정 요약\n"
        f"- 측정 구간: {look_str}\n"
        f"- RSI 조건: {rsi_mode}\n"
        f"- BB 조건: {bb_cond}\n"
        f"- 2차 조건: {sec_cond}\n"
        f"- 성공 판정: 종가 기준\n"
        f"- 미도달: 마지막 종가 양수=중립, 0 이하=실패\n"
        f"- 데이터 최신 캔들(KST): {pd.to_datetime(df['time'].max()).strftime('%Y-%m-%d %H:%M')}"
    )

    # --- 시뮬레이션 ---
    res_all = simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, threshold_pct,
                       bb_cond, "중복 포함 (연속 신호 모두)", bar_min, selected_code, bb_window, bb_dev, sec_cond)
    res_dedup = simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, threshold_pct,
                         bb_cond, "중복 제거 (연속 동일 결과 1개)", bar_min, selected_code, bb_window, bb_dev, sec_cond)
    res = res_all if dup_mode.startswith("중복 포함") else res_dedup

    # --- 결과 요약 ---
    succ = (res["결과"] == "성공").sum()
    fail = (res["결과"] == "실패").sum()
    neu  = (res["결과"] == "중립").sum()
    st.metric("성공", succ)
    st.metric("실패", fail)
    st.metric("중립", neu)

    # --- 차트 ---
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="가격", increasing_line_color="red", decreasing_line_color="blue"
    ))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_up"], mode="lines", name="BB 상단"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_low"], mode="lines", name="BB 하단"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_mid"], mode="lines", name="BB 중앙"))
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"오류: {e}")
