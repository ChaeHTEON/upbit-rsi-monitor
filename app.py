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
# 마우스 휠 버튼 더블 클릭 → 소프트 리프레시 (구버전 호환: key 인자 사용 안함)
# ──────────────────────────────────────────────────────────────────────────────
refresh_token = components.html("""
<script src="https://unpkg.com/@streamlit/component-lib/dist/index.js"></script>
<script>
(function(){
  document.addEventListener('contextmenu', e => e.preventDefault(), true);

  // 휠 버튼(중간 버튼) 더블 클릭 감지
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
      if (e.button === 1) {  // 1 = middle button
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
""", height=0)  # ← key 제거

# 토큰/플래그 초기화
if "soft_refresh_token" not in st.session_state:
    st.session_state["soft_refresh_token"] = 0
if "soft_refresh_pending" not in st.session_state:
    st.session_state["soft_refresh_pending"] = False

# 값이 실제로 바뀐 최초 한 번만 rerun (무한루프 방지)
if refresh_token is not None:
    if refresh_token != st.session_state["soft_refresh_token"]:
        st.session_state["soft_refresh_token"] = refresh_token
        st.cache_data.clear()
        st.experimental_rerun()

if "soft_refresh_token" not in st.session_state:
    st.session_state["soft_refresh_token"] = None

if refresh_token and refresh_token != st.session_state["soft_refresh_token"]:
    st.session_state["soft_refresh_token"] = refresh_token
    st.cache_data.clear()
    st.experimental_rerun()

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
    selected = st.selectbox("종목 선택", MARKET_LIST, index=idx, key="market_sel", format_func=lambda x: x[0])
    market_label, market_code = selected
    # 항상 세션에 저장
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

# ──────────────────────────────────────────────────────────────────────────────
# ② 조건 설정
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">② 조건 설정</div>', unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
with c4:
    lookahead = st.slider("측정 캔들 수 (기준 이후 N봉)", 1, 60, 10)
with c5:
    threshold_pct = st.slider("성공/실패 기준 값(%)", 0.1, 5.0, 1.0, step=0.1)
    st.caption(f"현재 설정: **{threshold_pct:.1f}%** (성공 판정 = 종가 기준)")
with c6:
    r1, r2, r3 = st.columns(3)
    with r1:
        rsi_mode = st.selectbox(
            "RSI 조건",
            ["없음", "현재(과매도/과매수 중 하나)", "과매도 기준", "과매수 기준"],
            index=0,
        )
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

st.markdown('<div class="hint">2차 조건: 선택한 조건만 적용</div>', unsafe_allow_html=True)
sec_cond = st.selectbox(
    "2차 조건 선택",
    ["없음", "양봉 2개 연속 상승", "양봉 2개 (범위 내)", "BB 기반 첫 양봉 50% 진입"],
    index=0,
)
st.session_state["bb_cond"] = bb_cond
st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# ③ 데이터 수집
# ──────────────────────────────────────────────────────────────────────────────
def estimate_calls(start_dt, end_dt, minutes_per_bar):
    mins = max(1, int((end_dt - start_dt).total_seconds() // 60))
    bars = max(1, mins // minutes_per_bar)
    return bars // 200 + 1

_session = requests.Session()
_retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
_session.mount("https://", HTTPAdapter(max_retries=_retries))

@st.cache_data(ttl=120, show_spinner=False)
def fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars: int = 0):
    start_cutoff = start_dt - timedelta(minutes=(warmup_bars or 0) * minutes_per_bar)
    if "minutes/" in interval_key:
        unit = interval_key.split("/")[1]
        url = f"https://api.upbit.com/v1/candles/minutes/{unit}"
    else:
        url = "https://api.upbit.com/v1/candles/days"

    calls_est = estimate_calls(start_cutoff, end_dt, minutes_per_bar)
    max_calls = min(calls_est + 2, 60)
    req_count = 200
    all_data = []
    to_time = None

    try:
        for _ in range(max_calls):
            params = {"market": market_code, "count": req_count}
            if to_time is not None:
                params["to"] = to_time.strftime("%Y-%m-%d %H:%M:%S")
            r = _session.get(url, params=params, headers={"Accept": "application/json"}, timeout=10)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            all_data.extend(batch)
            last_ts = pd.to_datetime(batch[-1]["candle_date_time_kst"])
            if last_ts <= start_cutoff:
                break
            to_time = last_ts - timedelta(seconds=1)
    except Exception:
        return pd.DataFrame()

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data).rename(columns={
        "candle_date_time_kst": "time", "opening_price": "open", "high_price": "high",
        "low_price": "low", "trade_price": "close", "candle_acc_trade_volume": "volume"})
    df["time"] = pd.to_datetime(df["time"])
    df = df[["time", "open", "high", "low", "close", "volume"]].sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return df[(df["time"] >= start_cutoff) & (df["time"] <= end_dt)]

# ──────────────────────────────────────────────────────────────────────────────
# ④ 지표
# ──────────────────────────────────────────────────────────────────────────────
def add_indicators(df, bb_window, bb_dev):
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=bb_window, window_dev=bb_dev)
    out["BB_up"]  = bb.bollinger_hband().fillna(method="bfill").fillna(method="ffill")
    out["BB_low"] = bb.bollinger_lband().fillna(method="bfill").fillna(method="ffill")
    out["BB_mid"] = bb.bollinger_mavg().fillna(method="bfill").fillna(method="ffill")
    return out

# ──────────────────────────────────────────────────────────────────────────────
# ⑤ 시뮬레이션
# ──────────────────────────────────────────────────────────────────────────────
def simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, thr_pct, bb_cond, dedup_mode,
             minutes_per_bar, market_code, bb_window, bb_dev, sec_cond="없음"):
    res = []
    n = len(df)
    thr = float(thr_pct)

    # 1차 조건
    if rsi_mode == "없음":
        rsi_idx = []
    elif rsi_mode == "현재(과매도/과매수 중 하나)":
        rsi_idx = sorted(set(df.index[df["RSI13"] <= float(rsi_low)].tolist()) |
                         set(df.index[df["RSI13"] >= float(rsi_high)].tolist()))
    elif rsi_mode == "과매도 기준":
        rsi_idx = df.index[df["RSI13"] <= float(rsi_low)].tolist()
    else:
        rsi_idx = df.index[df["RSI13"] >= float(rsi_high)].tolist()

    def bb_ok(i):
        c = float(df.at[i, "close"])
        up, lo, mid = df.at[i, "BB_up"], df.at[i, "BB_low"], df.at[i, "BB_mid"]
        if bb_cond == "상한선": return pd.notna(up) and (c > float(up))
        if bb_cond == "하한선": return pd.notna(lo) and (c <= float(lo))
        if bb_cond == "중앙선": return pd.notna(mid) and (c >= float(mid))
        return False

    bb_idx = [i for i in df.index if bb_cond != "없음" and bb_ok(i)]

    if rsi_mode != "없음" and bb_cond != "없음":
        base_sig_idx = sorted(set(rsi_idx) & set(bb_idx))
    elif rsi_mode != "없음":
        base_sig_idx = rsi_idx
    elif bb_cond != "없음":
        base_sig_idx = bb_idx
    else:
        base_sig_idx = list(range(n)) if sec_cond != "없음" else []

    def is_bull(idx):
        return float(df.at[idx, "close"]) > float(df.at[idx, "open"])

    def first_bull_50_over_bb(start_i):
        for j in range(start_i + 1, n):
            if not is_bull(j): continue
            if bb_cond == "하한선":   ref = df.at[j, "BB_low"]
            elif bb_cond == "중앙선": ref = df.at[j, "BB_mid"]
            else:                     ref = df.at[j, "BB_up"]
            if pd.isna(ref): continue
            if float(df.at[j, "close"]) >= float(ref):
                return j, float(df.at[j, "close"])
        return None, None

    i = 0
    while i < n:
        if i not in base_sig_idx:
            i += 1
            continue

        anchor_idx  = i
        signal_time = df.at[i, "time"]
        base_price  = float(df.at[i, "close"])

        if sec_cond == "양봉 2개 연속 상승":
            if i + 2 >= n:
                i += 1; continue
            c1, o1 = float(df.at[i + 1, "close"]), float(df.at[i + 1, "open"])
            c2, o2 = float(df.at[i + 2, "close"]), float(df.at[i + 2, "open"])
            if not ((c1 > o1) and (c2 > o2) and (c2 > c1)):
                i += 1; continue

        elif sec_cond == "양봉 2개 (범위 내)":
            bull_cnt = 0
            for j in range(i + 1, min(i + 1 + lookahead, n)):
                if float(df.at[j, "close"]) > float(df.at[j, "open"]):
                    bull_cnt += 1
                    if bull_cnt >= 2:
                        break
            if bull_cnt < 2:
                i += 1; continue

        elif sec_cond == "BB 기반 첫 양봉 50% 진입":
            B1_idx, B1_close = first_bull_50_over_bb(i)
            if B1_idx is None:
                i += 1; continue
            bull_cnt, B3_idx = 0, None
            scan_end = min(B1_idx + lookahead, n - 1)
            for j in range(B1_idx + 1, scan_end + 1):
                if is_bull(j):
                    bull_cnt += 1
                    if bull_cnt == 2:
                        B3_idx = j; break
            if B3_idx is None:
                i += 1; continue
            T_idx = None
            for j in range(B3_idx + 1, n):
                if pd.notna(df.at[j, "close"]) and float(df.at[j, "close"]) >= B1_close:
                    T_idx = j; break
            if T_idx is None:
                i += 1; continue
            anchor_idx  = T_idx
            signal_time = df.at[T_idx, "time"]
            base_price  = float(df.at[T_idx, "close"])

        end_idx = anchor_idx + lookahead
        if end_idx >= n:
            i += 1
            continue

        win_slice = df.iloc[anchor_idx + 1:end_idx + 1]
        end_time  = df.at[end_idx, "time"]
        end_close = float(df.at[end_idx, "close"])
        final_ret = (end_close / base_price - 1) * 100

        target = base_price * (1.0 + thr / 100.0)
        hit_idx = None
        for j in range(anchor_idx + 1, end_idx + 1):
            if float(df.at[j, "close"]) >= target:
                hit_idx = j
                break

        if hit_idx is not None:
            end_time  = df.at[hit_idx, "time"]
            end_close = target
            final_ret = thr
            result    = "성공"
            reach_min = (hit_idx - anchor_idx) * minutes_per_bar
        else:
            result    = "중립" if final_ret > 0 else "실패"
            reach_min = None

        if bb_cond == "상한선":
            bb_value = df.at[anchor_idx, "BB_up"]
        elif bb_cond == "중앙선":
            bb_value = df.at[anchor_idx, "BB_mid"]
        elif bb_cond == "하한선":
            bb_value = df.at[anchor_idx, "BB_low"]
        else:
            bb_value = None

        res.append({
            "신호시간": signal_time,
            "종료시간": end_time,
            "기준시가": int(round(base_price)),
            "종료가": end_close,
            "RSI(13)": round(float(df.at[anchor_idx, "RSI13"]), 1) if pd.notna(df.at[anchor_idx, "RSI13"]) else None,
            "BB값": round(float(bb_value), 1) if (bb_value is not None and pd.notna(bb_value)) else None,
            "성공기준(%)": round(thr, 1),
            "결과": result,
            "도달분": reach_min,
            "최종수익률(%)": round(final_ret, 2),
            "최저수익률(%)": round(((win_slice["close"].min()/base_price - 1)*100) if not win_slice.empty else 0.0, 2),
            "최고수익률(%)": round(((win_slice["close"].max()/base_price - 1)*100) if not win_slice.empty else 0.0, 2),
        })

        i = end_idx if dup_mode.startswith("중복 제거") else i + 1

    return pd.DataFrame(res)

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
    def _summarize(df_in):
        if df_in.empty:
            return 0, 0, 0
        succ = (df_in["결과"] == "성공").sum()
        fail = (df_in["결과"] == "실패").sum()
        neu  = (df_in["결과"] == "중립").sum()
        return succ, fail, neu

    succ, fail, neu = _summarize(res)
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
