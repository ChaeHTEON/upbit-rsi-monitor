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
# 소프트 리프레시: 마우스 휠 버튼(중간 버튼) "짧은 더블 클릭" 감지
#  - components.html 값 변경 자체가 rerun을 발생시키므로, 여기서는 rerun()을 절대 호출하지 않음
#  - 동일 timestamp 재진입 방지용으로 세션에 마지막 값을 기록만 함
# ──────────────────────────────────────────────────────────────────────────────
refresh_ts = components.html("""
<script>
(function(){
  document.addEventListener('contextmenu', e => e.preventDefault(), true);
  let lastClick = 0, streak = 0;

  function triggerRefresh(e){
    const payload = Date.now();  // 고유 timestamp
    if (window.Streamlit && window.Streamlit.setComponentValue) {
      window.Streamlit.setComponentValue(payload); // 값이 바뀌면 Streamlit이 자동 rerun
    }
    if (e) e.preventDefault();
  }

  // 휠 버튼(중간 버튼) 더블 클릭(≤400ms) 감지
  document.addEventListener('mousedown', function(e){
    if (e.button === 1) { // 1 = wheel click
      const now = Date.now();
      if (now - lastClick <= 400) {
        streak += 1;
        if (streak >= 2) { streak = 0; triggerRefresh(e); }
      } else { streak = 1; }
      lastClick = now;
    }
  }, true);

  if (window.Streamlit && window.Streamlit.setFrameHeight) {
    window.Streamlit.setFrameHeight(0); // 보이지 않게
  }
})();
</script>
""", height=0)

# 새 이벤트면 캐시만 비우고 진행(재실행은 Streamlit이 이미 해 줌)
if refresh_ts and refresh_ts != st.session_state.get("soft_refresh_ts"):
    st.session_state["soft_refresh_ts"] = refresh_ts
    st.cache_data.clear()

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
        rows.sort(key=lambda x: (x[1] != "KRW-BTC", x[1]))  # BTC 우선
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
    # 항상 세션에 저장(기본값 회귀 방지)
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
# ③ 데이터 수집
# ──────────────────────────────────────────────────────────────────────────────
_session = requests.Session()
_session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.4,
                                                         status_forcelist=[429,500,502,503,504])))

@st.cache_data(ttl=120, show_spinner=False)
def fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars=0):
    start_cutoff = start_dt - timedelta(minutes=(warmup_bars or 0) * minutes_per_bar)
    if "minutes/" in interval_key:
        unit = interval_key.split("/")[1]
        url = f"https://api.upbit.com/v1/candles/minutes/{unit}"
    else:
        url = "https://api.upbit.com/v1/candles/days"

    to_time = None
    all_rows = []
    for _ in range(60):  # 안전 상한
        params = {"market": market_code, "count": 200}
        if to_time is not None:
            params["to"] = to_time.strftime("%Y-%m-%d %H:%M:%S")
        r = _session.get(url, params=params, timeout=10)
        r.raise_for_status()
        batch = r.json()
        if not batch: break
        all_rows.extend(batch)
        last_ts = pd.to_datetime(batch[-1]["candle_date_time_kst"])
        if last_ts <= start_cutoff: break
        to_time = last_ts - timedelta(seconds=1)

    if not all_rows: return pd.DataFrame()
    df = (pd.DataFrame(all_rows)
            .rename(columns={"candle_date_time_kst":"time","opening_price":"open",
                             "high_price":"high","low_price":"low","trade_price":"close",
                             "candle_acc_trade_volume":"volume"}))
    df["time"] = pd.to_datetime(df["time"])
    df = df[["time","open","high","low","close","volume"]].sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return df[(df["time"] >= start_cutoff) & (df["time"] <= end_dt)]

# ──────────────────────────────────────────────────────────────────────────────
# ④ 지표
# ──────────────────────────────────────────────────────────────────────────────
def add_indicators(df, bb_window, bb_dev):
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=int(bb_window), window_dev=float(bb_dev))
    out["BB_up"]  = bb.bollinger_hband()
    out["BB_low"] = bb.bollinger_lband()
    out["BB_mid"] = bb.bollinger_mavg()
    return out

# ──────────────────────────────────────────────────────────────────────────────
# ⑤ 시뮬레이션 (프로젝트 규칙 반영)
#   - 성공 판정: 종가 기준 고정(목표 종가 도달 시 조기 성공)
#   - 미도달 처리: N번째 종가 수익률이 +면 '중립', 0 이하 '실패'
#   - 2차 조건: '양봉 2개 (범위 내)' 포함
# ──────────────────────────────────────────────────────────────────────────────
def simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, thr_pct, bb_cond, dedup_mode,
             minutes_per_bar, market_code, bb_window, bb_dev, sec_cond="없음"):
    res, n = [], len(df)
    thr = float(thr_pct)

    # 1차 조건
    if rsi_mode == "없음":
        sig_rsi = set(range(n))
    elif rsi_mode == "현재(과매도/과매수 중 하나)":
        sig_rsi = set(df.index[(df["RSI13"] <= float(rsi_low)) | (df["RSI13"] >= float(rsi_high))])
    elif rsi_mode == "과매도 기준":
        sig_rsi = set(df.index[df["RSI13"] <= float(rsi_low)])
    else:
        sig_rsi = set(df.index[df["RSI13"] >= float(rsi_high)])

    def bb_ok(i):
        c = float(df.at[i,"close"])
        up, lo, mid = df.at[i,"BB_up"], df.at[i,"BB_low"], df.at[i,"BB_mid"]
        if bb_cond == "상한선": return pd.notna(up)  and (c >  float(up))
        if bb_cond == "하한선": return pd.notna(lo)  and (c <= float(lo))
        if bb_cond == "중앙선": return pd.notna(mid) and (c >= float(mid))
        return True  # "없음"

    sig_idx = [i for i in sig_rsi if bb_ok(i)]

    def is_bull(k): return float(df.at[k,"close"]) > float(df.at[k,"open"])

    i = 0
    while i < n:
        if i not in sig_idx:
            i += 1; continue

        anchor = i
        if sec_cond == "양봉 2개 연속 상승":
            if i+2 >= n: i += 1; continue
            c1,o1 = float(df.at[i+1,"close"]), float(df.at[i+1,"open"])
            c2,o2 = float(df.at[i+2,"close"]), float(df.at[i+2,"open"])
            if not ((c1>o1) and (c2>o2) and (c2>c1)): i += 1; continue
        elif sec_cond == "양봉 2개 (범위 내)":
            bulls = 0
            for j in range(i+1, min(i+1+lookahead, n)):
                if is_bull(j):
                    bulls += 1
                    if bulls >= 2: break
            if bulls < 2: i += 1; continue
        elif sec_cond == "BB 기반 첫 양봉 50% 진입":
            # (간결 구현) 기준봉 이후 첫 양봉이 BB 중앙선 이상 종가이면 기준 이동
            new_idx = None
            for j in range(i+1, n):
                if is_bull(j) and pd.notna(df.at[j,"BB_mid"]) and float(df.at[j,"close"]) >= float(df.at[j,"BB_mid"]):
                    new_idx = j; break
            if new_idx is None: i += 1; continue
            anchor = new_idx

        base = float(df.at[anchor,"close"])
        end  = min(anchor + lookahead, n-1)

        # 목표가(종가 기준)
        target = base * (1.0 + thr/100.0)
        hit = None
        for j in range(anchor+1, end+1):
            if float(df.at[j,"close"]) >= target:
                hit = j; break

        window = df.iloc[anchor+1:end+1]
        if hit is not None:
            end_time  = df.at[hit,"time"]
            end_close = target
            ret = thr
            result = "성공"
            reach_min = (hit - anchor) * minutes_per_bar
        else:
            end_time  = df.at[end,"time"]
            end_close = float(df.at[end,"close"])
            ret = (end_close/base - 1)*100.0
            result = "중립" if ret > 0 else "실패"
            reach_min = None

        res.append({
            "신호시간": df.at[anchor,"time"],
            "종료시간": end_time,
            "기준시가": int(round(base)),
            "종료가": end_close,
            "RSI(13)": round(float(df.at[anchor,"RSI13"]),1) if pd.notna(df.at[anchor,"RSI13"]) else None,
            "성공기준(%)": round(thr,1),
            "결과": result,
            "도달분": reach_min,
            "최종수익률(%)": round(ret,2),
            "최저수익률(%)": round(((window["close"].min()/base - 1)*100) if not window.empty else 0.0, 2),
            "최고수익률(%)": round(((window["close"].max()/base - 1)*100) if not window.empty else 0.0, 2),
        })

        i = end if dup_mode.startswith("중복 제거") else i+1

    return pd.DataFrame(res)

# ──────────────────────────────────────────────────────────────────────────────
# ⑥ 실행
# ──────────────────────────────────────────────────────────────────────────────
try:
    if start_date > end_date:
        st.error("시작 날짜가 종료 날짜보다 이후입니다.")
        st.stop()

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt   = datetime.combine(end_date,   datetime.max.time())

    selected_code = st.session_state.get("market_code", market_code)
    warmup_bars   = max(13, bb_window) * 5

    # 데이터 수집
    df_raw = fetch_upbit_paged(selected_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars)
    if df_raw.empty:
        st.info(f"{selected_code} 구간에 데이터가 없습니다.")
        st.stop()

    # 지표
    df = add_indicators(df_raw, bb_window, bb_dev)

    # 실제 바 간격(분) 추정 → UI 표시 정확도 개선
    _diff = df["time"].diff().dropna()
    bar_min = int(round(_diff.median().total_seconds()/60)) if not _diff.empty else minutes_per_bar
    if bar_min <= 0: bar_min = minutes_per_bar
    hh, mm = divmod(lookahead*bar_min, 60)
    look_str = f"{lookahead}봉 / {hh:02d}:{mm:02d}"

    # 요약
    st.markdown('<div class="section-title">③ 요약 & 차트</div>', unsafe_allow_html=True)
    st.info(
        "설정 요약\n"
        f"- 종목: {selected_code}\n"
        f"- 측정 구간: {look_str}\n"
        f"- 1차 조건: RSI={rsi_mode}, BB={bb_cond}\n"
        f"- 2차 조건: {sec_cond}\n"
        f"- 성공 판정 기준: 종가 기준(고정)\n"
        f"- 미도달 규칙: 마지막 종가 수익 +면 중립, 0 이하 실패"
    )

    # 시뮬레이션
    res_all = simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, threshold_pct,
                       bb_cond, "중복 포함 (연속 신호 모두)", bar_min, selected_code, bb_window, bb_dev, sec_cond)
    res_dedup = simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, threshold_pct,
                         bb_cond, "중복 제거 (연속 동일 결과 1개)", bar_min, selected_code, bb_window, bb_dev, sec_cond)
    res = res_all if dup_mode.startswith("중복 포함") else res_dedup

    # 차트
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                 name="가격", increasing_line_color="red", decreasing_line_color="blue"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_up"],  mode="lines", name="BB 상단"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_low"], mode="lines", name="BB 하단"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_mid"], mode="lines", name="BB 중앙"))
    st.plotly_chart(fig, use_container_width=True)

    # 표
    if res.empty:
        st.info("조건을 만족하는 신호가 없습니다.")
    else:
        tbl = res.sort_values("신호시간", ascending=False).reset_index(drop=True)
        # 도달 캔들/시간 계산
        s = pd.to_datetime(tbl["신호시간"]); e = pd.to_datetime(tbl["종료시간"])
        diff_min = ((e - s).dt.total_seconds()/60).round().astype(int)
        bars_after = (diff_min / bar_min).round().astype(int)
        tbl["도달캔들"] = np.where(tbl["결과"] == "성공", bars_after, lookahead)
        tbl["도달시간"] = diff_min.apply(lambda m: f"{m//60:02d}:{m%60:02d}")
        st.dataframe(tbl[["신호시간","기준시가","RSI(13)","성공기준(%)","결과",
                          "최종수익률(%)","최저수익률(%)","최고수익률(%)","도달캔들","도달시간"]],
                     use_container_width=True)

except Exception as e:
    st.error(f"오류: {e}")
