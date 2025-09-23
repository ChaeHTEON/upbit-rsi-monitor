### 최종 파이널 app.py
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
import streamlit.components.v1 as components

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
  table {border-collapse:collapse; width:100%;}
  th, td {border:1px solid #ddd; padding:6px; text-align:center;}
</style>
""", unsafe_allow_html=True)

st.title("📊 코인 시뮬레이션")
st.markdown("<div style='margin-bottom:10px; color:gray;'>※ 차트 점선: 신호~판정 구간, 성공 시 도달 지점에 ⭐ 마커</div>", unsafe_allow_html=True)

# -----------------------------
# Soft Refresh (PC/모바일 이벤트)
# -----------------------------
if "last_soft_refresh_ts" not in st.session_state:
    st.session_state["last_soft_refresh_ts"] = 0

components.html(
    """
    <script>
      (function(){
        let lastClick = 0;
        let locked = false;
        function triggerSoftRefresh(){
          if(locked) return;
          locked = true;
          const url = new URL(window.location.href);
          url.searchParams.set("soft_refresh", Date.now().toString());
          window.history.replaceState({}, "", url.toString());
          window.parent.postMessage({isRefresh:true}, "*");
          setTimeout(()=>{ locked = false; }, 1200);
        }
        // PC: 중간 버튼 더블클릭
        document.addEventListener("mousedown", function(e){
          if(e.button === 1){
            const now = Date.now();
            if(now - lastClick < 400){ triggerSoftRefresh(); }
            lastClick = now;
          }
        }, {passive:true});
        // Mobile: 세 손가락 터치
        document.addEventListener("touchstart", function(e){
          if(e.touches && e.touches.length === 3){ triggerSoftRefresh(); }
        }, {passive:true});
      })();
    </script>
    """,
    height=0
)

qp = st.experimental_get_query_params()
if "soft_refresh" in qp:
    try:
        ts = int(qp.get("soft_refresh")[0])
    except Exception:
        ts = 0
    if ts != st.session_state.get("last_soft_refresh_ts", 0):
        st.cache_data.clear()
        st.session_state["last_soft_refresh_ts"] = ts
        st.experimental_rerun()

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
    "일봉": ("days", 24 * 60),
}

# -----------------------------
# 상단: 신호 중복 처리
# -----------------------------
dup_mode = st.radio(
    "신호 중복 처리",
    ["중복 포함 (연속 신호 모두)", "중복 제거 (연속 동일 결과 1개)"],
    horizontal=True,
)

# -----------------------------
# 기본 설정
# -----------------------------
st.markdown('<div class="section-title">① 기본 설정</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    market_label, market_code = st.selectbox("종목 선택", MARKET_LIST, index=default_idx, format_func=lambda x: x[0])
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

# -----------------------------
# 조건 설정
# -----------------------------
st.markdown('<div class="section-title">② 조건 설정</div>', unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
with c4:
    lookahead = st.slider("측정 캔들 수 (기준 이후 N봉)", 1, 60, 10)
with c5:
    threshold_pct = st.slider("성공/실패 기준 값(%)", 0.1, 5.0, 1.0, step=0.1)
    hit_basis = st.selectbox("성공 판정 기준", ["종가 기준", "고가 기준(스침 인정)", "종가 또는 고가"], index=0)
with c6:
    r1, r2, r3 = st.columns(3)
    with r1:
        rsi_mode = st.selectbox("RSI 조건", ["없음", "현재(과매도/과매수 중 하나)", "과매도 기준", "과매수 기준"], index=0)
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
sec_cond = st.selectbox("2차 조건 선택", ["없음", "양봉 2개 연속 상승", "BB 기반 첫 양봉 50% 진입"], index=0)
st.session_state["bb_cond"] = bb_cond
st.markdown("---")

# -----------------------------
# 데이터 수집 (Upbit)
# -----------------------------
_session = requests.Session()
_retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
_session.mount("https://", HTTPAdapter(max_retries=_retries))

@st.cache_data(ttl=120, show_spinner=False)
def fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars: int = 0):
    start_cutoff = start_dt - timedelta(minutes=max(0, warmup_bars) * minutes_per_bar)
    url = f"https://api.upbit.com/v1/candles/minutes/{interval_key.split('/')[1]}" if "minutes/" in interval_key else "https://api.upbit.com/v1/candles/days"
    req_count, all_data, to_time = 200, [], None
    try:
        for _ in range(60):
            params = {"market": market_code, "count": req_count}
            if to_time is not None:
                params["to"] = to_time.strftime("%Y-%m-%d %H:%M:%S")
            r = _session.get(url, params=params, headers={"Accept": "application/json"}, timeout=10)
            r.raise_for_status()
            batch = r.json()
            if not batch: break
            all_data.extend(batch)
            last_ts = pd.to_datetime(batch[-1]["candle_date_time_kst"])
            if last_ts <= start_cutoff: break
            to_time = last_ts - timedelta(seconds=1)
    except Exception:
        return pd.DataFrame()
    if not all_data: return pd.DataFrame()
    df = pd.DataFrame(all_data).rename(columns={
        "candle_date_time_kst": "time", "opening_price": "open", "high_price": "high",
        "low_price": "low", "trade_price": "close", "candle_acc_trade_volume": "volume"})
    df["time"] = pd.to_datetime(df["time"])
    df = df[["time","open","high","low","close","volume"]].sort_values("time").reset_index(drop=True)
    return df[(df["time"] >= start_cutoff) & (df["time"] <= end_dt)]

# -----------------------------
# 지표
# -----------------------------
def add_indicators(df, bb_window, bb_dev):
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=bb_window, window_dev=bb_dev)
    out["BB_up"]  = bb.bollinger_hband()
    out["BB_low"] = bb.bollinger_lband()
    out["BB_mid"] = bb.bollinger_mavg()
    return out

# -----------------------------
# 시뮬레이션 (미도달 자동 판정)
# -----------------------------
def simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, thr_pct, bb_cond, dedup_mode,
             minutes_per_bar, market_code, bb_window, bb_dev, sec_cond="없음", hit_basis="종가 기준"):
    res, n, thr = [], len(df), float(thr_pct)

    # 1) 1차 조건 인덱스
    if rsi_mode == "없음":
        rsi_idx = []
    elif rsi_mode == "현재(과매도/과매수 중 하나)":
        rsi_idx = sorted(set(df.index[df["RSI13"] <= float(rsi_low)]) | set(df.index[df["RSI13"] >= float(rsi_high)]))
    elif rsi_mode == "과매도 기준":
        rsi_idx = df.index[df["RSI13"] <= float(rsi_low)].tolist()
    else:
        rsi_idx = df.index[df["RSI13"] >= float(rsi_high)].tolist()

    def bb_ok(i):
        c = float(df.at[i,"close"]); up, lo, mid = df.at[i,"BB_up"], df.at[i,"BB_low"], df.at[i,"BB_mid"]
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

    i = 0
    while i < n:
        if i not in base_sig_idx: i += 1; continue
        anchor_idx = i
        signal_time = df.at[i,"time"]
        base_price = float(df.at[i,"close"])

        # 2차 조건
        if sec_cond == "양봉 2개 연속 상승":
            if i+2 >= n: i += 1; continue
            c1,o1 = float(df.at[i+1,"close"]), float(df.at[i+1,"open"])
            c2,o2 = float(df.at[i+2,"close"]), float(df.at[i+2,"open"])
            if not ((c1>o1) and (c2>o2) and (c2>c1)): i += 1; continue

        end_idx = anchor_idx + lookahead
        if end_idx >= n: i += 1; continue

        # 목표가 탐색
        target = base_price * (1.0 + thr/100.0)
        def price_for_hit(j):
            c, h = float(df.at[j,"close"]), float(df.at[j,"high"])
            if hit_basis.startswith("고가"): return h
            if hit_basis.startswith("종가 또는 고가"): return max(c,h)
            return c
        hit_idx = None
        for j in range(anchor_idx+1, end_idx+1):
            if price_for_hit(j) >= target: hit_idx = j; break

        if hit_idx is not None:
            end_time = df.at[hit_idx,"time"]
            end_close = target
            final_ret = thr
            result = "성공"
        else:
            end_time = df.at[end_idx,"time"]
            end_close = float(df.at[end_idx,"close"])
            final_ret = (end_close / base_price - 1) * 100
            result = "실패" if final_ret < 0 else "중립"

        res.append({
            "신호시간": signal_time, "종료시간": end_time,
            "기준시가": int(round(base_price)), "종료가": end_close,
            "성공기준(%)": round(thr,1), "결과": result, "최종수익률(%)": round(final_ret,2),
        })
        i = end_idx if dup_mode.startswith("중복 제거") else i+1

    return pd.DataFrame(res)

# -----------------------------
# 실행
# -----------------------------
try:
    if start_date > end_date:
        st.error("시작 날짜가 종료 날짜보다 이후입니다.")
        st.stop()

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt   = datetime.combine(end_date,   datetime.max.time())
    warmup_bars = max(13, bb_window) * 5

    df_raw = fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars)
    if df_raw.empty:
        st.error("데이터가 없습니다.")
        st.stop()

    df_ind = add_indicators(df_raw, bb_window, bb_dev)
    df = df_ind[(df_ind["time"] >= start_dt) & (df_ind["time"] <= end_dt)].reset_index(drop=True)
    bb_cond = st.session_state.get("bb_cond", bb_cond)

    # 시뮬레이션
    res_all = simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, threshold_pct,
                       bb_cond, "중복 포함 (연속 신호 모두)", minutes_per_bar, market_code, bb_window, bb_dev,
                       sec_cond=sec_cond, hit_basis=hit_basis)
    res_dedup = simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, threshold_pct,
                         bb_cond, "중복 제거 (연속 동일 결과 1개)", minutes_per_bar, market_code, bb_window, bb_dev,
                         sec_cond=sec_cond, hit_basis=hit_basis)
    res = res_all if dup_mode.startswith("중복 포함") else res_dedup

    # 요약 메트릭
    def summarize(df_in):
        if df_in is None or df_in.empty: return 0,0,0,0,0.0
        total = len(df_in)
        succ  = (df_in["결과"]=="성공").sum()
        fail  = (df_in["결과"]=="실패").sum()
        neu   = (df_in["결과"]=="중립").sum()
        win   = succ/total*100 if total else 0.0
        return total,succ,fail,neu,win

    total,succ,fail,neu,win = summarize(res)
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("신호 수", f"{total}")
    m2.metric("성공", f"{succ}")
    m3.metric("실패", f"{fail}")
    m4.metric("중립", f"{neu}")
    m5.metric("승률", f"{win:.1f}%")
    st.markdown("---")

    # 차트
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="가격", increasing_line_color="red", decreasing_line_color="blue", line=dict(width=1.1)
    ))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_up"],  mode="lines", line=dict(width=1.4), name="BB 상단"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_low"], mode="lines", line=dict(width=1.4), name="BB 하단"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_mid"], mode="lines", line=dict(width=1.1, dash="dot"), name="BB 중앙"))

    # 신호 시각화(점선)
    if not res.empty:
        for _, row in res.iterrows():
            sx, sy = pd.to_datetime(row["신호시간"]), float(row["기준시가"])
            ex, ey = pd.to_datetime(row["종료시간"]), float(row["종료가"])
            grp = row["결과"]; color = "red" if grp=="성공" else ("blue" if grp=="실패" else "#FF9800")
            fig.add_trace(go.Scatter(x=[sx,ex], y=[sy,ey], mode="lines",
                                     line=dict(width=1.4, dash="dot", color=color),
                                     name=f"신호-{grp}", showlegend=False))
            if grp=="성공":
                fig.add_trace(go.Scatter(x=[ex], y=[ey], mode="markers",
                                         marker=dict(size=14, symbol="star", line=dict(width=1,color="black")),
                                         name="목표 도달", showlegend=False))

    fig.update_layout(
        title=f"{market_label.split(' — ')[0]} · {tf_label} · RSI(13)+BB",
        dragmode="zoom", xaxis_rangeslider_visible=False, height=620,
        legend_orientation="h", legend_y=1.05, margin=dict(l=50,r=30,t=60,b=40),
        uirevision="constant"
    )
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "doubleClick": "reset"})

    # 결과 테이블
    st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)
    if res is None or res.empty:
        st.info("조건을 만족하는 신호가 없습니다.")
    else:
        tbl = res.sort_values("신호시간", ascending=False).reset_index(drop=True).copy()
        tbl["신호시간"] = pd.to_datetime(tbl["신호시간"]).dt.strftime("%Y-%m-%d %H:%M")
        tbl["기준시가"] = tbl["기준시가"].map(lambda v: f"{int(v):,}")
        for col in ["성공기준(%)","최종수익률(%)"]:
            tbl[col] = tbl[col].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "")
        st.dataframe(tbl, use_container_width=True)

except Exception as e:
    st.error(f"오류: {e}")
