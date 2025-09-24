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
# 페이지/스타일 (예전 UI/UX 유지)
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
  table {border-collapse:collapse; width:100%;}
  th, td {border:1px solid #ddd; padding:6px; text-align:center;}
</style>
""", unsafe_allow_html=True)

st.title("📊 코인 시뮬레이션")
st.markdown("<div style='margin-bottom:10px; color:gray;'>※ 차트 점선: 신호~판정 구간, 성공 시 도달 지점에 ⭐ 마커</div>", unsafe_allow_html=True)

# -----------------------------
# 마켓 목록 (예전 로직 유지)
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
# 차트-근처 선택으로 바꾼 값이 있으면 상단 select의 기본 인덱스에 반영 (UI 동일, 동작만 동기화)
def _index_for(code: str):
    return next((i for i, (_, c) in enumerate(MARKET_LIST) if c == code), 0)

default_idx = _index_for("KRW-BTC")
if "chart_market_override" in st.session_state:
    default_idx = _index_for(st.session_state["chart_market_override"])

# -----------------------------
# 타임프레임 (예전 로직 유지)
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
# 상단: 신호 중복 처리 (예전 UI 유지)
# -----------------------------
dup_mode = st.radio("신호 중복 처리", ["중복 포함 (연속 신호 모두)", "중복 제거 (연속 동일 결과 1개)"], horizontal=True)

# -----------------------------
# ① 기본 설정 (예전 UI 유지)
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
# ② 조건 설정 (예전 UI 유지)
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

miss_policy = st.selectbox("미도달 처리", ["실패(권장)","중립(미도달=항상 중립)","중립(예전: -thr 이하면 실패)"], index=0)
sec_cond = st.selectbox("2차 조건 선택", ["없음","양봉 2개 연속 상승","BB 기반 첫 양봉 50% 진입"], index=0)
st.session_state["bb_cond"] = bb_cond
st.markdown("---")

# -----------------------------
# 데이터 수집 (예전 로직 유지)
# -----------------------------
_session = requests.Session()
_retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])
_session.mount("https://", HTTPAdapter(max_retries=_retries))

@st.cache_data(ttl=120, show_spinner=False)
def fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars: int = 0):
    start_cutoff = start_dt - timedelta(minutes=max(0, warmup_bars) * minutes_per_bar)
    if "minutes/" in interval_key:
        unit = interval_key.split("/")[1]
        url = f"https://api.upbit.com/v1/candles/minutes/{unit}"
    else:
        url = "https://api.upbit.com/v1/candles/days"
    all_rows, to_time = [], None
    try:
        for _ in range(60):
            params = {"market": market_code, "count": 200}
            if to_time is not None:
                params["to"] = to_time.strftime("%Y-%m-%d %H:%M:%S")
            r = _session.get(url, params=params, headers={"Accept":"application/json"}, timeout=10)
            r.raise_for_status()
            batch = r.json()
            if not batch: break
            all_rows.extend(batch)
            last_kst = pd.to_datetime(batch[-1]["candle_date_time_kst"])
            if last_kst <= start_cutoff: break
            to_time = last_kst - timedelta(seconds=1)
    except Exception:
        return pd.DataFrame()
    if not all_rows: return pd.DataFrame()
    df = pd.DataFrame(all_rows).rename(columns={
        "candle_date_time_kst":"time",
        "opening_price":"open","high_price":"high","low_price":"low","trade_price":"close",
        "candle_acc_trade_volume":"volume"
    })
    df["time"] = pd.to_datetime(df["time"])
    df = df[["time","open","high","low","close","volume"]].sort_values("time").reset_index(drop=True)
    return df[(df["time"] >= start_cutoff) & (df["time"] <= end_dt)]

# -----------------------------
# 지표 (예전 로직 유지)
# -----------------------------
def add_indicators(df, bb_window, bb_dev):
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=int(bb_window), window_dev=float(bb_dev))
    out["BB_up"]  = bb.bollinger_hband()
    out["BB_low"] = bb.bollinger_lband()
    out["BB_mid"] = bb.bollinger_mavg()
    for c in ["RSI13","BB_up","BB_low","BB_mid"]:
        out[c] = out[c].fillna(method="bfill").fillna(method="ffill")
    return out

# -----------------------------
# 시뮬레이션 (예전 UI/UX에 맞춘 상세 결과 산출)
# -----------------------------
def simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, thr_pct, bb_cond, dedup_mode,
             minutes_per_bar, market_code, bb_window, bb_dev, sec_cond="없음",
             hit_basis="종가 기준", miss_policy="실패(권장)"):
    res = []
    n = len(df); thr = float(thr_pct)

    # 1) 1차 조건 인덱스
    if rsi_mode == "없음":
        rsi_idx = []
    elif rsi_mode == "현재(과매도/과매수 중 하나)":
        rsi_idx = sorted(set(df.index[df["RSI13"] <= float(rsi_low)].tolist()) |
                         set(df.index[df["RSI13"] >= float(rsi_high)].tolist()))
    elif rsi_mode == "과매도 기준":
        rsi_idx = df.index[df["RSI13"] <= float(rsi_low)].tolist()
    else:  # 과매수 기준
        rsi_idx = df.index[df["RSI13"] >= float(rsi_high)].tolist()

    def bb_ok(i):
        c = float(df.at[i,"close"])
        up, lo, mid = df.at[i,"BB_up"], df.at[i,"BB_low"], df.at[i,"BB_mid"]
        if bb_cond == "상한선":
            return pd.notna(up) and (c > float(up))
        if bb_cond == "하한선":
            return pd.notna(lo) and (c <= float(lo))
        if bb_cond == "중앙선":
            return pd.notna(mid) and (c >= float(mid))
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

    # 2) 보조 함수
    def is_bull(idx): return float(df.at[idx,"close"]) > float(df.at[idx,"open"])

    # 3) 메인 루프
    i = 0
    while i < n:
        if i not in base_sig_idx:
            i += 1
            continue

        anchor_idx = i
        signal_time = df.at[i,"time"]
        base_price = float(df.at[i,"close"])

        # 2차 조건들
        if sec_cond == "양봉 2개 연속 상승":
            if i + 2 >= n:
                i += 1; continue
            c1, o1 = float(df.at[i+1,"close"]), float(df.at[i+1,"open"])
            c2, o2 = float(df.at[i+2,"close"]), float(df.at[i+2,"open"])
            if not ((c1 > o1) and (c2 > o2) and (c2 > c1)):
                i += 1; continue

        elif sec_cond == "BB 기반 첫 양봉 50% 진입":
            # 간략화: 첫 양봉이 BB 참조선 이상이면 진입, 이후 2번째 양봉 확인 후 진행 (원본 동작 재현)
            ref_series = {"상한선":"BB_up","중앙선":"BB_mid","하한선":"BB_low"}.get(bb_cond,"BB_mid")
            B1_idx = None
            for j in range(i+1, min(i+lookahead+1, n)):
                if is_bull(j) and pd.notna(df.at[j, ref_series]) and float(df.at[j,"close"]) >= float(df.at[j, ref_series]):
                    B1_idx = j; break
            if B1_idx is None:
                i += 1; continue
            bull_cnt, B3_idx = 0, None
            for j in range(B1_idx+1, min(B1_idx+1+lookahead, n)):
                if is_bull(j):
                    bull_cnt += 1
                    if bull_cnt == 2:
                        B3_idx = j; break
            if B3_idx is None:
                i += 1; continue
            # 진입을 B3 이후로 본다
            anchor_idx = B3_idx
            signal_time = df.at[anchor_idx,"time"]
            base_price = float(df.at[anchor_idx,"close"])

        end_idx = anchor_idx + lookahead
        if end_idx >= n:
            i += 1; continue

        win = df.iloc[anchor_idx+1:end_idx+1]
        end_time = df.at[end_idx,"time"]
        end_close = float(df.at[end_idx,"close"])
        final_ret = (end_close/base_price - 1) * 100
        min_ret = (win["close"].min()/base_price - 1) * 100 if not win.empty else 0.0
        max_ret = (win["close"].max()/base_price - 1) * 100 if not win.empty else 0.0

        # 목표가 도달 확인
        target = base_price * (1.0 + thr/100.0)
        def _price_for_hit(j):
            c = float(df.at[j,"close"]); h = float(df.at[j,"high"])
            if hit_basis.startswith("고가"): return h
            if hit_basis.startswith("종가 또는 고가"): return max(c, h)
            return c
        hit_idx = None
        for j in range(anchor_idx+1, end_idx+1):
            if _price_for_hit(j) >= target:
                hit_idx = j; break

        if hit_idx is not None:
            end_time = df.at[hit_idx,"time"]
            end_close = target
            final_ret = thr
            result = "성공"
        else:
            if miss_policy.startswith("실패"):
                result = "실패"
            elif "항상 중립" in miss_policy:
                result = "중립"
            else:
                result = "실패" if final_ret <= -thr else "중립"

        # BB 표시값
        bb_val = None
        if bb_cond == "상한선": bb_val = df.at[anchor_idx,"BB_up"]
        elif bb_cond == "중앙선": bb_val = df.at[anchor_idx,"BB_mid"]
        elif bb_cond == "하한선": bb_val = df.at[anchor_idx,"BB_low"]

        res.append({
            "신호시간": signal_time,
            "종료시간": end_time,
            "기준시가": int(round(base_price)),
            "종료가": end_close,
            "RSI(13)": round(float(df.at[anchor_idx,"RSI13"]), 1) if pd.notna(df.at[anchor_idx,"RSI13"]) else None,
            "BB값": round(float(bb_val), 1) if (bb_val is not None and pd.notna(bb_val)) else None,
            "성공기준(%)": round(thr, 1),
            "결과": result,
            "최종수익률(%)": round(final_ret, 2),
            "최저수익률(%)": round(min_ret, 2),
            "최고수익률(%)": round(max_ret, 2),
            "anchor_idx": anchor_idx  # 차트 표시용
        })

        i = end_idx if dup_mode.startswith("중복 제거") else i + 1

    return pd.DataFrame(res)

# -----------------------------
# 실행
# -----------------------------
try:
    if start_date > end_date:
        st.error("시작 날짜가 종료 날짜보다 이후입니다.")
        st.stop()

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt   = datetime.combine(end_date, datetime.max.time())
    warmup_bars = max(13, bb_window) * 5

    df_raw = fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars=warmup_bars)
    if df_raw.empty:
        st.error("데이터가 없습니다.")
        st.stop()

    df_ind = add_indicators(df_raw, bb_window, bb_dev)
    df = df_ind[(df_ind["time"] >= start_dt) & (df_ind["time"] <= end_dt)].reset_index(drop=True)
    bb_cond = st.session_state.get("bb_cond", bb_cond)

    # -----------------------------
    # 🔄 차트 컨트롤 (추가) — UI 배치는 기존 레이아웃 존중
    # -----------------------------
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = datetime.now()
    st.markdown("### 🔄 차트 컨트롤")
    cc1, cc2 = st.columns([1,2])
    with cc1:
        # 3초 딜레이
        if st.button("🔄 새로고침", use_container_width=True):
            now = datetime.now()
            if (now - st.session_state["last_refresh"]).total_seconds() >= 3:
                st.session_state["last_refresh"] = now
                st.rerun()  # Streamlit 1.50.0
            else:
                st.warning("새로고침은 3초 간격으로만 가능합니다.")
    with cc2:
        sel_idx2 = _index_for(market_code)
        market_label2, market_code2 = st.selectbox("차트 근처 종목 선택", MARKET_LIST, index=sel_idx2, format_func=lambda x: x[0], key="chart_market_select")
        if market_code2 != market_code:
            st.session_state["chart_market_override"] = market_code2
            st.rerun()

    st.markdown("---")

    # -----------------------------
    # 차트 (예전 UI/UX 그대로)
    # -----------------------------
    fig = make_subplots(rows=1, cols=1)
    # 캔들
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="가격", increasing_line_color="red", decreasing_line_color="blue", line=dict(width=1.1)
    ))
    # BB 3선
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_up"],  mode="lines", line=dict(color="#FFB703", width=1.4), name="BB 상단"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_low"], mode="lines", line=dict(color="#219EBC", width=1.4), name="BB 하한"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_mid"], mode="lines", line=dict(color="#8D99AE", width=1.1, dash="dot"), name="BB 중앙"))
    # RSI (보조축)
    fig.add_trace(go.Scatter(x=df["time"], y=df["RSI13"], mode="lines",
                             line=dict(color="rgba(42,157,143,0.30)", width=6), yaxis="y2", showlegend=False))
    fig.add_trace(go.Scatter(x=df["time"], y=df["RSI13"], mode="lines",
                             line=dict(color="#2A9D8F", width=2.4, dash="dot"), name="RSI(13)", yaxis="y2"))
    fig.add_hline(y=70, line_dash="dash", line_color="#E63946", line_width=1.1, yref="y2")
    fig.add_hline(y=30, line_dash="dash", line_color="#457B9D", line_width=1.1, yref="y2")

    # 신호 시각화(점선 + ⭐)
    res_all = simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, threshold_pct,
                       bb_cond, "중복 포함 (연속 신호 모두)", minutes_per_bar, market_code, bb_window, bb_dev,
                       sec_cond=sec_cond, hit_basis=hit_basis, miss_policy=miss_policy)
    res_dedup = simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, threshold_pct,
                         bb_cond, "중복 제거 (연속 동일 결과 1개)", minutes_per_bar, market_code, bb_window, bb_dev,
                         sec_cond=sec_cond, hit_basis=hit_basis, miss_policy=miss_policy)
    res = res_all if dup_mode.startswith("중복 포함") else res_dedup

    if not res.empty:
        # 신호 마커
        for _label, _color in [("성공","red"), ("실패","blue"), ("중립","#FF9800")]:
            sub = res[res["결과"] == _label]
            if sub.empty: continue
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(sub["신호시간"]), y=sub["기준시가"], mode="markers",
                name=f"신호({_label})",
                marker=dict(size=9, color=_color, symbol="circle", line=dict(width=1, color="black"))
            ))

        legend_emitted = {"성공": False, "실패": False, "중립": False}
        for _, row in res.iterrows():
            start_x = pd.to_datetime(row["신호시간"]); start_y = float(row["기준시가"])
            end_x = pd.to_datetime(row["종료시간"]);   end_y = float(row["종료가"])
            grp = row["결과"]; color = "red" if grp=="성공" else ("blue" if grp=="실패" else "#FF9800")
            # 점선 연결
            fig.add_trace(go.Scatter(
                x=[start_x, end_x], y=[start_y, end_y], mode="lines",
                line=dict(color=color, width=1.6 if grp=="성공" else 1.0, dash="dot"),
                opacity=0.9 if grp=="성공" else 0.5,
                showlegend=(not legend_emitted[grp]),
                name=f"신호(점선)-{grp}"
            ))
            legend_emitted[grp] = True
            # 도달 마커
            if grp == "성공":
                # ⭐
                hit_row = df.loc[df["time"] == end_x]
                star_y = float(hit_row.iloc[0]["high"]) if not hit_row.empty else end_y
                fig.add_trace(go.Scatter(
                    x=[end_x], y=[star_y], mode="markers", name="목표 도달",
                    marker=dict(size=15, color="orange", symbol="star", line=dict(width=1, color="black")),
                    showlegend=False
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=[end_x], y=[end_y], mode="markers", name=f"도착-{grp}",
                    marker=dict(size=8, color=color, symbol="x", line=dict(width=1, color="black")),
                    showlegend=False
                ))

    fig.update_layout(
        title=f"{market_label.split(' — ')[0]} · {tf_label} · RSI(13) + BB 시뮬레이션",
        dragmode="zoom", xaxis_rangeslider_visible=False, height=600,
        legend_orientation="h", legend_y=1.05,
        margin=dict(l=60, r=40, t=60, b=40),
        yaxis=dict(title="가격"),
        yaxis2=dict(overlaying="y", side="right", showgrid=False, title="RSI(13)", range=[0, 100]),
        uirevision="chart-view"  # 🔒 뷰(줌/스크롤) 유지
    )
    # 예전 코드와 동일하게 컨테이너 폭 사용
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "doubleClick": "reset"})

    # -----------------------------
    # ④ 신호 결과 (예전 UI/UX 그대로)
    # -----------------------------
    st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)
    if res.empty:
        st.info("조건을 만족하는 신호가 없습니다. (데이터는 정상 처리됨)")
    else:
        tbl = res.sort_values("신호시간", ascending=False).reset_index(drop=True).copy()
        # 표시 형식 (예전과 동일)
        tbl["신호시간"] = pd.to_datetime(tbl["신호시간"]).dt.strftime("%Y-%m-%d %H:%M")
        tbl["기준시가"] = tbl["기준시가"].map(lambda v: f"{int(v):,}")
        for col in ["RSI(13)","BB값"]:
            if col in tbl:
                tbl[col] = tbl[col].map(lambda v: f"{v:.1f}" if pd.notna(v) else "")
        for col in ["성공기준(%)","최종수익률(%)","최저수익률(%)","최고수익률(%)"]:
            if col in tbl:
                tbl[col] = tbl[col].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "")

        # 도달시간/도달캔들 (표시용)
        def fmt_hhmm(s, e):
            try:
                s = pd.to_datetime(s); e = pd.to_datetime(e)
                m = int((e - s).total_seconds() // 60); h, mm = divmod(m, 60)
                return f"{h:02d}:{mm:02d}"
            except Exception:
                return "-"
        def bars_after(s, e):
            try:
                s = pd.to_datetime(s); e = pd.to_datetime(e)
                mins = int(round((e - s).total_seconds() / 60))
                return int(round(mins / minutes_per_bar))
            except Exception:
                return None
        tbl["도달시간"] = [fmt_hhmm(res.loc[i,"신호시간"], res.loc[i,"종료시간"]) for i in range(len(res))]
        tbl["도달캔들"] = [bars_after(res.loc[i,"신호시간"], res.loc[i,"종료시간"]) for i in range(len(res))]

        # 내부 계산 컬럼 제거
        if "anchor_idx" in tbl: tbl = tbl.drop(columns=["anchor_idx"])

        # 컬럼 순서 (예전 표 구성)
        cols_order = ["신호시간","기준시가","RSI(13)","성공기준(%)","결과","최종수익률(%)","최저수익률(%)","최고수익률(%)","도달캔들","도달시간"]
        tbl = [c for c in cols_order if c in tbl.columns]
        tbl = res.sort_values("신호시간", ascending=False).reset_index(drop=True)
        # 위에서 만든 형식 적용
        tbl["신호시간"] = pd.to_datetime(tbl["신호시간"]).dt.strftime("%Y-%m-%d %H:%M")
        try:
            tbl["기준시가"] = tbl["기준시가"].map(lambda v: f"{int(v):,}")
        except Exception:
            pass
        if "RSI(13)" in tbl: tbl["RSI(13)"] = tbl["RSI(13)"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "")
        if "BB값"   in tbl: tbl["BB값"]   = tbl["BB값"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "")
        for col in ["성공기준(%)","최종수익률(%)","최저수익률(%)","최고수익률(%)"]:
            if col in tbl: tbl[col] = tbl[col].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "")
        # 스타일 (성공/실패/중립 색상)
        def style_result(v):
            if v == "성공": return "background-color:#FFF59D; color:#E53935;"
            if v == "실패": return "color:#1E40AF;"
            if v == "중립": return "color:#FF9800;"
            return ""
        styled_tbl = tbl.style.applymap(style_result, subset=["결과"])
        st.dataframe(styled_tbl, use_container_width=True)

except Exception as e:
    st.error(f"오류: {e}")
