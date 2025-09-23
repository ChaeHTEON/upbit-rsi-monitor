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

st.set_page_config(page_title="Upbit RSI(13) + Bollinger Band 시뮬레이터", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 0.8rem; padding-bottom: 0.8rem; max-width: 1100px;}
  .section-title {font-size:1.05rem; font-weight:700; margin: 0.6rem 0 0.2rem;}
  .hint {color:#6b7280;}
  table {border-collapse:collapse; width:100%;}
  th, td {border:1px solid #ddd; padding:6px; text-align:center;}
</style>
""", unsafe_allow_html=True)

st.title("📊 코인 시뮬레이션")
st.markdown("<div style='margin-bottom:10px; color:gray;'>점선=신호~판정, ⭐=목표 도달</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 마켓 목록
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
                rows.append((f'{it.get("korean_name","")} ({sym}) — {mk}', mk))
        rows.sort(key=lambda x: (x[1] != "KRW-BTC", x[1]))
        if rows: return rows
    except Exception:
        pass
    return [("비트코인 (BTC) — KRW-BTC", "KRW-BTC")]

MARKET_LIST = get_upbit_krw_markets()
default_idx = next((i for i, (_, code) in enumerate(MARKET_LIST) if code == "KRW-BTC"), 0)

TF_MAP = {
    "1분": ("minutes/1", 1), "3분": ("minutes/3", 3), "5분": ("minutes/5", 5),
    "15분": ("minutes/15", 15), "30분": ("minutes/30", 30), "60분": ("minutes/60", 60),
    "일봉": ("days", 24*60),
}

dup_mode = st.radio("신호 중복 처리", ["중복 포함 (연속 신호 모두)", "중복 제거 (연속 동일 결과 1개)"], horizontal=True)

st.markdown('<div class="section-title">① 기본 설정</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    market_label, market_code = st.selectbox("종목 선택", MARKET_LIST, index=default_idx, format_func=lambda x: x[0])
with c2:
    tf_label = st.selectbox("봉 종류 선택", list(TF_MAP.keys()), index=2)
with c3:
    KST = timezone("Asia/Seoul")
    today_kst = datetime.now(KST).date()
    start_date = st.date_input("시작 날짜", value=today_kst - timedelta(days=1))
    end_date = st.date_input("종료 날짜", value=today_kst)

interval_key, minutes_per_bar = TF_MAP[tf_label]
st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# 조건
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">② 조건 설정</div>', unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
with c4:
    lookahead = st.slider("측정 캔들 수 (기준 이후 N봉)", 1, 60, 10)
with c5:
    threshold_pct = st.slider("성공/실패 기준 값(%)", 0.1, 5.0, 1.0, step=0.1)
    hit_basis = st.selectbox("성공 판정 기준",
        ["종가 기준", "고가 기준(스침 인정)", "종가 또는 고가"], index=0)
with c6:
    r1, r2, r3 = st.columns(3)
    with r1:
        rsi_mode = st.selectbox("RSI 조건",
            ["없음", "현재(과매도/과매수 중 하나)", "과매도 기준", "과매수 기준"], index=0)
    with r2: rsi_low = st.slider("과매도 RSI 기준", 0, 100, 30, step=1)
    with r3: rsi_high = st.slider("과매수 RSI 기준", 0, 100, 70, step=1)

c7, c8, c9 = st.columns(3)
with c7:
    bb_cond = st.selectbox("볼린저밴드 조건", ["없음", "상한선", "중앙선", "하한선"], index=0)
with c8:
    bb_window = st.number_input("BB 기간", min_value=5, max_value=100, value=30, step=1)
with c9:
    bb_dev = st.number_input("BB 승수", min_value=1.0, max_value=4.0, value=2.0, step=0.1)

miss_policy = st.selectbox(
    "미도달 처리",
    ["실패(권장)", "중립(미도달=항상 중립)", "중립(예전: -thr 이하면 실패)"],
    index=0
)

st.markdown('<div class="hint">2차 조건</div>', unsafe_allow_html=True)
sec_cond = st.selectbox(
    "2차 조건 선택",
    ["없음", "양봉 2개 연속 상승", "양봉 2개 (범위 내)", "BB 기반 첫 양봉 50% 진입"],
    index=0
)
st.session_state["bb_cond"] = bb_cond
st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# 데이터 수집(워밍업 포함)
# ──────────────────────────────────────────────────────────────────────────────
def estimate_calls(start_dt, end_dt, minutes_per_bar):
    mins = max(1, int((end_dt - start_dt).total_seconds() // 60))
    bars = max(1, mins // minutes_per_bar)
    return bars // 200 + 1

_session = requests.Session()
_session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.5,
                    status_forcelist=[429, 500, 502, 503, 504])))

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
    req_count, all_data, to_time = 200, [], None
    try:
        for _ in range(max_calls):
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
        "candle_date_time_kst": "time",
        "opening_price":"open","high_price":"high","low_price":"low",
        "trade_price":"close","candle_acc_trade_volume":"volume"})
    df["time"] = pd.to_datetime(df["time"])
    df = df[["time","open","high","low","close","volume"]].sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return df[(df["time"] >= start_cutoff) & (df["time"] <= end_dt)]

def add_indicators(df, bb_window, bb_dev):
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=bb_window, window_dev=bb_dev)
    out["BB_up"] = bb.bollinger_hband().fillna(method="bfill").fillna(method="ffill")
    out["BB_low"] = bb.bollinger_lband().fillna(method="bfill").fillna(method="ffill")
    out["BB_mid"] = bb.bollinger_mavg().fillna(method="bfill").fillna(method="ffill")
    return out

# ──────────────────────────────────────────────────────────────────────────────
# 시뮬레이션
# ──────────────────────────────────────────────────────────────────────────────
def simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, thr_pct, bb_cond, dedup_mode,
             minutes_per_bar, market_code, bb_window, bb_dev, sec_cond="없음",
             hit_basis="종가 기준", miss_policy="실패(권장)"):
    res, n, thr = [], len(df), float(threshold_pct)

    # 1차 조건 인덱스
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
        c = float(df.at[i, "close"]); up, lo, mid = df.at[i, "BB_up"], df.at[i, "BB_low"], df.at[i, "BB_mid"]
        if bb_cond == "상한선": return pd.notna(up) and (c > float(up))
        if bb_cond == "하한선": return pd.notna(lo) and (c <= float(lo))
        if bb_cond == "중앙선": return pd.notna(mid) and (c >= float(mid))
        return False

    bb_idx = [i for i in df.index if bb_cond != "없음" and bb_ok(i)]
    if rsi_mode != "없음" and bb_cond != "없음": base_sig_idx = sorted(set(rsi_idx) & set(bb_idx))
    elif rsi_mode != "없음":                  base_sig_idx = rsi_idx
    elif bb_cond != "없음":                   base_sig_idx = bb_idx
    else:                                     base_sig_idx = list(range(n)) if sec_cond != "없음" else []

    def is_bull(idx): return float(df.at[idx, "close"]) > float(df.at[idx, "open"])

    def first_bull_50_over_bb(start_i):
        for j in range(start_i + 1, n):
            if not is_bull(j): continue
            ref = df.at[j, {"하한선":"BB_low","중앙선":"BB_mid","상한선":"BB_up"}[bb_cond]] if bb_cond!="없음" else np.nan
            if pd.isna(ref): continue
            if float(df.at[j, "close"]) >= float(ref): return j, float(df.at[j, "close"])
        return None, None

    i = 0
    while i < n:
        if i not in base_sig_idx:
            i += 1; continue

        anchor_idx = i
        signal_time = df.at[i, "time"]
        base_price = float(df.at[i, "close"])

        # ── 2차 조건들
        if sec_cond == "양봉 2개 연속 상승":
            if i + 2 >= n: i += 1; continue
            c1, o1 = float(df.at[i+1,"close"]), float(df.at[i+1,"open"])
            c2, o2 = float(df.at[i+2,"close"]), float(df.at[i+2,"open"])
            if not ((c1 > o1) and (c2 > o2) and (c2 > c1)):
                i += 1; continue

        elif sec_cond == "양봉 2개 (범위 내)":
            bull_cnt, scan_end = 0, min(i + lookahead, n - 1)
            for j in range(i + 1, scan_end + 1):
                if float(df.at[j,"close"]) > float(df.at[j,"open"]):
                    bull_cnt += 1
                    if bull_cnt >= 2: break
            if bull_cnt < 2:
                i += 1; continue

        elif sec_cond == "BB 기반 첫 양봉 50% 진입":
            B1_idx, B1_close = first_bull_50_over_bb(i)
            if B1_idx is None: i += 1; continue
            bull_cnt, B3_idx = 0, None
            for j in range(B1_idx + 1, min(B1_idx + lookahead, n - 1) + 1):
                if is_bull(j):
                    bull_cnt += 1
                    if bull_cnt == 2:
                        B3_idx = j; break
            if B3_idx is None: i += 1; continue
            T_idx = None
            for j in range(B3_idx + 1, n):
                if pd.notna(df.at[j,"close"]) and float(df.at[j,"close"]) >= B1_close:
                    T_idx = j; break
            if T_idx is None: i += 1; continue
            anchor_idx, signal_time, base_price = T_idx, df.at[T_idx,"time"], float(df.at[T_idx,"close"])

        # ── 성과 측정
        end_idx = anchor_idx + lookahead
        if end_idx >= n:
            i += 1; continue

        win_slice = df.iloc[anchor_idx + 1:end_idx + 1]
        end_time  = df.at[end_idx, "time"]
        end_close = float(df.at[end_idx, "close"])
        final_ret = (end_close / base_price - 1) * 100

        target = base_price * (1.0 + thr / 100.0)
        result, reach_min, hit_idx = "중립", None, None

        def price_for_hit(j):
            c, h = float(df.at[j,"close"]), float(df.at[j,"high"])
            if hit_basis.startswith("고가"): return h
            if hit_basis.startswith("종가 또는 고가"): return max(c, h)
            return c

        for j in range(anchor_idx + 1, end_idx + 1):
            if price_for_hit(j) >= target:
                hit_idx = j; break

        if hit_idx is not None:
            bars_after = hit_idx - anchor_idx
            reach_min  = bars_after * minutes_per_bar
            end_time   = df.at[hit_idx, "time"]
            end_close  = target
            final_ret  = thr
            result     = "성공"
        else:
            if miss_policy.startswith("실패"):
                result = "실패"
            elif "항상 중립" in miss_policy:
                result = "중립"
            else:
                result = "실패" if final_ret <= -thr else "중립"

        # Debug (최대 5개)
        if len(res) < 5:
            st.write({
                "anchor_idx": anchor_idx, "lookahead": lookahead, "hit_idx": hit_idx,
                "bars_after": (hit_idx - anchor_idx) if hit_idx is not None else None,
                "signal_time": str(signal_time), "end_time": str(end_time),
                "hit_basis": hit_basis, "miss_policy": miss_policy, "result": result
            })

        # 결과 적재
        bb_value = (df.at[anchor_idx,"BB_up"] if bb_cond=="상한선" else
                    df.at[anchor_idx,"BB_mid"] if bb_cond=="중앙선" else
                    df.at[anchor_idx,"BB_low"] if bb_cond=="하한선" else None)

        res.append({
            "신호시간": signal_time, "종료시간": end_time,
            "기준시가": int(round(base_price)), "종료가": end_close,
            "RSI(13)": round(float(df.at[anchor_idx,"RSI13"]), 1) if pd.notna(df.at[anchor_idx,"RSI13"]) else None,
            "BB값": round(float(bb_value), 1) if (bb_value is not None and pd.notna(bb_value)) else None,
            "성공기준(%)": round(thr, 1), "결과": result,
            "최종수익률(%)": round(final_ret, 2),
            "최저수익률(%)": round(((win_slice["close"].min()/base_price - 1)*100) if not win_slice.empty else 0.0, 2),
            "최고수익률(%)": round(((win_slice["close"].max()/base_price - 1)*100) if not win_slice.empty else 0.0, 2),
        })

        i = end_idx if dup_mode.startswith("중복 제거") else i + 1

    return pd.DataFrame(res)

# ──────────────────────────────────────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────────────────────────────────────
try:
    if start_date > end_date:
        st.error("시작 날짜가 종료 날짜보다 이후입니다."); st.stop()

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt   = datetime.combine(end_date,   datetime.max.time())

    warmup_bars = max(13, bb_window) * 5
    df_raw = fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars=warmup_bars)
    if df_raw.empty:
        st.error("데이터가 없습니다."); st.stop()

    df = add_indicators(df_raw, bb_window, bb_dev)
    df = df[(df["time"] >= start_dt) & (df["time"] <= end_dt)].reset_index(drop=True)
    bb_cond = st.session_state.get("bb_cond", bb_cond)

    hh, mm = divmod(int(lookahead * minutes_per_bar), 60)
    rsi_txt = ("없음" if rsi_mode=="없음" else
               f"현재(≤{int(rsi_low)} or ≥{int(rsi_high)})" if rsi_mode.startswith("현재") else
               f"과매도≤{int(rsi_low)}" if rsi_mode=="과매도 기준" else f"과매수≥{int(rsi_high)}")
    st.markdown('<div class="section-title">③ 요약 & 차트</div>', unsafe_allow_html=True)
    st.info(
        "설정 요약\n"
        f"- 측정 구간: {lookahead}봉 / {hh:02d}:{mm:02d}\n"
        f"- 1차 조건 · RSI: {rsi_txt} · BB: {bb_cond if bb_cond!='없음' else '없음'}\n"
        f"- 2차 조건 · {sec_cond}\n"
        f"- 성공 판정 기준: {hit_basis}\n"
        f"- 미도달 처리: {miss_policy}\n"
        f"- 워밍업: {warmup_bars}봉"
    )

    res_all = simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, threshold_pct,
                       bb_cond, "중복 포함 (연속 신호 모두)", minutes_per_bar, market_code, bb_window, bb_dev,
                       sec_cond=sec_cond, hit_basis=hit_basis, miss_policy=miss_policy)
    res_dedup = simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, threshold_pct,
                         bb_cond, "중복 제거 (연속 동일 결과 1개)", minutes_per_bar, market_code, bb_window, bb_dev,
                         sec_cond=sec_cond, hit_basis=hit_basis, miss_policy=miss_policy)
    res = res_all if dup_mode.startswith("중복 포함") else res_dedup

    # ── 차트
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="가격", increasing_line_color="red", decreasing_line_color="blue", line=dict(width=1.1)
    ))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_up"],  mode="lines", line=dict(color="#FFB703", width=1.4), name="BB 상단"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_low"], mode="lines", line=dict(color="#219EBC", width=1.4), name="BB 하단"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_mid"], mode="lines", line=dict(color="#8D99AE", width=1.1, dash="dot"), name="BB 중앙"))

    if not res.empty:
        for _label, _color in [("성공","red"),("실패","blue"),("중립","#FF9800")]:
            sub = res[res["결과"]==_label]
            if sub.empty: continue
            fig.add_trace(go.Scatter(
                x=sub["신호시간"], y=sub["기준시가"], mode="markers",
                name=f"신호({_label})", marker=dict(size=9, color=_color, symbol="circle", line=dict(width=1, color="black"))
            ))
        legend_emitted = {"성공":False,"실패":False,"중립":False}
        for _, row in res.iterrows():
            sx, sy = pd.to_datetime(row["신호시간"]), float(row["기준시가"])
            ex, ec = pd.to_datetime(row["종료시간"]), float(row["종료가"])
            grp = row["결과"]; color = "red" if grp=="성공" else ("blue" if grp=="실패" else "#FF9800")
            fig.add_trace(go.Scatter(x=[sx,ex], y=[sy,ec], mode="lines",
                                     line=dict(color=color, width=1.6 if grp=="성공" else 1.0, dash="dot"),
                                     opacity=0.9 if grp=="성공" else 0.5,
                                     showlegend=(not legend_emitted[grp]),
                                     name=f"신호(점선)-{grp}"))
            legend_emitted[grp]=True
            if grp=="성공":
                hit_row = df.loc[df["time"]==ex]
                star_y = float(hit_row.iloc[0]["high"]) if not hit_row.empty else ec
                fig.add_trace(go.Scatter(x=[ex], y=[star_y], mode="markers", name="목표 도달",
                                         marker=dict(size=15, color="orange", symbol="star", line=dict(width=1, color="black")),
                                         showlegend=False))
            else:
                fig.add_trace(go.Scatter(x=[ex], y=[ec], mode="markers", name=f"도착-{grp}",
                                         marker=dict(size=8, color=color, symbol="x", line=dict(width=1, color="black")),
                                         showlegend=False))

    fig.add_trace(go.Scatter(x=df["time"], y=df["RSI13"], mode="lines",
                             line=dict(color="#2A9D8F", width=2.4, dash="dot"),
                             name="RSI(13)", yaxis="y2"))
    fig.update_layout(
        title=f"{market_label.split(' — ')[0]} · {tf_label} · RSI(13) + BB 시뮬레이션",
        xaxis_rangeslider_visible=False, height=600, legend_orientation="h", legend_y=1.05,
        margin=dict(l=60, r=40, t=60, b=40),
        yaxis=dict(title="가격"),
        yaxis2=dict(overlaying="y", side="right", showgrid=False, title="RSI(13)", range=[0,100])
    )
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "doubleClick": "reset"})

    # ── 표
    st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)
    if res is None or res.empty:
        st.info("조건을 만족하는 신호가 없습니다.")
    else:
        tbl = res.sort_values("신호시간", ascending=False).reset_index(drop=True).copy()

        # 도달시간/도달캔들: 정렬된 tbl로 직접 계산(버그픽스)
        s_series = pd.to_datetime(tbl["신호시간"])
        e_series = pd.to_datetime(tbl["종료시간"])
        diff_min = ((e_series - s_series).dt.total_seconds() / 60).round().astype(int)
        bars_after = (diff_min / minutes_per_bar).round().astype(int)
        tbl["도달캔들"] = bars_after
        tbl.loc[tbl["결과"] != "성공", "도달캔들"] = lookahead
        tbl["도달시간"] = diff_min.apply(lambda m: f"{m//60:02d}:{m%60:02d}")

        # 표시 포맷
        tbl["신호시간"] = s_series.dt.strftime("%Y-%m-%d %H:%M")
        tbl["기준시가"] = tbl["기준시가"].map(lambda v: f"{int(v):,}")
        if "RSI(13)" in tbl: tbl["RSI(13)"] = tbl["RSI(13)"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "")
        if "BB값" in tbl:    tbl["BB값"] = tbl["BB값"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "")
        for col in ["성공기준(%)","최종수익률(%)","최저수익률(%)","최고수익률(%)"]:
            if col in tbl: tbl[col] = tbl[col].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "")

        tbl = tbl[["신호시간","기준시가","RSI(13)","성공기준(%)","결과",
                   "최종수익률(%)","최저수익률(%)","최고수익률(%)","도달캔들","도달시간"]]
        st.dataframe(tbl, use_container_width=True)

except Exception as e:
    st.error(f"오류: {e}")
