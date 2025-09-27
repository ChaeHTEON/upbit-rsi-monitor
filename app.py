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
  .success-cell {background-color:#FFF59D; color:#E53935; font-weight:600;}
  .fail-cell {color:#1E40AF; font-weight:600;}
  .neutral-cell {color:#FF9800; font-weight:600;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# API 호출 (업비트 캔들)
# -----------------------------
def fetch_upbit(symbol, interval, count, to=None):
    url = f"https://api.upbit.com/v1/candles/{interval}"
    headers = {"Accept": "application/json"}
    params = {"market": symbol, "count": count}
    if to:
        params["to"] = to
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    r = session.get(url, headers=headers, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul")
    df = df.rename(columns={"opening_price": "open", "high_price": "high", "low_price": "low", "trade_price": "close", "candle_acc_trade_volume": "volume"})
    df = df[["time", "open", "high", "low", "close", "volume"]]
    df = df.iloc[::-1].reset_index(drop=True)
    return df

# -----------------------------
# 보조지표 계산
# -----------------------------
def add_indicators(df, bb_window=20, bb_k=2):
    if df.empty: 
        return df
    df["RSI13"] = ta.momentum.RSIIndicator(df["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(df["close"], window=bb_window, window_dev=bb_k)
    df["BB_up"] = bb.bollinger_hband()
    df["BB_mid"] = bb.bollinger_mavg()
    df["BB_low"] = bb.bollinger_lband()
    return df

# -----------------------------
# 신호 판정 함수 (process_one)
# -----------------------------
def process_one(df, i0, thr, lookahead, minutes_per_bar, hit_basis, bb_cond, sec_cond, manual_supply_levels):
    n = len(df)
    if i0 >= n - 1:
        return None, None

    anchor_idx = i0
    signal_time = df.at[i0, "time"]
    base_price = float(df.at[i0, "close"])

    # (2차 조건 처리부는 최신 코드 그대로 유지)

    # --- 성과 측정 (단일 공식) ---
    end_scan = min(anchor_idx + lookahead, n - 1)
    win_slice = df.iloc[anchor_idx + 1:end_scan + 1]
    min_ret = (win_slice["close"].min() / base_price - 1) * 100 if not win_slice.empty else 0.0
    max_ret = (win_slice["close"].max() / base_price - 1) * 100 if not win_slice.empty else 0.0

    target = base_price * (1.0 + thr / 100.0)
    hit_idx = None
    for j in range(anchor_idx + 1, end_scan + 1):
        c_ = float(df.at[j, "close"])
        h_ = float(df.at[j, "high"])
        price_for_hit = max(c_, h_) if hit_basis.startswith("종가 또는 고가") else (h_ if hit_basis.startswith("고가") else c_)
        if price_for_hit >= target * 0.9999:
            hit_idx = j
            break

    if hit_idx is not None:
        end_i = hit_idx
        end_close = target
        final_ret = thr
        result = "성공"
    else:
        end_i = end_scan
        end_close = float(df.at[end_i, "close"])
        final_ret = (end_close / base_price - 1) * 100
        result = "실패" if final_ret <= 0 else "중립"

    bars_after = int(end_i - anchor_idx)
    reach_min = bars_after * minutes_per_bar
    end_time = df.at[end_i, "time"]

    bb_value = None
    if bb_cond == "상한선":
        bb_value = df.at[anchor_idx, "BB_up"]
    elif bb_cond == "중앙선":
        bb_value = df.at[anchor_idx, "BB_mid"]
    elif bb_cond == "하한선":
        bb_value = df.at[anchor_idx, "BB_low"]

    row = {
        "신호시간": signal_time,
        "종료시간": end_time,
        "기준시가": int(round(base_price)),
        "종료가": end_close,
        "RSI(13)": round(float(df.at[anchor_idx, "RSI13"]), 1) if pd.notna(df.at[anchor_idx, "RSI13"]) else None,
        "BB값": round(float(bb_value), 1) if (bb_value is not None and pd.notna(bb_value)) else None,
        "성공기준(%)": round(thr, 1),
        "결과": result,
        "도달분": reach_min,
        "도달캔들(bars)": bars_after,
        "최종수익률(%)": round(final_ret, 2),
        "최저수익률(%)": round(min_ret, 2),
        "최고수익률(%)": round(max_ret, 2),
        "anchor_i": int(anchor_idx),
        "end_i": int(end_i),
    }
    return row, end_i

# -----------------------------
# 메인 실행부
# -----------------------------
try:
    # ===== 사이드바 입력 =====
    symbol = st.sidebar.selectbox(
        "마켓 선택",
        ["KRW-BTC", "KRW-ETH", "KRW-XRP"],
        index=0
    )

    # Upbit API 규칙에 맞는 interval 값 사용
    interval = st.sidebar.selectbox(
        "봉 간격",
        ["minutes/1", "minutes/5", "minutes/15", "minutes/60", "days"],
        index=2
    )

    count = st.sidebar.slider("조회 캔들 수", min_value=50, max_value=500, value=200, step=10)
    to = None  # 최신 시점까지 불러오기

    # 데이터 수집 및 지표
    df = fetch_upbit(symbol, interval, count, to)
    df = add_indicators(df)

    # (시뮬레이션 및 신호 계산 루프 → res_all, res_dedup 생성)

    res = res_all if dup_mode.startswith("중복 포함") else res_dedup

    # 🔒 차트-표 정합성 보정
    if res is not None and not res.empty:
        for col in ("anchor_i", "end_i"):
            if col in res.columns:
                res[col] = pd.to_numeric(res[col], errors="coerce").fillna(-1).astype(int)
        res = res[
            (res["anchor_i"] >= 0) & (res["end_i"] >= 0) &
            (res["anchor_i"] < len(df)) & (res["end_i"] < len(df))
        ]

    # ===== 신호 마커/점선 =====
    if res is not None and not res.empty:
        # 1) anchor 마커
        for _label, _color in [("성공", "red"), ("실패", "blue"), ("중립", "#FF9800")]:
            sub = res[res["결과"] == _label]
            if sub.empty: continue
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(sub["신호시간"]), y=sub["기준시가"],
                mode="markers", name=f"신호({_label})",
                marker=dict(size=9, color=_color, symbol="circle", line=dict(width=1, color="black"))
            ))

        legend_emitted = {"성공": False, "실패": False, "중립": False}

        # 2) 점선/종료 마커 (anchor_i/end_i 직접 사용)
        for _, row in res.iterrows():
            a_i = int(row["anchor_i"]); e_i = int(row["end_i"])
            if a_i < 0 or e_i < 0 or a_i >= len(df) or e_i >= len(df): continue

            x_seg = [df.at[a_i, "time"], df.at[e_i, "time"]]
            y_seg = [float(df.at[a_i, "close"]), float(df.at[e_i, "close"])]

            fig.add_trace(go.Scatter(
                x=x_seg, y=y_seg, mode="lines",
                line=dict(color="rgba(0,0,0,0.5)", width=1.2, dash="dot"),
                showlegend=False, hoverinfo="skip"
            ))

            if row["결과"] == "성공":
                showlegend = not legend_emitted["성공"]; legend_emitted["성공"] = True
                fig.add_trace(go.Scatter(
                    x=[df.at[e_i, "time"]], y=[float(df.at[e_i, "close"])],
                    mode="markers", name="도달⭐",
                    marker=dict(size=12, color="orange", symbol="star", line=dict(width=1, color="black")),
                    showlegend=showlegend
                ))
            elif row["결과"] == "실패":
                showlegend = not legend_emitted["실패"]; legend_emitted["실패"] = True
                fig.add_trace(go.Scatter(
                    x=[df.at[e_i, "time"]], y=[float(df.at[e_i, "close"])],
                    mode="markers", name="실패❌",
                    marker=dict(size=12, color="blue", symbol="x", line=dict(width=1, color="black")),
                    showlegend=showlegend
                ))
            else:
                showlegend = not legend_emitted["중립"]; legend_emitted["중립"] = True
                fig.add_trace(go.Scatter(
                    x=[df.at[e_i, "time"]], y=[float(df.at[e_i, "close"])],
                    mode="markers", name="중립❌",
                    marker=dict(size=12, color="orange", symbol="x", line=dict(width=1, color="black")),
                    showlegend=showlegend
                ))

    # (이후 ③ 요약, 차트 표시, ④ 신호결과 테이블 표시: 최신 코드 유지)

except Exception as e:
    st.error(f"오류: {e}")
