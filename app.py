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
  .neutral-cell {color:#FB8C00; font-weight:600;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 유틸 함수
# -----------------------------
def fetch_upbit_data(market="KRW-BTC", unit=15, to=None, count=200):
    url = f"https://api.upbit.com/v1/candles/minutes/{unit}"
    headers = {"Accept": "application/json"}
    params = {"market": market, "count": count}
    if to: params["to"] = to
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.3)
    session.mount("https://", HTTPAdapter(max_retries=retries))
    res = session.get(url, headers=headers, params=params)
    res.raise_for_status()
    data = res.json()
    df = pd.DataFrame(data)
    df = df.rename(columns={"candle_date_time_kst":"time","opening_price":"open",
                            "high_price":"high","low_price":"low","trade_price":"close","candle_acc_trade_volume":"volume"})
    df["time"] = pd.to_datetime(df["time"])
    return df.iloc[::-1].reset_index(drop=True)

def process_one(df, i0, lookahead, thr, minutes_per_bar):
    n = len(df)
    base_price = df.at[i0, "close"]
    anchor_idx = i0
    hit_idx = None
    end_idx = min(i0 + lookahead, n - 1)

    # 수익률 계산
    max_ret, min_ret = -999, 999
    for j in range(i0 + 1, end_idx + 1):
        ret = (df.at[j, "close"] / base_price - 1) * 100
        max_ret = max(max_ret, ret)
        min_ret = min(min_ret, ret)
        if ret >= thr and hit_idx is None:
            hit_idx = j

    if hit_idx is not None:
        bars_after = hit_idx - anchor_idx
        end_idx_final = hit_idx
        end_close = float(df.at[hit_idx, "close"])
        final_ret = thr
        result = "성공"
    else:
        bars_after = lookahead
        end_idx_final = end_idx
        end_close = float(df.at[end_idx_final, "close"])
        final_ret = (end_close / base_price - 1) * 100
        result = "실패" if final_ret <= 0 else "중립"

    # ✅ anchor/end 인덱스 기반으로 확정 저장
    signal_time_fix = df.at[anchor_idx, "time"]
    end_time_fix    = df.at[end_idx_final, "time"]
    bars_after_fix  = int(end_idx_final - anchor_idx)
    reach_min_fix   = bars_after_fix * minutes_per_bar

    row = {
        "신호시간": signal_time_fix,
        "종료시간": end_time_fix,
        "기준시가": int(round(base_price)),
        "종료가": end_close,
        "성공기준(%)": round(thr, 1),
        "결과": result,
        "도달분": reach_min_fix,
        "도달캔들(bars)": bars_after_fix,
        "최종수익률(%)": round(final_ret, 2),
        "최저수익률(%)": round(min_ret, 2),
        "최고수익률(%)": round(max_ret, 2),
        "anchor_i": int(anchor_idx),
        "end_i": int(end_idx_final),
    }
    return row, end_idx_final

# -----------------------------
# 시뮬레이션
# -----------------------------
def simulate(df, lookahead=10, thr=1.0, minutes_per_bar=15):
    res_rows = []
    i = 0
    n = len(df)
    while i < n:
        row, end_i = process_one(df, i, lookahead, thr, minutes_per_bar)
        res_rows.append(row)
        i += 1
    return pd.DataFrame(res_rows)

# -----------------------------
# 실행부
# -----------------------------
try:
    market = "KRW-XRP"
    unit = 15
    lookahead = 10
    thr = 1.0
    df = fetch_upbit_data(market=market, unit=unit, count=200)
    minutes_per_bar = unit

    res = simulate(df, lookahead=lookahead, thr=thr, minutes_per_bar=minutes_per_bar)

    # -----------------------------
    # 차트
    # -----------------------------
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Candlestick(x=df["time"], open=df["open"], high=df["high"],
                                 low=df["low"], close=df["close"], name="가격"))
    fig.add_trace(go.Scatter(x=df["time"], y=ta.momentum.RSIIndicator(df["close"], 13).rsi(),
                             mode="lines", name="RSI(13)", line=dict(color="green", width=2, dash="dot")),
                             secondary_y=True)

    # ===== 신호 마커/점선 =====
    if not res.empty:
        for _label, _color in [("성공","red"),("실패","blue"),("중립","#FF9800")]:
            sub = res[res["결과"] == _label]
            if sub.empty: continue
            fig.add_trace(go.Scatter(
                x=sub["신호시간"], y=sub["기준시가"], mode="markers",
                name=f"신호({_label})",
                marker=dict(size=9, color=_color, symbol="circle", line=dict(width=1,color="black"))
            ))

        legend_emitted = {"성공":False,"실패":False,"중립":False}
        for _, row in res.iterrows():
            anchor_idx = int(row["anchor_i"]); end_idx = int(row["end_i"])
            start_x = df.at[anchor_idx,"time"]; end_x = df.at[end_idx,"time"]
            start_y = float(row["기준시가"]); end_y = float(df.at[end_idx,"close"])
            grp = row["결과"]; color = "red" if grp=="성공" else ("blue" if grp=="실패" else "#FF9800")

            fig.add_trace(go.Scatter(
                x=[start_x,end_x], y=[start_y,end_y], mode="lines",
                line=dict(color=color,width=1.6 if grp=="성공" else 1.0,dash="dot"),
                opacity=0.9 if grp=="성공" else 0.5,
                showlegend=(not legend_emitted[grp]), name=f"신호(점선)-{grp}"
            ))
            legend_emitted[grp] = True

            if grp=="성공":
                fig.add_trace(go.Scatter(
                    x=[end_x], y=[float(row["종료가"])], mode="markers", name="목표 도달",
                    marker=dict(size=15,color="orange",symbol="star",line=dict(width=1,color="black")),
                    showlegend=False
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=[end_x], y=[end_y], mode="markers", name=f"도착-{grp}",
                    marker=dict(size=8,color=color,symbol="x",line=dict(width=1,color="black")),
                    showlegend=False
                ))

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # 신호 결과 테이블
    # -----------------------------
    if not res.empty:
        st.markdown("### ④ 신호 결과 (최신 순)")
        st.dataframe(res[::-1].reset_index(drop=True))

except Exception as e:
    st.error(f"오류: {e}")
