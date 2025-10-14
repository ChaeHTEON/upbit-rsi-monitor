# app_pair_complete.py  (제태크_코인 — 커스텀 페어 백테스트 완성본)
# 실행 예: streamlit run app_pair_complete.py

import os
import math
from datetime import datetime, timedelta, date
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

from plotly.subplots import make_subplots
import plotly.graph_objs as go

# ------------------------------------------------------------
# 페이지/스타일
# ------------------------------------------------------------
st.set_page_config(page_title="커스텀 페어 백테스트 (거래량순)", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 0.6rem; padding-bottom: 0.6rem; max-width: 1100px;}
  .section-title {font-size:1.05rem; font-weight:700; margin:0.6rem 0 0.4rem;}
  .hint {color:#6b7280;}
  .success-cell {background-color:#FFF59D; color:#E53935; font-weight:600;}
  .fail-cell {color:#1E40AF; font-weight:600;}
  .neutral-cell {color:#FF9800; font-weight:600;}
  table {border-collapse:collapse; width:100%;}
  th, td {border:1px solid #ddd; padding:6px; text-align:center;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# 유틸: 업비트 마켓 (거래대금순 정렬)
# ------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_upbit_krw_markets_sorted() -> List[Tuple[str, str]]:
    """
    KRW-마켓만 불러와 24h 거래대금(acc_trade_price_24h) 기준으로 정렬.
    label: '비트코인 (BTC) — KRW-BTC'
    value: 'KRW-BTC'
    """
    try:
        r = requests.get("https://api.upbit.com/v1/market/all",
                         params={"isDetails": "false"}, timeout=8)
        r.raise_for_status()
        items = r.json()

        codes = []
        code2name = {}
        for it in items:
            mk = it.get("market", "")
            if mk.startswith("KRW-"):
                codes.append(mk)
                code2name[mk] = it.get("korean_name", mk[4:])

        # 티커(거래대금) 조회
        def _fetch_tickers(batch):
            r2 = requests.get("https://api.upbit.com/v1/ticker",
                              params={"markets": ",".join(batch)}, timeout=8)
            r2.raise_for_status()
            return r2.json()

        vol_map = {}
        for i in range(0, len(codes), 50):
            subset = codes[i:i+50]
            for t in _fetch_tickers(subset):
                vol_map[t["market"]] = float(t.get("acc_trade_price_24h", 0.0))

        ordered = sorted(codes, key=lambda c: (-vol_map.get(c, 0.0), c))
        rows = []
        for mk in ordered:
            sym = mk[4:]
            knm = code2name.get(mk, sym)
            label = f"{knm} ({sym}) — {mk}"
            rows.append((label, mk))
        return rows if rows else [("비트코인 (BTC) — KRW-BTC", "KRW-BTC")]
    except Exception:
        return [("비트코인 (BTC) — KRW-BTC", "KRW-BTC")]

# ------------------------------------------------------------
# 유틸: 캔들 로더 (load_ohlcv)
# ------------------------------------------------------------
def _to_interval_key(tf_label: str) -> Tuple[str, int]:
    """
    '1분','3분','5분','15분','30분','60분','일봉' -> (interval_key, minutes_per_bar)
    """
    mapping = {
        "1분": ("minutes/1", 1),
        "3분": ("minutes/3", 3),
        "5분": ("minutes/5", 5),
        "15분": ("minutes/15", 15),
        "30분": ("minutes/30", 30),
        "60분": ("minutes/60", 60),
        "일봉": ("days", 1440),
    }
    return mapping[tf_label]

def load_ohlcv(symbol: str, tframe_label: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    업비트에서 [start, end] 구간의 OHLCV를 페이징 수집 (최신→과거 순 API를 뒤집어 정렬).
    tframe_label 은 UI 라벨('5분','일봉') 형식.
    """
    interval_key, mpb = _to_interval_key(tframe_label)
    if "minutes/" in interval_key:
        unit = interval_key.split("/")[1]
        url = f"https://api.upbit.com/v1/candles/minutes/{unit}"
    else:
        url = "https://api.upbit.com/v1/candles/days"

    # 업비트는 'to' 기준 역순으로 200개 반환 → 여러번 내려가며 모으기
    KST = timedelta(hours=9)
    to_dt = end  # naive (local) datetime
    rows = []
    for _ in range(200):  # 충분한 상한
        params = {"market": symbol, "count": 200}
        # 'to'는 KST 기준 문자열 필요
        to_kst = (to_dt + KST).strftime("%Y-%m-%d %H:%M:%S")
        params["to"] = to_kst
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            batch = r.json()
        except Exception:
            break
        if not batch:
            break
        rows.extend(batch)
        last_kst = pd.to_datetime(batch[-1]["candle_date_time_kst"])
        # 다음 페이지는 그 직전 시각으로 이동
        to_dt = (last_kst - pd.Timedelta(seconds=1)).to_pydatetime() - KST
        # 범위를 벗어났으면 중단
        if last_kst.tz_localize(None).to_pydatetime() - KST <= start:
            break

    if not rows:
        return pd.DataFrame(columns=["time","open","high","low","close","volume"])

    df = pd.DataFrame(rows).rename(columns={
        "candle_date_time_kst": "time",
        "opening_price": "open",
        "high_price": "high",
        "low_price": "low",
        "trade_price": "close",
        "candle_acc_trade_volume": "volume",
    })
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df = df[["time","open","high","low","close","volume"]].sort_values("time").reset_index(drop=True)
    df = df[(df["time"] >= start) & (df["time"] <= end)].reset_index(drop=True)
    return df

# ------------------------------------------------------------
# 보조지표
# ------------------------------------------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(period).mean()
    md = (tp - ma).abs().rolling(period).mean()
    return (tp - ma) / (0.015 * (md + 1e-12))

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def bb(df: pd.DataFrame, window: int = 20, dev: float = 2.0):
    ma = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()
    up = ma + dev * std
    low = ma - dev * std
    return ma, up, low

# ------------------------------------------------------------
# 9개 전략: 베이스 자산에서 신호 생성
# ------------------------------------------------------------
def generate_base_signals(df: pd.DataFrame, strategy: str) -> List[int]:
    sig = []
    df = df.copy()
    df["rsi"] = rsi(df["close"], 14)
    df["cci"] = cci(df, 20)
    df["ema5"] = ema(df["close"], 5)
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    ma, up, low = bb(df, 20, 2.0)
    df["bb_mid"], df["bb_up"], df["bb_low"] = ma, up, low
    df["vol_mean"] = df["volume"].rolling(20).mean()

    for i in range(2, len(df)):
        o, h, l, c = df.loc[i, ["open","high","low","close"]]
        prev_h = df.loc[i-1, "high"]
        if strategy == "TGV":
            if (df.loc[i, "volume"] > df.loc[i, "vol_mean"] * 2.5) and (c > prev_h) and (df.loc[i, "rsi"] > 55):
                sig.append(i)
        elif strategy == "RVB":
            if (df.loc[i, "rsi"] <= 30) and (df.loc[i, "cci"] <= -100) and (c > o):
                sig.append(i)
        elif strategy == "PR":
            drop = (df.loc[i-1, "close"] / df.loc[i-2, "close"] - 1.0)
            if (drop <= -0.015) and (df.loc[i, "rsi"] <= 30) and (c > o):
                sig.append(i)
        elif strategy == "LCT":
            if (df.loc[i, "ema50"] > df.loc[i, "ema200"]) and (df.loc[i, "cci"] > -100) and (df.loc[i, "rsi"] > 50):
                sig.append(i)
        elif strategy == "4D_Sync":
            if (c >= df.loc[i, "bb_mid"]) and (df.loc[i, "rsi"] >= 55):
                sig.append(i)
        elif strategy == "240m_Sync":
            if (df.loc[i-1, "cci"] <= -200) and (df.loc[i, "cci"] > df.loc[i-1, "cci"]):
                sig.append(i)
        elif strategy == "Composite_Confirm":
            if (c >= df.loc[i, "bb_mid"]) and (df.loc[i, "rsi"] >= 60) and (c > df["high"].rolling(3).max().shift(1).iloc[i]):
                sig.append(i)
        elif strategy == "Divergence_RVB":
            if (df.loc[i, "rsi"] > df.loc[i-1, "rsi"]) and (c <= df.loc[i-1, "close"] * 0.999):
                sig.append(i)
        elif strategy == "Market_Divergence":
            if (c >= df.loc[i, "bb_low"]) and (df.loc[i, "rsi"] > df.loc[i-1, "rsi"]) and (df.loc[i, "rsi"] >= 45):
                sig.append(i)
    return sig

# ------------------------------------------------------------
# 페어 백테스트
# ------------------------------------------------------------
def pair_backtest(
    df_base: pd.DataFrame,
    df_follow: pd.DataFrame,
    strategies: List[str],
    lookahead: int,
    tp: float,
    sl: float,
) -> pd.DataFrame:
    """
    베이스 자산의 신호가 발생한 직후 캔들 기준으로 팔로워 자산의 N봉 내 TP/SL 도달 여부 평가.
    수익률은 팔로워 자산의 종가 기준.
    """
    if df_base.empty or df_follow.empty:
        return pd.DataFrame()

    # 시간 정렬 & 병합을 위해 인덱스화
    df_b = df_base.copy().reset_index(drop=True)
    df_f = df_follow.copy().reset_index(drop=True)

    out_rows = []
    for strat in strategies:
        sig_idx = generate_base_signals(df_b, strat)
        for i in sig_idx:
            anchor_time = df_b.loc[i, "time"]
            # 팔로워에서 anchor_time 이후 첫 캔들 찾기
            j = df_f.index[df_f["time"] >= anchor_time].min() if (df_f["time"] >= anchor_time).any() else None
            if j is None or j+1 >= len(df_f):
                continue
            anchor_j = j + 1  # 다음 봉부터 측정
            base_price = float(df_f.loc[anchor_j, "close"])
            end_j = min(anchor_j + lookahead, len(df_f)-1)

            target_up = base_price * (1.0 + tp)
            target_dn = base_price * (1.0 - sl)

            hit_j = None
            label = "중립"
            final_ret = 0.0
            for k in range(anchor_j+1, end_j+1):
                c = float(df_f.loc[k, "close"])
                if c >= target_up:
                    hit_j = k; label = "성공"; final_ret = (target_up/base_price - 1) * 100.0; break
                if c <= target_dn:
                    hit_j = k; label = "실패"; final_ret = (target_dn/base_price - 1) * 100.0; break

            if hit_j is None:
                c_end = float(df_f.loc[end_j, "close"])
                final_ret = (c_end/base_price - 1) * 100.0
                label = "실패" if final_ret <= 0 else "중립"
                hit_j = end_j

            out_rows.append({
                "전략": strat,
                "신호시간(베이스)": anchor_time,
                "측정시작(팔로워)": df_f.loc[anchor_j, "time"],
                "측정종료": df_f.loc[hit_j, "time"],
                "기준가": round(base_price, 4),
                "최종수익률(%)": round(final_ret, 3),
                "결과": label,
                "도달캔들": int(hit_j - anchor_j),
            })

    if not out_rows:
        return pd.DataFrame()

    df_res = pd.DataFrame(out_rows).sort_values(["전략","신호시간(베이스)"]).reset_index(drop=True)
    return df_res

# ------------------------------------------------------------
# UI — ② 커스텀 페어 백테스트 (거래량순)
# ------------------------------------------------------------
st.markdown('<div class="section-title">② 커스텀 페어 백테스트 (거래량순)</div>', unsafe_allow_html=True)

markets = get_upbit_krw_markets_sorted()
c1, c2, c3, c4, c5 = st.columns([2,2,1.2,1.2,1.4])
with c1:
    base_label, base_code = st.selectbox("기준 종목 선택", markets, index=0, format_func=lambda x: x[0])
with c2:
    follow_label, follow_code = st.selectbox("추종 종목 선택", markets, index=1, format_func=lambda x: x[0])
with c3:
    tf_label = st.selectbox("분봉", ["1분","3분","5분","15분","30분","60분","일봉"], index=2)
with c4:
    # 기본: 어제
    today = date.today()
    start_date = st.date_input("시작 날짜", value=today - timedelta(days=1))
with c5:
    end_date = st.date_input("종료 날짜", value=today)

st.markdown("### 커스텀 페어 백테스트 실행")

cA, cB, cC, cD = st.columns([1.2,1.2,1.2,2])
with cA:
    lookahead = st.number_input("측정 N봉", min_value=3, max_value=100, value=10, step=1)
with cB:
    tp = st.number_input("목표수익(TP, %)", min_value=0.1, max_value=10.0, value=1.0, step=0.1) / 100.0
with cC:
    sl = st.number_input("손절폭(SL, %)", min_value=0.1, max_value=10.0, value=0.5, step=0.1) / 100.0
with cD:
    strategies = st.multiselect(
        "매매기법 선택 (9개)",
        ["TGV","RVB","PR","LCT","4D_Sync","240m_Sync","Composite_Confirm","Divergence_RVB","Market_Divergence"],
        default=["TGV","RVB","PR","Divergence_RVB"]
    )

run_btn = st.button("▶ 실행", use_container_width=True)
if run_btn:
    try:
        sdt = datetime.combine(start_date, datetime.min.time())
        edt = datetime.combine(end_date, datetime.max.time())

        with st.status(f"{base_code} ➜ {follow_code} ({sdt:%Y-%m-%d}~{edt:%Y-%m-%d}) 페어 백테스트 실행 중...", expanded=True) as st_status:
            st.write("① 데이터 수집 중...")
            df_base = load_ohlcv(base_code, tf_label, sdt, edt)
            df_follow = load_ohlcv(follow_code, tf_label, sdt, edt)
            if df_base.empty or df_follow.empty:
                st.error("데이터가 없습니다. 기간/분봉을 변경해 보세요.")
            else:
                st.write("② 결과 계산 중...")
                res = pair_backtest(df_base, df_follow, strategies, lookahead, tp, sl)
                if res.empty:
                    st.info("조건을 만족하는 신호가 없습니다.")
                else:
                    st.write("③ 요약 지표")
                    total = len(res)
                    succ = (res["결과"]=="성공").sum()
                    fail = (res["결과"]=="실패").sum()
                    neu  = (res["결과"]=="중립").sum()
                    win  = succ/total*100 if total else 0.0
                    m1,m2,m3,m4 = st.columns(4)
                    m1.metric("신호 수", f"{total}")
                    m2.metric("성공", f"{succ}")
                    m3.metric("실패", f"{fail}")
                    m4.metric("승률", f"{win:.1f}%")

                    st.write("④ 상세 결과")
                    show = res.copy()
                    show["신호시간(베이스)"] = pd.to_datetime(show["신호시간(베이스)"]).dt.strftime("%Y-%m-%d %H:%M")
                    show["측정시작(팔로워)"] = pd.to_datetime(show["측정시작(팔로워)"]).dt.strftime("%Y-%m-%d %H:%M")
                    show["측정종료"]       = pd.to_datetime(show["측정종료"]).dt.strftime("%Y-%m-%d %H:%M")
                    st.dataframe(show, use_container_width=True)

                    csv_bytes = show.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("⬇ 결과 CSV 다운로드", data=csv_bytes,
                                       file_name="pair_backtest_results.csv", mime="text/csv")
            st_status.update(state="complete", label="완료")
    except Exception as e:
        st.error(f"오류 발생: {e}")
        st.stop()

st.caption("※ 본 화면은 '② 커스텀 페어 백테스트 (거래량순)' 단일 앱 예제입니다. 기존 메인 앱에서는 ④ 신호 결과 아래 섹션으로 그대로 붙여 넣으면 됩니다.")
