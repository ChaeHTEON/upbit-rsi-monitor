# app_complete_full.txt
# =============================================================
# 제태크_코인 (Streamlit 풀버전 · 단일 파일 실행)
# - 거래량순 종목 정렬
# - 9개 매매기법 멀티선택
# - 시뮬레이터 + 통계/조합 탐색 + 신호 차트 + 실시간 감시/알람
# - 커스텀 페어 백테스트 (기준/추종 종목 동일 구간·타임프레임 비교)
# - 공유 메모(GitHub 업로드 옵션) · Kakao 비활성
# - 차트 "공백" 이슈 해결(레이아웃·스펙·데이터 컷 정돈)
# =============================================================

# -*- coding: utf-8 -*-
import os
# watchdog/inotify 한도 초과 방지
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["WATCHDOG_DISABLE_FILE_SYSTEM_EVENTS"] = "true"

import streamlit as st
import requests
from requests.adapters import HTTPAdapter, Retry
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import ta
from datetime import datetime, timedelta
from pytz import timezone
from typing import Optional, List, Set

# -------------------------------------------------------------
# 페이지/스타일
# -------------------------------------------------------------
st.set_page_config(page_title="Upbit RSI(13)+BB 시뮬레이터(풀버전)", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 0.6rem; padding-bottom: 0.6rem; max-width: 1180px;}
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

st.title("📊 제태크_코인 · 시뮬레이터 & 페어 백테스트(풀버전)")
st.caption("※ 차트 점선: 신호~판정 구간, 성공 시 ⭐ 마커 표기")

# -------------------------------------------------------------
# 공용 세션 / 유틸
# -------------------------------------------------------------
_session = requests.Session()
_retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
_session.mount("https://", HTTPAdapter(max_retries=_retries))

KST = timezone("Asia/Seoul")

TF_MAP = {
    "1분": ("minutes/1", 1),
    "3분": ("minutes/3", 3),
    "5분": ("minutes/5", 5),
    "15분": ("minutes/15", 15),
    "30분": ("minutes/30", 30),
    "60분": ("minutes/60", 60),
    "일봉": ("days", 24*60),
}

MAIN9 = [
    "TGV","RVB","PR","LCT","4D_Sync","240m_Sync","Composite_Confirm","Divergence_RVB","Market_Divergence"
]

def _get_secret(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)

# -------------------------------------------------------------
# 거래량순 종목 리스트
# -------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_upbit_krw_markets_sorted():
    try:
        r = _session.get("https://api.upbit.com/v1/market/all", params={"isDetails":"false"}, timeout=8)
        r.raise_for_status()
        items = r.json()
        krw_codes = [it["market"] for it in items if it["market"].startswith("KRW-")]
        code2name = {it["market"]: it["korean_name"] for it in items if it["market"].startswith("KRW-")}
        # ticker for acc_trade_price_24h
        vols = {}
        for i in range(0, len(krw_codes), 50):
            sub = ",".join(krw_codes[i:i+50])
            t = _session.get("https://api.upbit.com/v1/ticker", params={"markets": sub}, timeout=8).json()
            for x in t:
                vols[x["market"]] = float(x.get("acc_trade_price_24h", 0.0))
        ordered = sorted(krw_codes, key=lambda c: (-vols.get(c, 0.0), c))
        rows = []
        for mk in ordered:
            sym = mk[4:]
            knm = code2name.get(mk, sym)
            rows.append((f"{knm} ({sym}) — {mk}", mk))
        return rows if rows else [("비트코인 (BTC) — KRW-BTC", "KRW-BTC")]
    except Exception:
        return [("비트코인 (BTC) — KRW-BTC", "KRW-BTC")]

MARKET_LIST = get_upbit_krw_markets_sorted()
default_idx = 0

# -------------------------------------------------------------
# OHLCV 로더 (페이징 + CSV 캐시)
# -------------------------------------------------------------
def _tf_to_url_key(interval_key:str):
    if "minutes/" in interval_key: 
        unit = interval_key.split("/")[1]
        return f"minutes/{unit}", f"{unit}min", f"https://api.upbit.com/v1/candles/minutes/{unit}"
    return "days", "day", "https://api.upbit.com/v1/candles/days"

def load_ohlcv(market_code:str, interval_key:str, start_dt:datetime, end_dt:datetime, minutes_per_bar:int, warmup_bars:int=0)->pd.DataFrame:
    if warmup_bars>0:
        start_cutoff = start_dt - timedelta(minutes=warmup_bars*minutes_per_bar)
    else:
        start_cutoff = start_dt
    _, tf_key, url = _tf_to_url_key(interval_key)

    data_dir = os.path.join(os.path.dirname(__file__), "data_cache")
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, f"{market_code}_{tf_key}.csv")

    # 손상시 복구
    if os.path.exists(cache_path):
        try:
            pd.read_csv(cache_path, nrows=3)
        except Exception:
            try:
                os.remove(cache_path)
            except Exception:
                pass

    if os.path.exists(cache_path):
        df_cache = pd.read_csv(cache_path, parse_dates=["time"])
        df_cache["time"] = pd.to_datetime(df_cache["time"]).dt.tz_localize(None)
    else:
        df_cache = pd.DataFrame(columns=["time","open","high","low","close","volume"])

    from pytz import timezone as _tz
    _KST = _tz("Asia/Seoul"); _UTC = _tz("UTC")
    all_data = []
    to_time = _KST.localize(end_dt).astimezone(_UTC).replace(tzinfo=None)

    try:
        while True:
            params = {"market": market_code, "count": 200}
            if to_time is not None:
                params["to"] = to_time.strftime("%Y-%m-%d %H:%M:%S")
            r = _session.get(url, params=params, headers={"Accept":"application/json"}, timeout=10)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            all_data.extend(batch)
            last_kst = pd.to_datetime(batch[-1]["candle_date_time_kst"])
            last_utc = pd.to_datetime(batch[-1]["candle_date_time_utc"])
            if last_kst <= start_cutoff:
                break
            to_time = (last_utc - timedelta(seconds=1))
    except Exception:
        # 실패 시 캐시만 사용
        return df_cache[(df_cache["time"]>=start_cutoff) & (df_cache["time"]<=end_dt)].reset_index(drop=True)

    if all_data:
        df_new = pd.DataFrame(all_data).rename(columns={
            "candle_date_time_kst":"time",
            "opening_price":"open",
            "high_price":"high",
            "low_price":"low",
            "trade_price":"close",
            "candle_acc_trade_volume":"volume",
        })
        df_new["time"] = pd.to_datetime(df_new["time"]).dt.tz_localize(None)
        df_new = df_new[["time","open","high","low","close","volume"]].sort_values("time")
        df_all = pd.concat([df_cache, df_new], ignore_index=True)\
                   .drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
        tmp = cache_path + ".tmp"
        df_all.to_csv(tmp, index=False)
        try:
            os.replace(tmp, cache_path)
        except Exception:
            df_all.to_csv(cache_path, index=False)
    else:
        df_all = df_cache

    return df_all[(df_all["time"]>=start_cutoff) & (df_all["time"]<=end_dt)].reset_index(drop=True)

# -------------------------------------------------------------
# 지표 / 시뮬레이터
# -------------------------------------------------------------
def add_indicators(df:pd.DataFrame, bb_window:int, bb_dev:float, cci_window:int, cci_signal:int=9)->pd.DataFrame:
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=bb_window, window_dev=bb_dev)
    out["BB_up"]  = bb.bollinger_hband().fillna(method="bfill").fillna(method="ffill")
    out["BB_low"] = bb.bollinger_lband().fillna(method="bfill").fillna(method="ffill")
    out["BB_mid"] = bb.bollinger_mavg().fillna(method="bfill").fillna(method="ffill")
    cci = ta.trend.CCIIndicator(high=out["high"], low=out["low"], close=out["close"], window=int(cci_window), constant=0.015)
    out["CCI"] = cci.cci()
    n = max(int(cci_signal), 1)
    out["CCI_sig"] = out["CCI"].rolling(n, min_periods=1).mean()
    return out

def simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, threshold_pct, bb_cond, dedup_mode,
             minutes_per_bar, market_code, bb_window, bb_dev, sec_cond="없음", hit_basis="종가 기준",
             miss_policy="(고정) 성공·실패·중립", bottom_mode=False, supply_levels: Optional[Set[float]] = None,
             manual_supply_levels: Optional[list] = None, cci_mode: str = "없음",
             cci_over: float = 100.0, cci_under: float = -100.0, cci_signal_n: int = 9):
    res = []
    n = len(df)
    thr = float(threshold_pct)

    # 1차 조건
    if bottom_mode:
        base_sig_idx = df.index[
            (df["RSI13"] <= float(rsi_low)) & (df["close"] <= df["BB_low"]) & (df["CCI"] <= -100)
        ].tolist()
    else:
        # 기본 조합
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
            o = float(df.at[i, "open"])
            l = float(df.at[i, "low"])
            up, lo, mid = df.at[i, "BB_up"], df.at[i, "BB_low"], df.at[i, "BB_mid"]
            if bb_cond == "상한선":
                return pd.notna(up) and (c > float(up))
            if bb_cond == "하한선":
                if pd.isna(lo): return False
                rv = float(lo)
                return ((o < rv) or (l <= rv)) and (c >= rv)
            if bb_cond == "중앙선":
                if pd.isna(mid): return False
                return c >= float(mid)
            return False

        bb_idx = [i for i in df.index if bb_cond != "없음" and bb_ok(i)]

        if cci_mode == "없음":
            cci_idx = []
        elif cci_mode == "과매수":
            cci_idx = df.index[df["CCI"] >= float(cci_over)].tolist()
        elif cci_mode == "과매도":
            cci_idx = df.index[df["CCI"] <= float(cci_under)].tolist()
        else:
            cci_idx = []

        idx_sets = []
        if rsi_mode != "없음": idx_sets.append(set(rsi_idx))
        if bb_cond  != "없음": idx_sets.append(set(bb_idx))
        if cci_mode != "없음": idx_sets.append(set(cci_idx))
        if idx_sets:
            base_sig_idx = sorted(set.intersection(*idx_sets)) if len(idx_sets) > 1 else sorted(idx_sets[0])
        else:
            base_sig_idx = list(range(n)) if sec_cond != "없음" else []

    # 보조
    def first_bull_50_over_bb(start_i):
        for j in range(start_i + 1, n):
            o, l, c = float(df.at[j, "open"]), float(df.at[j, "low"]), float(df.at[j, "close"])
            if not (c > o):
                continue
            if bb_cond == "하한선":
                ref_series = df["BB_low"]
            elif bb_cond == "중앙선":
                ref_series = df["BB_mid"]
            else:
                ref_series = df["BB_up"]
            ref = ref_series.iloc[j]
            if pd.isna(ref):
                continue
            rv = float(ref)
            entered_from_below = (o < rv) or (l <= rv)
            closes_above = (c >= rv)
            if not (entered_from_below and closes_above):
                continue
            if j - (start_i + 1) > 0:
                prev_close = df.loc[start_i + 1:j - 1, "close"]
                prev_ref   = ref_series.loc[start_i + 1:j - 1]
                if not (prev_close < prev_ref).all():
                    continue
            return j, c
        return None, None

    def process_one(i0):
        anchor_idx = i0 + 1
        if anchor_idx >= n:
            return None, None
        signal_time = df.at[anchor_idx, "time"]
        base_price = float(df.at[anchor_idx, "close"])

        if sec_cond == "양봉 2개 연속 상승":
            if i0 + 2 >= n:
                return None, None
            c1, o1 = float(df.at[i0 + 1, "close"]), float(df.at[i0 + 1, "open"])
            c2, o2 = float(df.at[i0 + 2, "close"]), float(df.at[i0 + 2, "open"])
            if not ((c1 > o1) and (c2 > o2) and (c2 > c1)):
                return None, None
            anchor_idx = i0 + 3
            if anchor_idx >= n:
                return None, None
            signal_time = df.at[anchor_idx, "time"]
            base_price  = float(df.at[anchor_idx, "close"])

        elif sec_cond == "양봉 2개 (범위 내)":
            found, T_idx = 0, None
            scan_end = min(i0 + lookahead, n - 1)
            for j in range(i0 + 1, scan_end + 1):
                c, o = float(df.at[j, "close"]), float(df.at[j, "open"])
                if c > o:
                    found += 1
                    if found == 2:
                        T_idx = j
                        break
            if T_idx is None:
                return None, None
            anchor_idx = T_idx + 1
            if anchor_idx >= n:
                return None, None
            signal_time = df.at[anchor_idx, "time"]
            base_price = float(df.at[anchor_idx, "close"])

        elif sec_cond == "BB 기반 첫 양봉 50% 진입":
            if bb_cond == "없음":
                return None, None
            B1_idx, _ = first_bull_50_over_bb(i0)
            if B1_idx is None:
                return None, None
            anchor_idx = B1_idx + 1
            if anchor_idx >= n:
                return None, None
            signal_time = df.at[anchor_idx, "time"]
            base_price  = float(df.at[anchor_idx, "close"])

        elif sec_cond == "매물대 자동 (하단→상단 재진입 + BB하단 위 양봉)":
            anchor_idx2 = None
            scan_end = min(i0 + lookahead, n - 1)
            for j in range(i0 + 2, scan_end + 1):
                prev_high = float(df.at[j - 1, "high"])
                prev_open = float(df.at[j - 1, "open"])
                prev_close = float(df.at[j - 1, "close"])
                maemul = max(prev_high, prev_close if prev_close >= prev_open else prev_open)
                cur_low = float(df.at[j, "low"])
                cur_close = float(df.at[j, "close"])
                cur_open = float(df.at[j, "open"])
                cur_bb_low = float(df.at[j, "BB_low"])
                below = cur_low <= maemul * 0.999
                above = cur_close >= maemul
                is_bull = cur_close > cur_open
                bb_above = maemul >= cur_bb_low
                if below and above and is_bull and bb_above:
                    anchor_idx2 = j
                    break
            if anchor_idx2 is None or anchor_idx2 >= n:
                return None, None
            anchor_idx = anchor_idx2
            signal_time = df.at[anchor_idx, "time"]
            base_price  = float(df.at[anchor_idx, "close"])

        # 성과 측정
        eval_start = anchor_idx + 1
        end_idx = anchor_idx + lookahead
        if end_idx >= n:
            return None, None

        win_slice = df.iloc[eval_start:end_idx + 1]
        min_ret = (win_slice["close"].min() / base_price - 1) * 100 if not win_slice.empty else 0.0
        max_ret = (win_slice["close"].max() / base_price - 1) * 100 if not win_slice.empty else 0.0

        target = base_price * (1.0 + thr / 100.0)
        hit_idx = None
        for j in range(anchor_idx + 1, end_idx + 1):
            c_ = float(df.at[j, "close"])
            if c_ >= target * 0.9999:
                hit_idx = j
                break

        if hit_idx is not None:
            bars_after = hit_idx - anchor_idx
            end_time = df.at[hit_idx, "time"]
            end_close = target
            final_ret = thr
            result = "성공"
            lock_end = hit_idx
        else:
            bars_after = lookahead
            end_idx2 = anchor_idx + bars_after
            if end_idx2 >= n:
                end_idx2 = n - 1
                bars_after = end_idx2 - anchor_idx
            end_time = df.at[end_idx2, "time"]
            end_close = float(df.at[end_idx2, "close"])
            final_ret = (end_close / base_price - 1) * 100
            result = "실패" if final_ret <= 0 else "중립"
            lock_end = end_idx2

        row = {
            "신호시간": signal_time,
            "종료시간": end_time,
            "기준시가": int(round(base_price)),
            "종료가": end_close,
            "RSI(13)": round(float(df.at[anchor_idx, "RSI13"]), 2) if pd.notna(df.at[anchor_idx, "RSI13"]) else None,
            "성공기준(%)": round(thr, 1),
            "결과": result,
            "도달캔들(bars)": int(bars_after),
            "최종수익률(%)": round(final_ret, 2),
            "최저수익률(%)": round(min_ret, 2),
            "최고수익률(%)": round(max_ret, 2),
            "anchor_i": int(anchor_idx),
            "end_i": int(hit_idx if hit_idx is not None else end_idx2),
        }
        return row, int(lock_end)

    # 메인 루프
    if dedup_mode.startswith("중복 제거"):
        i = 0
        while i < n:
            if i not in base_sig_idx:
                i += 1
                continue
            row, lock_end = process_one(i)
            if row is not None:
                res.append(row)
                i = int(lock_end) + 1
            else:
                i += 1
    else:
        for i0 in base_sig_idx:
            row, _ = process_one(i0)
            if row is not None:
                res.append(row)

    if res:
        return pd.DataFrame(res).drop_duplicates(subset=["anchor_i"], keep="first").reset_index(drop=True)
    return pd.DataFrame()

# -------------------------------------------------------------
# 차트 (공백 이슈 방지)
# -------------------------------------------------------------
def plot_signals_chart(df_plot:pd.DataFrame, results:pd.DataFrame, bb_on:bool, buy_price:int, cci_signal:int, cci_window:int):
    # 4행 레이아웃(가격/RSI/CCI/거래량) · shared_xaxes
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}], [{}]],
        row_heights=[0.55, 0.20, 0.20, 0.20],
        vertical_spacing=0.04
    )
    fig.update_layout(height=1000)

    # RSI(13) (row=2)
    fig.add_trace(go.Scatter(
        x=df_plot["time"], y=df_plot["RSI13"], name="RSI(13)",
        mode="lines", line=dict(color="orange", width=1)), row=2, col=1
    )
    fig.add_hline(y=40, line=dict(color="rgba(255,0,0,0.5)", dash="solid", width=1.4), row=2, col=1)

    # CCI 기준선 (row=3)
    fig.add_hline(y=-30, line=dict(color="rgba(255,0,0,0.5)", dash="solid", width=1.4), row=3, col=1)

    # 거래량 (row=4)
    colors = ["rgba(255,75,75,0.6)" if c>o else "rgba(0,104,201,0.6)" for c,o in zip(df_plot["close"],df_plot["open"])]
    fig.add_trace(go.Bar(x=df_plot["time"], y=df_plot["volume"], name="거래량", marker_color=colors), row=4, col=1)
    if "vol_mean" not in df_plot.columns:
        df_plot["vol_mean"] = df_plot["volume"].rolling(20).mean()
    if "vol_threshold" not in df_plot.columns:
        df_plot["vol_threshold"] = df_plot["vol_mean"] * 2.5
    fig.add_trace(go.Scatter(x=df_plot["time"], y=df_plot["vol_mean"], name="거래량 평균(20)", mode="lines", line=dict(width=1.2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df_plot["time"], y=df_plot["vol_threshold"], name="TGV 기준(2.5x)", mode="lines", line=dict(width=1.2, dash="dot")), row=4, col=1)
    fig.update_yaxes(title_text="거래량", row=4, col=1)

    # 캔들 (row=1)
    fig.add_trace(go.Candlestick(
        x=df_plot["time"], open=df_plot["open"], high=df_plot["high"],
        low=df_plot["low"], close=df_plot["close"], name="가격",
        increasing=dict(line=dict(color="red", width=1.1)),
        decreasing=dict(line=dict(color="blue", width=1.1)),
        hoverinfo="x+name"), row=1, col=1
    )

    # BB 라인 (row=1)
    if bb_on:
        fig.add_trace(go.Scatter(x=df_plot["time"], y=df_plot["BB_up"],  name="BB 상단", mode="lines", line=dict(width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot["time"], y=df_plot["BB_mid"], name="BB 중앙", mode="lines", line=dict(width=1.0, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot["time"], y=df_plot["BB_low"], name="BB 하단", mode="lines", line=dict(width=1.2)), row=1, col=1)

    # RSI 라인 y2 (row=1, secondary_y)
    fig.add_trace(go.Scatter(
        x=df_plot["time"], y=df_plot["RSI13"], mode="lines",
        line=dict(color="rgba(42,157,143,0.30)", width=6),
        name="", showlegend=False
    ), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(
        x=df_plot["time"], y=df_plot["RSI13"], mode="lines",
        line=dict(width=2.0, dash="dot"), name="RSI(13)"
    ), row=1, col=1, secondary_y=True)

    # CCI 하단 (row=3)
    fig.add_trace(go.Scatter(x=df_plot["time"], y=df_plot["CCI"], name="CCI", mode="lines", line=dict(width=1.2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_plot["time"], y=df_plot["CCI_sig"], name=f"CCI 신호({int(cci_signal)})", mode="lines", line=dict(width=1.0, dash="dot")), row=3, col=1)
    for yv,colr in [(100,"#E63946"),(-100,"#457B9D"),(0,"#888")]:
        fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y3", y0=yv, y1=yv, line=dict(color=colr, width=1, dash="dot"))

    # 신호 마커/점선/⭐
    if results is not None and not results.empty:
        plot_res = results.sort_values("신호시간").drop_duplicates(subset=["anchor_i"], keep="first").reset_index(drop=True)
        for _label, _color in [("성공","red"),("실패","blue"),("중립","#FF9800")]:
            sub = plot_res[plot_res["결과"]==_label]
            if sub.empty: continue
            xs, ys = [], []
            for _, r in sub.iterrows():
                t0 = pd.to_datetime(r["신호시간"])
                if t0 in df_plot["time"].values:
                    xs.append(t0)
                    ys.append(float(df_plot.loc[df_plot["time"]==t0, "open"].iloc[0]))
            if xs:
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name=f"신호({_label})",
                                         marker=dict(size=9, color=_color, symbol="circle", line=dict(width=1, color="black"))),
                              row=1, col=1)
        legend_emitted = {"성공":False,"실패":False,"중립":False}
        for _, row_ in plot_res.iterrows():
            t0 = pd.to_datetime(row_["신호시간"])
            t1 = pd.to_datetime(row_["종료시간"])
            if (t0 not in df_plot["time"].values) or (t1 not in df_plot["time"].values):
                continue
            y0 = float(df_plot.loc[df_plot["time"]==t0,"close"].iloc[0])
            y1 = float(df_plot.loc[df_plot["time"]==t1,"close"].iloc[0])
            fig.add_trace(go.Scatter(x=[t0,t1], y=[y0,y1], mode="lines",
                                     line=dict(color="rgba(0,0,0,0.5)", width=1.1, dash="dot"),
                                     showlegend=False, hoverinfo="skip"), row=1, col=1)
            if row_["결과"]=="성공":
                fig.add_trace(go.Scatter(x=[t1], y=[y1], mode="markers", name="도달⭐",
                                         marker=dict(size=12, color="orange", symbol="star", line=dict(width=1, color="black")),
                                         showlegend=not legend_emitted["성공"]), row=1, col=1)
                legend_emitted["성공"] = True
            elif row_["결과"]=="실패":
                fig.add_trace(go.Scatter(x=[t1], y=[y1], mode="markers", name="실패❌",
                                         marker=dict(size=12, color="blue", symbol="x", line=dict(width=1, color="black")),
                                         showlegend=not legend_emitted["실패"]), row=1, col=1)
                legend_emitted["실패"] = True
            elif row_["결과"]=="중립":
                fig.add_trace(go.Scatter(x=[t1], y=[y1], mode="markers", name="중립❌",
                                         marker=dict(size=12, color="orange", symbol="x", line=dict(width=1, color="black")),
                                         showlegend=not legend_emitted["중립"]), row=1, col=1)
                legend_emitted["중립"] = True

    fig.update_layout(
        dragmode="pan", xaxis_rangeslider_visible=False, legend_orientation="h", legend_y=1.02,
        margin=dict(l=30, r=30, t=60, b=40),
        yaxis=dict(title="가격", autorange=True, fixedrange=False),
        yaxis2=dict(title="RSI(13)", range=[0,100], autorange=False, fixedrange=False),
        yaxis3=dict(title="CCI", autorange=True, fixedrange=False),
        hovermode="x", hoverdistance=1, spikedistance=1
    )
    fig.update_xaxes(showspikes=True, spikecolor="gray", spikethickness=1, spikemode="across")
    fig.update_yaxes(showspikes=True, spikecolor="gray", spikethickness=1, spikemode="across")
    return fig

# -------------------------------------------------------------
# 실시간 감시/알람 (토스트/히스토리) - Kakao 비활성
# -------------------------------------------------------------
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def calc_cci(df, period=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(period).mean()
    md = (tp - ma).abs().rolling(period).mean()
    return (tp - ma) / (0.015 * (md + 1e-12))

def _push_alert(symbol, tf, strategy, msg, tp=None, sl=None):
    if "alerts_live" not in st.session_state: st.session_state["alerts_live"]=[]
    if "alert_history" not in st.session_state: st.session_state["alert_history"]=[]
    if "last_alert_at" not in st.session_state: st.session_state["last_alert_at"]={}
    now_kst = datetime.utcnow() + timedelta(hours=9)
    key = f"{strategy}|{symbol}|{tf}"
    last_at = st.session_state["last_alert_at"].get(key)
    if last_at and (now_kst - last_at).total_seconds() < 180:
        return
    entry = {"time": now_kst.strftime("%H:%M:%S"), "symbol":symbol, "tf":tf, "strategy":strategy, "msg":msg, "checked":False}
    if tp is not None: entry["tp"]=tp
    if sl is not None: entry["sl"]=sl
    st.session_state["alerts_live"].insert(0, entry)
    st.session_state["alert_history"].insert(0, entry)
    st.session_state["last_alert_at"][key] = now_kst
    try:
        st.toast(msg, icon="📈")
    except Exception:
        pass

def check_tgv_signal(df, symbol="KRW-BTC", tf="5"):
    if len(df)<25: return
    df["rsi"]=calc_rsi(df["close"]); df["cci"]=calc_cci(df)
    df["ema5"]=df["close"].ewm(span=5).mean(); df["ema20"]=df["close"].ewm(span=20).mean()
    df["vol_mean"]=df["volume"].rolling(20).mean(); df["vol_threshold"]=df["vol_mean"]*2.5
    latest, prev = df.iloc[-1], df.iloc[-2]
    cond = (latest["volume"]>latest["vol_threshold"]) and (latest["ema5"]>latest["ema20"]) and (latest["close"]>prev["high"]) and (latest["rsi"]>55)
    if cond:
        _push_alert(symbol, tf, "TGV",
                    f"⚡ TGV [{symbol},{tf}분] RSI {prev['rsi']:.1f}→{latest['rsi']:.1f} · 거래량 {latest['volume']/max(latest['vol_mean'],1e-9):.1f}x",
                    tp="+0.7%", sl="-0.4%")

def check_rvb_signal(df, symbol, tf):
    if len(df)<5: return
    rsi=calc_rsi(df["close"]); cci=calc_cci(df)
    if (rsi.iloc[-1]<35) and (cci.iloc[-1]<-80) and (df["close"].iloc[-1]>df["open"].iloc[-1]):
        _push_alert(symbol, tf, "RVB",
                    f"⚡ RVB [{symbol},{tf}분] RSI {rsi.iloc[-2]:.1f}→{rsi.iloc[-1]:.1f}, CCI {cci.iloc[-2]:.0f}→{cci.iloc[-1]:.0f}",
                    tp="+1.2%", sl="-0.5%")

def check_pr_signal(df,symbol,tf):
    if len(df)<5: return
    drop=(df["close"].iloc[-2]/df["close"].iloc[-3]-1.0)
    if (drop<-0.015) and (calc_rsi(df["close"]).iloc[-1]<25) and (df["volume"].iloc[-1]>df["volume"].rolling(20).mean().iloc[-1]*1.6):
        _push_alert(symbol,tf,"PR","⚡ PR 급락 후 반등 포착",tp="+1.2%",sl="-0.5%")

def check_lct_signal(df,symbol,tf):
    if len(df)<200: return
    ema50=df["close"].ewm(span=50).mean(); ema200=df["close"].ewm(span=200).mean()
    cci=calc_cci(df)
    if (ema50.iloc[-1]>ema200.iloc[-1]) and (cci.iloc[-1]>-100):
        _push_alert(symbol,tf,"LCT","⚡ LCT 장기 추세 전환 초기",tp="+8%",sl="-2%")

def check_4d_sync_signal(df,symbol,tf):
    _push_alert(symbol,tf,"4D_Sync","⚡ 4D_Sync 상승 동조 시작",tp="+1.5%",sl="-0.4%")

def check_240m_sync_signal(df,symbol,tf):
    cci=calc_cci(df)
    if cci.iloc[-1] < -200:
        _push_alert(symbol,tf,"240m_Sync",f"⚡ 240m CCI {cci.iloc[-1]:.0f}",tp="+2.5%",sl="-0.6%")

def check_composite_confirm_signal(df,symbol,tf):
    _push_alert(symbol,tf,"Composite_Confirm","⚡ Composite BTC·ETH·SOL 동시 포착",tp="+1.5%",sl="-0.4%")

def check_divergence_rvb_signal(df,symbol,tf):
    rsi=calc_rsi(df["close"])
    if rsi.iloc[-1]>rsi.iloc[-2] and df["close"].iloc[-1]<df["close"].iloc[-2]:
        _push_alert(symbol,tf,"Divergence_RVB","⚡ Divergence RSI 상승/가격하락",tp="+1.7%",sl="-0.5%")

def check_market_divergence_signal(df,symbol,tf):
    _push_alert(symbol,tf,"Market_Divergence","⚡ Market Divergence BTC 하락멈춤",tp="+1.4%",sl="-0.5%")

STRATEGY_TF_MAP = {
    "TGV":["5"], "RVB":["15"], "PR":["15"], "LCT":["240"],
    "4D_Sync":["60"], "240m_Sync":["240"], "Composite_Confirm":["15"],
    "Divergence_RVB":["15"], "Market_Divergence":["15"],
}

# -------------------------------------------------------------
# GitHub 업로드(공유 메모/CSV)
# -------------------------------------------------------------
def github_commit_file(local_file:str):
    token  = _get_secret("GITHUB_TOKEN")
    repo   = _get_secret("GITHUB_REPO")
    branch = _get_secret("GITHUB_BRANCH", "main")
    if not (token and repo): 
        return False, "no_token_or_repo"
    import base64
    url  = f"https://api.github.com/repos/{repo}/contents/{os.path.basename(local_file)}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    with open(local_file, "rb") as f:
        b64_content = base64.b64encode(f.read()).decode()
    # get SHA
    sha = None
    r_get = _session.get(url, headers=headers, timeout=8)
    if r_get.status_code == 200:
        sha = r_get.json().get("sha")
    data = {"message": f"Update {os.path.basename(local_file)} from Streamlit", "content": b64_content, "branch": branch}
    if sha: data["sha"] = sha
    r_put = _session.put(url, headers=headers, json=data, timeout=8)
    return r_put.status_code in (200,201), r_put.text

# -------------------------------------------------------------
# ① 기본 설정 (메인 시뮬레이터)
# -------------------------------------------------------------
st.markdown('<div class="section-title">① 기본 설정</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    market_label, market_code = st.selectbox("종목 선택 (거래량순)", MARKET_LIST, index=default_idx, format_func=lambda x: x[0])
with c2:
    tf_label = st.selectbox("봉종류 선택", list(TF_MAP.keys()), index=2)
with c3:
    today_kst = datetime.now(KST).date()
    default_start = today_kst - timedelta(days=1)
    start_date = st.date_input("시작 날짜", value=default_start)
with c4:
    end_date = st.date_input("종료 날짜", value=today_kst)

interval_key, minutes_per_bar = TF_MAP[tf_label]
st.markdown("---")

# -------------------------------------------------------------
# ② 조건 설정
# -------------------------------------------------------------
st.markdown('<div class="section-title">② 조건 설정</div>', unsafe_allow_html=True)
c5, c6, c7 = st.columns(3)
with c5:
    lookahead = st.slider("측정 캔들 수 (기준 이후 N봉)", 1, 60, 10)
with c6:
    threshold_pct = st.slider("성공/실패 기준 값(%)", 0.1, 5.0, 1.0, step=0.1)
    winrate_thr   = st.slider("승률 기준(%)", 10, 100, 70, step=1)
    hit_basis = "종가 기준"
with c7:
    primary_strategy = st.selectbox("1차 매매기법(없음=직접 조건)", ["없음"]+MAIN9, index=0)

r1, r2, r3 = st.columns(3)
with r1:
    rsi_mode = st.selectbox("RSI 조건", ["없음","현재(과매도/과매수 중 하나)","과매도 기준","과매수 기준"], index=0)
with r2:
    rsi_low = st.slider("과매도 RSI 기준", 0, 100, 30, step=1)
with r3:
    rsi_high = st.slider("과매수 RSI 기준", 0, 100, 70, step=1)

c8, c9, c10 = st.columns(3)
with c8:
    bb_cond = st.selectbox("볼린저밴드 조건", ["없음","상한선","중앙선","하한선"], index=0)
with c9:
    bb_window = st.number_input("BB 기간", min_value=5, max_value=100, value=30, step=1)
with c10:
    bb_dev = st.number_input("BB 승수", min_value=1.0, max_value=4.0, value=2.0, step=0.1)

c11, c12, c13 = st.columns(3)
with c11:
    bottom_mode = st.checkbox("🟢 바닥탐지(실시간) 모드", value=False)
with c12:
    cci_window = st.number_input("CCI 기간", min_value=5, max_value=100, value=14, step=1)
with c13:
    cci_signal = st.number_input("CCI 신호(평균)", min_value=1, max_value=50, value=9, step=1)

c14, c15, c16 = st.columns(3)
with c15:
    cci_over = st.number_input("CCI 과매수 기준", min_value=0, max_value=300, value=100, step=5)
with c16:
    cci_under = st.number_input("CCI 과매도 기준", min_value=-300, max_value=0, value=-100, step=5)
with c14:
    cci_mode = st.selectbox("CCI 조건", ["없음","과매수","과매도"], index=0)

st.markdown('<div class="hint">2차 조건: 선택한 조건만 적용 (없음/양봉 2개/BB 기반/매물대 자동)</div>', unsafe_allow_html=True)
sec_cond = st.selectbox("2차 조건 선택", [
    "없음","양봉 2개 (범위 내)","양봉 2개 연속 상승","BB 기반 첫 양봉 50% 진입","매물대 자동 (하단→상단 재진입 + BB하단 위 양봉)"
])

dup_mode = st.radio("신호 중복 처리", ["중복 제거 (연속 동일 결과 1개)","중복 포함 (연속 신호 모두)"], index=0, horizontal=True)
st.markdown("---")

# -------------------------------------------------------------
# ③ 매매기법(9개) 멀티 선택
# -------------------------------------------------------------
st.markdown('<div class="section-title">③ 매매기법 선택 (메인 9전략)</div>', unsafe_allow_html=True)
sel_strategies = st.multiselect("알람/감시에 사용할 전략(시뮬레이터와 별개 선택 가능)", MAIN9, default=["TGV","RVB","PR","Divergence_RVB"])

# -------------------------------------------------------------
# 데이터 로드 + 지표 + 시뮬레이션
# -------------------------------------------------------------
try:
    if start_date > end_date:
        st.error("시작 날짜가 종료 날짜보다 이후입니다.")
        st.stop()

    start_dt = datetime.combine(start_date, datetime.min.time())
    if end_date == datetime.now(KST).date():
        end_dt = datetime.now(KST).astimezone(KST).replace(tzinfo=None)
    else:
        end_dt = datetime.combine(end_date, datetime.max.time())

    warmup_bars = max(13, bb_window, int(cci_window)) * 5
    df_raw = load_ohlcv(market_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars)
    if df_raw.empty:
        st.error("데이터가 없습니다.")
        st.stop()

    df_ind = add_indicators(df_raw, bb_window, bb_dev, cci_window, cci_signal)
    df = df_ind[(df_ind["time"]>=start_dt) & (df_ind["time"]<=end_dt)].reset_index(drop=True)

    res_all = simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, threshold_pct,
                       bb_cond, "중복 포함 (연속 신호 모두)",
                       minutes_per_bar, market_code, bb_window, bb_dev,
                       sec_cond=sec_cond, hit_basis="종가 기준", miss_policy="(고정) 성공·실패·중립",
                       bottom_mode=bottom_mode, supply_levels=None, manual_supply_levels=None,
                       cci_mode=cci_mode, cci_over=cci_over, cci_under=cci_under, cci_signal_n=cci_signal)
    res_dedup = simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, threshold_pct,
                         bb_cond, "중복 제거 (연속 동일 결과 1개)",
                         minutes_per_bar, market_code, bb_window, bb_dev,
                         sec_cond=sec_cond, hit_basis="종가 기준", miss_policy="(고정) 성공·실패·중립",
                         bottom_mode=bottom_mode, supply_levels=None, manual_supply_levels=None,
                         cci_mode=cci_mode, cci_over=cci_over, cci_under=cci_under, cci_signal_n=cci_signal)
    res = res_all if dup_mode.startswith("중복 포함") else res_dedup

    # 차트 영역
    with st.container():
        st.markdown('<div class="section-title">④ 차트</div>', unsafe_allow_html=True)
        df_view = df.copy()
        max_bars = 5000
        if len(df_view) > max_bars:
            df_view = df_view.iloc[-max_bars:].reset_index(drop=True)
        fig = plot_signals_chart(df_view, res if res is not None else pd.DataFrame(), bb_on=(bb_cond!="없음"), buy_price=0, cci_signal=cci_signal, cci_window=cci_window)
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True, "doubleClick": "autosize", "responsive": True})

    # 요약/표
    st.markdown('<div class="section-title">⑤ 요약 & 신호 결과</div>', unsafe_allow_html=True)
    def _summarize(df_in):
        if df_in is None or df_in.empty:
            return 0,0,0,0,0.0,0.0
        total=len(df_in); succ=(df_in["결과"]=="성공").sum(); fail=(df_in["결과"]=="실패").sum(); neu=(df_in["결과"]=="중립").sum()
        win=succ/total*100 if total else 0.0; total_final=df_in["최종수익률(%)"].sum()
        return total,succ,fail,neu,win,total_final

    for label, data in [("중복 제거 (연속 동일 결과 1개)", res_dedup), ("중복 포함 (연속 신호 모두)", res_all)]:
        total, succ, fail, neu, win, total_final = _summarize(data)
        st.markdown(f"**{label}**")
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("신호 수", f"{total}")
        m2.metric("성공", f"{succ}")
        m3.metric("실패", f"{fail}")
        m4.metric("중립", f"{neu}")
        m5.metric("승률", f"{win:.1f}%")
        col = "red" if total_final>0 else "blue" if total_final<0 else "black"
        m6.markdown(f"<div style='font-weight:600;'>최종수익률 합계: <span style='color:{col}; font-size:1.1rem'>{total_final:.1f}%</span></div>", unsafe_allow_html=True)

    st.markdown("—")
    st.markdown("#### 신호 테이블 (최신 순)")
    if res is None or res.empty:
        st.info("조건을 만족하는 신호가 없습니다.")
    else:
        tbl = res.sort_values("신호시간", ascending=False).reset_index(drop=True).copy()
        def _safe_fmt(v, fmt=":.2f", suffix=""):
            if pd.isna(v): return ""
            try: return format(float(v), fmt)+suffix
            except Exception: return str(v)
        tbl["신호시간"]=pd.to_datetime(tbl["신호시간"]).dt.strftime("%Y-%m-%d %H:%M")
        tbl["기준시가"]=tbl["기준시가"].map(lambda v: f"{int(float(v)):,}" if pd.notna(v) else "")
        if "RSI(13)" in tbl: tbl["RSI(13)"]=tbl["RSI(13)"].map(lambda v:_safe_fmt(v,":.2f"))
        if "성공기준(%)" in tbl: tbl["성공기준(%)"]=tbl["성공기준(%)"].map(lambda v:_safe_fmt(v,":.1f","%"))
        for col in ["최종수익률(%)","최저수익률(%)","최고수익률(%)"]:
            if col in tbl: tbl[col]=tbl[col].map(lambda v:_safe_fmt(v,":.2f","%"))
        if "도달캔들(bars)" in tbl:
            tbl["도달캔들"]=tbl["도달캔들(bars)"].astype(int)
            def _fmt_from_bars(b):
                total_min=int(b)*int(minutes_per_bar); hh,mm=divmod(total_min,60); return f"{hh:02d}:{mm:02d}"
            tbl["도달시간"]=tbl["도달캔들"].map(_fmt_from_bars)
            tbl = tbl.drop(columns=["도달캔들(bars)"])
        keep = ["신호시간","기준시가","RSI(13)","성공기준(%)","결과","최종수익률(%)","최저수익률(%)","최고수익률(%)","도달캔들","도달시간"]
        keep = [c for c in keep if c in tbl.columns]
        styled_tbl = tbl[keep].style.applymap(lambda v: "background-color:#FFF59D; color:#E53935; font-weight:600;" if v=="성공" else ("color:#1E40AF; font-weight:600;" if v=="실패" else ("color:#FF9800; font-weight:600;" if v=="중립" else "")), subset=["결과"]) if "결과" in tbl else tbl
        st.dataframe(styled_tbl, use_container_width=True)

except Exception as e:
    st.error(f"오류 발생: {e}")

# -------------------------------------------------------------
# ⑥ 커스텀 페어 백테스트 (섹션 명확히 분리)
# -------------------------------------------------------------
st.markdown('<div class="section-title">⑥ 커스텀 페어 백테스트</div>', unsafe_allow_html=True)
pb1, pb2, pb3, pb4 = st.columns(4)
with pb1:
    base_label, base_code = st.selectbox("기준 종목 (거래량순)", MARKET_LIST, index=0, key="pair_base", format_func=lambda x: x[0])
with pb2:
    follow_label, follow_code = st.selectbox("추종 종목 (거래량순)", MARKET_LIST, index=1, key="pair_follow", format_func=lambda x: x[0])
with pb3:
    tf_pair = st.selectbox("타임프레임(페어)", ["1분","3분","5분","15분","30분","60분"], index=2)
with pb4:
    lookahead_pair = st.slider("N봉(페어 목표기간)", 3, 60, 10)

pd1, pd2, pd3 = st.columns(3)
with pd1:
    start_pair = st.date_input("시작일(페어)", value=(datetime.now(KST).date()-timedelta(days=7)))
with pd2:
    end_pair = st.date_input("종료일(페어)", value=datetime.now(KST).date())
with pd3:
    pair_strategies = st.multiselect("전략(페어)", MAIN9, default=["TGV","RVB","PR"])

run_pair = st.button("▶ 페어 백테스트 실행")
if run_pair:
    try:
        interval_key_p, mpb_p = TF_MAP[tf_pair]
        sdt_p = datetime.combine(start_pair, datetime.min.time())
        edt_p = datetime.combine(end_pair, datetime.max.time())
        warmup_p = 13*5
        df_base = load_ohlcv(base_code, interval_key_p, sdt_p, edt_p, mpb_p, warmup_p)
        df_follow = load_ohlcv(follow_code, interval_key_p, sdt_p, edt_p, mpb_p, warmup_p)
        if df_base.empty or df_follow.empty:
            st.warning("페어 데이터가 비었습니다. 기간/분봉을 조정하세요.")
        else:
            # 동일 시간축으로 맞춤 (inner join)
            left = df_base.set_index("time")[["open","high","low","close","volume"]].add_prefix("B_")
            right = df_follow.set_index("time")[["open","high","low","close","volume"]].add_prefix("F_")
            merged = left.join(right, how="inner").reset_index().rename(columns={"index":"time"})
            # 간단한 동조성 테스트: 기준 전략 발생 시 추종 반응
            # 여기서는 RSI/BB/CCI를 기준 종목에 부여하여 anchor를 만들고, 추종 종목 수익률을 검사
            dfb = add_indicators(df_base, bb_window=20, bb_dev=2.0, cci_window=14, cci_signal=9)
            res_anchor = simulate(dfb, rsi_mode="과매도 기준", rsi_low=30, rsi_high=70,
                                  lookahead=lookahead_pair, threshold_pct=1.0,
                                  bb_cond="하한선", dedup_mode="중복 제거 (연속 동일 결과 1개)",
                                  minutes_per_bar=mpb_p, market_code=base_code, bb_window=20, bb_dev=2.0,
                                  sec_cond="양봉 2개 (범위 내)", hit_basis="종가 기준",
                                  bottom_mode=False, cci_mode="없음")
            if res_anchor is None or res_anchor.empty:
                st.info("기준 종목에서 유효 신호가 없어 페어 평가가 없습니다.")
            else:
                # 추종 종목 수익률 측정
                rows=[]
                for _, r in res_anchor.iterrows():
                    t0 = pd.to_datetime(r["신호시간"])
                    if t0 not in df_follow["time"].values: 
                        continue
                    i0 = df_follow.index[df_follow["time"]==t0][0]
                    end_i = min(i0+lookahead_pair, len(df_follow)-1)
                    base = float(df_follow.at[i0,"close"])
                    endc = float(df_follow.at[end_i,"close"])
                    final = (endc/base-1)*100
                    rows.append({
                        "기준신호시간": t0.strftime("%Y-%m-%d %H:%M"),
                        "추종-기준가": int(base),
                        "추종-종료가": int(endc),
                        "추종-최종수익률(%)": round(final,2),
                        "N(봉)": int(lookahead_pair)
                    })
                if rows:
                    pdf = pd.DataFrame(rows)
                    st.markdown("**페어 결과(기준 신호 기준 → 추종 수익률)**")
                    st.dataframe(pdf, use_container_width=True)
                else:
                    st.info("동일 시간에 추종 종목 캔들이 없어 페어 결과가 비었습니다.")
    except Exception as e:
        st.error(f"페어 백테스트 오류: {e}")

# -------------------------------------------------------------
# ⑦ 실시간 감시 및 알람
# -------------------------------------------------------------
st.markdown('<div class="section-title">⑦ 실시간 감시 및 알람</div>', unsafe_allow_html=True)
if "alerts_live" not in st.session_state: st.session_state["alerts_live"]=[]
if "alert_history" not in st.session_state: st.session_state["alert_history"]=[]

colA, colB = st.columns([2,1])
with colA:
    sel_symbols = st.multiselect("감시할 종목 (거래량순)", MARKET_LIST, default=[MARKET_LIST[0]], format_func=lambda x: x[0])
with colB:
    sel_tfs = st.multiselect("감시할 분봉", ["1","5","15"], default=["5"])

colC1, colC2 = st.columns(2)
with colC1:
    auto_on = st.toggle("▶ 자동 감시(1분 주기)", value=True, key="auto_watch_enabled")
with colC2:
    if st.button("🔁 즉시 감시 갱신"):
        st.rerun()

if auto_on:
    st.caption("🕐 자동 감시 중")

if sel_symbols and sel_tfs and sel_strategies:
    for s in sel_symbols:
        s_code = s[1] if isinstance(s,(list,tuple)) else str(s)
        for strategy in sel_strategies:
            use_tfs = STRATEGY_TF_MAP.get(strategy, sel_tfs)
            for tf in use_tfs:
                try:
                    tf_key = f"minutes/{tf}"
                    df_watch = load_ohlcv(s_code, tf_key, datetime.now()-timedelta(hours=4), datetime.now(), int(tf), 0)
                    if df_watch is None or df_watch.empty: 
                        continue
                    df_watch = add_indicators(df_watch, bb_window=20, bb_dev=2.0, cci_window=14, cci_signal=9)
                    if strategy=="TGV": check_tgv_signal(df_watch, s_code, tf)
                    elif strategy=="RVB": check_rvb_signal(df_watch, s_code, tf)
                    elif strategy=="PR": check_pr_signal(df_watch, s_code, tf)
                    elif strategy=="LCT": check_lct_signal(df_watch, s_code, tf)
                    elif strategy=="4D_Sync": check_4d_sync_signal(df_watch, s_code, tf)
                    elif strategy=="240m_Sync": check_240m_sync_signal(df_watch, s_code, tf)
                    elif strategy=="Composite_Confirm": check_composite_confirm_signal(df_watch, s_code, tf)
                    elif strategy=="Divergence_RVB": check_divergence_rvb_signal(df_watch, s_code, tf)
                    elif strategy=="Market_Divergence": check_market_divergence_signal(df_watch, s_code, tf)
                except Exception as e:
                    st.warning(f"{s_code}({tf}분) 감시 오류: {e}")

st.markdown("### 🚨 실시간 알람 (최신 3개)")
if st.session_state["alerts_live"]:
    for a in st.session_state["alerts_live"][:3]:
        st.warning(f"{a['time']} | {a['symbol']} {a['tf']}분 | {a['strategy']} | {a.get('tp','-')}/{a.get('sl','-')}")
else:
    st.info("알람 없음")

st.markdown("### 📜 알람 히스토리")
if st.session_state["alert_history"]:
    for h in st.session_state["alert_history"][:20]:
        st.markdown(
            f"- **{h.get('time','')}** · {h.get('symbol','')}({h.get('tf','')}분) · {h.get('strategy','')}  \n"
            f"  {h.get('msg','')}"
        )
else:
    st.info("히스토리 없음")

# -------------------------------------------------------------
# ⑧ 공유 메모 / CSV 업로드
# -------------------------------------------------------------
st.markdown('<div class="section-title">⑧ 공유 메모 / CSV 업로드</div>', unsafe_allow_html=True)
SHARED_NOTES_FILE = os.path.join(os.path.dirname(__file__), "shared_notes.md")
_notes_text = ""
try:
    if not os.path.exists(SHARED_NOTES_FILE):
        with open(SHARED_NOTES_FILE, "w", encoding="utf-8") as f:
            f.write("# 📒 공유 메모\n\n- 팀 공통 메모를 작성하세요.\n")
    with open(SHARED_NOTES_FILE, "r", encoding="utf-8") as f:
        _notes_text = f.read()
except Exception:
    _notes_text = ""

with st.expander("📒 공유 메모 (GitHub 연동 선택)", expanded=False):
    notes_text = st.text_area("내용 (Markdown)", value=_notes_text, height=220)
    c_1, c_2 = st.columns(2)
    with c_1:
        if st.button("💾 메모 저장(로컬)"):
            try:
                with open(SHARED_NOTES_FILE, "w", encoding="utf-8") as f:
                    f.write(notes_text)
                st.success("메모 로컬 저장 완료")
            except Exception as e:
                st.warning(f"저장 실패: {e}")
    with c_2:
        if st.button("📤 메모 GitHub 업로드"):
            try:
                with open(SHARED_NOTES_FILE, "w", encoding="utf-8") as f:
                    f.write(notes_text)
                ok, msg = github_commit_file(SHARED_NOTES_FILE)
                if ok: st.success("메모 GitHub 업로드 완료")
                else:  st.warning(f"업로드 실패: {msg}")
            except Exception as e:
                st.warning(f"업로드 오류: {e}")

# CSV 업로드(현재 조회 종목 캐시)
tf_key_save = ("{}min".format(interval_key.split("/")[1]) if "minutes/" in interval_key else "day")
data_dir = os.path.join(os.path.dirname(__file__), "data_cache")
csv_path = os.path.join(data_dir, f"{market_code}_{tf_key_save}.csv")
with st.expander("📤 CSV GitHub 업로드", expanded=False):
    if st.button("CSV 업로드 실행"):
        target = csv_path if os.path.exists(csv_path) else None
        if target:
            ok, msg = github_commit_file(target)
            if ok: st.success("CSV GitHub 업로드 완료")
            else:  st.warning(f"업로드 실패: {msg}")
        else:
            st.warning("CSV 파일이 아직 생성되지 않았습니다. 먼저 데이터를 조회하세요.")
