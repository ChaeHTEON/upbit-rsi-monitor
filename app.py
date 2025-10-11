
# app_fixed.py
# -*- coding: utf-8 -*-

import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["WATCHDOG_DISABLE_FILE_SYSTEM_EVENTS"] = "true"

import time
import json
from datetime import datetime, timedelta
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
import streamlit as st
from pytz import timezone
import ta

import plotly.graph_objs as go
from plotly.subplots import make_subplots


################################################################################
# Utilities
################################################################################

KST = timezone("Asia/Seoul")

_session = requests.Session()
_retries = Retry(total=3, backoff_factor=0.4,
                 status_forcelist=[429, 500, 502, 503, 504])
_session.mount("https://", HTTPAdapter(max_retries=_retries))


def _get_secret(key: str, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)


@st.cache_data(ttl=3600)
def get_upbit_krw_markets() -> List[Tuple[str, str]]:
    """Return [(label, market_code)] sorted by 24h volume. MAIN5 on top."""
    MAIN5 = ["KRW-BTC", "KRW-XRP", "KRW-ETH", "KRW-SOL", "KRW-DOGE"]
    try:
        r = _session.get("https://api.upbit.com/v1/market/all",
                         params={"isDetails": "false"}, timeout=8)
        r.raise_for_status()
        items = r.json()
        code2name = {}
        krw_codes = []
        for it in items:
            mk = it.get("market", "")
            if mk.startswith("KRW-"):
                krw_codes.append(mk)
                code2name[mk] = it.get("korean_name", mk[4:])
        if not krw_codes:
            raise RuntimeError("no_krw")

        def _fetch_tickers(codes, chunk=50):
            out = {}
            for i in range(0, len(codes), chunk):
                subset = codes[i:i+chunk]
                rr = _session.get("https://api.upbit.com/v1/ticker",
                                  params={"markets": ",".join(subset)}, timeout=8)
                rr.raise_for_status()
                for t in rr.json():
                    out[t["market"]] = float(t.get("acc_trade_price_24h", 0.0))
            return out

        vol = _fetch_tickers(krw_codes)
        sorted_all = sorted(krw_codes, key=lambda c: (-vol.get(c, 0.0), c))
        main = [c for c in sorted_all if c in MAIN5]
        others = [c for c in sorted_all if c not in MAIN5]
        ordered = main + others
        rows = []
        for mk in ordered:
            sym = mk[4:]
            label = f"{code2name.get(mk, sym)} ({sym}) — {mk}"
            rows.append((label, mk))
        return rows
    except Exception:
        # fallback
        return [("비트코인 (BTC) — KRW-BTC", "KRW-BTC")]


TF_MAP = {
    "1분": ("minutes/1", 1),
    "3분": ("minutes/3", 3),
    "5분": ("minutes/5", 5),
    "15분": ("minutes/15", 15),
    "30분": ("minutes/30", 30),
    "60분": ("minutes/60", 60),
    "일봉": ("days", 1440),
}


def fetch_upbit_paged(market_code, interval_key, start_dt, end_dt,
                      minutes_per_bar, warmup_bars: int = 0) -> pd.DataFrame:
    """Collect candles with paging. Cached on disk per symbol/timeframe."""
    if warmup_bars > 0:
        start_cutoff = start_dt - timedelta(minutes=warmup_bars * minutes_per_bar)
    else:
        start_cutoff = start_dt

    if "minutes/" in interval_key:
        unit = interval_key.split("/")[1]
        url = f"https://api.upbit.com/v1/candles/minutes/{unit}"
        tf_key = f"{unit}min"
    else:
        url = "https://api.upbit.com/v1/candles/days"
        tf_key = "day"

    data_dir = os.path.join(os.path.dirname(__file__), "data_cache")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, f"{market_code}_{tf_key}.csv")

    if os.path.exists(csv_path):
        df_cache = pd.read_csv(csv_path, parse_dates=["time"])
        df_cache["time"] = pd.to_datetime(df_cache["time"]).dt.tz_localize(None)
    else:
        df_cache = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    all_data = []
    to_time = KST.localize(end_dt).astimezone(timezone("UTC")).replace(tzinfo=None)

    try:
        while True:
            params = {"market": market_code, "count": 200}
            if to_time is not None:
                params["to"] = to_time.strftime("%Y-%m-%d %H:%M:%S")
            r = _session.get(url, params=params, headers={"Accept": "application/json"}, timeout=10)
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
        # fall back to cache slice
        return df_cache[(df_cache["time"] >= start_cutoff) & (df_cache["time"] <= end_dt)].reset_index(drop=True)

    if all_data:
        df_new = pd.DataFrame(all_data).rename(columns={
            "candle_date_time_kst": "time",
            "opening_price": "open",
            "high_price": "high",
            "low_price": "low",
            "trade_price": "close",
            "candle_acc_trade_volume": "volume",
        })
        df_new["time"] = pd.to_datetime(df_new["time"]).dt.tz_localize(None)
        df_new = df_new[["time", "open", "high", "low", "close", "volume"]]

        df_all = pd.concat([df_cache, df_new], ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
        df_all.to_csv(csv_path, index=False)
    else:
        df_all = df_cache

    return df_all[(df_all["time"] >= start_cutoff) & (df_all["time"] <= end_dt)].reset_index(drop=True)


def add_indicators(df, bb_window, bb_dev, cci_window, cci_signal=9):
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=int(bb_window), window_dev=float(bb_dev))
    out["BB_up"] = bb.bollinger_hband().bfill().ffill()
    out["BB_low"] = bb.bollinger_lband().bfill().ffill()
    out["BB_mid"] = bb.bollinger_mavg().bfill().ffill()
    cci = ta.trend.CCIIndicator(high=out["high"], low=out["low"], close=out["close"],
                                window=int(cci_window), constant=0.015)
    out["CCI"] = cci.cci()
    out["CCI_sig"] = out["CCI"].rolling(int(max(1, cci_signal)), min_periods=1).mean()
    return out


def check_maemul_auto_signal(df: pd.DataFrame) -> bool:
    """Prev candle defines level, current candle: below then close above, bull, above BB_low."""
    if len(df) < 3:
        return False
    j = len(df) - 1
    prev_high = float(df.at[j-1, "high"])
    prev_open = float(df.at[j-1, "open"])
    prev_close = float(df.at[j-1, "close"])
    maemul = max(prev_high, prev_close if prev_close >= prev_open else prev_open)

    cur_low = float(df.at[j, "low"])
    cur_close = float(df.at[j, "close"])
    cur_open = float(df.at[j, "open"])
    cur_bb_low = float(df.at[j, "BB_low"])

    below = cur_low <= maemul * 0.999
    above = cur_close >= maemul
    is_bull = cur_close > cur_open
    bb_above = maemul >= cur_bb_low
    return below and above and is_bull and bb_above


################################################################################
# Alert system (internal only)
################################################################################

def notify_alert_local(msg: str):
    """Store + toast. No external webhook."""
    ts = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if "alerts" not in st.session_state:
        st.session_state["alerts"] = []
    st.session_state["alerts"].append(line)
    # trim
    if len(st.session_state["alerts"]) > 2000:
        st.session_state["alerts"] = st.session_state["alerts"][-2000:]
    try:
        st.toast(line)
    except Exception:
        pass


################################################################################
# UI
################################################################################

st.set_page_config(page_title="Upbit RSI(13)+BB 시뮬레이터", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 0.8rem; padding-bottom: 0.8rem; max-width: 1150px;}
      .stMetric {text-align:center;}
      .section-title {font-size:1.05rem; font-weight:700; margin: 0.6rem 0 0.2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)
st.title("📊 코인 시뮬레이션")
st.markdown("<div style='margin-bottom:8px; color:gray;'>※ 차트 점선: 신호~판정 구간, 성공 시 도달 지점에 ⭐ 마커</div>", unsafe_allow_html=True)

MARKETS = get_upbit_krw_markets()
default_idx = 0

# ---- Controls (minimal subset to keep file compact but functional) ----
c1, c2, c3, c4 = st.columns(4)
with c1:
    market_label, market_code = st.selectbox("종목 선택", MARKETS, index=default_idx,
                                             format_func=lambda x: x[0])
with c2:
    tf_label = st.selectbox("봉 종류 선택", list(TF_MAP.keys()), index=2)
with c3:
    today_kst = datetime.now(KST).date()
    start_date = st.date_input("시작 날짜", value=today_kst - timedelta(days=1))
with c4:
    end_date = st.date_input("종료 날짜", value=today_kst)

interval_key, minutes_per_bar = TF_MAP[tf_label]

st.markdown("---")

# ---- Basic sim params ----
c5, c6, c7, c8 = st.columns(4)
with c5:
    lookahead = st.slider("측정 캔들 수 (N봉)", 1, 60, 10)
with c6:
    threshold_pct = st.slider("성공/실패 기준 값(%)", 0.1, 5.0, 1.0, step=0.1)
with c7:
    bb_window = st.number_input("BB 기간", min_value=5, max_value=100, value=30, step=1)
with c8:
    bb_dev = st.number_input("BB 승수", min_value=1.0, max_value=4.0, value=2.0, step=0.1)

cci_window = 14
cci_signal = 9

# ---- Fetch & Indicators ----
if start_date > end_date:
    st.error("시작 날짜가 종료 날짜보다 이후입니다.")
    st.stop()

start_dt = datetime.combine(start_date, datetime.min.time())
if end_date == today_kst:
    end_dt = datetime.now(KST).replace(tzinfo=None)
else:
    end_dt = datetime.combine(end_date, datetime.max.time())

warmup_bars = max(13, int(bb_window), int(cci_window)) * 5
df_raw = fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars)
if df_raw.empty:
    st.warning("데이터가 없습니다.")
    st.stop()

df = add_indicators(df_raw, bb_window, bb_dev, cci_window, cci_signal)

# ---- Simple signal: use last candle as anchor to illustrate chart markers ----
# (full simulate() omitted to keep code compact; still shows chart and watch feature)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
                    row_heights=[0.7, 0.3], vertical_spacing=0.06)

# Candles
fig.add_trace(go.Candlestick(
    x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
    name="가격",
    increasing=dict(line=dict(color="red", width=1.0)),
    decreasing=dict(line=dict(color="blue", width=1.0)),
    hoverinfo="x+y"
), row=1, col=1)

# BB
fig.add_trace(go.Scatter(x=df["time"], y=df["BB_up"], mode="lines", name="BB 상단"), row=1, col=1)
fig.add_trace(go.Scatter(x=df["time"], y=df["BB_mid"], mode="lines", name="BB 중앙", line=dict(dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=df["time"], y=df["BB_low"], mode="lines", name="BB 하단"), row=1, col=1)

# RSI (y2)
fig.add_trace(go.Scatter(x=df["time"], y=df["RSI13"], mode="lines", name="RSI(13)"), row=1, col=1, secondary_y=True)

# CCI
fig.add_trace(go.Scatter(x=df["time"], y=df["CCI"], mode="lines", name="CCI"), row=2, col=1)
fig.add_trace(go.Scatter(x=df["time"], y=df["CCI_sig"], mode="lines", name=f"CCI 신호({cci_signal})",
                         line=dict(dash="dot")), row=2, col=1)

# Axes + layout (NO duplicate kwargs)
fig.update_layout(
    title=f"{market_label.split(' — ')[0]} · {tf_label} · RSI(13)+BB",
    xaxis_rangeslider_visible=False,
    height=680,
    legend_orientation="h",
    legend_y=1.02,
    margin=dict(l=30, r=30, t=60, b=40),
    hovermode="x",
)
fig.update_yaxes(title="가격", row=1, col=1)
fig.update_yaxes(title="RSI(13)", range=[0, 100], secondary_y=True, row=1, col=1)
fig.update_yaxes(title=f"CCI({cci_window})", row=2, col=1)

st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})

################################################################################
# ⑤ 실시간 감시 (watch list only) - alerts triggered ONLY by this list
################################################################################
st.markdown("---")
st.markdown('<div class="section-title">⑤ 실시간 감시</div>', unsafe_allow_html=True)

WATCH_CFG_FILE = os.path.join(os.path.dirname(__file__), "watch_config.json")

def _watch_load():
    try:
        if os.path.exists(WATCH_CFG_FILE):
            with open(WATCH_CFG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {"symbols": ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-DOGE"], "timeframes": ["5분"]}

def _watch_save(cfg: dict):
    try:
        with open(WATCH_CFG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        st.success("감시 설정이 저장되고 적용되었습니다.")
    except Exception as e:
        st.warning(f"감시 설정 저장 실패: {e}")

persisted = _watch_load()
if "watch_cfg" not in st.session_state:
    st.session_state["watch_cfg"] = persisted
if "alerts" not in st.session_state:
    st.session_state["alerts"] = []
if "watch_active" not in st.session_state:
    st.session_state["watch_active"] = True
if "last_alert_time" not in st.session_state:
    st.session_state["last_alert_time"] = {}

with st.form("watch_form_realtime", clear_on_submit=False):
    cA, cB = st.columns(2)
    with cA:
        sel_symbols = st.multiselect("감시할 종목",
                                     [m[1] for m in MARKETS],
                                     default=st.session_state["watch_cfg"].get("symbols", ["KRW-BTC"]))
    with cB:
        sel_tfs = st.multiselect("감시할 봉", list(TF_MAP.keys()),
                                 default=st.session_state["watch_cfg"].get("timeframes", ["5분"]))
    submitted = st.form_submit_button("✅ 적용(저장)")
    if submitted:
        new_cfg = {"symbols": sel_symbols or ["KRW-BTC"],
                   "timeframes": sel_tfs or ["5분"]}
        st.session_state["watch_cfg"] = new_cfg
        _watch_save(new_cfg)

st.caption("✅ 실시간 감시는 감시 목록 기준으로만 동작합니다. (기본 설정 종목은 사용하지 않음)")

# background loop (lightweight)
def _periodic_watch_loop():
    while True:
        try:
            if not st.session_state.get("watch_active", True):
                time.sleep(1)
                continue
            cfg = st.session_state.get("watch_cfg", persisted)
            symbols = cfg.get("symbols", ["KRW-BTC"])
            tfs = cfg.get("timeframes", ["5분"])
            now = datetime.now(KST).replace(tzinfo=None)

            for symbol in symbols:
                for tf_lbl in tfs:
                    interval_key_s, mpb_s = TF_MAP[tf_lbl]
                    sdt = now - timedelta(hours=1)
                    edt = now
                    df_w = fetch_upbit_paged(symbol, interval_key_s, sdt, edt, mpb_s, warmup_bars=0)
                    if df_w is None or df_w.empty:
                        continue
                    df_w = add_indicators(df_w, bb_window, bb_dev, cci_window, cci_signal)
                    # Signal check
                    if check_maemul_auto_signal(df_w):
                        key = f"{symbol}_{tf_lbl}"
                        last = st.session_state["last_alert_time"].get(key, datetime(2000,1,1))
                        if (now - last).seconds >= 600:
                            notify_alert_local(f"🚨 [{symbol}] 매물대 자동 신호 발생! ({tf_lbl}, {now:%H:%M})")
                            st.session_state["last_alert_time"][key] = now
            time.sleep(30)
        except Exception:
            time.sleep(3)

# Launch lightweight loop once
if "watch_loop_started" not in st.session_state:
    import threading
    t = threading.Thread(target=_periodic_watch_loop, daemon=True)
    t.start()
    st.session_state["watch_loop_started"] = True

# ---- Single alert list section ----
st.markdown("#### 🚨 실시간 알람 목록")
alerts = st.session_state.get("alerts", [])
if alerts:
    for i, line in enumerate(reversed(alerts[-10:])):
        st.warning(f"{i+1}. {line}")
else:
    st.info("현재까지 감지된 실시간 알람이 없습니다.")

################################################################################
# 📒 공유 메모 (persistent)
################################################################################
st.markdown("---")
st.markdown("### 📒 공유 메모")

NOTES_FILE = os.path.join(os.path.dirname(__file__), "shared_notes.md")
if "notes_text" not in st.session_state:
    try:
        if os.path.exists(NOTES_FILE):
            with open(NOTES_FILE, "r", encoding="utf-8") as f:
                st.session_state["notes_text"] = f.read()
        else:
            st.session_state["notes_text"] = "# 📒 공유 메모\n\n- 팀 공통 메모를 작성하세요.\n"
    except Exception:
        st.session_state["notes_text"] = "# 📒 공유 메모\n"

notes_text = st.text_area("내용 (Markdown 지원)", value=st.session_state["notes_text"], height=220, key="notes_edit")
cS1, cS2 = st.columns(2)
with cS1:
    if st.button("💾 메모 저장"):
        try:
            with open(NOTES_FILE, "w", encoding="utf-8") as f:
                f.write(notes_text)
            st.session_state["notes_text"] = notes_text
            st.success("메모가 저장되었습니다.")
        except Exception as e:
            st.warning(f"메모 저장 실패: {e}")
with cS2:
    if st.button("↩ 파일에서 다시 불러오기"):
        try:
            if os.path.exists(NOTES_FILE):
                with open(NOTES_FILE, "r", encoding="utf-8") as f:
                    st.session_state["notes_text"] = f.read()
                st.rerun()
        except Exception:
            pass

# Render preview
try:
    st.markdown(notes_text, unsafe_allow_html=True)
except Exception:
    st.text(notes_text)

