# === AUTO-PATCH (minimal) ===
# - Removed duplicate hovermode in fig.update_layout calls
# - Removed Kakao test button block
# - Disabled base-signal toast; alerts now only via ⑤ 실시간 감시
# - Deduped duplicate '실시간 알람 목록' header
# - Source: app (8).py
# - Source SHA256: 2906ed2ba9404c9b3d95396b8adde7771946e5aea9a88a39a3e2e884116c132c
# - Patched: 2025-10-11T10:59:05.660225
# =============================


# app.py
# -*- coding: utf-8 -*-

import os
# watchdog/inotify 한도 초과 방지
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["WATCHDOG_DISABLE_FILE_SYSTEM_EVENTS"] = "true"

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
from typing import Optional, Set, List, Tuple

# -----------------------------
# 공용 유틸
# -----------------------------
def _get_secret(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)

# 외부 전송은 현재 단계 비활성 (요청사항)
def send_kakao_alert(msg: str):
    return False

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
  table {border-collapse:collapse; width:100%;}
  th, td {border:1px solid #ddd; padding:6px; text-align:center;}
</style>
""", unsafe_allow_html=True)

st.title("📊 코인 시뮬레이션")
st.markdown("<div style='margin-bottom:10px; color:gray;'>※ 차트 점선: 신호~판정 구간, 성공 시 도달 지점에 ⭐ 마커</div>", unsafe_allow_html=True)

# -----------------------------
# 업비트 마켓 로드
# -----------------------------
@st.cache_data(ttl=3600)
def get_upbit_krw_markets() -> List[Tuple[str,str]]:
    try:
        r = requests.get("https://api.upbit.com/v1/market/all",
                         params={"isDetails": "false"}, timeout=8)
        r.raise_for_status()
        items = r.json()

        code2name = {}
        krw_codes = []
        for it in items:
            mk = it.get("market", "")
            if mk.startswith("KRW-"):
                krw_codes.append(mk)
                code2name[mk] = it.get("korean_name", "")

        if not krw_codes:
            raise RuntimeError("no_krw_markets")

        # 24h 거래대금으로 정렬
        def _fetch_tickers(codes, chunk=50):
            out = {}
            for i in range(0, len(codes), chunk):
                subset = codes[i:i+chunk]
                rr = requests.get(
                    "https://api.upbit.com/v1/ticker",
                    params={"markets": ",".join(subset)},
                    timeout=8
                )
                rr.raise_for_status()
                for t in rr.json():
                    mk = t.get("market")
                    out[mk] = float(t.get("acc_trade_price_24h", 0.0))
            return out
        vol_krw = _fetch_tickers(krw_codes)

        sorted_all = sorted(krw_codes, key=lambda c: (-vol_krw.get(c, 0.0), c))
        MAIN5 = ["KRW-BTC", "KRW-XRP", "KRW-ETH", "KRW-SOL", "KRW-DOGE"]
        main_sorted   = [c for c in sorted_all if c in MAIN5]
        others_sorted = [c for c in sorted_all if c not in MAIN5]
        ordered = main_sorted + others_sorted

        rows = []
        for mk in ordered:
            sym = mk[4:]
            knm = code2name.get(mk, sym)
            label = f"{knm} ({sym}) — {mk}"
            rows.append((label, mk))
        if rows:
            return rows
    except Exception:
        pass

    # 폴백
    rows = [("비트코인 (BTC) — KRW-BTC", "KRW-BTC")]
    return rows

MARKET_LIST = get_upbit_krw_markets()
default_idx = 0

# -----------------------------
# 타임프레임 맵
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
# 상단 UI - 기본 설정
# -----------------------------
dup_mode = st.radio(
    "신호 중복 처리",
    options=["중복 제거 (연속 동일 결과 1개)", "중복 포함 (연속 신호 모두)"],
    index=0, horizontal=True
)

st.markdown('<div class="section-title">① 기본 설정</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    market_label, market_code = st.selectbox("종목 선택", MARKET_LIST, index=default_idx, format_func=lambda x: x[0])
with c2:
    tf_label = st.selectbox("봉 종류 선택", list(TF_MAP.keys()), index=2)
with c3:
    KST = timezone("Asia/Seoul")
    today_kst = datetime.now(KST).date()
    default_start = today_kst - timedelta(days=1)
    start_date = st.date_input("시작 날짜", value=default_start)
with c4:
    end_date = st.date_input("종료 날짜", value=today_kst)

interval_key, minutes_per_bar = TF_MAP[tf_label]
st.markdown("---")
chart_box = st.container()

# -----------------------------
# ② 조건 설정
# -----------------------------
st.markdown('<div class="section-title">② 조건 설정</div>', unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
with c4:
    lookahead = st.slider("측정 캔들 수 (기준 이후 N봉)", 1, 60, 10)
with c5:
    threshold_pct = st.slider("성공/실패 기준 값(%)", 0.1, 5.0, 1.0, step=0.1)
    winrate_thr   = st.slider("승률 기준(%)", 10, 100, 70, step=1)
    hit_basis = "종가 기준"
with c6:
    r1, r2, r3 = st.columns(3)
    with r1:
        rsi_mode = st.selectbox(
            "RSI 조건",
            ["없음", "현재(과매도/과매수 중 하나)", "과매도 기준", "과매수 기준"],
            index=0
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

# 바닥탐지 + CCI
c10, c11, c12 = st.columns(3)
with c10:
    bottom_mode = st.checkbox("🟢 바닥탐지(실시간) 모드", value=False)
with c11:
    cci_window = st.number_input("CCI 기간", min_value=5, max_value=100, value=14, step=1)
with c12:
    cci_signal = st.number_input("CCI 신호(평균)", min_value=1, max_value=50, value=9, step=1)

c13, c14, c15 = st.columns(3)
with c14:
    cci_over = st.number_input("CCI 과매수 기준", min_value=0, max_value=300, value=100, step=5)
with c15:
    cci_under = st.number_input("CCI 과매도 기준", min_value=-300, max_value=0, value=-100, step=5)
with c13:
    cci_mode = st.selectbox(
        "CCI 조건",
        options=["없음", "과매수", "과매도"],
        format_func=lambda x: (
            "없음" if x == "없음" else
            f"과매수(≥{cci_over})" if x == "과매수" else
            f"과매도(≤{cci_under})"
        ),
        index=0
    )

st.markdown('<div class="hint">2차 조건: 선택한 조건만 적용 (없음/양봉 2개/BB 기반/매물대)</div>', unsafe_allow_html=True)
sec_cond = st.selectbox(
    "2차 조건 선택",
    [
        "없음",
        "양봉 2개 (범위 내)",
        "양봉 2개 연속 상승",
        "BB 기반 첫 양봉 50% 진입",
        "매물대 터치 후 반등(위→아래→반등)",
        "매물대 자동 (하단→상단 재진입 + BB하단 위 양봉)"
    ]
)

# 매물대 수동 입력 (간소화: 로컬 CSV만)
CSV_FILE = os.path.join(os.path.dirname(__file__), "supply_levels.csv")
if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=["market", "level"]).to_csv(CSV_FILE, index=False)

def load_supply_levels(market_code):
    try:
        df = pd.read_csv(CSV_FILE)
        df_market = df[df["market"] == market_code]
        return df_market["level"].tolist()
    except Exception:
        return []

def save_supply_levels(market_code, levels):
    df = pd.read_csv(CSV_FILE) if os.path.exists(CSV_FILE) else pd.DataFrame(columns=["market","level"])
    df = df[df["market"] != market_code]
    new_df = pd.DataFrame({"market": [market_code]*len(levels), "level": levels})
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

manual_supply_levels = []
if sec_cond == "매물대 터치 후 반등(위→아래→반등)":
    current_levels = load_supply_levels(market_code)
    st.markdown("**매물대 가격대 입력**")
    supply_df = st.data_editor(
        pd.DataFrame({"매물대": current_levels if current_levels else [0]}),
        num_rows="dynamic",
        use_container_width=True,
        height=180
    )
    manual_supply_levels = supply_df["매물대"].dropna().astype(float).tolist()
    if st.button("💾 매물대 저장"):
        try:
            save_supply_levels(market_code, manual_supply_levels)
            st.success("로컬 저장 완료")
        except Exception as _e:
            st.warning(f"매물대 저장 실패: {_e}")

st.session_state["bb_cond"] = bb_cond
st.markdown("---")

# -----------------------------
# 데이터 로딩 & 지표/시뮬
# -----------------------------
_session = requests.Session()
_retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
_session.mount("https://", HTTPAdapter(max_retries=_retries))

def fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars: int = 0):
    import shutil
    if warmup_bars and warmup_bars > 0:
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
        root_csv = os.path.join(os.path.dirname(__file__), f"{market_code}_{tf_key}.csv")
        if os.path.exists(root_csv):
            df_cache = pd.read_csv(root_csv, parse_dates=["time"])
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
        return df_cache[(df_cache["time"] >= start_cutoff) & (df_cache["time"] <= end_dt)]

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
        tmp_path = csv_path + ".tmp"
        df_all.to_csv(tmp_path, index=False)
        try:
            shutil.move(tmp_path, csv_path)
        except FileNotFoundError:
            df_all.to_csv(csv_path, index=False)
    else:
        df_all = df_cache

    return df_all[(df_all["time"] >= start_cutoff) & (df_all["time"] <= end_dt)].reset_index(drop=True)

def add_indicators(df, bb_window, bb_dev, cci_window, cci_signal=9):
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
             minutes_per_bar, market_code, bb_window, bb_dev, sec_cond="없음",
             hit_basis="종가 기준", miss_policy="(고정) 성공·실패·중립", bottom_mode=False,
             supply_levels: Optional[Set[float]] = None,
             manual_supply_levels: Optional[list] = None,
             cci_mode: str = "없음", cci_over: float = 100.0, cci_under: float = -100.0, cci_signal_n: int = 9):

    res = []
    n = len(df)
    thr = float(threshold_pct)

    if bottom_mode:
        base_sig_idx = df.index[
            (df["RSI13"] <= float(rsi_low)) &
            (df["close"] <= df["BB_low"]) &
            (df["CCI"] <= -100)
        ].tolist()
    else:
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
                if pd.isna(lo):
                    return False
                rv = float(lo)
                entered_from_below = (o < rv) or (l <= rv)
                closes_above       = c >= rv
                return entered_from_below and closes_above
            if bb_cond == "중앙선":
                if pd.isna(mid):
                    return False
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
            closes_above       = (c >= rv)
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
        base_price = float(df.at[anchor_idx, "close"])  # ✅ 신호 캔들의 종가

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
            base_price  = float(df.at[anchor_idx, "close"])

        elif sec_cond == "BB 기반 첫 양봉 50% 진입":
            if bb_cond == "없음":
                return None, None
            B1_idx, B1_close = first_bull_50_over_bb(i0)
            if B1_idx is None:
                return None, None
            anchor_idx = B1_idx + 1
            if anchor_idx >= n:
                return None, None
            signal_time = df.at[anchor_idx, "time"]
            base_price  = float(df.at[anchor_idx, "close"])

        elif sec_cond == "매물대 터치 후 반등(위→아래→반등)":
            rebound_idx = None
            scan_end = min(i0 + lookahead, n - 1)
            for j in range(i0 + 1, scan_end + 1):
                if manual_supply_levels:
                    low_j   = float(df.at[j, "low"])
                    close_j = float(df.at[j, "close"])
                    touched = any(low_j <= float(L) for L in manual_supply_levels)
                    is_nbar_low = False
                    lookback_n = 50
                    past_n = df.loc[:j-1].tail(lookback_n)
                    if not past_n.empty:
                        min_price = past_n["low"].min()
                        if low_j <= min_price * 1.001:
                            is_nbar_low = True
                    if touched and is_nbar_low and close_j > max(manual_supply_levels):
                        rebound_idx = j
                        break
            if rebound_idx is None:
                return None, None
            anchor_idx = rebound_idx + 1
            if anchor_idx >= n:
                return None, None
            signal_time = df.at[anchor_idx, "time"]
            base_price  = float(df.at[anchor_idx, "close"])

        elif sec_cond == "매물대 자동 (하단→상단 재진입 + BB하단 위 양봉)":
            # 본 모드는 실시간 감시에서만 사용(알림). 백테스트도 가능하도록 로직 유지.
            anchor_idx = None
            scan_end = min(i0 + lookahead, n - 1)
            for j in range(i0 + 2, scan_end + 1):
                prev_high = float(df.at[j - 1, "high"])
                prev_open = float(df.at[j - 1, "open"])
                prev_close = float(df.at[j - 1, "close"])
                cur_low = float(df.at[j, "low"])
                cur_close = float(df.at[j, "close"])
                cur_open = float(df.at[j, "open"])
                cur_bb_low = float(df.at[j, "BB_low"])
                maemul = max(prev_high, prev_close if prev_close >= prev_open else prev_open)
                below = cur_low <= maemul * 0.999
                above = cur_close >= maemul
                is_bull = cur_close > cur_open
                bb_above = maemul >= cur_bb_low
                if below and above and is_bull and bb_above:
                    anchor_idx = j
                    break
            if anchor_idx is None or anchor_idx >= n:
                return None, None
            signal_time = df.at[anchor_idx, "time"]
            base_price  = float(df.at[anchor_idx, "close"])

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
            bars_after = end_idx - anchor_idx
            end_time = df.at[end_idx, "time"]
            end_close = float(df.at[end_idx, "close"])
            final_ret = (end_close / base_price - 1) * 100
            result = "실패" if final_ret <= 0 else "중립"
            lock_end = end_idx

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
            "end_i": int(hit_idx if hit_idx is not None else end_idx),
        }
        return row, int(lock_end)

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
        df_res = pd.DataFrame(res).drop_duplicates(subset=["anchor_i"], keep="first").reset_index(drop=True)
        return df_res
    return pd.DataFrame()

# -----------------------------
# 실행
# -----------------------------
try:
    if start_date > end_date:
        st.error("시작 날짜가 종료 날짜보다 이후입니다.")
        st.stop()

    KST = timezone("Asia/Seoul")
    start_dt = datetime.combine(start_date, datetime.min.time())
    if end_date == datetime.now(KST).date():
        end_dt = datetime.now(KST).astimezone(KST).replace(tzinfo=None)
    else:
        end_dt = datetime.combine(end_date, datetime.max.time())
    warmup_bars = max(13, bb_window, int(cci_window)) * 5

    df_raw = fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars)
    if df_raw.empty:
        st.error("데이터가 없습니다.")
        st.stop()

    df_ind = add_indicators(df_raw, bb_window, bb_dev, cci_window, cci_signal)
    df = df_ind[(df_ind["time"] >= start_dt) & (df_ind["time"] <= end_dt)].reset_index(drop=True)

    # ❌ [중요] 기본 종목에서의 즉시 알림(2차조건=매물대 자동) 제거
    #    -> 실시간 감시는 ⑤ 섹션의 다중 감시 스레드에서만 수행

    # 보기 요약
    total_min = lookahead * int(minutes_per_bar)
    hh, mm = divmod(total_min, 60)
    look_str = f"{lookahead}봉 / {hh:02d}:{mm:02d}"

    if rsi_mode == "없음":
        rsi_txt = "없음"
    elif rsi_mode == "현재(과매도/과매수 중 하나)":
        rsi_txt = f"현재: (과매도≤{int(rsi_low)}) 또는 (과매수≥{int(rsi_high)})"
    elif rsi_mode == "과매도 기준":
        rsi_txt = f"과매도≤{int(rsi_low)}"
    else:
        rsi_txt = f"과매수≥{int(rsi_high)}"

    bb_txt = bb_cond if bb_cond != "없음" else "없음"
    sec_txt = f"{sec_cond}"
    bottom_txt = "ON" if bottom_mode else "OFF"
    cci_txt = ("없음" if cci_mode == "없음"
               else f"{'과매수≥' + str(int(cci_over)) if cci_mode.startswith('과매수') else '과매도≤' + str(int(cci_under))} · 기간 {int(cci_window)} · 신호 {int(cci_signal)}")

    # 매수가 입력 + 최적화뷰 버튼
    if "opt_view" not in st.session_state:
        st.session_state.opt_view = False
    if "buy_price" not in st.session_state:
        st.session_state.buy_price = 0
    if "buy_price_text" not in st.session_state:
        st.session_state.buy_price_text = "0"
    buy_price = st.session_state.get("buy_price", 0)

    def _toggle_opt_view():
        st.session_state.opt_view = not st.session_state.get("opt_view", False)
        st.rerun()

    # 시뮬레이션 (중복 포함/제거)
    res_all = simulate(
        df, rsi_mode, rsi_low, rsi_high, lookahead, threshold_pct,
        bb_cond, "중복 포함 (연속 신호 모두)",
        minutes_per_bar, market_code, bb_window, bb_dev,
        sec_cond=sec_cond, hit_basis="종가 기준", miss_policy="(고정) 성공·실패·중립",
        bottom_mode=bottom_mode, supply_levels=None, manual_supply_levels=manual_supply_levels,
        cci_mode=cci_mode, cci_over=cci_over, cci_under=cci_under, cci_signal_n=cci_signal
    )
    res_dedup = simulate(
        df, rsi_mode, rsi_low, rsi_high, lookahead, threshold_pct,
        bb_cond, "중복 제거 (연속 동일 결과 1개)",
        minutes_per_bar, market_code, bb_window, bb_dev,
        sec_cond=sec_cond, hit_basis="종가 기준", miss_policy="(고정) 성공·실패·중립",
        bottom_mode=bottom_mode, supply_levels=None, manual_supply_levels=manual_supply_levels,
        cci_mode=cci_mode, cci_over=cci_over, cci_under=cci_under, cci_signal_n=cci_signal
    )
    res = res_all if dup_mode.startswith("중복 포함") else res_dedup

    # 차트 빌드
    max_bars = 5000
    plot_res = (res.sort_values("신호시간").drop_duplicates(subset=["anchor_i"], keep="first").reset_index(drop=True)
                if res is not None and not res.empty else pd.DataFrame())
    df_view = df.copy()
    if len(df_view) > max_bars:
        df_view = df_view.iloc[-max_bars:].reset_index(drop=True)
    else:
        df_view = df_view.reset_index(drop=True)

    df_plot = df_view.copy()
    if buy_price > 0:
        df_plot["수익률(%)"] = (df_plot["close"] / buy_price - 1) * 100
        df_plot["_pnl_str"] = df_plot["수익률(%)"].apply(lambda v: f"{'+' if v>=0 else ''}{v:.2f}%")
    else:
        df_plot["수익률(%)"] = np.nan
        df_plot["_pnl_str"] = ""

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        row_heights=[0.72, 0.28], vertical_spacing=0.06
    )

    def _fmt_ohlc_tooltip(t, o, h, l, c, pnl_str=None):
        if pnl_str is None or pnl_str == "":
            return "시간: " + t + "<br>시가: " + str(o) + "<br>고가: " + str(h) + "<br>저가: " + str(l) + "<br>종가: " + str(c)
        else:
            return "시간: " + t + "<br>시가: " + str(o) + "<br>고가: " + str(h) + "<br>저가: " + str(l) + "<br>종가: " + str(c) + "<br>수익률(%): " + pnl_str

    def _make_candle_hovertexts(dfp, has_buy):
        if has_buy:
            return [
                _fmt_ohlc_tooltip(
                    t, o, h, l, c, pnl_str=s
                )
                for t, o, h, l, c, s in zip(
                    dfp["time"].dt.strftime("%Y-%m-%d %H:%M"),
                    dfp["open"], dfp["high"], dfp["low"], dfp["close"], dfp["_pnl_str"]
                )
            ]
        else:
            return [
                _fmt_ohlc_tooltip(t, o, h, l, c, pnl_str=None)
                for t, o, h, l, c in zip(
                    dfp["time"].dt.strftime("%Y-%m-%d %H:%M"),
                    dfp["open"], dfp["high"], dfp["low"], dfp["close"]
                )
            ]

    candle_hovertext = _make_candle_hovertexts(df_plot, buy_price > 0)
    fig.add_trace(go.Candlestick(
        x=df_plot["time"], open=df_plot["open"], high=df_plot["high"],
        low=df_plot["low"], close=df_plot["close"], name="가격",
        increasing=dict(line=dict(color="red", width=1.1)),
        decreasing=dict(line=dict(color="blue", width=1.1)),
        hovertext=candle_hovertext, hoverinfo="text"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_plot["time"], y=df_plot["BB_up"], mode="lines",
        line=dict(color="#FFB703", width=1.4), name="BB 상단"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_plot["time"], y=df_plot["BB_low"], mode="lines",
        line=dict(color="#219EBC", width=1.4), name="BB 하단"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_plot["time"], y=df_plot["BB_mid"], mode="lines",
        line=dict(color="#8D99AE", width=1.4, dash="dot"), name="BB 중앙"
    ), row=1, col=1)

    if not plot_res.empty:
        for _label, _color in [("성공", "red"), ("실패", "blue"), ("중립", "#FF9800")]:
            sub = plot_res[plot_res["결과"] == _label]
            if sub.empty:
                continue
            xs, ys = [], []
            for _, r in sub.iterrows():
                t0 = pd.to_datetime(r["신호시간"])
                if t0 in df_plot["time"].values:
                    xs.append(t0)
                    ys.append(float(df_plot.loc[df_plot["time"] == t0, "open"].iloc[0]))
            if xs:
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="markers",
                    name=f"신호({_label})",
                    marker=dict(size=9, color=_color, symbol="circle", line=dict(width=1, color="black"))
                ), row=1, col=1)

        legend_emitted = {"성공": False, "실패": False, "중립": False}
        for _, row_ in plot_res.iterrows():
            t0 = pd.to_datetime(row_["신호시간"])
            t1 = pd.to_datetime(row_["종료시간"])
            if (t0 not in df_plot["time"].values) or (t1 not in df_plot["time"].values):
                continue
            y0 = float(df_plot.loc[df_plot["time"] == t0, "close"].iloc[0])
            y1 = float(df_plot.loc[df_plot["time"] == t1, "close"].iloc[0])
            fig.add_trace(go.Scatter(
                x=[t0, t1], y=[y0, y1], mode="lines",
                line=dict(color="rgba(0,0,0,0.5)", width=1.2, dash="dot"),
                showlegend=False, hoverinfo="skip"
            ), row=1, col=1)
            if row_["결과"] == "성공":
                fig.add_trace(go.Scatter(
                    x=[t1], y=[y1], mode="markers", name="도달⭐",
                    marker=dict(size=12, color="orange", symbol="star", line=dict(width=1, color="black")),
                    showlegend=not legend_emitted["성공"]
                ), row=1, col=1); legend_emitted["성공"] = True
            elif row_["결과"] == "실패":
                fig.add_trace(go.Scatter(
                    x=[t1], y=[y1], mode="markers", name="실패❌",
                    marker=dict(size=12, color="blue", symbol="x", line=dict(width=1, color="black")),
                    showlegend=not legend_emitted["실패"]
                ), row=1, col=1); legend_emitted["실패"] = True
            elif row_["결과"] == "중립":
                fig.add_trace(go.Scatter(
                    x=[t1], y=[y1], mode="markers", name="중립❌",
                    marker=dict(size=12, color="orange", symbol="x", line=dict(width=1, color="black")),
                    showlegend=not legend_emitted["중립"]
                ), row=1, col=1); legend_emitted["중립"] = True

    fig.add_trace(go.Scatter(
        x=df_plot["time"], y=df_plot["RSI13"], mode="lines",
        line=dict(color="rgba(42,157,143,0.30)", width=6),
        name="", showlegend=False
    ), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(
        x=df_plot["time"], y=df_plot["RSI13"], mode="lines",
        line=dict(color="#2A9D8F", width=2.4, dash="dot"),
        name="RSI(13)"
    ), row=1, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(
        x=df_plot["time"], y=df_plot["CCI"], mode="lines",
        line=dict(width=1.6), name="CCI"
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df_plot["time"], y=df_plot["CCI_sig"], mode="lines",
        line=dict(width=1.2, dash="dot"),
        name=f"CCI 신호({int(cci_signal)})"
    ), row=2, col=1)
    for yv, colr in [(100, "#E63946"), (-100, "#457B9D"), (0, "#888")]:
        fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y3", y0=yv, y1=yv, line=dict(color=colr, width=1, dash="dot"))

    fig.update_layout(
        hovermode="x", hoverdistance=1, spikedistance=1,
        title=f"{market_label.split(' — ')[0]} · {tf_label} · RSI(13) + BB 시뮬레이션",
        dragmode="pan", xaxis_rangeslider_visible=False, height=680,
        legend_orientation="h", legend_y=1.02, margin=dict(l=30, r=30, t=60, b=40),
        yaxis=dict(title="가격", autorange=True,  fixedrange=False),
        yaxis2=dict(title="RSI(13)", range=[0, 100], autorange=False, fixedrange=False),
        yaxis3=dict(title=f"CCI({int(cci_window)})", autorange=True,  fixedrange=False),
        uirevision=f"opt-{int(st.session_state.get('opt_view', False))}-{np.random.randint(1_000_000_000)}"
    )

    with chart_box:
        top_l, top_r = st.columns([4, 1])
        def _format_buy_price():
            raw = st.session_state.get("buy_price_text", "0")
            digits = "".join(ch for ch in raw if ch.isdigit())
            if digits == "": digits = "0"
            val = int(digits)
            st.session_state.buy_price = val
            st.session_state.buy_price_text = f"{val:,}"

        with top_l:
            st.text_input("💰 매수가 입력", key="buy_price_text", on_change=_format_buy_price)
            buy_price = st.session_state.get("buy_price", 0)
        with top_r:
            label = "↩ 되돌아가기" if st.session_state.get("opt_view", False) else "📈 최적화뷰"
            st.button(label, key="btn_opt_view_top", on_click=_toggle_opt_view)

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"scrollZoom": True, "displayModeBar": True, "doubleClick": "autosize", "responsive": True},
        )

    # ③ 요약
    st.markdown('<div class="section-title">③ 요약 & 차트</div>', unsafe_allow_html=True)
    st.info(
        "설정 요약\n"
        f"- 측정 구간: {look_str}\n"
        f"- 1차 조건 · RSI: {rsi_txt} · BB: {bb_txt} · CCI: {cci_txt}\n"
        f"- 바닥탐지(실시간): {bottom_txt}\n"
        f"- 2차 조건 · {sec_txt}\n"
        f"- 워밍업: {warmup_bars}봉"
    )

    # 메트릭 요약
    def _summarize(df_in):
        if df_in is None or df_in.empty:
            return 0, 0, 0, 0, 0.0, 0.0
        total = len(df_in)
        succ = (df_in["결과"] == "성공").sum()
        fail = (df_in["결과"] == "실패").sum()
        neu  = (df_in["결과"] == "중립").sum()
        win  = succ / total * 100 if total else 0.0
        total_final = df_in["최종수익률(%)"].sum()
        return total, succ, fail, neu, win, total_final

    for label, data in [("중복 제거 (연속 동일 결과 1개)", res_dedup), ("중복 포함 (연속 신호 모두)", res_all)]:
        total, succ, fail, neu, win, total_final = _summarize(data)
        st.markdown(f"**{label}**")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("신호 수", f"{total}")
        m2.metric("성공", f"{succ}")
        m3.metric("실패", f"{fail}")
        m4.metric("중립", f"{neu}")
        m5.metric("승률", f"{win:.1f}%")
        col = "red" if total_final > 0 else "blue" if total_final < 0 else "black"
        m6.markdown(
            f"<div style='font-weight:600;'>최종수익률 합계: "
            f"<span style='color:{col}; font-size:1.1rem'>{total_final:.1f}%</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ④ 신호 결과 테이블
    st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)
    if res is None or res.empty:
        st.info("조건을 만족하는 신호가 없습니다. (데이터는 정상 처리됨)")
    else:
        tbl = res.sort_values("신호시간", ascending=False).reset_index(drop=True).copy()
        def _safe_fmt(v, fmt=":.2f", suffix=""):
            if pd.isna(v):
                return ""
            try:
                return format(float(v), fmt) + suffix
            except Exception:
                return str(v)
        tbl["신호시간"] = pd.to_datetime(tbl["신호시간"]).dt.strftime("%Y-%m-%d %H:%M")
        tbl["기준시가"] = tbl["기준시가"].map(lambda v: f"{int(float(v)):,}" if pd.notna(v) else "")
        if "RSI(13)" in tbl: tbl["RSI(13)"] = tbl["RSI(13)"].map(lambda v: _safe_fmt(v, ":.2f"))
        if "성공기준(%)" in tbl: tbl["성공기준(%)"] = tbl["성공기준(%)"].map(lambda v: _safe_fmt(v, ":.1f", "%"))
        for col in ["최종수익률(%)", "최저수익률(%)", "최고수익률(%)"]:
            if col in tbl: tbl[col] = tbl[col].map(lambda v: _safe_fmt(v, ":.2f", "%"))
        if "도달캔들(bars)" in tbl.columns:
            tbl["도달캔들"] = tbl["도달캔들(bars)"].astype(int)
            def _fmt_from_bars(b):
                total_min = int(b) * int(minutes_per_bar)
                hh, mm = divmod(total_min, 60)
                return f"{hh:02d}:{mm:02d}"
            tbl["도달시간"] = tbl["도달캔들"].map(_fmt_from_bars)
        else:
            tbl["도달캔들"] = 0
            tbl["도달시간"] = "-"
        drop_cols = [c for c in ["BB값", "도달분", "도달캔들(bars)"] if c in tbl.columns]
        if drop_cols: tbl = tbl.drop(columns=drop_cols)
        keep_cols = ["신호시간", "기준시가", "RSI(13)", "성공기준(%)", "결과",
                     "최종수익률(%)", "최저수익률(%)", "최고수익률(%)", "도달캔들", "도달시간"]
        keep_cols = [c for c in keep_cols if c in tbl.columns]
        tbl = tbl[keep_cols]

        def style_result(val):
            if val == "성공": return "background-color: #FFF59D; color: #E53935; font-weight:600;"
            if val == "실패": return "color: #1E40AF; font-weight:600;"
            if val == "중립": return "color: #FF9800; font-weight:600;"
            return ""
        styled_tbl = tbl.style.applymap(style_result, subset=["결과"]) if "결과" in tbl.columns else tbl
        st.dataframe(styled_tbl, width="stretch")

    # -----------------------------
    # ⑤ 실시간 감시 (다중 종목/봉) — 단일 알람 목록 + 토스트 큐
    # -----------------------------
    import threading, time, json

    WATCH_CFG_FILE = os.path.join(os.path.dirname(__file__), "watch_config.json")

    def _watch_load():
        try:
            if os.path.exists(WATCH_CFG_FILE):
                with open(WATCH_CFG_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {"symbols": ["KRW-BTC"], "timeframes": ["5분"]}

    def _watch_save(cfg: dict):
        try:
            with open(WATCH_CFG_FILE, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            st.caption("감시설정이 로컬에 저장되었습니다.")
        except Exception as _e:
            st.warning(f"감시 설정 저장 실패: {_e}")

    _persisted = _watch_load()

    # 상태 보장
    if "alerts" not in st.session_state:
        st.session_state["alerts"] = []
    if "toast_queue" not in st.session_state:
        st.session_state["toast_queue"] = []
    if "last_alert_time" not in st.session_state:
        st.session_state["last_alert_time"] = {}
    if "watch_active" not in st.session_state:
        st.session_state["watch_active"] = True
    if "watch_active_config" not in st.session_state:
        st.session_state["watch_active_config"] = _persisted.copy()
    if "watch_ui_symbols" not in st.session_state:
        st.session_state["watch_ui_symbols"] = _persisted.get("symbols", ["KRW-BTC"])
    if "watch_ui_tfs" not in st.session_state:
        st.session_state["watch_ui_tfs"] = _persisted.get("timeframes", ["5분"])

    # 실시간 신호 판정 로직
    def check_maemul_auto_signal(df):
        if len(df) < 3: return False
        j = len(df) - 1
        prev_high  = float(df.at[j - 1, "high"])
        prev_open  = float(df.at[j - 1, "open"])
        prev_close = float(df.at[j - 1, "close"])
        cur_low = float(df.at[j, "low"])
        cur_close = float(df.at[j, "close"])
        cur_open = float(df.at[j, "open"])
        cur_bb_low = float(df.at[j, "BB_low"])
        maemul = max(prev_high, prev_close if prev_close >= prev_open else prev_open)
        below = cur_low <= maemul * 0.999
        above = cur_close >= maemul
        is_bull = cur_close > cur_open
        bb_above = maemul >= cur_bb_low
        return below and above and is_bull and bb_above

    def _periodic_multi_check():
        TF_MAP_LOC = {
            "1분": ("minutes/1", 1),
            "3분": ("minutes/3", 3),
            "5분": ("minutes/5", 5),
            "15분": ("minutes/15", 15),
            "30분": ("minutes/30", 30),
            "60분": ("minutes/60", 60),
            "일봉": ("days", 1440)
        }
        os.makedirs("data_cache", exist_ok=True)
        alert_csv = "data_cache/realtime_alerts.csv"
        if not os.path.exists(alert_csv):
            pd.DataFrame(columns=["시간","코인","분봉","신호","현재가"]).to_csv(alert_csv, index=False)

        while True:
            try:
                if not st.session_state.get("watch_active"):
                    time.sleep(1); continue
                KST = timezone("Asia/Seoul")
                now = datetime.now(KST).replace(tzinfo=None)
                cfg = st.session_state.get("watch_active_config") or _persisted or {"symbols": ["KRW-BTC"], "timeframes": ["5분"]}
                symbols = cfg.get("symbols", ["KRW-BTC"])
                tfs     = cfg.get("timeframes", ["5분"])

                for symbol in symbols:
                    for tf_lbl in tfs:
                        interval_key_s, mpb_s = TF_MAP_LOC[tf_lbl]
                        start_dt = now - timedelta(hours=1)
                        end_dt   = now
                        try:
                            df_w = fetch_upbit_paged(symbol, interval_key_s, start_dt, end_dt, mpb_s)
                            if df_w is None or df_w.empty:
                                continue
                            if "time" in df_w.columns:
                                df_w["time"] = pd.to_datetime(df_w["time"], errors="coerce")
                            df_w = add_indicators(df_w, bb_window, bb_dev, cci_window, cci_signal)
                            if check_maemul_auto_signal(df_w):
                                key = f"{symbol}_{tf_lbl}"
                                last_time = st.session_state["last_alert_time"].get(key, datetime(2000,1,1))
                                if (now - last_time).seconds >= 600:
                                    msg = f"🚨 [{symbol}] 매물대 자동 신호 발생! ({tf_lbl}, {now:%H:%M})"
                                    st.session_state["toast_queue"].append(msg)
                                    st.session_state["alerts"].append(msg)
                                    st.session_state["last_alert_time"][key] = now
                                    # CSV 로깅
                                    try:
                                        prev = pd.read_csv(alert_csv)
                                        prev = pd.concat([prev, pd.DataFrame([{
                                            "시간": now.strftime("%Y-%m-%d %H:%M:%S"),
                                            "코인": symbol,
                                            "분봉": tf_lbl,
                                            "신호": "매물대 자동",
                                            "현재가": float(df_w.iloc[-1]["close"])
                                        }])], ignore_index=True)
                                        prev.to_csv(alert_csv, index=False)
                                    except Exception:
                                        pass
                        except Exception:
                            continue
                time.sleep(30)
            except Exception:
                time.sleep(3)

    if "watch_bg_thread" not in st.session_state:
        t = threading.Thread(target=_periodic_multi_check, daemon=True)
        t.start()
        st.session_state["watch_bg_thread"] = True

    st.markdown("---")
    st.markdown('<div class="section-title">⑤ 실시간 감시</div>', unsafe_allow_html=True)

    # 적용 폼
    with st.form("watch_form_realtime", clear_on_submit=False):
        ui_cols = st.columns(2)
        with ui_cols[0]:
            sel_symbols = st.multiselect(
                "감시할 종목",
                [m[1] for m in MARKET_LIST],
                default=st.session_state.get("watch_ui_symbols", ["KRW-BTC"]),
            )
        with ui_cols[1]:
            sel_tfs = st.multiselect(
                "감시할 봉",
                ["1분", "3분", "5분", "15분", "30분", "60분", "일봉"],
                default=st.session_state.get("watch_ui_tfs", ["5분"]),
            )
        submitted = st.form_submit_button("✅ 적용(저장)", use_container_width=True)
        if submitted:
            new_cfg = {"symbols": sel_symbols or ["KRW-BTC"], "timeframes": sel_tfs or ["5분"]}
            _watch_save(new_cfg)
            st.session_state["watch_ui_symbols"] = new_cfg["symbols"]
            st.session_state["watch_ui_tfs"] = new_cfg["timeframes"]
            st.session_state["watch_active_config"] = new_cfg
            st.success("감시 설정이 저장되고 적용되었습니다.")

    # 항상 실행
    st.caption("✅ 실시간 감시가 항상 실행 중입니다. (중지 기능 없음)")

    # ▶ 토스트 큐 처리 (단일 위치)
    if "toast_queue" not in st.session_state:
        st.session_state["toast_queue"] = []
    if len(st.session_state["toast_queue"]) > 0:
        for tmsg in st.session_state["toast_queue"]:
            try:
                st.toast(tmsg)
            except Exception:
                pass
        st.session_state["toast_queue"].clear()

    # ▶ 실시간 알람 목록 (단일 섹션만 유지)
    st.markdown("#### 🚨 실시간 알람 목록")
    alerts_list = st.session_state.get("alerts", [])
    if len(alerts_list) > 0:
        for i, alert in enumerate(reversed(alerts_list[-10:])):
            st.warning(f"{i+1}. {alert}")
    else:
        st.info("현재까지 감지된 실시간 알람이 없습니다.")

except Exception as e:
    st.error(f"오류 발생: {str(e)}")
