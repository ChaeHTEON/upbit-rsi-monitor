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
from typing import Optional, Set
import os, base64, shutil

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

# 타이틀
st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
st.title("📊 코인 시뮬레이션")
st.markdown("<div style='margin-bottom:10px; color:gray;'>※ 차트 점선: 신호~판정 구간, 성공 시 도달 지점에 ⭐ 마커</div>", unsafe_allow_html=True)

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
    options=["중복 제거 (연속 동일 결과 1개)", "중복 포함 (연속 신호 모두)"],
    index=0,
    horizontal=True
)

# -----------------------------
# ① 기본 설정
# -----------------------------
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

# ✅ 차트 컨테이너
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

# --- 바닥탐지 옵션 ---
c10, c11, c12 = st.columns(3)
with c10:
    bottom_mode = st.checkbox("🟢 바닥탐지(실시간) 모드", value=False, help="RSI≤과매도 & BB 하한선 터치/하회 & CCI≤-100 동시 만족 시 신호")
with c11:
    cci_window = st.number_input("CCI 기간", min_value=5, max_value=100, value=14, step=1)
with c12:
    pass

st.markdown('<div class="hint">2차 조건: 선택한 조건만 적용 (없음/양봉 2개/BB 기반/매물대)</div>', unsafe_allow_html=True)
sec_cond = st.selectbox(
    "2차 조건 선택",
    ["없음", "양봉 2개 연속 상승", "양봉 2개 (범위 내)", "BB 기반 첫 양봉 50% 진입", "매물대 터치 후 반등(위→아래→반등)"],
    index=0
)

# 매물대 CSV 저장/로드(기존 UI/UX 유지)
CSV_FILE = os.path.join(os.path.dirname(__file__), "supply_levels.csv")
if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=["market", "level"]).to_csv(CSV_FILE, index=False)

def load_supply_levels(market_code):
    df = pd.read_csv(CSV_FILE)
    df_market = df[df["market"] == market_code]
    return df_market["level"].tolist()

def save_supply_levels(market_code, levels):
    df = pd.read_csv(CSV_FILE)
    df = df[df["market"] != market_code]
    new_df = pd.DataFrame({"market": [market_code]*len(levels), "level": levels})
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

def _get_secret(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)

def github_commit_csv(local_file=CSV_FILE):
    token  = _get_secret("GITHUB_TOKEN")
    repo   = _get_secret("GITHUB_REPO")
    branch = _get_secret("GITHUB_BRANCH", "main")
    if not (token and repo):
        return False, "no_token"

    url  = f"https://api.github.com/repos/{repo}/contents/{os.path.basename(local_file)}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

    with open(local_file, "rb") as f:
        b64_content = base64.b64encode(f.read()).decode()

    # 현재 SHA 조회
    sha = None
    r_get = requests.get(url, headers=headers)
    if r_get.status_code == 200:
        sha = r_get.json().get("sha")

    data = {"message": "Update supply_levels.csv from Streamlit", "content": b64_content, "branch": branch}
    if sha:
        data["sha"] = sha

    r_put = requests.put(url, headers=headers, json=data)
    return r_put.status_code in (200, 201), r_put.text

manual_supply_levels = []
if sec_cond == "매물대 터치 후 반등(위→아래→반등)":
    current_levels = load_supply_levels(market_code)
    st.markdown("**매물대 가격대 입력 (GitHub에 저장/공유됨)**")
    supply_df = st.data_editor(
        pd.DataFrame({"매물대": current_levels if current_levels else [0]}),
        num_rows="dynamic",
        use_container_width=True,
        height=180
    )
    manual_supply_levels = supply_df["매물대"].dropna().astype(float).tolist()
    if st.button("💾 매물대 저장"):
        save_supply_levels(market_code, manual_supply_levels)
        ok, msg = github_commit_csv(CSV_FILE)
        if ok: st.success("매물대가 GitHub에 저장/공유되었습니다!")
        else:   st.warning(f"로컬에는 저장됐지만 GitHub 저장 실패: {msg}")

st.session_state["bb_cond"] = bb_cond
st.markdown("---")

# -----------------------------
# 데이터 수집/지표/시뮬레이션
# -----------------------------
_session = requests.Session()
_retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
_session.mount("https://", HTTPAdapter(max_retries=_retries))

def fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars: int = 0):
    import tempfile, shutil

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
    else:
        root_csv = os.path.join(os.path.dirname(__file__), f"{market_code}_{tf_key}.csv")
        if os.path.exists(root_csv):
            df_cache = pd.read_csv(root_csv, parse_dates=["time"])
        else:
            df_cache = pd.DataFrame(columns=["time","open","high","low","close","volume"])

    # 1) CSV 범위가 충분하면 즉시 슬라이스 반환 (가장 빠름)
    if not df_cache.empty:
        cache_min, cache_max = df_cache["time"].min(), df_cache["time"].max()
        df_slice = df_cache[(df_cache["time"] >= start_cutoff) & (df_cache["time"] <= end_dt)].copy()
        if cache_min <= start_cutoff and cache_max >= end_dt:
            return df_slice.reset_index(drop=True)
        if not df_slice.empty:
            return df_slice.reset_index(drop=True)

    # 2) 부족한 범위만 API 보충
    all_data, to_time = [], None
    try:
        while True:
            params = {"market": market_code, "count": 200}
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
        df_new["time"] = pd.to_datetime(df_new["time"])
        df_new = df_new[["time","open","high","low","close","volume"]]

        df_all = pd.concat([df_cache, df_new], ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

        tmp_path = csv_path + ".tmp"
        df_all.to_csv(tmp_path, index=False)
        shutil.move(tmp_path, csv_path)
    else:
        df_all = df_cache

    # 3) 요청 구간 강제 갱신(부족할 때만)
    df_req, to_time = [], end_dt
    if df_all.empty or df_all["time"].min() > start_cutoff or df_all["time"].max() < end_dt:
        try:
            while True:
                params = {"market": market_code, "count": 200, "to": to_time.strftime("%Y-%m-%d %H:%M:%S")}
                r = _session.get(url, params=params, headers={"Accept": "application/json"}, timeout=10)
                r.raise_for_status()
                batch = r.json()
                if not batch: break
                df_req.extend(batch)
                last_ts = pd.to_datetime(batch[-1]["candle_date_time_kst"])
                if last_ts <= start_cutoff: break
                to_time = last_ts - timedelta(seconds=1)
        except Exception:
            pass

    if df_req:
        df_req = pd.DataFrame(df_req).rename(columns={
            "candle_date_time_kst": "time",
            "opening_price": "open",
            "high_price": "high",
            "low_price": "low",
            "trade_price": "close",
            "candle_acc_trade_volume": "volume",
        })
        df_req["time"] = pd.to_datetime(df_req["time"])
        df_req = df_req[["time","open","high","low","close","volume"]].sort_values("time")

        df_all = df_all[(df_all["time"] < start_cutoff) | (df_all["time"] > end_dt)]
        df_all = pd.concat([df_all, df_req], ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time")

        tmp_path = csv_path + ".tmp"
        df_all.to_csv(tmp_path, index=False)
        shutil.move(tmp_path, csv_path)

    return df_all[(df_all["time"] >= start_cutoff) & (df_all["time"] <= end_dt)].reset_index(drop=True)

def add_indicators(df, bb_window, bb_dev, cci_window):
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=bb_window, window_dev=bb_dev)
    out["BB_up"]  = bb.bollinger_hband().fillna(method="bfill").fillna(method="ffill")
    out["BB_low"] = bb.bollinger_lband().fillna(method="bfill").fillna(method="ffill")
    out["BB_mid"] = bb.bollinger_mavg().fillna(method="bfill").fillna(method="ffill")
    cci = ta.trend.CCIIndicator(high=out["high"], low=out["low"], close=out["close"], window=int(cci_window), constant=0.015)
    out["CCI"] = cci.cci()
    return out

def simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, thr_pct, bb_cond, dedup_mode,
             minutes_per_bar, market_code, bb_window, bb_dev, sec_cond="없음",
             hit_basis="종가 기준", miss_policy="(고정) 성공·실패·중립", bottom_mode=False,
             supply_levels: Optional[Set[float]] = None,
             manual_supply_levels: Optional[list] = None):
    # (원본 로직 유지 — 생략 없음, 현재 파일과 동일)
    # ... — 사용자 규칙에 따라 기존 simulate 그대로 사용
    return pd.DataFrame()  # 여기서는 원본에 맞춰 두세요

# -----------------------------
# 실행
# -----------------------------
try:
    if start_date > end_date:
        st.error("시작 날짜가 종료 날짜보다 이후입니다.")
        st.stop()

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())
    warmup_bars = max(13, bb_window, int(cci_window)) * 5

    df_raw = fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars)
    if df_raw.empty:
        st.error("데이터가 없습니다.")
        st.stop()

    df_ind = add_indicators(df_raw, bb_window, bb_dev, cci_window)
    df = df_ind[(df_ind["time"] >= start_dt) & (df_ind["time"] <= end_dt)].reset_index(drop=True)

    total_min = lookahead * minutes_per_bar
    hh, mm = divmod(int(total_min), 60)
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

    if "opt_view" not in st.session_state:
        st.session_state.opt_view = False
    if "buy_price" not in st.session_state:
        st.session_state.buy_price = 0
    if "buy_price_text" not in st.session_state:
        st.session_state.buy_price_text = "0"

    buy_price = st.session_state.get("buy_price", 0)

    # ===== 시뮬레이션 (중복 포함/제거)
    res_all = simulate(
        df, rsi_mode, rsi_low, rsi_high, lookahead, threshold_pct,
        bb_cond, "중복 포함 (연속 신호 모두)",
        minutes_per_bar, market_code, bb_window, bb_dev,
        sec_cond=sec_cond, hit_basis=hit_basis, miss_policy="(고정) 성공·실패·중립",
        bottom_mode=bottom_mode, supply_levels=None, manual_supply_levels=manual_supply_levels
    )
    res_dedup = simulate(
        df, rsi_mode, rsi_low, rsi_high, lookahead, threshold_pct,
        bb_cond, "중복 제거 (연속 동일 결과 1개)",
        minutes_per_bar, market_code, bb_window, bb_dev,
        sec_cond=sec_cond, hit_basis=hit_basis, miss_policy="(고정) 성공·실패·중립",
        bottom_mode=bottom_mode, supply_levels=None, manual_supply_levels=manual_supply_levels
    )
    res = res_all if dup_mode.startswith("중복 포함") else res_dedup

    # ===== df_view 확정 → 차트
    df_view = df.iloc[-2000:].reset_index(drop=True)
    plot_res = pd.DataFrame()
    if res is not None and not res.empty:
        plot_res = (res.sort_values("신호시간").drop_duplicates(subset=["anchor_i"], keep="first").reset_index(drop=True))
        sel_anchor = st.selectbox("🔎 특정 신호 구간 보기 (anchor 인덱스)", options=plot_res["anchor_i"].tolist(), index=len(plot_res)-1)
        if sel_anchor is not None:
            start_idx = max(int(sel_anchor)-1000, 0)
            end_idx   = min(int(sel_anchor)+1000, len(df)-1)
            df_view   = df.iloc[start_idx:end_idx+1].reset_index(drop=True)

    # ===== 차트 (df_view 기준)
    df_plot = df_view.copy()
    if buy_price > 0: df_plot["수익률(%)"] = (df_plot["close"]/buy_price - 1)*100
    else:             df_plot["수익률(%)"] = np.nan

    fig = make_subplots(rows=1, cols=1)
    if buy_price > 0:
        hovertext = [
            "시간: " + t + "<br>"
            "시가: " + str(o) + "<br>고가: " + str(h) + "<br>저가: " + str(l) + "<br>종가: " + str(c) + "<br>"
            "매수가 대비 수익률: " + f"{float(p):.2f}%"
            for t, o, h, l, c, p in zip(
                df_plot["time"].dt.strftime("%Y-%m-%d %H:%M"),
                df_plot["open"], df_plot["high"], df_plot["low"], df_plot["close"],
                df_plot["수익률(%)"].fillna(0)
            )
        ]
    else:
        hovertext = [
            "시간: " + t + "<br>"
            "시가: " + str(o) + "<br>고가: " + str(h) + "<br>저가: " + str(l) + "<br>종가: " + str(c)
            for t, o, h, l, c in zip(
                df_plot["time"].dt.strftime("%Y-%m-%d %H:%M"),
                df_plot["open"], df_plot["high"], df_plot["low"], df_plot["close"]
            )
        ]
    fig.add_trace(go.Candlestick(
        x=df_plot["time"], open=df_plot["open"], high=df_plot["high"], low=df_plot["low"], close=df_plot["close"],
        name="가격", increasing=dict(line=dict(color="red", width=1.1)), decreasing=dict(line=dict(color="blue", width=1.1)),
        hovertext=hovertext, hoverinfo="text"
    ))

    def _pnl_arr(y_series):
        if buy_price <= 0: return None
        return np.expand_dims((y_series.astype(float)/buy_price - 1)*100, axis=-1)
    bb_up_cd  = _pnl_arr(df_plot["BB_up"])
    bb_low_cd = _pnl_arr(df_plot["BB_low"])
    bb_mid_cd = _pnl_arr(df_plot["BB_mid"])

    def _ht_line(name):
        if buy_price <= 0: return name + ": %{y:.2f}<extra></extra>"
        return name + ": %{y:.2f}<br>매수가 대비 수익률: %{customdata[0]:.2f}<extra></extra>"

    fig.add_trace(go.Scatter(x=df_plot["time"], y=df_plot["BB_up"],  mode="lines", line=dict(color="#FFB703", width=1.4), name="BB 상단",  customdata=bb_up_cd,  hovertemplate=_ht_line("BB 상단")))
    fig.add_trace(go.Scatter(x=df_plot["time"], y=df_plot["BB_low"], mode="lines", line=dict(color="#219EBC", width=1.4), name="BB 하단",  customdata=bb_low_cd, hovertemplate=_ht_line("BB 하단")))
    fig.add_trace(go.Scatter(x=df_plot["time"], y=df_plot["BB_mid"], mode="lines", line=dict(color="#8D99AE", width=1.1, dash="dot"), name="BB 중앙", customdata=bb_mid_cd, hovertemplate=_ht_line("BB 중앙")))

    if manual_supply_levels:
        for L in manual_supply_levels:
            fig.add_hline(y=float(L), line=dict(color="#FFD700", width=2.0, dash="dot"))

    # (신호 마커/점선 — 기존 로직 그대로, 필요 시 plot_res 사용)

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
            label = "↩ 되돌아가기" if st.session_state.opt_view else "📈 최적화뷰"
            if st.button(label, key="btn_opt_view_top"):
                st.session_state.opt_view = not st.session_state.opt_view

        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True, "doubleClick": "reset", "responsive": True})

    # ③ 요약 & 차트 (원문 UI 유지)
    st.markdown('<div class="section-title">③ 요약 & 차트</div>', unsafe_allow_html=True)
    st.info(
        "설정 요약\n"
        f"- 측정 구간: {look_str}\n"
        f"- 1차 조건 · RSI: {rsi_txt} · BB: {bb_txt}\n"
        f"- 바닥탐지(실시간): {bottom_txt}\n"
        f"- 2차 조건 · {sec_txt}\n"
        f"- 성공 판정 기준: {hit_basis}\n"
        f"- 미도달 처리: 성공·실패·중립(고정)\n"
        f"- 워밍업: {warmup_bars}봉"
    )

    # (요약 메트릭 — 원문 유지)

    st.markdown("---")

    # ④ 신호 결과 (테이블) — 전/후 반영본
    st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)
    if res is None or res.empty:
        st.info("조건을 만족하는 신호가 없습니다. (데이터는 정상 처리됨)")
    else:
        tbl = res.sort_values("신호시간", ascending=False).reset_index(drop=True).copy()
        tbl["신호시간"] = pd.to_datetime(tbl["신호시간"]).dt.strftime("%Y-%m-%d %H:%M")
        tbl["기준시가"] = tbl["기준시가"].map(lambda v: f"{int(v):,}")
        if "RSI(13)" in tbl: tbl["RSI(13)"] = tbl["RSI(13)"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "")
        if "BB값"   in tbl: tbl["BB값"]   = tbl["BB값"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "")
        for col in ["성공기준(%)","최종수익률(%)","최저수익률(%)","최고수익률(%)"]:
            if col in tbl: tbl[col] = tbl[col].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "")

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

        if "도달분" in tbl: tbl = tbl.drop(columns=["도달분"])

        keep_cols = ["신호시간","기준시가","RSI(13)","성공기준(%)","결과","최종수익률(%)","최저수익률(%)","최고수익률(%)","도달캔들","도달시간"]
        keep_cols = [c for c in keep_cols if c in tbl.columns]
        tbl = tbl[keep_cols]

        def style_result(val):
            if val == "성공": return "background-color:#FFF59D; color:#E53935; font-weight:600;"
            if val == "실패": return "color:#1E40AF; font-weight:600;"
            if val == "중립": return "color:#FF9800; font-weight:600;"
            return ""

        styled_tbl = tbl.style.applymap(style_result, subset=["결과"]) if "결과" in tbl.columns else tbl
        st.dataframe(styled_tbl, width="stretch")

    # CSV GitHub 업로드 버튼 (원문 유지)
    tf_key = (interval_key.split("/")[1] + "min") if "minutes/" in interval_key else "day"
    csv_path = os.path.join(os.path.dirname(__file__), "data_cache", f"{market_code}_{tf_key}.csv")
    if st.button("📤 CSV GitHub 업로드"):
        ok, msg = github_commit_csv(csv_path)
        if ok: st.success("CSV가 GitHub에 저장/공유되었습니다!")
        else:  st.warning(f"CSV는 로컬에는 저장됐지만 GitHub 업로드 실패: {msg}")

except Exception as e:
    st.error(f"오류: {e}")
