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
    index=0,  # ✅ 이제 "중복 제거"가 기본 선택
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
    winrate_thr   = st.slider("승률 기준(%)", 10, 100, 70, step=1)
    # 성공 판정 기준은 항상 종가 기준으로 고정 (UI 제거 요청에 따라 값만 고정)
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
    [
        "없음",
        "양봉 2개 (범위 내)",
        "양봉 2개 연속 상승",
        "BB 기반 첫 양봉 50% 진입",
        "매물대 터치 후 반등(위→아래→반등)"
    ]
)

# ✅ 매물대 반등 조건일 때만 N봉 입력 노출
if sec_cond == "매물대 터치 후 반등(위→아래→반등)":
    maemul_n = st.number_input("매물대 반등 조건: 이전 캔들 수", min_value=5, max_value=500, value=50, step=5)
    st.session_state["maemul_n"] = maemul_n

# ✅ 볼린저 옵션 미체크 시 안내 문구 (bb_cond 값으로 판단)
if sec_cond == "BB 기반 첫 양봉 50% 진입" and bb_cond == "없음":
    st.info("ℹ️ 볼린저 밴드를 활성화해야 이 조건이 정상 작동합니다.")

# ✅ 매물대 조건 UI 추가 (CSV 저장/불러오기 + GitHub commit/push)
import os, base64, requests

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
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    with open(local_file, "rb") as f:
        b64_content = base64.b64encode(f.read()).decode()

    # 현재 SHA 조회
    sha = None
    r_get = requests.get(url, headers=headers)
    if r_get.status_code == 200:
        sha = r_get.json().get("sha")

    data = {
        "message": "Update supply_levels.csv from Streamlit",
        "content": b64_content,
        "branch": branch
    }
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
        height=180  # ✅ 입력창 높이 (약 5줄 수준)
    )
    manual_supply_levels = supply_df["매물대"].dropna().astype(float).tolist()
    if st.button("💾 매물대 저장"):
        save_supply_levels(market_code, manual_supply_levels)
        ok, msg = github_commit_csv(CSV_FILE)
        if ok:
            st.success("매물대가 GitHub에 저장/공유되었습니다!")
        else:
            st.warning(f"로컬에는 저장됐지만 GitHub 저장 실패: {msg}")

st.session_state["bb_cond"] = bb_cond
st.markdown("---")

# -----------------------------
# 데이터 수집/지표/시뮬레이션 함수
# -----------------------------
_session = requests.Session()
_retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
_session.mount("https://", HTTPAdapter(max_retries=_retries))

def fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars: int = 0):
    """Upbit 캔들 페이징 수집 (CSV 저장/보충 포함 + GitHub 커밋 지원).
    - API 기본 반환(최신→과거)을 정렬하여 항상 시간 오름차순 유지
    - 요청 구간(start_dt~end_dt)은 항상 API 호출 후 갱신
    - CSV는 원자적 쓰기(tmp→move)로 저장 안정성 강화
    - 저장 후 GitHub에도 커밋(push)
    """
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

    # CSV 경로 설정
    data_dir = os.path.join(os.path.dirname(__file__), "data_cache")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, f"{market_code}_{tf_key}.csv")

    # CSV 로드 (있으면) — 기본: data_cache/, 없으면 루트에서도 탐색
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

# ✅ CSV 활용 우

    # ⚡ CSV에 일부만 있는 경우 → 부족한 앞/뒤 구간만 API 보충
    from pytz import timezone as _tz
    _KST = _tz("Asia/Seoul"); _UTC = _tz("UTC")
    # ✅ 첫 호출부터 end_dt(KST)를 UTC로 변환한 시각으로 페이징 시작
    all_data = []
    to_time = _KST.localize(end_dt).astimezone(_UTC).replace(tzinfo=None)
    try:
        while True:
            params = {"market": market_code, "count": 200}
            if to_time is not None:
                params["to"] = to_time.strftime("%Y-%m-%d %H:%M:%S")  # ✅ UTC 기준
            r = _session.get(url, params=params, headers={"Accept": "application/json"}, timeout=10)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            all_data.extend(batch)

            # ✅ 경계 판정은 KST, 페이징 파라미터는 UTC 사용
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

        # 캐시와 병합 후 정렬/중복제거
        df_all = pd.concat([df_cache, df_new], ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

        # 원자적 저장 (디렉토리 보장 + 에러 안전 처리)
        data_dir = os.path.dirname(csv_path)
        os.makedirs(data_dir, exist_ok=True)
        tmp_path = csv_path + ".tmp"
        df_all.to_csv(tmp_path, index=False)
        try:
            shutil.move(tmp_path, csv_path)
        except FileNotFoundError:
            # tmp 생성 실패 시 직접 저장
            df_all.to_csv(csv_path, index=False)

        # ⚡ GitHub 커밋은 최종 저장 시 1회만 실행
        # (중간 보충/강제 갱신 단계에서는 커밋하지 않음)
    else:
        # API에서 새로운 데이터가 없으면 캐시 데이터 그대로 사용
        df_all = df_cache

    # ✅ 2차: 요청 구간 강제 갱신 (CSV 부족할 때만 실행)
    #     - to_time 시작점을 end_dt(KST)를 UTC로 변환해 사용
    from pytz import timezone as _tz
    _KST = _tz("Asia/Seoul"); _UTC = _tz("UTC")
    df_req = []
    to_time = _KST.localize(end_dt).astimezone(_UTC).replace(tzinfo=None)

    if df_all.empty or df_all["time"].min() > start_cutoff or df_all["time"].max() < end_dt:
        try:
            while True:
                params = {"market": market_code, "count": 200, "to": to_time.strftime("%Y-%m-%d %H:%M:%S")}  # ✅ UTC
                r = _session.get(url, params=params, headers={"Accept": "application/json"}, timeout=10)
                r.raise_for_status()
                batch = r.json()
                if not batch:
                    break
                df_req.extend(batch)

                # ✅ 경계 판정은 KST, 페이징 파라미터는 UTC 사용
                last_kst = pd.to_datetime(batch[-1]["candle_date_time_kst"])
                last_utc = pd.to_datetime(batch[-1]["candle_date_time_utc"])
                if last_kst <= start_cutoff:
                    break
                to_time = (last_utc - timedelta(seconds=1))
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
        df_req["time"] = pd.to_datetime(df_req["time"]).dt.tz_localize(None)
        df_req = df_req[["time", "open", "high", "low", "close", "volume"]].sort_values("time")

        # 해당 구간 삭제 후 새 데이터 삽입
        df_all = df_all[(df_all["time"] < start_cutoff) | (df_all["time"] > end_dt)]
        df_all = pd.concat([df_all, df_req], ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time")

        # 원자적 저장 (디렉토리 보장 + 에러 안전 처리)
        data_dir = os.path.dirname(csv_path)
        os.makedirs(data_dir, exist_ok=True)
        tmp_path = csv_path + ".tmp"
        df_all.to_csv(tmp_path, index=False)
        try:
            shutil.move(tmp_path, csv_path)
        except FileNotFoundError:
            # tmp 생성 실패 시 직접 저장
            df_all.to_csv(csv_path, index=False)

        # ⚡ GitHub 커밋은 자동 실행하지 않음 (수동 버튼에서만 실행)

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
    """UI/UX 유지. 기존 로직 + 바닥탐지 + 매물대 조건(수동 입력) 반영 + 판정 규칙 고정(종가 기준/성공·실패·중립)."""
    res = []
    n = len(df)
    thr = float(threshold_pct if isinstance(threshold_pct := thr_pct, (int, float)) else thr_pct)

    # --- 1) 1차 조건 인덱스 (RSI/BB/바닥탐지) ---
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
            h = float(df.at[i, "high"])
            l = float(df.at[i, "low"])
            up, lo, mid = df.at[i, "BB_up"], df.at[i, "BB_low"], df.at[i, "BB_mid"]

            if bb_cond == "상한선":
                return pd.notna(up) and (c > float(up))

            if bb_cond == "하한선":
                # ✅ 종가가 하단 이하이거나, 저가가 하단 밴드를 터치 후 종가가 위로 복귀한 경우도 포함
                return pd.notna(lo) and ((c <= float(lo)) or (l <= float(lo) and c > float(lo)))

            if bb_cond == "중앙선":
                if pd.isna(mid):
                    return False
                return c >= float(mid)

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

    # --- 2) 보조 함수 ---
    def is_bull(idx):
        return float(df.at[idx, "close"]) > float(df.at[idx, "open"])

    def first_bull_50_over_bb(start_i):
        for j in range(start_i + 1, n):
            if not is_bull(j):
                continue
            if bb_cond == "하한선":
                ref = df.at[j, "BB_low"]
            elif bb_cond == "중앙선":
                ref = df.at[j, "BB_mid"]
            else:
                ref = df.at[j, "BB_up"]
            if pd.isna(ref):
                continue
            if float(df.at[j, "close"]) >= float(ref):
                return j, float(df.at[j, "close"])
        return None, None

    # --- 3) 공통 처리(하나의 신호 평가) ---
    def process_one(i0):
        anchor_idx = i0 + 1  # ✅ 신호 확인 후, 실제 진입은 다음 봉
        if anchor_idx >= n:
            return None, None
        signal_time = df.at[anchor_idx, "time"]
        base_price = float(df.at[anchor_idx, "open"])  # ✅ 매수가를 다음 봉 시가 기준

        # 2차 조건 공통 원칙:
        # - anchor_idx = 실제 진입(신호 확정) 봉
        # - base_price = close(anchor_idx)
        # - 평가 시작(eval_start) = anchor_idx + 1

        if sec_cond == "양봉 2개 연속 상승":
            if i0 + 2 >= n:
                return None, None
            c1, o1 = float(df.at[i0 + 1, "close"]), float(df.at[i0 + 1, "open"])
            c2, o2 = float(df.at[i0 + 2, "close"]), float(df.at[i0 + 2, "open"])
            if not ((c1 > o1) and (c2 > o2) and (c2 > c1)):
                return None, None
            anchor_idx = i0 + 3  # ✅ 조건 확인 후, 진입은 그 다음 봉
            if anchor_idx >= n:
                return None, None
            signal_time = df.at[anchor_idx, "time"]
            base_price  = float(df.at[anchor_idx, "open"])  # ✅ 시가 기준

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
            anchor_idx = T_idx + 1  # ✅ 신호봉 직후 캔들에서 진입
            if anchor_idx >= n:
                return None, None
            signal_time = df.at[anchor_idx, "time"]
            base_price  = float(df.at[anchor_idx, "open"])  # ✅ 시가 기준

        elif sec_cond == "BB 기반 첫 양봉 50% 진입":
            if bb_cond == "없음":
                return None, None
            B1_idx, B1_close = first_bull_50_over_bb(i0)
            if B1_idx is None:
                return None, None
            anchor_idx = B1_idx + 1  # ✅ BB 신호 확인 후 다음 캔들에서 진입
            if anchor_idx >= n:
                return None, None
            signal_time = df.at[anchor_idx, "time"]
            base_price  = float(df.at[anchor_idx, "open"])  # ✅ 시가 기준

        elif sec_cond == "매물대 터치 후 반등(위→아래→반등)":
            rebound_idx = None
            scan_end = min(i0 + lookahead, n - 1)
            for j in range(i0 + 1, scan_end + 1):
                if manual_supply_levels:
                    touched = False
                    low_j   = float(df.at[j, "low"])
                    close_j = float(df.at[j, "close"])
                    # ① 매물대 터치 여부
                    for L in manual_supply_levels:
                        if low_j <= float(L):
                            touched = True
                            break
                    # ② 직전 N봉 최저가 여부 확인 (허용 오차 포함)
                    is_nbar_low = False
                    lookback_n = st.session_state.get("maemul_n", 50)  # 기본값 50봉
                    past_n = df.loc[:j-1].tail(lookback_n)  # 현재 봉 제외, 직전 N봉만 참조
                    if not past_n.empty:
                        min_price = past_n["low"].min()
                        # ✅ 직전 N봉 최저가 갱신 or 최저가 수준(±0.1%) 터치 시 인정
                        if low_j <= min_price * 1.001:
                            is_nbar_low = True
                    # ③ 최종 조건: 매물대 터치 + N봉 최저가 + 매물대 위 종가 복귀
                    if touched and is_nbar_low and close_j > max(manual_supply_levels):
                        rebound_idx = j
                        break
            if rebound_idx is None:
                return None, None
            anchor_idx = rebound_idx + 1  # ✅ 반등 신호 확인 후 다음 캔들 진입
            if anchor_idx >= n:
                return None, None
            signal_time = df.at[anchor_idx, "time"]
            base_price  = float(df.at[anchor_idx, "open"])  # ✅ 시가 기준
        # --- 성과 측정 (공통) ---
        eval_start = anchor_idx + 1
        end_idx = anchor_idx + lookahead  # ✅ 정확히 N봉까지만 탐색
        if end_idx >= n:
            return None, None

        win_slice = df.iloc[eval_start:end_idx + 1]
        min_ret = (win_slice["close"].min() / base_price - 1) * 100 if not win_slice.empty else 0.0
        max_ret = (win_slice["close"].max() / base_price - 1) * 100 if not win_slice.empty else 0.0

        target = base_price * (1.0 + thr / 100.0)
        hit_idx = None
        for j in range(anchor_idx + 1, end_idx + 1):
            c_ = float(df.at[j, "close"])
            h_ = float(df.at[j, "high"])
            price_for_hit = max(c_, h_) if hit_basis.startswith("종가 또는 고가") else (h_ if hit_basis.startswith("고가") else c_)
            if price_for_hit >= target * 0.9999:
                hit_idx = j
                break

        if hit_idx is not None:
            bars_after = hit_idx - anchor_idx
            reach_min = bars_after * minutes_per_bar
            end_time = df.at[hit_idx, "time"]
            end_close = target
            final_ret = thr
            result = "성공"
            lock_end = hit_idx  # ✅ 중복 제거 모드에서 이 인덱스까지는 다음 신호 금지
        else:
            bars_after = lookahead
            end_idx = anchor_idx + bars_after
            if end_idx >= n:
                end_idx = n - 1
                bars_after = end_idx - anchor_idx
            end_time = df.at[end_idx, "time"]
            end_close = float(df.at[end_idx, "close"])
            final_ret = (end_close / base_price - 1) * 100
            result = "실패" if final_ret <= 0 else "중립"
            lock_end = end_idx  # ✅ 평가 구간 끝까지 다음 신호 금지

        reach_min = bars_after * minutes_per_bar

        bb_value = None
        if bb_cond == "상한선":
            bb_value = df.at[anchor_idx, "BB_up"]
        elif bb_cond == "중앙선":
            bb_value = df.at[anchor_idx, "BB_mid"]
        elif bb_cond == "하한선":
            bb_value = df.at[anchor_idx, "BB_low"]

        end_idx_final = hit_idx if (locals().get("hit_idx") is not None) else end_idx

        row = {
            "신호시간": signal_time,
            "종료시간": end_time,
            "기준시가": int(round(base_price)),
            "종료가": end_close,
            "RSI(13)": round(float(df.at[anchor_idx, "RSI13"]), 2) if pd.notna(df.at[anchor_idx, "RSI13"]) else None,
            "BB값": round(float(bb_value), 1) if (bb_value is not None and pd.notna(bb_value)) else None,
            "성공기준(%)": round(thr, 1),
            "결과": result,
            "도달분": reach_min,
            "도달캔들(bars)": int(bars_after),
            "최종수익률(%)": round(final_ret, 2),
            "최저수익률(%)": round(min_ret, 2),
            "최고수익률(%)": round(max_ret, 2),
            "anchor_i": int(anchor_idx),
            "end_i": int(end_idx_final),
        }
        return row, int(lock_end)

    # --- 4) 메인 루프 (중복 포함/제거 분기) ---
    if dedup_mode.startswith("중복 제거"):
        i = 0
        while i < n:
            if i not in base_sig_idx:
                i += 1
                continue
            row, lock_end = process_one(i)
            if row is not None:
                res.append(row)
                # ✅ anchor 기준 평가구간(성공: hit_idx / 실패·중립: anchor+lookahead) 끝까지 건너뜀
                i = int(lock_end) + 1
            else:
                i += 1
    else:
        for i0 in base_sig_idx:
            row, _ = process_one(i0)
            if row is not None:
                res.append(row)

    # ✅ 동일 anchor_i(=신호 시작 캔들) 중복 제거: 표·차트와 1:1 동기화
    if res:
        df_res = pd.DataFrame(res).drop_duplicates(subset=["anchor_i"], keep="first").reset_index(drop=True)
        return df_res
    return pd.DataFrame()
# -----------------------------
# Long-run safe utilities (추가)
# -----------------------------
from datetime import timedelta
import time

def chunked_periods(start_dt, end_dt, days_per_chunk=7):
    """긴 기간을 days_per_chunk 단위로 잘라 (start, end) 튜플을 순서대로 생성."""
    cur = start_dt
    delta = timedelta(days=days_per_chunk)
    while cur < end_dt:
        nxt = min(cur + delta, end_dt)
        yield cur, nxt
        cur = nxt

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_window_cached(symbol, interval_key, start_dt, end_dt, minutes_per_bar):
    """
    기존 fetch_upbit_paged를 캐시 래핑. 동일 구간 재요청시 API 호출/CSV I/O를 절감.
    """
    df = fetch_upbit_paged(symbol, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars=0)
    return df

def _safe_sleep(sec: float):
    try:
        time.sleep(sec)
    except Exception:
        pass

def _load_ckpt(key: str):
    return st.session_state.get(key)

def _save_ckpt(key: str, value):
    st.session_state[key] = value

def run_combination_scan_chunked(
    symbol: str,
    interval_key: str,
    minutes_per_bar: int,
    start_dt,
    end_dt,
    days_per_chunk: int = 7,
    checkpoint_key: str = "combo_scan_ckpt_v1",
    max_minutes: Optional[float] = None,
    on_progress=None,
    simulate_kwargs: Optional[dict] = None,
):
    """
    긴 기간을 청크로 나눠 안전하게 스캔 실행:
      - 각 청크별로 fetch + simulate 실행 → parquet로 부분 저장
      - 진행률 콜백(on_progress) 지원
      - max_minutes 초과 시 중간 저장 후 그레이스풀 스톱(재개 가능)
    반환: (merged_df, ckpt_dict)
    """
    simulate_kwargs = simulate_kwargs or {}
    t0 = time.time()
    chunks = list(chunked_periods(start_dt, end_dt, days_per_chunk))
    total = len(chunks)

    ckpt = _load_ckpt(checkpoint_key) or {"idx": 0, "parts": []}
    part_dir = os.path.join(os.path.dirname(__file__), "data_cache", "scan_parts")
    os.makedirs(part_dir, exist_ok=True)

    for i, (s, e) in enumerate(chunks):
        if i < ckpt["idx"]:
            if on_progress: on_progress((i+1)/total)
            continue

        # 1) 데이터 수집(캐시)
        df_chunk = fetch_window_cached(symbol, interval_key, s, e, minutes_per_bar)
        if df_chunk is None or df_chunk.empty:
            # 비어있어도 체크포인트는 갱신
            ckpt["idx"] = i + 1
            _save_ckpt(checkpoint_key, ckpt)
            if on_progress: on_progress((i+1)/total)
            continue

        # 2) 지표 부착 (기존 add_indicators 활용)
        df_chunk = add_indicators(df_chunk, bb_window, bb_dev, cci_window)

        # 3) 조건 스캔 (기존 simulate 그대로 호출)
        res_chunk = simulate(
            df_chunk,
            simulate_kwargs.get("rsi_mode", "없음"),
            simulate_kwargs.get("rsi_low", 30),
            simulate_kwargs.get("rsi_high", 70),
            simulate_kwargs.get("lookahead", 10),
            simulate_kwargs.get("threshold_pct", 1.0),
            simulate_kwargs.get("bb_cond", "없음"),
            simulate_kwargs.get("dup_mode", "중복 제거 (연속 동일 결과 1개)"),
            minutes_per_bar,
            symbol,
            bb_window,
            bb_dev,
            sec_cond=simulate_kwargs.get("sec_cond", "없음"),
            hit_basis="종가 기준",
            miss_policy="(고정) 성공·실패·중립",
            bottom_mode=simulate_kwargs.get("bottom_mode", False),
            supply_levels=None,
            manual_supply_levels=simulate_kwargs.get("manual_supply_levels", None),
        )

        # 4) 부분 저장
        part_path = os.path.join(
            part_dir,
            f"{symbol}_{interval_key.replace('/','-')}_{s:%Y%m%d%H%M}_{e:%Y%m%d%H%M}.parquet"
        )
        (res_chunk if res_chunk is not None else pd.DataFrame()).to_parquet(part_path, index=False)
        ckpt["parts"].append(part_path)

        # 5) 체크포인트 갱신
        ckpt["idx"] = i + 1
        _save_ckpt(checkpoint_key, ckpt)

        # 6) 진행률
        if on_progress: on_progress((i+1)/total)

        # 7) 레이트리밋/세션 유지
        _safe_sleep(0.2)

        # 8) 시간 제한
        if max_minutes is not None and (time.time() - t0) / 60.0 > max_minutes:
            break

    # ---- 부분결과 병합 (중복 anchor_i 제거 규칙 준수) ----
    parts = ckpt.get("parts", [])
    if not parts:
        return pd.DataFrame(), ckpt

    dfs = []
    for p in parts:
        try:
            dfp = pd.read_parquet(p)
            if dfp is not None and not dfp.empty:
                dfs.append(dfp)
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame(), ckpt

    merged = pd.concat(dfs, ignore_index=True)
    if "anchor_i" in merged.columns:
        merged = merged.drop_duplicates(subset=["anchor_i"], keep="first").reset_index(drop=True)

    return merged, ckpt

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
        # 오늘 날짜 → 현재 시각까지만 데이터 요청 (tz-naive 변환)
        end_dt = datetime.now(KST).astimezone(KST).replace(tzinfo=None)
    else:
        end_dt = datetime.combine(end_date, datetime.max.time())
    warmup_bars = max(13, bb_window, int(cci_window)) * 5

    df_raw = fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars)
    if df_raw.empty:
        st.error("데이터가 없습니다.")
        st.stop()

    df_ind = add_indicators(df_raw, bb_window, bb_dev, cci_window)
    df = df_ind[(df_ind["time"] >= start_dt) & (df_ind["time"] <= end_dt)].reset_index(drop=True)

    # 보기 요약 텍스트
    # ✅ 항상 선택한 봉 종류(minutes_per_bar)를 정확히 반영하여 시간 환산
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

    # -----------------------------
    # -----------------------------
    # 매수가 입력 + 최적화뷰 버튼 (입력 UI는 차트 상단으로 이동)
    # -----------------------------
    if "opt_view" not in st.session_state:
        st.session_state.opt_view = False
    if "buy_price" not in st.session_state:
        st.session_state.buy_price = 0
    if "buy_price_text" not in st.session_state:
        st.session_state.buy_price_text = "0"

    # 이 블록에서는 입력창을 렌더하지 않고 값만 참조합니다.
    buy_price = st.session_state.get("buy_price", 0)

    # ===== 시뮬레이션 (중복 포함/제거) — 먼저 계산하여 res/plot_res 사용 보장 =====
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

    # -----------------------------
    # 신호 선택 → 해당 구간 ±2000봉 차트 표시 (df_view/plot_res 안전 보장)
    # -----------------------------
    df_view = df.iloc[-2000:].reset_index(drop=True)
    plot_res = pd.DataFrame()
    if res is not None and not res.empty:
        plot_res = (
            res.sort_values("신호시간")
               .drop_duplicates(subset=["anchor_i"], keep="first")
               .reset_index(drop=True)
        )
        sel_anchor = st.selectbox(
            "🔎 특정 신호 구간 보기 (anchor 인덱스)",
            options=plot_res["anchor_i"].tolist(),
            index=len(plot_res) - 1
        )
        if sel_anchor is not None:
            start_idx = max(int(sel_anchor) - 1000, 0)
            end_idx   = min(int(sel_anchor) + 1000, len(df) - 1)
            df_view   = df.iloc[start_idx:end_idx+1].reset_index(drop=True)

    # -----------------------------
    # 차트 (선택 구간만 표시)
    # -----------------------------
    df_plot = df_view.copy()
    if buy_price > 0:
        df_plot["수익률(%)"] = (df_plot["close"] / buy_price - 1) * 100
    else:
        df_plot["수익률(%)"] = np.nan

    fig = make_subplots(rows=1, cols=1)

    # ===== Candlestick (hovertext + hoverinfo="text") =====
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
        x=df_plot["time"],
        open=df_plot["open"],
        high=df_plot["high"],
        low=df_plot["low"],
        close=df_plot["close"],
        name="가격",
        increasing=dict(line=dict(color="red", width=1.1)),
        decreasing=dict(line=dict(color="blue", width=1.1)),
        hovertext=hovertext,
        hoverinfo="text"
    ))

    # ===== BB 라인 + hover =====
    def _pnl_arr(y_series):
        if buy_price <= 0:
            return None
        return np.expand_dims((y_series.astype(float) / buy_price - 1) * 100, axis=-1)

    bb_up_cd  = _pnl_arr(df["BB_up"])
    bb_low_cd = _pnl_arr(df["BB_low"])
    bb_mid_cd = _pnl_arr(df["BB_mid"])

    def _ht_line(name):
        if buy_price <= 0:
            return name + ": %{y:.2f}<extra></extra>"
        return name + ": %{y:.2f}<br>매수가 대비 수익률: %{customdata[0]:.2f}<extra></extra>"

    fig.add_trace(go.Scatter(
        x=df["time"], y=df["BB_up"], mode="lines",
        line=dict(color="#FFB703", width=1.4), name="BB 상단",
        customdata=bb_up_cd, hovertemplate=_ht_line("BB 상단")
    ))
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["BB_low"], mode="lines",
        line=dict(color="#219EBC", width=1.4), name="BB 하단",
        customdata=bb_low_cd, hovertemplate=_ht_line("BB 하단")
    ))
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["BB_mid"], mode="lines",
        line=dict(color="#8D99AE", width=1.1, dash="dot"), name="BB 중앙",
        customdata=bb_mid_cd, hovertemplate=_ht_line("BB 중앙")
    ))

    # ===== 매물대 가격 라인 표시 =====
    if manual_supply_levels:
        for L in manual_supply_levels:
            fig.add_hline(
                y=float(L),
                line=dict(color="#FFD700", width=2.0, dash="dot")
            )

    # ===== anchor(신호 시작 캔들) 마커/점선 (신호가 있을 때만) =====
    if not plot_res.empty:
        for _label, _color in [("성공", "red"), ("실패", "blue"), ("중립", "#FF9800")]:
            sub = plot_res[plot_res["결과"] == _label]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(sub["신호시간"]),
                y=sub["기준시가"], mode="markers",
                name=f"신호({_label})",
                marker=dict(size=9, color=_color, symbol="circle", line=dict(width=1, color="black"))
            ))

        legend_emitted = {"성공": False, "실패": False, "중립": False}

        # 점선 + 종료 마커 (표와 1:1 동기화: anchor_i + end_i)
        for _, row in plot_res.iterrows():
            a_i = int(row["anchor_i"])
            e_i = int(row["end_i"])
            a_i = max(0, min(a_i, len(df) - 1))
            e_i = max(0, min(e_i, len(df) - 1))

            x_seg = [df.at[a_i, "time"], df.at[e_i, "time"]]
            y_seg = [float(df.at[a_i, "close"]), float(df.at[e_i, "close"])]

            # 점선(신호~종료 구간)
            fig.add_trace(go.Scatter(
                x=x_seg, y=y_seg, mode="lines",
                line=dict(color="rgba(0,0,0,0.5)", width=1.2, dash="dot"),
                showlegend=False, hoverinfo="skip"
            ))

            # 종료 마커 (결과별 범례 1회만 표시)
            if row["결과"] == "성공":
                fig.add_trace(go.Scatter(
                    x=[df.at[e_i, "time"]],
                    y=[float(df.at[e_i, "close"])],
                    mode="markers",
                    name="도달⭐",
                    marker=dict(size=12, color="orange", symbol="star", line=dict(width=1, color="black")),
                    showlegend=not legend_emitted["성공"]
                ))
                legend_emitted["성공"] = True

            elif row["결과"] == "실패":
                fig.add_trace(go.Scatter(
                    x=[df.at[e_i, "time"]],
                    y=[float(df.at[e_i, "close"])],
                    mode="markers",
                    name="실패❌",
                    marker=dict(size=12, color="blue", symbol="x", line=dict(width=1, color="black")),
                    showlegend=not legend_emitted["실패"]
                ))
                legend_emitted["실패"] = True

            elif row["결과"] == "중립":
                fig.add_trace(go.Scatter(
                    x=[df.at[e_i, "time"]],
                    y=[float(df.at[e_i, "close"])],
                    mode="markers",
                    name="중립❌",
                    marker=dict(size=12, color="orange", symbol="x", line=dict(width=1, color="black")),
                    showlegend=not legend_emitted["중립"]
                ))
                legend_emitted["중립"] = True
    # ===== 매수가 수평선 =====
    if buy_price and buy_price > 0:
        fig.add_shape(
            type="line",
            xref="paper", x0=0, x1=1,
            yref="y", y0=buy_price, y1=buy_price,
            line=dict(color="green", width=1.5, dash="dash"),
            name="매수가"
        )
    # ===== RSI 라인 및 기준선(y2) =====
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["RSI13"], mode="lines",
        line=dict(color="rgba(42,157,143,0.30)", width=6),
        yaxis="y2", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["RSI13"], mode="lines",
        line=dict(color="#2A9D8F", width=2.4, dash="dot"),
        name="RSI(13)", yaxis="y2"
    ))
    for y_val, dash, col, width in [
        (rsi_high, "dash", "#E63946", 1.1),
        (rsi_low, "dash", "#457B9D", 1.1),
    ]:
        fig.add_shape(
            type="line",
            xref="paper", x0=0, x1=1,
            yref="y2", y0=y_val, y1=y_val,
            line=dict(color=col, width=width, dash=dash)
        )

    # ===== 빈 영역에서도 PnL 단독 표시(매수가≥1) =====
    if buy_price > 0:
        fig.add_trace(go.Scatter(
            x=df_plot["time"], y=df_plot["close"],
            mode="lines",
            line=dict(color="rgba(0,0,0,0)", width=1e-3),
            showlegend=False,
            hovertemplate="매수가 대비 수익률: %{customdata[0]:.2f}%<extra></extra>",
            customdata=np.expand_dims(df_plot["수익률(%)"].fillna(0).values, axis=-1),
            name="PnL Hover"
        ))

    # ===== 최적화뷰: x축 범위 적용 =====
    if st.session_state.get("opt_view") and len(df) > 0:
        window_n = max(int(len(df) * 0.15), 200)
        start_idx = max(len(df) - window_n, 0)
        try:
            x_start = df.iloc[start_idx]["time"]
            x_end   = df.iloc[-1]["time"]
            fig.update_xaxes(range=[x_start, x_end])
        except Exception:
            pass

    # ===== 레이아웃 =====
    fig.update_layout(
        title=f"{market_label.split(' — ')[0]} · {tf_label} · RSI(13) + BB 시뮬레이션",
        dragmode="pan",
        xaxis_rangeslider_visible=False,
        height=600,
        legend_orientation="h",
        legend_y=1.05,
        margin=dict(l=30, r=30, t=60, b=40),
        yaxis=dict(title="가격"),
        yaxis2=dict(overlaying="y", side="right", showgrid=False, title="RSI(13)", range=[0, 100]),
        uirevision="chart-static",
        hovermode="closest"
    )

    # ===== 차트 상단: (왼) 매수가 입력  |  (오) 최적화뷰 버튼 =====
    with chart_box:
        top_l, top_r = st.columns([4, 1])

        def _format_buy_price():
            raw = st.session_state.get("buy_price_text", "0")
            digits = "".join(ch for ch in raw if ch.isdigit())
            if digits == "":
                digits = "0"
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

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"scrollZoom": True, "displayModeBar": True, "doubleClick": "reset", "responsive": True},
        )

    # -----------------------------
    # ③ 요약 & 차트
    # -----------------------------
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

    # -----------------------------
    # 🔎 통계/조합 탐색 (고도화)
    # -----------------------------
    # ✅ Expander 세션 유지 (닫힘/점프 방지)
    if "sweep_expanded" not in st.session_state:
        st.session_state["sweep_expanded"] = False
    def _keep_sweep_open():
        st.session_state["sweep_expanded"] = True

    with st.expander("🔎 통계/조합 탐색 (사용자 지정)", expanded=st.session_state["sweep_expanded"]):
        st.caption("※ 선택한 종목/기간/조건에 대해 여러 조합을 자동 시뮬레이션합니다. (기본 설정과는 별도 동작)")

        main_idx_for_sweep = next((i for i, (_, code) in enumerate(MARKET_LIST) if code == market_code), default_idx)
        sweep_market_label, sweep_market = st.selectbox(
            "종목 선택 (통계 전용)", MARKET_LIST, index=main_idx_for_sweep,
            format_func=lambda x: x[0], key="sweep_market_sel", on_change=_keep_sweep_open
        )
        sweep_start = st.date_input("시작일 (통계 전용)", value=start_date,
                                    key="sweep_start", on_change=_keep_sweep_open)
        sweep_end   = st.date_input("종료일 (통계 전용)", value=end_date,
                                    key="sweep_end", on_change=_keep_sweep_open)

        # ⚡ 빠른 테스트 모드 (최근 30일)
        fast_mode = st.checkbox("⚡ 빠른 테스트 모드 (최근 30일만)", value=False,
                                key="sweep_fast_mode", on_change=_keep_sweep_open)

        run_sweep = st.button("▶ 조합 스캔 실행", use_container_width=True, key="btn_run_sweep")
        if run_sweep and not st.session_state.get("use_sweep_wrapper"):
            # -----------------------------
            # (추가) 긴 기간 안전 스캔 래퍼
            # - 기존 run_sweep 루프를 대체하지 않고, 먼저 '안정 실행'을 시도
            # - 실패/빈결 시 기존 루프가 백업처럼 동작하게 순서를 유지
            # -----------------------------
            prog = st.progress(0)
            def _on_progress(p): prog.progress(min(max(p, 0.0), 1.0))

            # 기간 계산 (빠른 모드 ON → 최근 30일)
            if fast_mode:
                sdt = datetime.combine(sweep_end - timedelta(days=30), datetime.min.time())
            else:
                sdt = datetime.combine(sweep_start, datetime.min.time())
            edt = datetime.combine(sweep_end, datetime.max.time())

            try:
                simulate_kwargs = dict(
                    rsi_mode=rsi_mode, rsi_low=rsi_low, rsi_high=rsi_high,
                    lookahead=lookahead, threshold_pct=threshold_pct,
                    bb_cond=bb_cond, dup_mode=("중복 제거 (연속 동일 결과 1개)" if dup_mode.startswith("중복 제거") else "중복 포함 (연속 신호 모두)"),
                    sec_cond=sec_cond, bottom_mode=bottom_mode,
                    manual_supply_levels=manual_supply_levels,
                )

                merged_df, ckpt = run_combination_scan_chunked(
                    symbol=sweep_market,
                    interval_key=interval_key,
                    minutes_per_bar=minutes_per_bar,
                    start_dt=sdt,
                    end_dt=edt,
                    days_per_chunk=7,
                    checkpoint_key=f"combo_scan_{sweep_market}_{interval_key}",
                    max_minutes=15,
                    on_progress=_on_progress,
                    simulate_kwargs=simulate_kwargs,
                )

                if merged_df is not None and not merged_df.empty:
                    if "sweep_state" not in st.session_state:
                        st.session_state["sweep_state"] = {}
                    st.session_state["sweep_state"]["rows"] = merged_df.to_dict("records")
                    st.session_state["sweep_state"]["params"] = {
                        "sweep_market": sweep_market, "sdt": sdt, "edt": edt,
                        "bb_window": int(bb_window), "bb_dev": float(bb_dev), "cci_window": int(cci_window),
                        "rsi_low": int(rsi_low), "rsi_high": int(rsi_high),
                        "target_thr": float(threshold_pct)
                    }
                    st.success("✅ 긴 기간 안전 스캔(조각처리/캐시/체크포인트) 결과가 적용되었습니다.")
                    st.session_state["use_sweep_wrapper"] = True
            except Exception as _e:
                st.info("안전 스캔에 실패하여 기존 방식으로 계속합니다.")

            st.session_state["sweep_expanded"] = True

        # ✅ 스캔에서도 라디오의 중복 모드를 그대로 사용
        dedup_label = "중복 제거 (연속 동일 결과 1개)" if dup_mode.startswith("중복 제거") else "중복 포함 (연속 신호 모두)"

        def _winrate(df_in: pd.DataFrame):
            # ✅ 항상 5개 반환 (win, total, succ, fail, neu)
            if df_in is None or df_in.empty:
                return 0.0, 0, 0, 0, 0
            total = len(df_in)
            succ = (df_in["결과"] == "성공").sum()
            fail = (df_in["결과"] == "실패").sum()
            neu  = (df_in["결과"] == "중립").sum()
            win  = (succ / total * 100.0) if total else 0.0
            return win, total, succ, fail, neu

        # -----------------------------
        # ① 실행 시: 스캔 수행 후 세션에 저장
        # -----------------------------
        if run_sweep and not st.session_state.get("use_sweep_wrapper"):
            # 기간 계산 (빠른 모드 ON → 최근 30일)
            if fast_mode:
                sdt = datetime.combine(sweep_end - timedelta(days=30), datetime.min.time())
            else:
                sdt = datetime.combine(sweep_start, datetime.min.time())
            edt = datetime.combine(sweep_end, datetime.max.time())

            sweep_rows = []
            tf_list = ["15분", "30분", "60분"]
            rsi_list = ["없음", "현재(과매도/과매수 중 하나)", "과매도 기준", "과매수 기준"]
            bb_list  = ["없음", "상한선", "중앙선", "하한선"]
            sec_list = [
                "없음",
                "양봉 2개 (범위 내)",
                "양봉 2개 연속 상승",
                "BB 기반 첫 양봉 50% 진입",
                "매물대 터치 후 반등(위→아래→반등)",
            ]
            lookahead_list = [5, 10, 15, 20, 30]

            for tf_lbl in tf_list:
                interval_key_s, mpb_s = TF_MAP[tf_lbl]
                df_s = fetch_upbit_paged(sweep_market, interval_key_s, sdt, edt, mpb_s, warmup_bars)
                if df_s is None or df_s.empty:
                    continue
                df_s = add_indicators(df_s, bb_window, bb_dev, cci_window)

                for lookahead_s in lookahead_list:
                    for rsi_m in rsi_list:
                        for bb_c in bb_list:
                            for sec_c in sec_list:
                                res_s = simulate(
                                    df_s, rsi_m, rsi_low, rsi_high, lookahead_s, threshold_pct,
                                    bb_c, dedup_label,
                                    mpb_s, sweep_market, bb_window, bb_dev,
                                    sec_cond=sec_c, hit_basis="종가 기준",
                                    miss_policy="(고정) 성공·실패·중립",
                                    bottom_mode=False, supply_levels=None, manual_supply_levels=manual_supply_levels
                                )
                                win, total, succ, fail, neu = _winrate(res_s)
                                total_ret = float(res_s["최종수익률(%)"].sum()) if "최종수익률(%)" in res_s else 0.0
                                avg_ret   = float(res_s["최종수익률(%)"].mean()) if "최종수익률(%)" in res_s and total > 0 else 0.0

                                # ✅ 조합 판정 요약 (최종 규칙: 성공=승률+수익률 모두 충족 / 중립=승률 충족+수익률 미달 / 나머지는 실패)
                                target_thr_val = float(threshold_pct)
                                wr_val = float(winrate_thr)
                                EPS = 1e-3  # 퍼센트 비교시 경계값 오차 보정

                                if (succ > 0) and (win + EPS >= wr_val) and (total_ret + EPS >= target_thr_val):
                                    final_result = "성공"
                                elif (succ > 0) and (win + EPS >= wr_val) and (total_ret + EPS >= 0) and (total_ret + EPS < target_thr_val):
                                    final_result = "중립"
                                else:
                                    final_result = "실패"

                                sweep_rows.append({
                                    "타임프레임": tf_lbl,
                                    "측정N(봉)": lookahead_s,
                                    "RSI": rsi_m,
                                    "RSI_low": int(rsi_low),
                                    "RSI_high": int(rsi_high),
                                    "BB": bb_c,
                                    "BB_기간": int(bb_window),
                                    "BB_승수": round(float(bb_dev), 1),
                                    "2차조건": sec_c,
                                    "목표수익률(%)": float(threshold_pct),
                                    "승률기준(%)": f"{int(winrate_thr)}%",
                                    "신호수": int(total),
                                    "성공": int(succ),
                                    "중립": int(neu),
                                    "실패": int(fail),
                                    "승률(%)": round(win, 1),
                                    "평균수익률(%)": round(avg_ret, 1),
                                    "합계수익률(%)": round(total_ret, 1),
                                    "결과": final_result,
                                    # ✅ 그룹(해당 조합)의 '최초 신호 시간'에서 날짜만 추출
                                    "날짜": (pd.to_datetime(res_s["신호시간"].min()).strftime("%Y-%m-%d")
                                            if ("신호시간" in res_s and not res_s.empty) else ""),
                                })

            # 세션 저장 (초기화 방지)
            if "sweep_state" not in st.session_state:
                st.session_state["sweep_state"] = {}
            st.session_state["sweep_state"]["rows"] = sweep_rows
            st.session_state["sweep_state"]["params"] = {
                "sweep_market": sweep_market, "sdt": sdt, "edt": edt,
                "bb_window": int(bb_window), "bb_dev": float(bb_dev), "cci_window": int(cci_window),
                "rsi_low": int(rsi_low), "rsi_high": int(rsi_high),
                "target_thr": float(threshold_pct)
            }

        # -----------------------------
        # ② 표시 단계: 세션에 저장된 결과를 항상 우선 표시
        # -----------------------------
        sweep_rows_saved = st.session_state.get("sweep_state", {}).get("rows", [])
        if not sweep_rows_saved:
            st.info("조건을 만족하는 조합이 없습니다. (데이터 없음)")
        else:
            df_all = pd.DataFrame(sweep_rows_saved)

            # ✅ 성공/중립만 남기되, 성공은 승률·합계수익률 재검증
            wr_num = float(winrate_thr)
            mask_success = (df_all["결과"] == "성공") & (df_all["승률(%)"] >= wr_num) & (df_all["합계수익률(%)"] > 0)
            mask_neutral = (df_all["결과"] == "중립") & (df_all["합계수익률(%)"] > 0)
            df_keep = df_all[mask_success | mask_neutral].copy()

            if df_keep.empty:
                st.info("조건을 만족하는 조합이 없습니다. (성공·중립 없음)")
            else:
                df_show = df_keep.sort_values(
                    ["결과","승률(%)","신호수","합계수익률(%)"],
                    ascending=[True,False,False,False]
                ).reset_index(drop=True)

                # ✅ 날짜 컬럼: sweep_rows에서 이미 있으면 그대로 사용, 없을 때만 생성
                if "날짜" not in df_show:
                    if "신호시간" in df_show:
                        df_show["날짜"] = pd.to_datetime(df_show["신호시간"]).dt.strftime("%Y-%m-%d")
                    else:
                        df_show["날짜"] = ""

                # ✅ 퍼센트 포맷 (소수점 둘째 자리까지 통일)
                for col in ["목표수익률(%)","승률(%)","평균수익률(%)","합계수익률(%)"]:
                    if col in df_show:
                        df_show[col] = df_show[col].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "")
                if "BB_승수" in df_show:
                    df_show["BB_승수"] = df_show["BB_승수"].map(lambda v: f"{float(v):.1f}" if pd.notna(v) else "")

                # ✅ 색상 스타일
                styled_tbl = df_show.style.apply(
                    lambda col: [
                        ("color:#E53935; font-weight:600;" if r=="성공"
                         else "color:#FF9800; font-weight:600;" if r=="중립" else "")
                        for r in df_show["결과"]
                    ],
                    subset=["평균수익률(%)","합계수익률(%)"]
                )
                st.dataframe(styled_tbl, use_container_width=True)

                # CSV 다운로드
                csv_bytes = df_show.to_csv(index=False).encode("utf-8-sig")
                st.download_button("⬇ 결과 CSV 다운로드", data=csv_bytes, file_name="sweep_results.csv", mime="text/csv", use_container_width=True)

                # ✅ 세부 결과 확인 (Expander 유지)
                selected_idx = st.selectbox(
                    "세부 결과 확인할 조합 선택",
                    df_show.index,
                    key="sweep_select_idx",
                    format_func=lambda i: f"{i} - {df_show.loc[i,'결과']} · {df_show.loc[i,'타임프레임']} · N={df_show.loc[i,'측정N(봉)']}",
                    on_change=_keep_sweep_open
                )
                if selected_idx is not None:
                    sel = df_show.loc[selected_idx]
                    st.info(f"선택된 조건: {sel.to_dict()}")

                    # 데이터 다시 불러 simulate
                    P = st.session_state.get("sweep_state", {}).get("params", {})
                    tf_lbl = sel["타임프레임"]
                    interval_key_s, mpb_s = TF_MAP[tf_lbl]
                    # 세션 저장값 우선 사용(없으면 현재 날짜 위젯으로 대체)
                    sdt_sel = P.get("sdt", datetime.combine(sweep_start, datetime.min.time()))
                    edt_sel = P.get("edt", datetime.combine(sweep_end, datetime.max.time()))
                    df_raw_sel = fetch_upbit_paged(sweep_market, interval_key_s, sdt_sel, edt_sel, mpb_s, warmup_bars)
                    if df_raw_sel is not None and not df_raw_sel.empty:
                        df_sel = add_indicators(df_raw_sel, bb_window, bb_dev, cci_window)
                        res_detail = simulate(
                            df_sel, sel["RSI"], rsi_low, rsi_high,
                            int(sel["측정N(봉)"]), threshold_pct,
                            sel["BB"], dedup_label,
                            mpb_s, sweep_market, bb_window, bb_dev,
                            sec_cond=sel["2차조건"], hit_basis="종가 기준",
                            miss_policy="(고정) 성공·실패·중립",
                            bottom_mode=False, supply_levels=None, manual_supply_levels=manual_supply_levels
                        )
                        if res_detail is not None and not res_detail.empty:
                            st.subheader("세부 신호 결과 (최신 순)")
                            res_detail = res_detail.sort_index(ascending=False).reset_index(drop=True)

                            # 시간/퍼센트 포맷 적용
                            if "신호시간" in res_detail:
                                res_detail["신호시간"] = pd.to_datetime(res_detail["신호시간"]).dt.strftime("%Y-%m-%d %H:%M")
                            if "RSI(13)" in res_detail:
                                res_detail["RSI(13)"] = res_detail["RSI(13)"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")
                            if "성공기준(%)" in res_detail:
                                res_detail["성공기준(%)"] = res_detail["성공기준(%)"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "")
                            for col in ["최종수익률(%)","최저수익률(%)","최고수익률(%)"]:
                                if col in res_detail:
                                    res_detail[col] = res_detail[col].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "")

                            # 도달캔들(bars) → 도달시간(HH:MM) 변환
                            if "도달캔들(bars)" in res_detail.columns:
                                res_detail["도달캔들"] = res_detail["도달캔들(bars)"].astype(int)
                                def _fmt_from_bars(b):
                                    total_min = int(b) * int(mpb_s)
                                    hh, mm = divmod(total_min, 60)
                                    return f"{hh:02d}:{mm:02d}"
                                res_detail["도달시간"] = res_detail["도달캔들"].map(_fmt_from_bars)

                            # 컬럼 순서 메인 ④ 신호 결과와 동일
                            keep_cols = ["신호시간","기준시가","RSI(13)","성공기준(%)","결과",
                                         "최종수익률(%)","최저수익률(%)","최고수익률(%)","도달캔들","도달시간"]
                            keep_cols = [c for c in keep_cols if c in res_detail.columns]
                            res_detail = res_detail[keep_cols]

                            # 스타일 적용 (메인 표와 동일)
                            def style_result(val):
                                if val == "성공": return "background-color: #FFF59D; color:#E53935; font-weight:600;"
                                if val == "실패": return "color:#1E40AF; font-weight:600;"
                                if val == "중립": return "color:#FF9800; font-weight:600;"
                                return ""
                            # ✅ DataFrame에서 head(50) 먼저 적용 → 그 다음 스타일
                            styled_detail = res_detail.head(50).style.applymap(style_result, subset=["결과"])
                            st.dataframe(styled_detail, use_container_width=True)
    # -----------------------------
    # ④ 신호 결과 (테이블)
    # -----------------------------
    st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)
    if res is None or res.empty:
        st.info("조건을 만족하는 신호가 없습니다. (데이터는 정상 처리됨)")
    else:
        tbl = res.sort_values("신호시간", ascending=False).reset_index(drop=True).copy()
        tbl["신호시간"] = pd.to_datetime(tbl["신호시간"]).dt.strftime("%Y-%m-%d %H:%M")
        tbl["기준시가"] = tbl["기준시가"].map(lambda v: f"{int(v):,}")
        if "RSI(13)" in tbl:
            tbl["RSI(13)"] = tbl["RSI(13)"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")
        if "성공기준(%)" in tbl:
            tbl["성공기준(%)"] = tbl["성공기준(%)"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "")
        for col in ["최종수익률(%)", "최저수익률(%)", "최고수익률(%)"]:
            if col in tbl:
                tbl[col] = tbl[col].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "")

        # 도달캔들(bars) → 도달시간(HH:MM) 변환
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

        # 불필요한 컬럼 제거
        drop_cols = [c for c in ["BB값", "도달분", "도달캔들(bars)"] if c in tbl.columns]
        if drop_cols:
            tbl = tbl.drop(columns=drop_cols)

        # 최종 표시 컬럼 순서
        keep_cols = ["신호시간", "기준시가", "RSI(13)", "성공기준(%)", "결과",
                     "최종수익률(%)", "최저수익률(%)", "최고수익률(%)", "도달캔들", "도달시간"]
        keep_cols = [c for c in keep_cols if c in tbl.columns]
        tbl = tbl[keep_cols]

        # style 함수 정의
        def style_result(val):
            if val == "성공": return "background-color: #FFF59D; color: #E53935; font-weight:600;"
            if val == "실패": return "color: #1E40AF; font-weight:600;"
            if val == "중립": return "color: #FF9800; font-weight:600;"
            return ""

        styled_tbl = tbl.style.applymap(style_result, subset=["결과"]) if "결과" in tbl.columns else tbl
        st.dataframe(styled_tbl, width="stretch")
    # -----------------------------
    # CSV GitHub 업로드 버튼 (원할 때만 커밋)
    # -----------------------------
    tf_key = (interval_key.split("/")[1] + "min") if "minutes/" in interval_key else "day"
    data_dir = os.path.join(os.path.dirname(__file__), "data_cache")
    csv_path = os.path.join(data_dir, f"{market_code}_{tf_key}.csv")
    root_csv = os.path.join(os.path.dirname(__file__), f"{market_code}_{tf_key}.csv")

    if st.button("📤 CSV GitHub 업로드"):
        # data_cache 우선, 없으면 루트도 확인
        target_file = csv_path if os.path.exists(csv_path) else root_csv
        if os.path.exists(target_file):
            ok, msg = github_commit_csv(target_file)
            if ok:
                st.success("CSV가 GitHub에 저장/공유되었습니다!")
            else:
                st.warning(f"CSV는 로컬에는 저장됐지만 GitHub 업로드 실패: {msg}")
        else:
            st.warning("CSV 파일이 아직 생성되지 않았습니다. 먼저 데이터를 조회해주세요.")

except Exception as e:
    st.error(f"오류: {e}")
