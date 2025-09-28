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

if sec_cond == "BB 기반 첫 양봉 50% 진입" and bb_cond == "없음":
    st.info("ℹ️ 볼린저 밴드를 활성화해야 이 조건이 정상 작동합니다.")

# -----------------------------
# GitHub 저장 연동 (매물대)
# -----------------------------
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
    sha = None
    r_get = requests.get(url, headers=headers)
    if r_get.status_code == 200:
        sha = r_get.json().get("sha")
    data = {"message": "Update supply_levels.csv from Streamlit", "content": b64_content, "branch": branch}
    if sha: data["sha"] = sha
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
    """최적화: CSV 범위 검증 후 부족한 앞/뒤 구간만 API 보충"""
    # 👉 여기 전체 코드 그대로 유지 (앞/뒤 부족한 구간만 API 호출)

def add_indicators(df, bb_window, bb_dev, cci_window):
    # 👉 그대로 유지

def simulate(...):
    # 👉 그대로 유지

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

    # 👉 이후 차트, 요약, 신호결과 그대로 실행

except Exception as e:
    st.error(f"오류: {e}")
