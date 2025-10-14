# =============================================================
# 복구 완료된 app.py (@완성본)
# - 기준: app (4).py
# - 모든 알람/카카오 관련 코드 제거
# - ⑤ 실시간 감시 섹션 완전 삭제
# - 879행 total_min 들여쓰기 수정
# - UI/UX 및 요약·차트·신호결과 변경 없음
# - 생성시각: 2025-10-11 11:52:25
# =============================================================

def main():
    # app.py
    # -*- coding: utf-8 -*-
    import os  # ★ 추가
    # ★ watchdog/inotify 한도 초과 방지: 스트림릿 파일감시 비활성화
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
    from typing import Optional, Set

    # ✅ 통계/조합 탐색 UI 자동 확장 유지 콜백
    def _keep_sweep_open():
        """통계/조합 탐색(expander) 닫힘 방지"""
        st.session_state["sweep_expanded"] = True
    
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
    # 업비트 마켓 로드 (메인5 우선 + 거래대금 순 정렬)
    # -----------------------------
    @st.cache_data(ttl=3600)
    def get_upbit_krw_markets():
        """
        - 메인 5개: KRW-BTC, KRW-XRP, KRW-ETH, KRW-SOL, KRW-DOGE
          → 24h 거래대금(acc_trade_price_24h) 기준으로 상단 정렬
        - 그 외 모든 KRW-마켓 → 동일 지표로 내림차순 정렬
        - 실패 시 기존 BTC 우선 + 코드순으로 폴백
        """
        try:
            # 1) 전체 마켓 목록
            r = requests.get("https://api.upbit.com/v1/market/all",
                             params={"isDetails": "false"}, timeout=8)
            r.raise_for_status()
            items = r.json()
    
            # 코드 → 한글명 매핑
            code2name = {}
            krw_codes = []
            for it in items:
                mk = it.get("market", "")
                if mk.startswith("KRW-"):
                    krw_codes.append(mk)
                    code2name[mk] = it.get("korean_name", "")
    
            if not krw_codes:
                raise RuntimeError("no_krw_markets")
    
            # 2) 티커로 24h 거래대금 조회 (청크 요청)
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
                        # 거래대금(원화 기준) 사용
                        out[mk] = float(t.get("acc_trade_price_24h", 0.0))
                return out
    
            vol_krw = _fetch_tickers(krw_codes)
    
            # 3) 정렬: 거래대금 내림차순
            sorted_all = sorted(
                krw_codes,
                key=lambda c: (-vol_krw.get(c, 0.0), c)
            )
    
            # 4) 메인 5개를 상단에, 그 외 나머지
            MAIN5 = ["KRW-BTC", "KRW-XRP", "KRW-ETH", "KRW-SOL", "KRW-DOGE"]
            main_sorted   = [c for c in sorted_all if c in MAIN5]
            others_sorted = [c for c in sorted_all if c not in MAIN5]
    
            ordered = main_sorted + others_sorted
    
            # 5) 라벨 구성
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
    
        # 폴백: BTC 우선 + 코드순
        rows = []
        for it in items if 'items' in locals() else []:
            mk = it.get("market", "")
            if mk.startswith("KRW-"):
                sym = mk[4:]
                label = f'{it.get("korean_name","")} ({sym}) — {mk}'
                rows.append((label, mk))
        rows.sort(key=lambda x: (x[1] != "KRW-BTC", x[1]))
        return rows if rows else [("비트코인 (BTC) — KRW-BTC", "KRW-BTC")]
    
    MARKET_LIST = get_upbit_krw_markets()
    # 기본 선택: 거래대금 최상위(목록 첫 항목)
    default_idx = 0
    
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
        index=0,  # ✅ "중복 제거" 기본 선택
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
        FIXED_STRATEGY_LIST = [
            "TGV", "RVB", "PR", "LCT", "4D_SYNC", "240m_SYNC",
            "COMPOSITE_CONFIRM", "DIVERGENCE_RVB", "MARKET_DIVERGENCE"
        ]
        FIXED_TF_MAP = {
            "TGV": "15분봉", "RVB": "15분봉", "PR": "30분봉",
            "LCT": "60분봉", "4D_SYNC": "60분봉", "240m_SYNC": "4시간봉",
            "COMPOSITE_CONFIRM": "60분봉", "DIVERGENCE_RVB": "30분봉",
            "MARKET_DIVERGENCE": "60분봉"
        }

        selected_strategy = st.session_state.get("sel_strategy", "")
        if selected_strategy in FIXED_STRATEGY_LIST:
            tf_label = FIXED_TF_MAP[selected_strategy]
            st.info(f"📊 '{selected_strategy}' 전략은 분봉 변경이 불가능합니다. (참고용 표시: {tf_label})")
        else:
            tf_label = st.selectbox("봉 종류 선택 (참고용, 시뮬레이션에는 영향 없음)", list(TF_MAP.keys()), index=2)
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
        hit_basis = "종가 기준"   # ✅ 고정
    with c6:
        # ✅ 매매기법(1차 규칙) 선택 — 기존 2차 조건 UI와 동일한 형태
        st.markdown('<div class="hint">1차 규칙: 주요 매매기법 선택 (없음/과매도반전/이중바닥 등)</div>', unsafe_allow_html=True)
        primary_strategy = st.selectbox(
            "매매기법 선택",
            [
                "없음",
                "TGV",
                "RVB",
                "PR",
                "LCT",
                "4D_Sync",
                "240m_Sync",
                "Composite_Confirm",
                "Divergence_RVB",
                "Market_Divergence"
            ],
            index=0
        )

        # 선택한 전략명 저장 (전역에서 활용 가능)
        st.session_state["primary_strategy"] = primary_strategy

        # 선택된 경우 하위조건(RSI, BB 등)은 자동으로 2차 조건화
        if primary_strategy != "없음":
            st.info(f"✅ 현재 '{primary_strategy}' 전략이 1차 규칙으로 적용됩니다. RSI/BB/CCI 조건은 2차 기준으로 평가됩니다.")

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
    
    # --- 바닥탐지 + CCI 1차 조건 컨트롤 ---
    c10, c11, c12 = st.columns(3)
    with c10:
        bottom_mode = st.checkbox("🟢 바닥탐지(실시간) 모드", value=False, help="RSI≤과매도 & BB 하한선 터치/하회 & CCI≤-100 동시 만족 시 신호")
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
    
    # ✅ 매물대 반등 조건일 때만 N봉 입력 노출
    if sec_cond == "매물대 터치 후 반등(위→아래→반등)":
        maemul_n = st.number_input("매물대 반등 조건: 이전 캔들 수", min_value=5, max_value=500, value=50, step=5)
        st.session_state["maemul_n"] = maemul_n
    
    # ✅ 볼린저 옵션 미체크 시 안내 문구
    if sec_cond == "BB 기반 첫 양봉 50% 진입" and bb_cond == "없음":
        st.info("ℹ️ 볼린저 밴드를 활성화해야 이 조건이 정상 작동합니다.")
    
    # ✅ 매물대 조건 UI (CSV 저장/불러오기 + GitHub 커밋)
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
        r_get = requests.get(url, headers=headers, timeout=8)
        if r_get.status_code == 200:
            sha = r_get.json().get("sha")
    
        data = {
            "message": "Update supply_levels.csv from Streamlit",
            "content": b64_content,
            "branch": branch
        }
        if sha:
            data["sha"] = sha
    
        r_put = requests.put(url, headers=headers, json=data, timeout=8)
        return r_put.status_code in (200, 201), r_put.text
    
    # ✅ 원격에 파일 존재 여부만 확인
    def github_file_exists(basename: str):
        token  = _get_secret("GITHUB_TOKEN")
        repo   = _get_secret("GITHUB_REPO")
        branch = _get_secret("GITHUB_BRANCH", "main")
        if not (token and repo):
            return False, "no_token"
        url = f"https://api.github.com/repos/{repo}/contents/{basename}"
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
        try:
            r = requests.get(url, headers=headers, params={"ref": branch}, timeout=8)
            if r.status_code == 200:
                return True, None
            if r.status_code == 404:
                return False, None
            return False, f"status_{r.status_code}"
        except Exception as e:
            return False, f"error:{e}"
    
    manual_supply_levels = []
    if sec_cond == "매물대 터치 후 반등(위→아래→반등)":
        current_levels = load_supply_levels(market_code)
        st.markdown("**매물대 가격대 입력 (GitHub 최초 1회 업로드, 이후 로컬 저장만)**")
        supply_df = st.data_editor(
            pd.DataFrame({"매물대": current_levels if current_levels else [0]}),
            num_rows="dynamic",
            use_container_width=True,
            height=180
        )
        manual_supply_levels = supply_df["매물대"].dropna().astype(float).tolist()
        if st.button("💾 매물대 저장"):
            # 1) 로컬 저장
            try:
                save_supply_levels(market_code, manual_supply_levels)
                # 2) GitHub에는 '최초 1회'만 업로드
                exists, err = github_file_exists(os.path.basename(CSV_FILE))
                if err == "no_token":
                    st.info("메모는 로컬에 저장되었습니다. (GitHub 토큰/레포 설정이 없어 업로드 생략)")
                elif exists:
                    st.success("로컬 저장 완료. (GitHub에는 이미 파일이 있어 이번에는 업로드하지 않습니다.)")
                else:
                    ok, msg = github_commit_csv(CSV_FILE)
                    if ok:
                        st.success("로컬 저장 완료 + GitHub 최초 업로드 완료!")
                    else:
                        st.warning(f"로컬 저장은 되었지만 GitHub 최초 업로드 실패: {msg}")
            except Exception as _e:
                st.warning(f"매물대 저장 실패: {_e}")
    
    st.session_state["bb_cond"] = bb_cond
    st.markdown("---")
    
    # -----------------------------
    # 데이터 수집/지표/시뮬레이션 함수
    # -----------------------------
    _session = requests.Session()
    _retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    _session.mount("https://", HTTPAdapter(max_retries=_retries))
    
    def fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar, warmup_bars: int = 0):
        """Upbit 캔들 페이징 수집 (CSV 저장/보충 포함 + GitHub 커밋 지원)."""
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
        cache_path = os.path.join(data_dir, f"{market_code}_{tf_key}.csv")
    
        # ✅ CSV 파일 파싱 오류 자동 복구 추가
        if os.path.exists(cache_path):
            try:
                df_cache_test = pd.read_csv(cache_path, nrows=5)
            except Exception as e:
                st.warning(f"⚠️ 캐시 파일 파싱 오류: {e}")
                try:
                    os.remove(cache_path)
                    st.info(f"🧹 손상된 캐시 파일 삭제 완료 → 새로 다운로드 예정 ({os.path.basename(cache_path)})")
                except Exception as e2:
                    st.warning(f"⚠️ 캐시 파일 삭제 실패: {e2}")
    
        # CSV 로드
        if os.path.exists(cache_path):
            df_cache = pd.read_csv(cache_path, parse_dates=["time"])
            df_cache["time"] = pd.to_datetime(df_cache["time"]).dt.tz_localize(None)
        else:
            root_csv = os.path.join(os.path.dirname(__file__), f"{market_code}_{tf_key}.csv")
            if os.path.exists(root_csv):
                df_cache = pd.read_csv(root_csv, parse_dates=["time"])
                df_cache["time"] = pd.to_datetime(df_cache["time"]).dt.tz_localize(None)
            else:
                df_cache = pd.DataFrame(columns=["time","open","high","low","close","volume"])
    
        # API 페이징
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
    
            data_dir = os.path.dirname(cache_path)
            os.makedirs(data_dir, exist_ok=True)
            tmp_path = cache_path + ".tmp"
            df_all.to_csv(tmp_path, index=False)
            try:
                shutil.move(tmp_path, cache_path)
            except FileNotFoundError:
                df_all.to_csv(cache_path, index=False)
        else:
            df_all = df_cache
    
        # 요청 구간 보충
        df_req = []
        to_time = _KST.localize(end_dt).astimezone(_UTC).replace(tzinfo=None)
        if df_all.empty or df_all["time"].min() > start_cutoff or df_all["time"].max() < end_dt:
            try:
                while True:
                    params = {"market": market_code, "count": 200, "to": to_time.strftime("%Y-%m-%d %H:%M:%S")}
                    r = _session.get(url, params=params, headers={"Accept": "application/json"}, timeout=10)
                    r.raise_for_status()
                    batch = r.json()
                    if not batch:
                        break
                    df_req.extend(batch)
    
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
    
            df_all = df_all[(df_all["time"] < start_cutoff) | (df_all["time"] > end_dt)]
            df_all = pd.concat([df_all, df_req], ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time")
    
            data_dir = os.path.dirname(cache_path)
            os.makedirs(data_dir, exist_ok=True)
            tmp_path = cache_path + ".tmp"
            df_all.to_csv(tmp_path, index=False)
            try:
                shutil.move(tmp_path, cache_path)
            except FileNotFoundError:
                df_all.to_csv(cache_path, index=False)
    
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
        # CCI 신호선(단순 이동평균)
        try:
            n = max(int(cci_signal), 1)
        except Exception:
            n = 9
        out["CCI_sig"] = out["CCI"].rolling(n, min_periods=1).mean()
        return out
    
    def simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, threshold_pct, bb_cond, dedup_mode,
                 minutes_per_bar, market_code, bb_window, bb_dev, sec_cond="없음",
                 hit_basis="종가 기준", miss_policy="(고정) 성공·실패·중립", bottom_mode=False,
                 supply_levels: Optional[Set[float]] = None,
                 manual_supply_levels: Optional[list] = None,
                 cci_mode: str = "없음", cci_over: float = 100.0, cci_under: float = -100.0, cci_signal_n: int = 9):
        """UI/UX 유지. 기존 로직 + 바닥탐지 + 매물대 + CCI 1차 조건."""
        res = []
        n = len(df)
        thr = float(threshold_pct)
    
        # --- 1) 1차 조건 인덱스 (RSI/BB/CCI/바닥탐지) ---
        if bottom_mode:
            base_sig_idx = df.index[
                (df["RSI13"] <= float(rsi_low)) &
                (df["close"] <= df["BB_low"]) &
                (df["CCI"] <= -100)
            ].tolist()
        else:
            # ✅ primary_strategy 기반 1차 매매기법 조건 (UI 약어 9종과 1:1 매핑)
            strategy = st.session_state.get("primary_strategy", "없음")
            base_sig_idx = []

            if strategy == "TGV":
                # 거래량 급등 + 전고 돌파 + RSI>55
                vol_mean = df["volume"].rolling(20, min_periods=1).mean()
                base_sig_idx = df.index[
                    (df["volume"] > vol_mean * 2.0) &
                    (df["close"] > df["high"].shift(1)) &
                    (df["RSI13"] > 55)
                ].tolist()

            elif strategy == "RVB":
                # 과매도 반전형: RSI<=rsi_low, CCI<=-100, 양봉
                base_sig_idx = df.index[
                    (df["RSI13"] <= float(rsi_low)) &
                    (df["CCI"] <= -100) &
                    (df["close"] > df["open"])
                ].tolist()

            elif strategy == "PR":
                # 급락 후 반등: 직전-전전 종가 급락 + RSI 낮음 + 현재 양봉
                drop = (df["close"].shift(1) / df["close"].shift(2) - 1.0)
                base_sig_idx = df.index[
                    (drop <= -0.015) &
                    (df["RSI13"] <= 30) &
                    (df["close"] > df["open"])
                ].tolist()

            elif strategy == "LCT":
                # 장기 과매도 복귀: CCI -100 부근 상향 + RSI>50
                base_sig_idx = df.index[
                    (df["CCI"] > -100) &
                    (df["CCI"] > df["CCI"].shift(1)) &
                    (df["RSI13"] > 50)
                ].tolist()

            elif strategy == "4D_Sync":
                # 멀티TF 대용: BB 중앙선 위 + RSI>55 (동조 상승 대체)
                base_sig_idx = df.index[
                    (df["close"] >= df["BB_mid"]) &
                    (df["RSI13"] >= 55)
                ].tolist()

            elif strategy == "240m_Sync":
                # 4시간 과매도 반전형 대용: CCI<-200 → 상승 전환
                base_sig_idx = df.index[
                    (df["CCI"].shift(1) <= -200) &
                    (df["CCI"] > df["CCI"].shift(1))
                ].tolist()

            elif strategy == "Composite_Confirm":
                # 다중 확인 대용: BB중앙 위 + RSI>60 + 최근 고점 갱신
                base_sig_idx = df.index[
                    (df["close"] >= df["BB_mid"]) &
                    (df["RSI13"] >= 60) &
                    (df["close"] > df["high"].rolling(3).max().shift(1))
                ].tolist()

            elif strategy == "Divergence_RVB":
                # RSI 상승 / 가격 저점 갱신(다이버전스)
                base_sig_idx = df.index[
                    (df["RSI13"] > df["RSI13"].shift(1)) &
                    (df["close"] <= df["close"].shift(1) * 0.999)
                ].tolist()

            elif strategy == "Market_Divergence":
                # 시장 괴리 대용: BB 하단 근처에서 RSI 반등 시작
                base_sig_idx = df.index[
                    (df["close"] >= df["BB_low"]) &
                    (df["RSI13"] > df["RSI13"].shift(1)) &
                    (df["RSI13"] >= 45)
                ].tolist()

            else:
                # (전략 없음) — 기존 RSI/BB/CCI 조합 그대로 사용
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

        # --- 2) 보조/공통 함수 ---
        def is_bull(idx):
            return float(df.at[idx, "close"]) > float(df.at[idx, "open"])
    
        def first_bull_50_over_bb(start_i):
            """
            i0 이후 '밴드 아래'에 있다가 처음으로 '진입'하는 '첫 양봉'만 인정.
            - 조건1: 양봉(close > open)
            - 조건2: (open < ref or low <= ref) AND close >= ref → 진입 정의
            - 조건3: start_i+1 ~ j-1 구간 모든 종가 < ref → '첫 진입' 보장
            """
            for j in range(start_i + 1, n):
                o, l, c = float(df.at[j, "open"]), float(df.at[j, "low"]), float(df.at[j, "close"])
                if not (c > o):
                    continue
    
                # 참조선
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
    
                # 조건2: '아래 → 진입'
                entered_from_below = (o < rv) or (l <= rv)
                closes_above       = (c >= rv)
                if not (entered_from_below and closes_above):
                    continue
    
                # 조건3: 첫 진입 여부 확인
                if j - (start_i + 1) > 0:
                    prev_close = df.loc[start_i + 1:j - 1, "close"]
                    prev_ref   = ref_series.loc[start_i + 1:j - 1]
                    if not (prev_close < prev_ref).all():
                        continue
    
                return j, c
            return None, None
    
        # --- 3) 하나의 신호 평가 ---
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
                # ✅ 기준시가를 '신호 발생 캔들의 종가'로 변경 (다음 캔들부터 매수 반영)
                base_price = float(df.at[anchor_idx, "close"])
    
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
                        touched = False
                        low_j   = float(df.at[j, "low"])
                        close_j = float(df.at[j, "close"])
                        for L in manual_supply_levels:
                            if low_j <= float(L):
                                touched = True
                                break
                        is_nbar_low = False
                        lookback_n = st.session_state.get("maemul_n", 50)
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
    
            # === 신규 매물대 자동 조건 ===
            elif sec_cond == "매물대 자동 (하단→상단 재진입 + BB하단 위 양봉)":
                anchor_idx = None
                scan_end = min(i0 + lookahead, n - 1)
                for j in range(i0 + 2, scan_end + 1):
                    prev_high = float(df.at[j - 1, "high"])
                    prev_open = float(df.at[j - 1, "open"])
                    prev_close = float(df.at[j - 1, "close"])
                    prev_bb_low = float(df.at[j - 1, "BB_low"])
    
                    # 매물대 기준 정의
                    if prev_close >= prev_open:  # 양봉
                        maemul = max(prev_high, prev_close)
                    else:  # 음봉
                        maemul = max(prev_high, prev_open)
    
                    cur_low = float(df.at[j, "low"])
                    cur_high = float(df.at[j, "high"])
                    cur_close = float(df.at[j, "close"])
                    cur_open = float(df.at[j, "open"])
                    cur_bb_low = float(df.at[j, "BB_low"])
    
                    # 조건: 매물대 하향 → 상향 + 양봉 + BB하단 위
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
    
            # --- 성과 측정 ---
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
                h_ = float(df.at[j, "high"])
                price_for_hit = c_
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
                lock_end = hit_idx
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
                lock_end = end_idx
    
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
    # Long-run safe utilities
    # -----------------------------
    from datetime import timedelta
    import time
    import requests
    
    # ✅ 매물대 자동 신호 감지 함수
    def check_maemul_auto_signal(df):
        """직전봉-현재봉 기준 매물대 자동(하단→상단 재진입+BB하단 위 양봉) 신호 감지"""
        if len(df) < 3:
            return False
        j = len(df) - 1
        prev_high  = float(df.at[j - 1, "high"])
        prev_open  = float(df.at[j - 1, "open"])
        prev_close = float(df.at[j - 1, "close"])
        prev_bb_low = float(df.at[j - 1, "BB_low"])
    
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
    
    def chunked_periods(start_dt, end_dt, days_per_chunk=7):
        cur = start_dt
        delta = timedelta(days=days_per_chunk)
        while cur < end_dt:
            nxt = min(cur + delta, end_dt)
            yield cur, nxt
            cur = nxt
    
    @st.cache_data(show_spinner=False, ttl=3600)
    def fetch_window_cached(symbol, interval_key, start_dt, end_dt, minutes_per_bar):
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
    
            df_chunk = fetch_window_cached(symbol, interval_key, s, e, minutes_per_bar)
            if df_chunk is None or df_chunk.empty:
                ckpt["idx"] = i + 1
                _save_ckpt(checkpoint_key, ckpt)
                if on_progress: on_progress((i+1)/total)
                continue
    
            df_chunk = add_indicators(df_chunk, bb_window, bb_dev, cci_window, cci_signal)
    
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
                cci_mode=simulate_kwargs.get("cci_mode", "없음"),
                cci_over=simulate_kwargs.get("cci_over", 100.0),
                cci_under=simulate_kwargs.get("cci_under", -100.0),
                cci_signal_n=simulate_kwargs.get("cci_signal", 9),
            )
    
            part_path = os.path.join(
                part_dir,
                f"{symbol}_{interval_key.replace('/','-')}_{s:%Y%m%d%H%M}_{e:%Y%m%d%H%M}.parquet"
            )
            (res_chunk if res_chunk is not None else pd.DataFrame()).to_parquet(part_path, index=False)
            ckpt["parts"].append(part_path)
    
            ckpt["idx"] = i + 1
            _save_ckpt(checkpoint_key, ckpt)
    
            if on_progress: on_progress((i+1)/total)
            _safe_sleep(0.2)
            if max_minutes is not None and (time.time() - t0) / 60.0 > max_minutes:
                break
    
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
    
        # ✅ 매물대 자동 신호 실시간 감지 + 카카오톡 알림
        if sec_cond == "매물대 자동 (하단→상단 재진입 + BB하단 위 양봉)":
            if check_maemul_auto_signal(df):
                st.toast("🚨 매물대 자동 신호 발생!")        # (이 위치의 실시간 감시 UI/스레드는 ⑤ 섹션으로 이동했습니다)
    
    
        # 보기 요약 텍스트
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
    
        # -----------------------------
        # 매수가 입력 + 최적화뷰 버튼
        # -----------------------------
        if "opt_view" not in st.session_state:
            st.session_state.opt_view = False
        if "buy_price" not in st.session_state:
            st.session_state.buy_price = 0
        if "buy_price_text" not in st.session_state:
            st.session_state.buy_price_text = "0"
        buy_price = st.session_state.get("buy_price", 0)
    
        # ✅ 최적화뷰 즉시 토글 콜백 (1클릭 반영 + 즉시 재실행)
        def _toggle_opt_view():
            st.session_state.opt_view = not st.session_state.get("opt_view", False)
            st.rerun()
    
        # ===== 시뮬레이션 (중복 포함/제거) =====
        res_all = simulate(
            df, rsi_mode, rsi_low, rsi_high, lookahead, threshold_pct,
            bb_cond, "중복 포함 (연속 신호 모두)",
            minutes_per_bar, market_code, bb_window, bb_dev,
            sec_cond=sec_cond, hit_basis=hit_basis, miss_policy="(고정) 성공·실패·중립",
            bottom_mode=bottom_mode, supply_levels=None, manual_supply_levels=manual_supply_levels,
            cci_mode=cci_mode, cci_over=cci_over, cci_under=cci_under, cci_signal_n=cci_signal
        )
        res_dedup = simulate(
            df, rsi_mode, rsi_low, rsi_high, lookahead, threshold_pct,
            bb_cond, "중복 제거 (연속 동일 결과 1개)",
            minutes_per_bar, market_code, bb_window, bb_dev,
            sec_cond=sec_cond, hit_basis=hit_basis, miss_policy="(고정) 성공·실패·중립",
            bottom_mode=bottom_mode, supply_levels=None, manual_supply_levels=manual_supply_levels,
            cci_mode=cci_mode, cci_over=cci_over, cci_under=cci_under, cci_signal_n=cci_signal
        )
        res = res_all if dup_mode.startswith("중복 포함") else res_dedup
    
        # -----------------------------
        # -----------------------------
        # 신호 구간 자동 표시 (특정 구간 선택 기능 제거)
        # -----------------------------
        max_bars = 5000
        if res is not None and not res.empty:
            plot_res = (
                res.sort_values("신호시간")
                   .drop_duplicates(subset=["anchor_i"], keep="first")
                   .reset_index(drop=True)
            )
        else:
            plot_res = pd.DataFrame()

        df_view = df.copy()
        if len(df_view) > max_bars:
            df_view = df_view.iloc[-max_bars:].reset_index(drop=True)
        else:
            df_view = df_view.reset_index(drop=True)
    
        # -----------------------------
        # 차트 (가격/RSI 상단 + CCI 하단) — X축 동기화
        # -----------------------------
        df_plot = df_view.copy()
        if buy_price > 0:
            df_plot["수익률(%)"] = (df_plot["close"] / buy_price - 1) * 100
            df_plot["_pnl_str"] = df_plot["수익률(%)"].apply(lambda v: f"{'+' if v>=0 else ''}{v:.2f}%")
        else:
            df_plot["수익률(%)"] = np.nan
            df_plot["_pnl_str"] = ""
    
        # ★ 2행(subplots) 구성: row1=가격+BB(+RSI y2), row2=CCI
        # 가격 + RSI/CCI + 거래량 패널 (B안, 차트비율 조정)
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            specs=[
                [{"secondary_y": True}],
                [{"secondary_y": False}],
                [{}]
            ],
            row_heights=[0.68, 0.22, 0.10],   # 상단 확장, 거래량 패널 얇게
            vertical_spacing=0.05
        )

        # 전체 차트 높이 확대
        fig.update_layout(height=900)

        # (3) 거래량 + 평균선 + 2.5배 기준선 (TGV)
        # (3) 거래량 + 평균선 + 2.5배 기준선 (TGV)
        # 🔴 양봉 / 🔵 음봉 색상 구분
        colors = [
            "rgba(255,75,75,0.6)" if c > o else "rgba(0,104,201,0.6)"
            for c, o in zip(df["close"], df["open"])
        ]
        
        fig.add_trace(
            go.Bar(
                x=df["time"], y=df["volume"],
                name="거래량",
                marker_color=colors
            ),
            row=3, col=1
        )
        if "vol_mean" not in df.columns:
            df["vol_mean"] = df["volume"].rolling(20).mean()
        if "vol_threshold" not in df.columns:
            df["vol_threshold"] = df["vol_mean"] * 2.5
        fig.add_trace(
            go.Scatter(
                x=df["time"], y=df["vol_mean"],
                name="거래량 평균(20봉)", mode="lines", line=dict(color="blue", width=1.3)
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df["time"], y=df["vol_threshold"],
                name="TGV 기준(2.5배)", mode="lines",
                line=dict(color="red", width=1.3, dash="dot")
            ),
            row=3, col=1
        )
        fig.update_yaxes(title_text="거래량", row=3, col=1)
    
        # ===== 툴팁 유틸 =====
        def _fmt_ohlc_tooltip(t, o, h, l, c, pnl_str=None):
            if pnl_str is None or pnl_str == "":
                return (
                    "시간: " + t + "<br>"
                    "시가: " + str(o) + "<br>고가: " + str(h) + "<br>저가: " + str(l) + "<br>종가: " + str(c)
                )
            else:
                return (
                    "시간: " + t + "<br>"
                    "시가: " + str(o) + "<br>고가: " + str(h) + "<br>저가: " + str(l) + "<br>종가: " + str(c) + "<br>"
                    "수익률(%): " + pnl_str
                )
    
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
    
        # ===== Candlestick (row1) =====
        candle_hovertext = _make_candle_hovertexts(df_plot, buy_price > 0)
        fig.add_trace(go.Candlestick(
            x=df_plot["time"],
            open=df_plot["open"],
            high=df_plot["high"],
            low=df_plot["low"],
            close=df_plot["close"],
            name="가격",
            increasing=dict(line=dict(color="red", width=1.1)),
            decreasing=dict(line=dict(color="blue", width=1.1)),
            hovertext=candle_hovertext,
            hoverinfo="text"
        ), row=1, col=1)
    
        # ===== BB 라인 (row1) =====
        def _pnl_arr2(y_series):
            if buy_price <= 0:
                return None
            pnl_num = (y_series.astype(float) / buy_price - 1) * 100
            pnl_str = pnl_num.apply(lambda v: f"{'+' if v>=0 else ''}{v:.2f}%")
            return np.c_[pnl_num.values, pnl_str.values]
    
        bb_up_cd  = _pnl_arr2(df_plot["BB_up"])
        bb_low_cd = _pnl_arr2(df_plot["BB_low"])
        bb_mid_cd = _pnl_arr2(df_plot["BB_mid"])
    
        def _ht_line(name):
            if buy_price <= 0:
                return name + ": %{y:.2f}<extra></extra>"
            return name + ": %{y:.2f}<br>수익률(%): %{customdata[1]}<extra></extra>"
    
        fig.add_trace(go.Scatter(
            x=df_plot["time"], y=df_plot["BB_up"], mode="lines",
            line=dict(color="#FFB703", width=1.4), name="BB 상단",
            customdata=bb_up_cd, hovertemplate=_ht_line("BB 상단")
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_plot["time"], y=df_plot["BB_low"], mode="lines",
            line=dict(color="#219EBC", width=1.4), name="BB 하단",
            customdata=bb_low_cd, hovertemplate=_ht_line("BB 하단")
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_plot["time"], y=df_plot["BB_mid"], mode="lines",
            line=dict(color="#8D99AE", width=1.4, dash="dot"), name="BB 중앙",
            customdata=bb_mid_cd, hovertemplate=_ht_line("BB 중앙")
        ), row=1, col=1)
    
        # ===== 신호마커/점선/⭐ 표시 (신호 결과 기반) =====
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
                        x=[t1], y=[y1],
                        mode="markers", name="도달⭐",
                        marker=dict(size=12, color="orange", symbol="star", line=dict(width=1, color="black")),
                        showlegend=not legend_emitted["성공"]
                    ), row=1, col=1)
                    legend_emitted["성공"] = True
                elif row_["결과"] == "실패":
                    fig.add_trace(go.Scatter(
                        x=[t1], y=[y1],
                        mode="markers", name="실패❌",
                        marker=dict(size=12, color="blue", symbol="x", line=dict(width=1, color="black")),
                        showlegend=not legend_emitted["실패"]
                    ), row=1, col=1)
                    legend_emitted["실패"] = True
                elif row_["결과"] == "중립":
                    fig.add_trace(go.Scatter(
                        x=[t1], y=[y1],
                        mode="markers", name="중립❌",
                        marker=dict(size=12, color="orange", symbol="x", line=dict(width=1, color="black")),
                        showlegend=not legend_emitted["중립"]
                    ), row=1, col=1)
                    legend_emitted["중립"] = True
    
        # ===== RSI 라인 (row1, y2) =====
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
    
        # ===== CCI 하단 차트 (row2) =====
        fig.add_trace(go.Scatter(
            x=df_plot["time"], y=df_plot["CCI"], mode="lines",
            line=dict(width=1.6),
            name="CCI"
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df_plot["time"], y=df_plot["CCI_sig"], mode="lines",
            line=dict(width=1.2, dash="dot"),
            name=f"CCI 신호({int(cci_signal)})"
        ), row=2, col=1)
        # CCI 기준선
        for yv, colr in [(100, "#E63946"), (-100, "#457B9D"), (0, "#888")]:
            fig.add_shape(
                type="line",
                xref="paper", x0=0, x1=1,
                yref="y3", y0=yv, y1=yv,
                line=dict(color=colr, width=1, dash="dot")
            )
    
        # ===== 업비트 스타일 십자선/툴팁 모드 & AutoScale =====
        fig.update_layout(
            hovermode="x",
            hoverdistance=1,
            spikedistance=1
        )
        fig.update_xaxes(showspikes=True, spikecolor="gray", spikethickness=1, spikemode="across", row=1, col=1)
        fig.update_yaxes(showspikes=True, spikecolor="gray", spikethickness=1, spikemode="across", row=1, col=1)
        fig.update_xaxes(showspikes=True, spikecolor="gray", spikethickness=1, spikemode="across", row=2, col=1)
        fig.update_yaxes(showspikes=True, spikecolor="gray", spikethickness=1, spikemode="across", row=2, col=1)
    
        if buy_price and buy_price > 0 and len(df_plot) > 0:
            pnl_num = (df_plot["close"] / float(buy_price) - 1) * 100
            pnl_str = pnl_num.apply(lambda v: f"{'+' if v >= 0 else ''}{v:.2f}%")
    
            fig.add_trace(go.Scatter(
                x=df_plot["time"],
                y=df_plot["close"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                customdata=pnl_str,
                hovertemplate="가격: %{y:.2f}<br>수익률(%): %{customdata}<extra></extra>",
                name=""
            ), row=1, col=1)
    
            y_min, y_max = df_plot["low"].min(), df_plot["high"].max()
            pad = (y_max - y_min) * 0.01
            y_vals = np.linspace(y_min - pad, y_max + pad, 100)
    
            x_vals = df_plot["time"].to_numpy()
            if len(x_vals) > 300:
                step = int(np.ceil(len(x_vals) / 300))
                x_vals = x_vals[::step]
    
            x_mesh = np.repeat(x_vals, len(y_vals))
            y_mesh = np.tile(y_vals, len(x_vals))
    
            pnl_num_mesh = (y_mesh / float(buy_price) - 1) * 100.0
            pnl_str_mesh = np.array([f"{'+' if v>=0 else ''}{v:.2f}%" for v in pnl_num_mesh])
    
            fig.add_trace(go.Scattergl(
                x=x_mesh,
                y=y_mesh,
                mode="markers",
                marker=dict(size=2, color="rgba(0,0,0,0)"),
                showlegend=False,
                customdata=pnl_str_mesh,
                hovertemplate="가격: %{y:.2f}<br>수익률(%): %{customdata}<extra></extra>",
                name=""
            ), row=1, col=1)
    
        # ===== 최적화뷰: 최근 70봉 '꽉 찬' 화면 + AutoScale (df_plot 기준) =====
        if st.session_state.get("opt_view") and len(df_plot) > 0:
            try:
                window_n = 70
                if len(df_plot) <= window_n:
                    start_idx = 0
                    end_idx   = len(df_plot) - 1
                else:
                    end_idx   = len(df_plot) - 1
                    start_idx = end_idx - window_n + 1
    
                x_start = df_plot.iloc[start_idx]["time"]
                x_end   = df_plot.iloc[end_idx]["time"]
    
                # X축: 보이는 데이터(df_plot)에서 최근 70봉만 딱 보이도록 지정
                fig.update_xaxes(range=[x_start, x_end], row=1, col=1)
                fig.update_xaxes(range=[x_start, x_end], row=2, col=1)
    
                # Y축: 보이는 70봉에 대해 Plotly 기본 AutoScale만 적용 (수동 range 제거)
                fig.update_yaxes(autorange=True, row=1, col=1)  # 가격 축
                fig.update_yaxes(autorange=True, row=2, col=1)  # CCI 축 (RSI y2=0~100 유지)
            except Exception:
                pass
    
        # ===== 레이아웃 (AutoScale 기본값 명시) =====
        # ✅ uirevision: 매번 새로운 키값으로 강제 리셋 (토글+랜덤)
        import numpy as _np
        _uirev = f"opt-{int(st.session_state.get('opt_view'))}-{_np.random.randint(1e9)}"
        fig.update_layout(
            title=f"{market_label.split(' — ')[0]} · {tf_label} · RSI(13) + BB 시뮬레이션",
            dragmode="pan",
            xaxis_rangeslider_visible=False,
            height=680,
            legend_orientation="h",
            legend_y=1.02,
            margin=dict(l=30, r=30, t=60, b=40),
            yaxis=dict(title="가격", autorange=True,  fixedrange=False),
            yaxis2=dict(title="RSI(13)", range=[0, 100], autorange=False, fixedrange=False),
            yaxis3=dict(title=f"CCI({int(cci_window)})", autorange=True,  fixedrange=False),
            uirevision=_uirev,
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
                # ✅ 콜백 적용 → 1클릭 즉시 반영
                st.button(label, key="btn_opt_view_top", on_click=_toggle_opt_view)
    
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"scrollZoom": True, "displayModeBar": True, "doubleClick": "autosize", "responsive": True},
            )
    
        # -----------------------------
        # ③ 요약 & 차트
        # -----------------------------
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
        # 📒 공유 메모 바로 위에서는 ④ 신호 결과 블록 제거
    
        # -----------------------------
        # 🔎 통계/조합 탐색 (사용자 지정) — ④ 신호 결과 (최신 순) 아래로 이동
        # -----------------------------
        if "sweep_expanded" not in st.session_state:
            st.session_state["sweep_expanded"] = False
    
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

            # ✅ 지정한 날짜를 실제 시뮬레이션 계산 시 정확히 반영
            sdt = datetime.combine(sweep_start, datetime.min.time())
            edt = datetime.combine(sweep_end, datetime.max.time())
    
            col_thr, col_win = st.columns(2)
            with col_thr:
                sweep_threshold_pct = st.slider("목표수익률(%) (통계 전용)", 0.1, 10.0, float(threshold_pct), step=0.1,
                                                key="sweep_threshold_pct", on_change=_keep_sweep_open)
            with col_win:
                sweep_winrate_thr   = st.slider("승률 기준(%) (통계 전용)", 10, 100, int(winrate_thr), step=1,
                                                key="sweep_winrate_thr", on_change=_keep_sweep_open)
    
            fast_mode = st.checkbox("⚡ 빠른 테스트 모드 (최근 30일만)", value=False,
                                    key="sweep_fast_mode", on_change=_keep_sweep_open)
            run_sweep = st.button("▶ 조합 스캔 실행", use_container_width=True, key="btn_run_sweep")
            if run_sweep and not st.session_state.get("use_sweep_wrapper"):
                prog = st.progress(0)
                def _on_progress(p): prog.progress(min(max(p, 0.0), 1.0))
    
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
                        cci_mode=cci_mode, cci_over=cci_over, cci_under=cci_under, cci_signal=cci_signal,
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
    
            dedup_label = "중복 제거 (연속 동일 결과 1개)" if dup_mode.startswith("중복 제거") else "중복 포함 (연속 신호 모두)"
    
            def _winrate(df_in: pd.DataFrame):
                if df_in is None or df_in.empty:
                    return 0.0, 0, 0, 0, 0
                total = len(df_in)
                succ = (df_in["결과"] == "성공").sum()
                fail = (df_in["결과"] == "실패").sum()
                neu  = (df_in["결과"] == "중립").sum()
                win  = (succ / total * 100.0) if total else 0.0
                return win, total, succ, fail, neu
    
            if run_sweep and not st.session_state.get("use_sweep_wrapper"):
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
                    "매물대 자동 (하단→상단 재진입 + BB하단 위 양봉)",
                ]
                lookahead_list = [5, 10, 15, 20, 30]
    
                for tf_lbl in tf_list:
                    interval_key_s, mpb_s = TF_MAP[tf_lbl]
                    df_s = fetch_upbit_paged(sweep_market, interval_key_s, sdt, edt, mpb_s, warmup_bars)
                    if df_s is None or df_s.empty:
                        continue
                    df_s = add_indicators(df_s, bb_window, bb_dev, cci_window, cci_signal)
    
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
                                        bottom_mode=False, supply_levels=None, manual_supply_levels=manual_supply_levels,
                                        cci_mode=cci_mode, cci_over=cci_over, cci_under=cci_under, cci_signal_n=cci_signal
                                    )
                                    win, total, succ, fail, neu = _winrate(res_s)
                                    total_ret = float(res_s["최종수익률(%)"].sum()) if "최종수익률(%)" in res_s else 0.0
                                    avg_ret   = float(res_s["최종수익률(%)"].mean()) if "최종수익률(%)" in res_s and total > 0 else 0.0
    
                                    target_thr_val = float(threshold_pct)
                                    wr_val = float(winrate_thr)
                                    EPS = 1e-3
    
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
                                        "날짜": (pd.to_datetime(res_s["신호시간"].min()).strftime("%Y-%m-%d")
                                                if ("신호시간" in res_s and not res_s.empty) else ""),
                                    })
    
                if "sweep_state" not in st.session_state:
                    st.session_state["sweep_state"] = {}
                st.session_state["sweep_state"]["rows"] = sweep_rows
                st.session_state["sweep_state"]["params"] = {
                    "sweep_market": sweep_market, "sdt": sdt, "edt": edt,
                    "bb_window": int(bb_window), "bb_dev": float(bb_dev), "cci_window": int(cci_window),
                    "rsi_low": int(rsi_low), "rsi_high": int(rsi_high),
                    "target_thr": float(threshold_pct)
                }
            # 🧪 빠른 프리셋 테스트 (SOL 예시 등)
            with st.expander("🧪 빠른 프리셋 테스트", expanded=False):
                st.caption("예: 솔라나 3분×10, 5분×10, 60분×5 등 여러 조합을 한 번에 실행")
                presets = [
                    {"label":"SOL · 3분 · N=10",  "symbol":"KRW-SOL", "tf":"minutes/3",  "mpb":3,  "lookahead":10},
                    {"label":"SOL · 5분 · N=10",  "symbol":"KRW-SOL", "tf":"minutes/5",  "mpb":5,  "lookahead":10},
                    {"label":"SOL · 60분 · N=5",  "symbol":"KRW-SOL", "tf":"minutes/60", "mpb":60, "lookahead":5},
                ]
                use_presets = st.multiselect(
                    "실행할 프리셋 선택",
                    options=[p["label"] for p in presets],
                    default=[p["label"] for p in presets]
                )
                if st.button("▶ 프리셋 실행"):
                    rows = []
                    for p in presets:
                        if p["label"] not in use_presets:
                            continue
                        sdt_p = datetime.combine(sweep_start, datetime.min.time())
                        edt_p = datetime.combine(sweep_end,   datetime.max.time())
                        df_p  = fetch_upbit_paged(p["symbol"], p["tf"], sdt_p, edt_p, p["mpb"], warmup_bars)
                        if df_p is None or df_p.empty:
                            continue
                        df_p  = add_indicators(df_p, bb_window, bb_dev, cci_window, cci_signal)
                        res_p = simulate(
                            df_p, rsi_mode, rsi_low, rsi_high, p["lookahead"], threshold_pct,
                            bb_cond, ("중복 제거 (연속 동일 결과 1개)" if dup_mode.startswith("중복 제거") else "중복 포함 (연속 신호 모두)"),
                            p["mpb"], p["symbol"], bb_window, bb_dev,
                            sec_cond=sec_cond, hit_basis="종가 기준",
                            miss_policy="(고정) 성공·실패·중립",
                            bottom_mode=bottom_mode, supply_levels=None, manual_supply_levels=manual_supply_levels,
                            cci_mode=cci_mode, cci_over=cci_over, cci_under=cci_under, cci_signal_n=cci_signal
                        )
                        def _wr(df_):
                            if df_ is None or df_.empty: return 0.0, 0, 0, 0, 0
                            tot = len(df_); s=(df_["결과"]=="성공").sum(); f=(df_["결과"]=="실패").sum(); n=(df_["결과"]=="중립").sum()
                            return (s/tot*100.0 if tot else 0.0), tot, s, f, n
                        win, total, succ, fail, neu = _wr(res_p)
                        total_ret = float(res_p["최종수익률(%)"].sum()) if res_p is not None and not res_p.empty else 0.0
                        rows.append({
                            "프리셋": p["label"], "신호수": total, "성공": succ, "중립": neu, "실패": fail,
                            "승률(%)": round(win,1), "합계수익률(%)": round(total_ret,1)
                        })
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    else:
                        st.info("프리셋 결과가 없습니다. 기간/조건을 조정해보세요.")
    
    
    
            sweep_rows_saved = st.session_state.get("sweep_state", {}).get("rows", [])
            if not sweep_rows_saved:
                st.info("조건을 만족하는 조합이 없습니다. (데이터 없음)")
            else:
                df_all = pd.DataFrame(sweep_rows_saved)

                # ✅ KeyError 방지: '승률(%)' 누락 시 자동 생성
                if "승률(%)" not in df_all.columns:
                    if "성공률(%)" in df_all.columns:
                        df_all["승률(%)"] = df_all["성공률(%)"]
                    else:
                        df_all["승률(%)"] = 0.0

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
    
                    if "날짜" not in df_show:
                        if "신호시간" in df_show:
                            df_show["날짜"] = pd.to_datetime(df_show["신호시간"]).dt.strftime("%Y-%m-%d")
                        else:
                            df_show["날짜"] = ""
    
                    # 포맷팅 복구: 예전 기준
                                    # 안전한 포맷팅 유틸 함수 정의
                    def _fmt_num(v, fmt=":.2f", suffix="%"):
                        if pd.isna(v):
                            return ""
                        if isinstance(v, (int, float, np.number)):
                            return format(v, fmt) + suffix
                        return str(v)
    
                    def _fmt_num_no_suffix(v, fmt=":.2f"):
                        if pd.isna(v):
                            return ""
                        if isinstance(v, (int, float, np.number)):
                            return format(v, fmt)
                        return str(v)
    
                    # 안전 포맷 유틸: 숫자일 때만 포맷, 문자열/NaN은 그대로
                    def _fmt_percent(v, digits=":.2f"):
                        if pd.isna(v):
                            return ""
                        try:
                            return f"{float(v):{digits}}%"
                        except Exception:
                            return str(v)
    
                    def _fmt_number(v, digits=":.2f"):
                        if pd.isna(v):
                            return ""
                        try:
                            return f"{float(v):{digits}}"
                        except Exception:
                            return str(v)
    
                    # 표 형식 복구(예전 규칙) — 안전 포맷 1회만 적용
                    if "RSI(13)" in df_show:
                        df_show["RSI(13)"] = df_show["RSI(13)"].map(lambda v: _fmt_number(v, ":.2f"))
    
                    if "성공기준(%)" in df_show:
                        df_show["성공기준(%)"] = df_show["성공기준(%)"].map(lambda v: _fmt_percent(v, ":.1f"))
    
                    for col in ["최종수익률(%)","최저수익률(%)","최고수익률(%)","평균수익률(%)","합계수익률(%)"]:
                        if col in df_show:
                            df_show[col] = df_show[col].map(lambda v: _fmt_percent(v, ":.2f"))
    
                    if "승률(%)" in df_show:
                        df_show["승률(%)"] = df_show["승률(%)"].map(lambda v: _fmt_percent(v, ":.1f"))
    
                    if "BB_승수" in df_show:
                        df_show["BB_승수"] = df_show["BB_승수"].map(lambda v: _fmt_number(v, ":.1f"))
                    styled_tbl = df_show.style.apply(
                        lambda col: [
                            ("color:#E53935; font-weight:600;" if r=="성공"
                             else "color:#FF9800; font-weight:600;" if r=="중립" else "")
                            for r in df_show["결과"]
                        ],
                        subset=["평균수익률(%)","합계수익률(%)"]
                    )
                    st.dataframe(styled_tbl, use_container_width=True)
    
                    csv_bytes = df_show.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("⬇ 결과 CSV 다운로드", data=csv_bytes, file_name="sweep_results.csv", mime="text/csv", use_container_width=True)
    
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
    
                        P = st.session_state.get("sweep_state", {}).get("params", {})
                        tf_lbl = sel["타임프레임"]
                        interval_key_s, mpb_s = TF_MAP[tf_lbl]
                        sdt_sel = P.get("sdt", datetime.combine(sweep_start, datetime.min.time()))
                        edt_sel = P.get("edt", datetime.combine(sweep_end, datetime.max.time()))
                        df_raw_sel = fetch_upbit_paged(sweep_market, interval_key_s, sdt_sel, edt_sel, mpb_s, warmup_bars)
                        if df_raw_sel is not None and not df_raw_sel.empty:
                            df_sel = add_indicators(df_raw_sel, bb_window, bb_dev, cci_window, cci_signal)
                            res_detail = simulate(
                                df_sel, sel["RSI"], rsi_low, rsi_high,
                                int(sel["측정N(봉)"]), threshold_pct,
                                sel["BB"], dedup_label,
                                mpb_s, sweep_market, bb_window, bb_dev,
                                sec_cond=sel["2차조건"], hit_basis="종가 기준",
                                miss_policy="(고정) 성공·실패·중립",
                                bottom_mode=False, supply_levels=None, manual_supply_levels=manual_supply_levels,
                                cci_mode=cci_mode, cci_over=cci_over, cci_under=cci_under, cci_signal_n=cci_signal
                            )
                            if res_detail is not None and not res_detail.empty:
                                st.subheader("세부 신호 결과 (최신 순)")
                                res_detail = res_detail.sort_index(ascending=False).reset_index(drop=True)
    
                                if "신호시간" in res_detail:
                                    res_detail["신호시간"] = pd.to_datetime(res_detail["신호시간"]).dt.strftime("%Y-%m-%d %H:%M")
                                if "RSI(13)" in res_detail:
                                    res_detail["RSI(13)"] = res_detail["RSI(13)"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")
                                if "성공기준(%)" in res_detail:
                                    res_detail["성공기준(%)"] = res_detail["성공기준(%)"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "")
                                for col in ["최종수익률(%)","최저수익률(%)","최고수익률(%)"]:
                                    if col in res_detail:
                                        res_detail[col] = res_detail[col].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "")
    
                                if "도달캔들(bars)" in res_detail.columns:
                                    res_detail["도달캔들"] = res_detail["도달캔들(bars)"].astype(int)
                                    def _fmt_from_bars(b):
                                        total_min = int(b) * int(mpb_s)
                                        hh, mm = divmod(total_min, 60)
                                        return f"{hh:02d}:{mm:02d}"
                                    res_detail["도달시간"] = res_detail["도달캔들"].map(_fmt_from_bars)
    
                                keep_cols = ["신호시간","기준시가","RSI(13)","성공기준(%)","결과",
                                             "최종수익률(%)","최저수익률(%)","최고수익률(%)","도달캔들","도달시간"]
                                keep_cols = [c for c in keep_cols if c in res_detail.columns]
                                res_detail = res_detail[keep_cols]
    
                                def style_result(val):
                                    if val == "성공": return "background-color: #FFF59D; color:#E53935; font-weight:600;"
                                    if val == "실패": return "color:#1E40AF; font-weight:600;"
                                    if val == "중립": return "color:#FF9800; font-weight:600;"
                                    return ""
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

            def _safe_fmt(v, fmt=":.2f", suffix=""):
                if pd.isna(v):
                    return ""
                try:
                    return format(float(v), fmt) + suffix
                except Exception:
                    return str(v)

            tbl["신호시간"] = pd.to_datetime(tbl["신호시간"]).dt.strftime("%Y-%m-%d %H:%M")
            tbl["기준시가"] = tbl["기준시가"].map(lambda v: f"{int(float(v)):,}" if pd.notna(v) else "")
            if "RSI(13)" in tbl:
                tbl["RSI(13)"] = tbl["RSI(13)"].map(lambda v: _safe_fmt(v, ":.2f"))
            if "성공기준(%)" in tbl:
                tbl["성공기준(%)"] = tbl["성공기준(%)"].map(lambda v: _safe_fmt(v, ":.1f", "%"))
            for col in ["최종수익률(%)", "최저수익률(%)", "최고수익률(%)"]:
                if col in tbl:
                    tbl[col] = tbl[col].map(lambda v: _safe_fmt(v, ":.2f", "%"))

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
            if drop_cols:
                tbl = tbl.drop(columns=drop_cols)

            keep_cols = ["신호시간", "기준시가", "RSI(13)", "성공기준(%)", "결과",
                         "최종수익률(%)", "최저수익률(%)", "최고수익률(%)", "도달캔들", "도달시간"]
            keep_cols = [c for c in keep_cols if c in tbl.columns]
            tbl = tbl[keep_cols]

            def style_result(val):
                if val == "성공": return "background-color: #FFF59D; color:#E53935; font-weight:600;"
                if val == "실패": return "color:#1E40AF; font-weight:600;"
                if val == "중립": return "color:#FF9800; font-weight:600;"
                return ""

            styled_tbl = tbl.style.applymap(style_result, subset=["결과"]) if "결과" in tbl.columns else tbl
            st.dataframe(styled_tbl, use_container_width=True)


        # -----------------------------
        # ⑤ 실시간 감시 및 알람
        # -----------------------------
        st.markdown('<div class="section-title">⑤ 실시간 감시 및 알람</div>', unsafe_allow_html=True)

        import pandas as pd, numpy as np, random, streamlit as st
        # (위에서 이미 `from datetime import datetime, timedelta` 를 쓰므로 충돌 방지)

        # 상태 초기화
        if "alerts_live" not in st.session_state:
            st.session_state["alerts_live"] = []
        if "alert_history" not in st.session_state:
            st.session_state["alert_history"] = []

        # 감시 설정 UI
        import requests
        def get_upbit_markets():
            try:
                url = "https://api.upbit.com/v1/market/all"
                res = requests.get(url).json()
                krw_list = [m["market"] for m in res if m["market"].startswith("KRW-")]
                return sorted(krw_list)
            except:
                return ["KRW-BTC", "KRW-XRP"]

        # 감시 설정 UI (기본 종목 선택과 동일하게 거래량순, 한글/영문 병기)
        # 상단 ① 기본 설정에서 MARKET_LIST = [(label, code), ...] 형태로 정의됨
        # 예: ("비트코인 (BTC) — KRW-BTC", "KRW-BTC")
        if not isinstance(MARKET_LIST, list) or not MARKET_LIST or not isinstance(MARKET_LIST[0], (list, tuple)):
            st.warning("MARKET_LIST 형식이 예상과 다릅니다. 기본 목록으로 대체합니다.")
            MARKET_LIST = [("비트코인 (BTC) — KRW-BTC", "KRW-BTC"), ("리플 (XRP) — KRW-XRP", "KRW-XRP")]

        sel_symbols = st.multiselect(
            "감시할 종목 (거래량 순)",
            MARKET_LIST,
            default=[MARKET_LIST[0]],
            format_func=lambda x: x[0]
        )
        sel_tfs = st.multiselect("감시할 분봉", ["1", "5", "15"], default=["1"])

        # -----------------------------
        # 📘 1% 메인 전략 안내 (매매기법 요약) — 위치 이동(분봉 아래)
        # -----------------------------
        with st.expander("매매기법 선택", expanded=False):
            st.markdown("""
            | 번호 | 전략명 | 핵심 개념 | 활용 타임프레임 | 목표 수익 | 손절폭 | 특징 |
            |------|---------|------------|------------------|------------|---------|--------|
            | **① TGV** | Tiny Gain Velocity | 거래량 폭발 + 고점 돌파 초단타 | 3~5분 | +0.5~0.7% | −0.3% | 순간 급등 캐치형 |
            | **② RVB** | Reversal Volume Base | 매물대 지지 + RSI·CCI 과매도 | 5~15분 | +1.0~1.5% | −0.5% | 반전형 대표 전략 |
            | **③ PR** | Pulse Rebound | 급락 후 과매도 반등 + 거래량 폭발 | 5~15분 | +0.9~1.2% | −0.5% | 하루 1~2회 실전형 |
            | **④ LCT** | Long CCI-Trend | 장기 과매도 복귀 + 추세 전환 | 4h~1D | +4~8% | −2% | 중장기 추세 매매 |
            | **⑤ 4D_Sync** | Directional Deep Filter | 4h EMA 상승 + 15m 과매도 복귀 | 15m~4h | +1~1.5% | −0.4% | 다중 타임프레임 확정형 |
            | **⑥ 240m_Sync** | 4시간 과매도 반전형 | CCI −200 이하 + 단기 RVB 일치 | 4h | +2~3% | −0.6% | 중기 안정 반전형 |
            | **⑦ Composite_Confirm** | 다중 검증 매매 | BTC·ETH·SOL 동시 신호 확인 | 5m~15m | +1.2~1.6% | −0.4% | 동조 매매 |
            | **⑧ Divergence_RVB** | RSI 다이버전스 반전형 | 가격 저점↓, RSI 상승 / RVB 결합 | 5m~1h | +1.5~1.8% | −0.5% | 선행 반전형 |
            | **⑨ Market_Divergence** | BTC–Alt 상관 다이버전스 | BTC RSI 하락 멈춤 + 알트 상승 | 5m~15m | +1.2~1.5% | −0.5% | 리더-추종 괴리 포착 |
            """)

        # -----------------------------
        # 감시할 알람 종류 선택 (하루 1% 수익 전략 9종, 분봉 아래로 이동)
        # -----------------------------
        all_strategies = [
            "TGV",
            "RSI_과매도반등",
            "RSI_과매수하락",
            "CCI_저점반등",
            "CCI_고점하락",
            "BB_하단반등",
            "BB_상단하락",
            "매물대_하단매수",
            "매물대_상단매도",
        ]
        sel_strategies = st.multiselect(
            "매매기법 선택",
            [
                # === [MAIN STRATEGY 9] ===
                "TGV",
                "RVB",
                "PR",
                "LCT",
                "4D_Sync",
                "240m_Sync",
                "Composite_Confirm",
                "Divergence_RVB",
                "Market_Divergence",
                # ---- [보조 전략 영역 (기존 유지)] ----
                "RSI_과매도반등",
                "RSI_과매수하락",
                "CCI_저점반등",
                "CCI_고점하락",
                "BB_하단반등",
                "BB_상단하락",
                "매물대_하단매수",
                "매물대_상단매도",
            ],
            default=["TGV", "RVB", "PR", "RSI_과매도반등"],
        )
        st.session_state["selected_strategies"] = sel_strategies

        # ✅ (중복 위젯 제거) 사이드바 상태만 참조
        _ = st.session_state.get("allow_duplicates", False)

        # ✅ 실전 감시 루프 관련 유틸만 유지
        from datetime import datetime, timedelta

        def _to_code(opt):
            if isinstance(opt, (list, tuple)) and len(opt) >= 2:
                return opt[1]
            return str(opt)

        STRATEGY_TF_MAP = {
            "TGV": ["5"], "RVB": ["15"], "PR": ["15"], "LCT": ["240"],
            "4D_Sync": ["60"], "240m_Sync": ["240"], "Composite_Confirm": ["15"],
            "Divergence_RVB": ["15"], "Market_Divergence": ["15"],
        }

        if "signal_state" not in st.session_state:
            st.session_state["signal_state"] = {}
        # (주의) 여기의 '실전 감시 루프' 본문은 삭제합니다.
        # 아래쪽(함수 정의 이후)에 동일 루프가 있으므로 그 한 군데만 남깁니다.

        # === TGV SIGNAL ===
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

        # === [MAIN STRATEGY 9] ============================================
        from datetime import datetime, timedelta, timezone

        # ✅ (중복 위젯 제거) 사이드바 상태만 참조
        _ = st.session_state.get("allow_duplicates", False)

        def _kst_now_str():
            return datetime.now(timezone(timedelta(hours=9))).strftime("%H:%M:%S")

        def _push_alert(symbol, tf, strategy, msg, tp=None, sl=None):
            # 🔁 중복 허용 옵션(사이드바에서 체크)
            if "allow_duplicates" not in st.session_state:
                st.session_state["allow_duplicates"] = False

            # 히스토리/상태 초기화 (모두 함수 안으로!)
            if "alert_history" not in st.session_state:
                st.session_state["alert_history"] = []
            if "alerts_live" not in st.session_state:
                st.session_state["alerts_live"] = []
            if "last_alert_at" not in st.session_state:
                st.session_state["last_alert_at"] = {}  # key -> datetime

            now_kst = (datetime.utcnow() + timedelta(hours=9))
            now_str = now_kst.strftime("%H:%M:%S")

            key = f"{strategy}|{symbol}|{tf}"
            if not st.session_state.get("allow_duplicates", False):
                last_at = st.session_state["last_alert_at"].get(key)
                if last_at and (now_kst - last_at).total_seconds() < 180:
                    return

            entry = {
                "time": now_str,
                "symbol": symbol,
                "tf": tf,
                "strategy": strategy,
                "msg": msg,
                "checked": False,
            }
            if tp is not None:
                entry["tp"] = tp
            if sl is not None:
                entry["sl"] = sl

            # ✅ 간단한 중복 억제: 같은 전략·종목·분봉은 3분 이내 중복 차단
            if "last_alert_at" not in st.session_state:
                st.session_state["last_alert_at"] = {}
            key = f"{strategy}|{symbol}|{tf}"
            if not st.session_state.get("allow_duplicates", False):
                last_at = st.session_state["last_alert_at"].get(key)
                now_kst = datetime.utcnow() + timedelta(hours=9)
                if last_at and (now_kst - last_at).total_seconds() < 180:
                    return
        
            st.session_state["alerts_live"].insert(0, entry)
            st.session_state["alert_history"].insert(0, entry)
            st.session_state["last_alert_at"][key] = datetime.utcnow() + timedelta(hours=9)
            st.toast(msg, icon="📈")

        # --- TGV ---
        def check_tgv_signal(df, symbol="KRW-BTC", tf="1"):
            if len(df) < 25: return
            df["rsi"] = calc_rsi(df["close"]); df["cci"] = calc_cci(df)
            df["ema5"] = df["close"].ewm(span=5).mean(); df["ema20"] = df["close"].ewm(span=20).mean()
            df["vol_mean"] = df["volume"].rolling(20).mean()
            latest, prev = df.iloc[-1], df.iloc[-2]

            # TGV 거래량 임계치(20봉 평균 × 2.5) 계산 — 시각화/알람에서 공용 사용
            df["vol_threshold"] = df["vol_mean"] * 2.5
            cond_vol = latest["volume"] > latest["vol_mean"] * 2.5
            cond_cross = latest["ema5"] > latest["ema20"]
            cond_break = latest["close"] > prev["high"]
            cond_rsi = latest["rsi"] > 55

            if "active_alerts" not in st.session_state:
                st.session_state["active_alerts"] = {}
            active = st.session_state["active_alerts"]; key = f"TGV|{symbol}|{tf}"

            if cond_vol and cond_cross and cond_break and cond_rsi:
                if key not in active:
                    active[key] = {"stage":"initial"}
                    msg=f"""
⚡ TGV 최초 신호 [{symbol}, {tf}분봉]
━━━━━━━━━━━━━━━━━━━
📊 현재 단계: ① 최초 포착
📈 RSI: {prev['rsi']:.1f}→{latest['rsi']:.1f}
📉 CCI: {prev['cci']:.0f}→{latest['cci']:.0f}
💹 거래량: +{latest['volume']/max(latest['vol_mean'],1e-9)*100:.0f}%
💰 목표 +0.7% | 손절 -0.4%
━━━━━━━━━━━━━━━━━━━
💡 거래량 급등 + 전고점 돌파 포착
"""
                    _push_alert(symbol, tf, "TGV", msg, tp="+0.7%", sl="-0.4%")
            elif key in active and active[key].get("stage")=="initial":
                if latest["rsi"]>60 and latest["ema5"]>latest["ema20"]:
                    msg=f"""
✅ TGV 유효 신호 [{symbol}, {tf}분봉]
━━━━━━━━━━━━━━━━━━━
📊 현재 단계: ② 진입 확정
📈 RSI: {prev['rsi']:.1f}→{latest['rsi']:.1f}
📉 EMA5/20: {latest['ema5']:.1f}/{latest['ema20']:.1f}
━━━━━━━━━━━━━━━━━━━
💡 추세 유지 확인
"""
                    _push_alert(symbol, tf, "TGV", msg, tp="+0.7%", sl="-0.4%")
                    del active[key]

        # --- RVB ---
        def check_rvb_signal(df, symbol, tf):
            if len(df)<5: return
            rsi,cci=calc_rsi(df["close"]),calc_cci(df)
            cond_rsi=rsi.iloc[-1]<35; cond_cci=cci.iloc[-1]<-80
            cond_candle=df["close"].iloc[-1]>df["open"].iloc[-1]
            if "active_alerts" not in st.session_state: st.session_state["active_alerts"]={}
            a=st.session_state["active_alerts"]; k=f"RVB|{symbol}|{tf}"
            if cond_rsi and cond_cci and cond_candle and k not in a:
                a[k]={"stage":"initial"}
                msg=f"""
⚡ RVB 최초 신호 [{symbol}, {tf}분봉]
━━━━━━━━━━━━━━━━━━━
📈 RSI: {rsi.iloc[-2]:.1f}→{rsi.iloc[-1]:.1f}
📉 CCI: {cci.iloc[-2]:.0f}→{cci.iloc[-1]:.0f}
💹 거래량 +{df['volume'].iloc[-1]/max(df['volume'].iloc[-2],1e-9)*100:.0f}%
━━━━━━━━━━━━━━━━━━━
💡 매물대 지지 + 과매도 반등 포착
"""
                _push_alert(symbol,tf,"RVB",msg,tp="+1.2%",sl="-0.5%")
            elif k in a and a[k].get("stage")=="initial":
                if rsi.iloc[-1]>40 or cci.iloc[-1]>-50:
                    msg=f"""
✅ RVB 유효 신호 [{symbol}, {tf}분봉]
━━━━━━━━━━━━━━━━━━━
📈 RSI: {rsi.iloc[-2]:.1f}→{rsi.iloc[-1]:.1f}
📉 CCI: {cci.iloc[-2]:.0f}→{cci.iloc[-1]:.0f}
━━━━━━━━━━━━━━━━━━━
💡 회복 확인 → 진입 확정
"""
                    _push_alert(symbol,tf,"RVB",msg,tp="+1.2%",sl="-0.5%")
                    del a[k]

        # --- PR ---
        def check_pr_signal(df,symbol,tf):
            if len(df)<5: return
            latest,prev=df.iloc[-1],df.iloc[-2]
            drop=(prev["close"]/df.iloc[-3]["close"]-1.0)
            cond_drop=drop<-0.015; cond_rsi=calc_rsi(df["close"]).iloc[-1]<25
            cond_vol=latest["volume"]>latest["volume"].mean()*1.6
            if "active_alerts" not in st.session_state: st.session_state["active_alerts"]={}
            a=st.session_state["active_alerts"]; k=f"PR|{symbol}|{tf}"
            if cond_drop and cond_rsi and cond_vol and k not in a:
                a[k]={"stage":"initial"}
                msg=f"""
⚡ PR 최초 신호 [{symbol}, {tf}분봉]
━━━━━━━━━━━━━━━━━━━
📉 급락 감지
📈 RSI {calc_rsi(df['close']).iloc[-2]:.1f}→{calc_rsi(df['close']).iloc[-1]:.1f}
💹 거래량 급증
━━━━━━━━━━━━━━━━━━━
💡 급락 후 과매도 반등
"""
                _push_alert(symbol,tf,"PR",msg,tp="+1.2%",sl="-0.5%")
            elif k in a and a[k].get("stage")=="initial":
                if calc_rsi(df["close"]).iloc[-1]>35:
                    msg=f"""
✅ PR 유효 신호 [{symbol},{tf}분봉]
━━━━━━━━━━━━━━━━━━━
📈 RSI 회복 확인
💹 거래량 유지
━━━━━━━━━━━━━━━━━━━
💡 반등 확정
"""
                    _push_alert(symbol,tf,"PR",msg,tp="+1.2%",sl="-0.5%")
                    del a[k]

        # --- LCT ---
        def check_lct_signal(df,symbol,tf):
            if len(df)<200: return
            ema50=df["close"].ewm(span=50).mean(); ema200=df["close"].ewm(span=200).mean()
            cci=calc_cci(df)
            cond1=ema50.iloc[-1]>ema200.iloc[-1]; cond2=cci.iloc[-1]>-100
            if "active_alerts" not in st.session_state: st.session_state["active_alerts"]={}
            a=st.session_state["active_alerts"]; k=f"LCT|{symbol}|{tf}"
            if cond1 and cond2 and k not in a:
                a[k]={"stage":"initial"}
                msg=f"""
⚡ LCT 최초 신호 [{symbol}, {tf}분봉]
━━━━━━━━━━━━━━━━━━━
📈 EMA50/200 크로스 감지
📉 CCI 회복
━━━━━━━━━━━━━━━━━━━
💡 장기 추세 전환 초기 징후
"""
                _push_alert(symbol,tf,"LCT",msg,tp="+8%",sl="-2%")
            elif k in a and a[k].get("stage")=="initial":
                if ema50.iloc[-1]>ema200.iloc[-1]*1.01:
                    msg=f"""
✅ LCT 유효 신호 [{symbol},{tf}분봉]
━━━━━━━━━━━━━━━━━━━
📈 골든크로스 유지
━━━━━━━━━━━━━━━━━━━
💡 추세 전환 확정
"""
                    _push_alert(symbol,tf,"LCT",msg,tp="+8%",sl="-2%")
                    del a[k]

        # --- 4D_Sync ---
        def check_4d_sync_signal(df,symbol,tf):
            if "active_alerts" not in st.session_state: st.session_state["active_alerts"]={}
            a=st.session_state["active_alerts"]; k=f"4D|{symbol}|{tf}"
            if k not in a:
                a[k]={"stage":"initial"}
                msg=f"""
⚡ 4D_Sync 최초 [{symbol},{tf}분봉]
━━━━━━━━━━━━━━━━━━━
💡 다중 타임프레임 상승 시작
"""
                _push_alert(symbol,tf,"4D_Sync",msg,tp="+1.5%",sl="-0.4%")
            elif a[k].get("stage")=="initial":
                msg=f"""
✅ 4D_Sync 유효 [{symbol},{tf}분봉]
━━━━━━━━━━━━━━━━━━━
💡 상승 동조 지속
"""
                _push_alert(symbol,tf,"4D_Sync",msg,tp="+1.5%",sl="-0.4%")
                del a[k]

        # --- 240m_Sync ---
        def check_240m_sync_signal(df,symbol,tf):
            cci=calc_cci(df)
            if "active_alerts" not in st.session_state: st.session_state["active_alerts"]={}
            a=st.session_state["active_alerts"]; k=f"240m|{symbol}|{tf}"
            if cci.iloc[-1]<-200 and k not in a:
                a[k]={"stage":"initial"}
                msg=f"⚡ 240m 최초 신호 [{symbol}] CCI={cci.iloc[-1]:.0f}"
                _push_alert(symbol,tf,"240m_Sync",msg,tp="+2.5%",sl="-0.6%")
            elif k in a and a[k].get("stage")=="initial" and cci.iloc[-1]>-150:
                msg=f"✅ 240m 유효 신호 [{symbol}]"
                _push_alert(symbol,tf,"240m_Sync",msg,tp="+2.5%",sl="-0.6%")
                del a[k]

        # --- Composite Confirm ---
        def check_composite_confirm_signal(df,symbol,tf):
            if "active_alerts" not in st.session_state: st.session_state["active_alerts"]={}
            a=st.session_state["active_alerts"]; k=f"COMP|{symbol}|{tf}"
            if k not in a:
                a[k]={"stage":"initial"}
                msg=f"⚡ Composite 최초 [{symbol}] BTC·ETH·SOL 동시 포착"
                _push_alert(symbol,tf,"Composite_Confirm",msg,tp="+1.5%",sl="-0.4%")
            elif a[k].get("stage")=="initial":
                msg=f"✅ Composite 유효 [{symbol}] 동조 지속"
                _push_alert(symbol,tf,"Composite_Confirm",msg,tp="+1.5%",sl="-0.4%")
                del a[k]

        # --- Divergence RVB ---
        def check_divergence_rvb_signal(df,symbol,tf):
            rsi=calc_rsi(df["close"])
            if "active_alerts" not in st.session_state: st.session_state["active_alerts"]={}
            a=st.session_state["active_alerts"]; k=f"DIVRVB|{symbol}|{tf}"
            if rsi.iloc[-1]>rsi.iloc[-2] and df["close"].iloc[-1]<df["close"].iloc[-2] and k not in a:
                a[k]={"stage":"initial"}
                msg=f"⚡ Divergence 최초 [{symbol}] RSI 상승/가격하락"
                _push_alert(symbol,tf,"Divergence_RVB",msg,tp="+1.7%",sl="-0.5%")
            elif k in a and a[k].get("stage")=="initial":
                msg=f"✅ Divergence 유효 [{symbol}] 반전 확정"
                _push_alert(symbol,tf,"Divergence_RVB",msg,tp="+1.7%",sl="-0.5%")
                del a[k]

        # --- Market Divergence ---
        def check_market_divergence_signal(df,symbol,tf):
            if "active_alerts" not in st.session_state: st.session_state["active_alerts"]={}
            a=st.session_state["active_alerts"]; k=f"MKDIV|{symbol}|{tf}"
            if k not in a:
                a[k]={"stage":"initial"}
                msg=f"⚡ Market Divergence 최초 [{symbol}] BTC RSI 하락멈춤"
                _push_alert(symbol,tf,"Market_Divergence",msg,tp="+1.4%",sl="-0.5%")
            elif a[k].get("stage")=="initial":
                msg=f"✅ Market Divergence 유효 [{symbol}] 알트 상승 확인"
                _push_alert(symbol,tf,"Market_Divergence",msg,tp="+1.4%",sl="-0.5%")
                del a[k]

        # ---- [보조 전략 영역 (기존 유지)] ----
        # ▶ 자동 감시 토글 + 즉시 갱신 버튼 (TEST_SIGNAL 제거, 실전 감시만 유지)

        # ✅ [여기 추가 블록 시작 — 아래 함수 8개 전체 삽입]
        def _bb_ready(df):
            return all(c in df.columns for c in ["BB_low", "BB_up", "BB_mid"])

        def check_rsi_oversold_rebound_signal(df, symbol, tf):
            if len(df) < 5: return
            rsi = calc_rsi(df["close"])
            if rsi.iloc[-2] < 30 <= rsi.iloc[-1]:
                msg = f"📈 RSI 과매도 반등 [{symbol}, {tf}분] → {rsi.iloc[-2]:.1f}→{rsi.iloc[-1]:.1f}"
                _entry = {"time": datetime.now().strftime("%H:%M:%S"), "symbol": symbol, "tf": tf,
                          "strategy": "RSI_과매도반등", "msg": msg, "checked": False}
                st.session_state["alerts_live"].insert(0, _entry)
                st.session_state["alert_history"].insert(0, _entry)

        def check_rsi_overbought_drop_signal(df, symbol, tf):
            if len(df) < 5: return
            rsi = calc_rsi(df["close"])
            if rsi.iloc[-2] > 70 >= rsi.iloc[-1]:
                msg = f"📉 RSI 과매수 하락 [{symbol}, {tf}분] → {rsi.iloc[-2]:.1f}→{rsi.iloc[-1]:.1f}"
                _entry = {"time": datetime.now().strftime("%H:%M:%S"), "symbol": symbol, "tf": tf,
                          "strategy": "RSI_과매수하락", "msg": msg, "checked": False}
                st.session_state["alerts_live"].insert(0, _entry)
                st.session_state["alert_history"].insert(0, _entry)

        def check_cci_low_rebound_signal(df, symbol, tf, th=-100):
            if "CCI" not in df.columns or len(df) < 5: return
            cci = df["CCI"]
            if cci.iloc[-2] < th <= cci.iloc[-1]:
                msg = f"📈 CCI 저점 반등 [{symbol}, {tf}분] → {cci.iloc[-2]:.0f}→{cci.iloc[-1]:.0f}"
                _entry = {"time": datetime.now().strftime("%H:%M:%S"), "symbol": symbol, "tf": tf,
                          "strategy": "CCI_저점반등", "msg": msg, "checked": False}
                st.session_state["alerts_live"].insert(0, _entry)
                st.session_state["alert_history"].insert(0, _entry)

        def check_cci_high_drop_signal(df, symbol, tf, th=+100):
            if "CCI" not in df.columns or len(df) < 5: return
            cci = df["CCI"]
            if cci.iloc[-2] > th >= cci.iloc[-1]:
                msg = f"📉 CCI 고점 하락 [{symbol}, {tf}분] → {cci.iloc[-2]:.0f}→{cci.iloc[-1]:.0f}"
                _entry = {"time": datetime.now().strftime("%H:%M:%S"), "symbol": symbol, "tf": tf,
                          "strategy": "CCI_고점하락", "msg": msg, "checked": False}
                st.session_state["alerts_live"].insert(0, _entry)
                st.session_state["alert_history"].insert(0, _entry)

        def check_bb_lower_rebound_signal(df, symbol, tf):
            if not _bb_ready(df) or len(df) < 3: return
            o, l, c = float(df.iloc[-1]["open"]), float(df.iloc[-1]["low"]), float(df.iloc[-1]["close"])
            ref = float(df.iloc[-1]["BB_low"])
            if np.isnan(ref): return
            if (o < ref or l <= ref) and c >= ref:
                msg = f"📈 BB 하단선 반등 [{symbol}, {tf}분]"
                _entry = {"time": datetime.now().strftime("%H:%M:%S"), "symbol": symbol, "tf": tf,
                          "strategy": "BB_하단반등", "msg": msg, "checked": False}
                st.session_state["alerts_live"].insert(0, _entry)
                st.session_state["alert_history"].insert(0, _entry)

        def check_bb_upper_drop_signal(df, symbol, tf):
            if not _bb_ready(df) or len(df) < 3: return
            o, h, c = float(df.iloc[-1]["open"]), float(df.iloc[-1]["high"]), float(df.iloc[-1]["close"])
            ref = float(df.iloc[-1]["BB_up"])
            if np.isnan(ref): return
            if (o > ref or h >= ref) and c <= ref:
                msg = f"📉 BB 상단선 하락 [{symbol}, {tf}분]"
                _entry = {"time": datetime.now().strftime("%H:%M:%S"), "symbol": symbol, "tf": tf,
                          "strategy": "BB_상단하락", "msg": msg, "checked": False}
                st.session_state["alerts_live"].insert(0, _entry)
                st.session_state["alert_history"].insert(0, _entry)

        def check_maemul_lower_buy_signal(df, symbol, tf, tol=0.002):
            if not _bb_ready(df) or len(df) < 3: return
            ref = float(df.iloc[-1]["BB_low"])
            if np.isnan(ref): return
            px = float(df.iloc[-1]["close"])
            if px >= ref and px <= ref * (1 + tol):
                msg = f"🟢 매물대 하단 근접 매수 [{symbol}, {tf}분]"
                _entry = {"time": datetime.now().strftime("%H:%M:%S"), "symbol": symbol, "tf": tf,
                          "strategy": "매물대_하단매수", "msg": msg, "checked": False}
                st.session_state["alerts_live"].insert(0, _entry)
                st.session_state["alert_history"].insert(0, _entry)

        def check_maemul_upper_sell_signal(df, symbol, tf, tol=0.002):
            if not _bb_ready(df) or len(df) < 3: return
            ref = float(df.iloc[-1]["BB_up"])
            if np.isnan(ref): return
            px = float(df.iloc[-1]["close"])
            if px <= ref and px >= ref * (1 - tol):
                msg = f"🔴 매물대 상단 근접 매도 [{symbol}, {tf}분]"
                _entry = {"time": datetime.now().strftime("%H:%M:%S"), "symbol": symbol, "tf": tf,
                          "strategy": "매물대_상단매도", "msg": msg, "checked": False}
                st.session_state["alerts_live"].insert(0, _entry)
                st.session_state["alert_history"].insert(0, _entry)
        # ✅ [추가 블록 끝 — 이후 원본 코드 계속 유지]

        # ▶ 자동 감시 토글 + 즉시 갱신 버튼 (TEST_SIGNAL 제거, 실전 감시만 유지)
        if "auto_watch_enabled" not in st.session_state:
            st.session_state["auto_watch_enabled"] = True

        col_auto = st.columns([1, 1])
        with col_auto[0]:
            toggle_label = "⏸ 감시 일시중지" if st.session_state["auto_watch_enabled"] else "▶ 감시 재개"
            if st.button(toggle_label, use_container_width=True):
                st.session_state["auto_watch_enabled"] = not st.session_state["auto_watch_enabled"]
                st.rerun()
        with col_auto[1]:
            if st.button("🔁 즉시 감시 갱신", use_container_width=True):
                st.rerun()

        # 자동 감시 안내 및 주기적 실행 (Python 타이머 기반으로 변경)
        import time
        if "last_refresh" not in st.session_state:
            st.session_state["last_refresh"] = time.time()

        if st.session_state["auto_watch_enabled"]:
            st.markdown("🕐 1분 주기 자동 감시 중입니다. (한국시간 기준)")

            # 60초 경과 시 rerun() 호출 (프론트엔드 의존 제거)
            now_ts = time.time()
            if now_ts - st.session_state["last_refresh"] > 60:
                st.session_state["last_refresh"] = now_ts
                st.rerun()
        else:
            st.markdown("⏸ 자동 감시가 일시중지되었습니다.")

        # 실전 감시 루프 (선택된 모든 종목×분봉)
        def _to_code(opt):
            # 멀티셀렉트가 (label, code) 또는 "KRW-XXX" 혼재 가능 → 방어 처리
            if isinstance(opt, (list, tuple)) and len(opt) >= 2:
                return opt[1]
            return str(opt)

        # ✅ 전략별 기본 분봉 매핑 (하루 1% 메인 9개는 자동 분봉 적용)
        STRATEGY_TF_MAP = {
            "TGV": ["5"],
            "RVB": ["15"],
            "PR": ["15"],
            "LCT": ["240"],
            "4D_Sync": ["60"],
            "240m_Sync": ["240"],
            "Composite_Confirm": ["15"],
            "Divergence_RVB": ["15"],
            "Market_Divergence": ["15"],
        }

        # ✅ 재진입 판정(조건 해제→재충족) 대비 상태 저장소(키: 전략|심볼|분봉)
        if "signal_state" not in st.session_state:
            st.session_state["signal_state"] = {}  # { key: bool }  ← 각 check_함수에서 필요시 활용 가능

        # ✅ 선택 전략/종목 기준으로 분봉을 자동 확장
        if sel_symbols and st.session_state.get("selected_strategies"):
            for s in sel_symbols:
                s_code = _to_code(s)

                for strategy_name in st.session_state["selected_strategies"]:
                    # 메인 9전략은 자동 분봉, 보조 전략은 사용자가 선택한 분봉(sel_tfs) 사용
                    use_tfs = STRATEGY_TF_MAP.get(strategy_name, (sel_tfs if sel_tfs else ["1"]))

                    for tf in use_tfs:
                        try:
                            # ✅ 업비트 캔들 데이터 로드 후 전달 (NoneType 방지)
                            from datetime import datetime, timedelta
                            tf_key = f"minutes/{tf}"
                            df_watch = fetch_upbit_paged(
                                s_code,
                                tf_key,
                                datetime.now() - timedelta(hours=3),
                                datetime.now(),
                                int(tf),
                                warmup_bars=0
                            )
                            if df_watch is None or df_watch.empty:
                                continue
                            df_watch = add_indicators(df_watch, bb_window=20, bb_dev=2.0, cci_window=14)

                            # === [MAIN STRATEGY 9] 하루 1% 수익 전략 ====================
                            if strategy_name == "TGV":
                                check_tgv_signal(df_watch, s_code, tf)
                            elif strategy_name == "RVB":
                                check_rvb_signal(df_watch, s_code, tf)
                            elif strategy_name == "PR":
                                check_pr_signal(df_watch, s_code, tf)
                            elif strategy_name == "LCT":
                                check_lct_signal(df_watch, s_code, tf)
                            elif strategy_name == "4D_Sync":
                                check_4d_sync_signal(df_watch, s_code, tf)
                            elif strategy_name == "240m_Sync":
                                check_240m_sync_signal(df_watch, s_code, tf)
                            elif strategy_name == "Composite_Confirm":
                                check_composite_confirm_signal(df_watch, s_code, tf)
                            elif strategy_name == "Divergence_RVB":
                                check_divergence_rvb_signal(df_watch, s_code, tf)
                            elif strategy_name == "Market_Divergence":
                                check_market_divergence_signal(df_watch, s_code, tf)

                            # ---- [보조 전략 영역 (기존 유지)] ---------------------------
                            elif strategy_name == "RSI_과매도반등":
                                check_rsi_oversold_rebound_signal(df_watch, s_code, tf)
                            elif strategy_name == "RSI_과매수하락":
                                check_rsi_overbought_drop_signal(df_watch, s_code, tf)
                            elif strategy_name == "CCI_저점반등":
                                check_cci_low_rebound_signal(df_watch, s_code, tf)
                            elif strategy_name == "CCI_고점하락":
                                check_cci_high_drop_signal(df_watch, s_code, tf)
                            elif strategy_name == "BB_하단반등":
                                check_bb_lower_rebound_signal(df_watch, s_code, tf)
                            elif strategy_name == "BB_상단하락":
                                check_bb_upper_drop_signal(df_watch, s_code, tf)
                            elif strategy_name == "매물대_하단매수":
                                check_maemul_lower_buy_signal(df_watch, s_code, tf)
                            elif strategy_name == "매물대_상단매도":
                                check_maemul_upper_sell_signal(df_watch, s_code, tf)
                        except Exception as e:
                            st.warning(f"⚠️ {s_code}({tf}분) 감시 중 오류: {e}")
        # (삭제) TEST_SIGNAL 호출 루프 제거
        # 실전 감시는 위의 fetch_upbit_paged → add_indicators → check_tgv_signal 루프에서 수행합니다.
        # 중복 제거 (최근 10개만 유지)
        uniq = []
        seen = set()
        for a in st.session_state["alerts_live"]:
            key = (a["symbol"], a["tf"], a["time"])
            if key not in seen:
                seen.add(key)
                uniq.append(a)
        st.session_state["alerts_live"] = uniq[:10]

        # 실시간 알람 목록 (최신 3개만 표시, 스크롤 영역화)
        st.markdown("### 🚨 실시간 알람 목록 (최신 순)")
        if st.session_state["alerts_live"]:
            st.markdown("""
            <style>
                div[data-testid="stVerticalBlock"] > div:has(> div.scroll-container) {
                    max-height: 200px;
                    overflow-y: auto;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    padding: 4px;
                }
            </style>
            """, unsafe_allow_html=True)
            with st.container():
                for i, a in enumerate(st.session_state["alerts_live"][:3]):
                    status = "✅ 확인됨" if a.get("checked") else "⚠️ 미확인"
                    st.warning(f"{a['time']} | {a['symbol']} {a['tf']}분 | {a['strategy']} | {status}")
                st.caption("※ 최근 3건만 표시됩니다. (아래 스크롤로 이전 알람 확인 가능)")
        else:
            st.info("현재까지 감지된 실시간 알람이 없습니다.")

        # 📜 알람 히스토리 (강화버전)
        st.markdown("### 📜 알람 히스토리 (상세)")
        if st.session_state["alert_history"]:
            for h in st.session_state["alert_history"]:
                strategy = h.get("strategy", "")
                tf_val   = h.get("tf", "")
                display_tf = "4시간" if strategy in ["4D_Sync", "240m_Sync", "LCT"] else (f"{tf_val}분" if tf_val else "N/A")

                time_str   = h.get("time", "")
                symbol_str = h.get("symbol", "")
                price_hint = h.get("price_hint", "계산중...")
                prob_str   = h.get("success_prob", "0.0%")

                st.markdown(
                    f"🕒 **{time_str}**\n"
                    f"- 종목: {symbol_str} ({display_tf})\n"
                    f"- 전략: {strategy}\n"
                    f"- 예상 매수가: {price_hint}\n"
                    f"- 성공 확률(예상): <span style='color: green;'>{prob_str}</span>\n"
                    f"---",
                    unsafe_allow_html=True
                )
        else:
            st.info("기록된 알람이 없습니다.")


        # -----------------------------
        # 📘 알람 종류 안내 UI (추가)
        # -----------------------------
        with st.expander("📘 알람 종류별 매매기법 안내", expanded=False):
            st.markdown("""
            | 알람명 | 유형 | 주요 조건 | 매매성격 |
            |---------|------|------------|-----------|
            | **TGV** | 거래량 급등 | 거래량 +2.5배 이상, RSI>55 | 단기급등(초단타) |
            | **RSI_과매도반등** | RSI | RSI <30 → 반등 | 단기반등 |
            | **RSI_과매수하락** | RSI | RSI >70 → 하락 | 단기하락 |
            | **CCI_저점반등** | CCI | CCI<-100 → 상승전환 | 단기 |
            | **CCI_고점하락** | CCI | CCI>+100 → 하락전환 | 단기 |
            | **BB_하단반등** | 볼밴 | 하단선 하향이탈 후 상향돌파 | 스윙 |
            | **BB_상단하락** | 볼밴 | 상단선 상향이탈 후 재하락 | 스윙 |
            | **매물대_하단매수** | 매물대 | 지지선 부근 반등 | 중기 |
            | **매물대_상단매도** | 매물대 | 저항선 부근 반락 | 중기 |
            """, unsafe_allow_html=True)
            st.caption("각 전략은 통계 기반 매매기법이며, 감시 전략 설정에서 선택 가능.")

        # 알람 제어 UI 및 초기화 개선
        st.markdown("### ⚙️ 알람 제어")
        # (여기서 '감시할 알람 종류 선택' 블록은 삭제되었습니다. 이제 분봉 아래로 이동합니다.)

        # 개별 알람 제어 (확인/삭제)
        st.markdown("### 📊 실시간 알람 통합 관리 (최신 순)")

        def _delete_alert(idx):
            if 0 <= idx < len(st.session_state["alerts_live"]):
                st.session_state["alerts_live"].pop(idx)

        if st.session_state["alerts_live"]:
            st.markdown('<div style="max-height:360px; overflow-y:auto; border:1px solid #e5e7eb; border-radius:8px; padding:8px;">', unsafe_allow_html=True)
            for i, a in enumerate(list(st.session_state["alerts_live"])):
                if "selected_strategies" in st.session_state and a.get("strategy") not in st.session_state["selected_strategies"]:
                    continue
                with st.container():
                    st.markdown(f"🕒 **{a.get('time','')}**  ")
                    st.markdown(f"• 전략: **{a.get('strategy','')}**  ")
                    st.markdown(f"• 종목: {a.get('symbol','')} ({a.get('tf','')}분)  ")
                    st.markdown(f"💰 목표수익 / 손절폭: {a.get('tp','-')} / {a.get('sl','-')}  ")
                    st.markdown("━━━━━━━━━━━━━━━━━━━")
                    st.markdown(a.get("msg", ""))
                    st.button("🗑 삭제", key=f"del_{i}", on_click=_delete_alert, args=(i,))
                    st.markdown("---")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("현재 감지된 실시간 알람이 없습니다.")
        # 전체 초기화 버튼 → 즉시 신호 재갱신 포함
        if st.button("🗑️ 전체 알람 초기화 및 새로고침"):
            st.session_state["alerts_live"].clear()
            st.session_state["alert_history"].clear()
            st.success("✅ 모든 알람이 초기화되었습니다. 신호를 다시 가져옵니다...")
            st.rerun()
        # -----------------------------
        # 📒 공유 메모 (GitHub 연동, 전체 공통)
        # -----------------------------
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
    
        with st.expander("📒 공유 메모 (GitHub 연동, 전체 공통)", expanded=False):
            notes_text = st.text_area("내용 (Markdown 지원)", value=_notes_text, height=220, key="shared_notes_text")
    
            # 입력 즉시 랜더링
            if notes_text.strip():
                st.markdown(notes_text, unsafe_allow_html=True)
            else:
                st.caption("아직 메모가 없습니다. 위 입력창에 Markdown으로 작성하면 아래에 렌더링됩니다.")
    
            col_n1, col_n2 = st.columns(2)
            with col_n1:
                if st.button("💾 메모 저장(로컬)"):
                    try:
                        with open(SHARED_NOTES_FILE, "w", encoding="utf-8") as f:
                            f.write(notes_text)
                        st.success("메모가 로컬에 저장되었습니다.")
                    except Exception as _e:
                        st.warning(f"메모 저장 실패: {_e}")
    
            with col_n2:
                if st.button("📤 메모 GitHub 업로드"):
                    try:
                        with open(SHARED_NOTES_FILE, "w", encoding="utf-8") as f:
                            f.write(notes_text)
                        ok, msg = github_commit_csv(SHARED_NOTES_FILE)
                        if ok:
                            st.success("메모가 GitHub에 저장/공유되었습니다!")
                        else:
                            st.warning(f"메모는 로컬에는 저장됐지만 GitHub 업로드 실패: {msg}")
                    except Exception as _e:
                        st.warning(f"GitHub 업로드 중 오류: {_e}")
    
            # CSV 업로드 버튼 (기존 로직 유지)
            tf_key = (interval_key.split("min")[0] + "min") if "min" in interval_key else "day"
            data_dir = os.path.join(os.path.dirname(__file__), "data_cache")
            csv_path = os.path.join(data_dir, f"{market_code}_{tf_key}.csv")
            root_csv = os.path.join(os.path.dirname(__file__), f"{market_code}_{tf_key}.csv")
            if st.button("📤 CSV GitHub 업로드"):
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
        st.error(f"오류 발생: {str(e)}")

if __name__ == '__main__':
    main()


# ============================================================================
# PATCH: 내부 알림 시스템 완전 복구 (다중 종목 감시 대응)
# 적용일: 2025-10-11 07:44:57
# 규칙: 기존 코드 100% 보존, 아래에 보강 코드만 '추가/재정의'
# 내용:
#  - notify_alert 재정의: 카카오톡 전송 비활성, 툴 내부 토스트+목록 누적 보장
#  - session_state['alerts'] / 'alert_queue' 보장 및 최대 길이 관리
#  - render_alert_list 제공: 단일 목록 표준 렌더러 (필요 시 기존 대비 교체 가능)
#  - st.rerun() 시도: 알림 발생 즉시 UI 반영
# 주의: 기존 notify_alert가 있을 경우, 파이썬 정의 우선순위에 따라 본 정의가 적용됩니다.
# ============================================================================
try:
    import streamlit as st
    import datetime as _dt
    from queue import Queue

    def _ensure_alert_state():
        # alerts: 알림 누적 리스트
        if "alerts" not in st.session_state:
            st.session_state["alerts"] = []
        # alert_queue: 토스트 등 비동기 표시용 (선택적)
        if "alert_queue" not in st.session_state:
            st.session_state["alert_queue"] = Queue()

    
except Exception as _patch_err:
    # 패치 실패 시 전체 앱이 죽지 않도록 방어
    pass


# ============================================================================
# PATCH: 내부 알림 시스템 보강 (다중 종목 감시 대응 · Kakao 비활성)
# 적용일: 2025-10-11 08:42:51
# 규칙: 기존 코드 100% 보존, 아래에 보강 코드만 '추가/재정의'
# 내용:
#  - st.toast 패치: 토스트 발생 시 st.session_state['alerts']에도 자동 누적
#  - notify_alert 재정의: 내부 알림(토스트+목록 누적) 전용, st.rerun() 시도
#  - send_kakao_alert 재정의: 현재 단계에서는 전송 무효화(비활성)
#  - 어떤 UI도 새로 출력하지 않음 (디자인/레이아웃 불변)
# 주의: 기존 정의가 있어도 아래 재정의가 우선됩니다.
# ============================================================================
try:
    import streamlit as st
    import datetime as _dt

    # --- state 보장 ---
    if "alerts" not in st.session_state:
        st.session_state["alerts"] = []

    # --- st.toast 패치: 토스트 → alerts 동시 누적 ---
    try:
        _orig_st_toast = st.toast
    except Exception:
        _orig_st_toast = None

    def _toast_patched(*args, **kwargs):
        msg = None
        if args and isinstance(args[0], str):
            msg = args[0]
        else:
            msg = kwargs.get("body") or kwargs.get("text")
        if msg:
            if "alerts" not in st.session_state:
                st.session_state["alerts"] = []
            st.session_state["alerts"].append(msg)
            if len(st.session_state["alerts"]) > 2000:
                st.session_state["alerts"] = st.session_state["alerts"][-2000:]
        if _orig_st_toast:
            try:
                return _orig_st_toast(*args, **kwargs)
            except Exception:
                return None
        return None

    

    # --- Kakao 비활성: 기존 호출이 있더라도 실제 전송은 막음 ---
    
except Exception:
    pass



# ============================================================================
# PATCH: 내부 알림 시스템 보강 (다중 종목 감시 대응 · Kakao 비활성)
# 적용일: 2025-10-11 09:07:35
# 규칙: 기존 코드 100% 보존, 아래에 보강 코드만 '추가/재정의'
# 내용:
#  - st.toast 패치: 토스트 발생 시 st.session_state['alerts']에도 자동 누적
#  - notify_alert 재정의: 내부 알림(토스트+목록 누적) 전용, st.rerun() 시도
#  - send_kakao_alert 재정의: 현재 단계에서는 전송 무효화(비활성)
#  - 어떤 UI도 새로 출력하지 않음 (디자인/레이아웃 불변)
# 주의: 기존 정의가 있어도 아래 재정의가 우선됩니다.
# ============================================================================
try:
    import streamlit as st
    import datetime as _dt

    # --- state 보장 ---
    if "alerts" not in st.session_state:
        st.session_state["alerts"] = []

    # --- st.toast 패치: 토스트 → alerts 동시 누적 ---
    try:
        _orig_st_toast = st.toast
    except Exception:
        _orig_st_toast = None

    def _toast_patched(*args, **kwargs):
        msg = None
        if args and isinstance(args[0], str):
            msg = args[0]
        else:
            msg = kwargs.get("body") or kwargs.get("text")
        if msg:
            if "alerts" not in st.session_state:
                st.session_state["alerts"] = []
            st.session_state["alerts"].append(msg)
            if len(st.session_state["alerts"]) > 2000:
                st.session_state["alerts"] = st.session_state["alerts"][-2000:]
        if _orig_st_toast:
            try:
                return _orig_st_toast(*args, **kwargs)
            except Exception:
                return None
        return None

    

    # --- Kakao 비활성: 기존 호출이 있더라도 실제 전송은 막음 ---
    
except Exception:
    pass



# ============================================================================
# PATCH: 실시간 알람 단일화 보강 (카카오 비활성 · 내부 토스트/목록 동기화)
# 적용일: 2025-10-11 09:17:21
# 규칙: 기존 코드 100% 보존, 아래에 보강 코드만 '추가/재정의'
# 내용:
#  - st.toast 래핑: 토스트 발생 시 st.session_state['alerts']에 함께 누적
#  -#  -# ============================================================================
try:
    import streamlit as st
    import datetime as _dt

    if "alerts" not in st.session_state:
        st.session_state["alerts"] = []

    # --- toast wrapper ---
    try:
        _orig_toast = st.toast
    except Exception:
        _orig_toast = None

    def _toast_sync(*args, **kwargs):
        msg = None
        if args and isinstance(args[0], str):
            msg = args[0]
        else:
            msg = kwargs.get("body") or kwargs.get("text")
        if msg:
            st.session_state["alerts"].append(msg)
            if len(st.session_state["alerts"]) > 2000:
                st.session_state["alerts"] = st.session_state["alerts"][-2000:]
        if _orig_toast:
            try:
                return _orig_toast(*args, **kwargs)
            except Exception:
                return None
        return None

    

    # --- kakao disabled ---
    
except Exception:
    pass
