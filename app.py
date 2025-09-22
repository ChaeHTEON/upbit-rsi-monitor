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
  .neutral-cell {color:#059669; font-weight:600;}
  table {border-collapse:collapse; width:100%;}
  th, td {border:1px solid #ddd; padding:6px; text-align:center;}
</style>
""", unsafe_allow_html=True)

# 제목 고정 (요청: "코인 시뮬레이션")
st.title("📊 코인 시뮬레이션")
st.markdown("<div style='margin-bottom:10px; color:gray;'>※ 차트 내 점선은 신호 흐름선, 성공 시 도달 지점에 ⭐ 별표 표시</div>", unsafe_allow_html=True)

# -----------------------------
# 업비트 마켓 로드 (네트워크 폴백 포함)
# -----------------------------
@st.cache_data(ttl=3600)
def get_upbit_krw_markets():
    url = "https://api.upbit.com/v1/market/all"
    try:
        r = requests.get(url, params={"isDetails":"false"}, timeout=8)
        r.raise_for_status()
        items = r.json()
        rows = []
        for it in items:
            mk = it.get("market","")
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
default_idx = next((i for i,(_,code) in enumerate(MARKET_LIST) if code=="KRW-BTC"), 0)

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
    "일봉": ("days", 24*60),
}

# -----------------------------
# 상단: 신호 중복 처리
# -----------------------------
dup_mode = st.radio(
    "신호 중복 처리",
    ["중복 포함 (연속 신호 모두)", "중복 제거 (연속 동일 결과 1개)"],
    horizontal=True,
)

# -----------------------------
# 기본 설정
# -----------------------------
st.markdown('<div class="section-title">① 기본 설정</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    market_label, market_code = st.selectbox("종목 선택", MARKET_LIST, index=default_idx, format_func=lambda x: x[0])
with c2:
    tf_label = st.selectbox("봉 종류 선택", list(TF_MAP.keys()), index=2)
with c3:
    default_start = datetime.today() - timedelta(days=1)
    start_date = st.date_input("시작 날짜", value=default_start)
    end_date = st.date_input("종료 날짜", value=datetime.today())

interval_key, minutes_per_bar = TF_MAP[tf_label]

# 구분선 추가
st.markdown("---")

# -----------------------------
# 조건 설정
# -----------------------------
st.markdown('<div class="section-title">② 조건 설정</div>', unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
with c4:
    lookahead = st.slider("측정 캔들 수 (기준 이후 N봉)", 1, 60, 10)
with c5:
    threshold_pct = st.slider("성공/실패 기준 값(%)", 0.1, 5.0, 1.0, step=0.1)
with c6:
    # RSI: 없음 / 현재(과매도·과매수 중 하나) / 과매도 기준 / 과매수 기준  (BB와 동일 정렬)
    r1, r2, r3 = st.columns(3)
    with r1:
        rsi_mode = st.selectbox(
            "RSI 조건",
            ["없음", "현재(과매도/과매수 중 하나)", "과매도 기준", "과매수 기준"],
            index=0,
            help="현재: RSI≤과매도 또는 RSI≥과매수 중 하나라도 충족"
        )
    with r2:
        rsi_low = st.slider("과매도 RSI 기준", 0, 100, 30, step=1, help="RSI ≤ 값")
    with r3:
        rsi_high = st.slider("과매수 RSI 기준", 0, 100, 70, step=1, help="RSI ≥ 값")

c7, c8, c9 = st.columns(3)
with c7:
    bb_cond = st.selectbox(
        "볼린저밴드 조건",
        ["없음", "상한선", "중앙선", "하한선"],
        index=0,
    )
with c8:
    bb_window = st.number_input("BB 기간", min_value=5, max_value=100, value=30, step=1)
with c9:
    bb_dev = st.number_input("BB 승수", min_value=1.0, max_value=4.0, value=2.0, step=0.1)

# 2차 조건 (양봉/기타)
st.markdown('<div class="hint">2차 조건: 1차 조건(RSI·볼린저밴드) 충족 후 추가 필터</div>', unsafe_allow_html=True)
sec1, sec2 = st.columns(2)
with sec1:
    use_bull2 = st.checkbox("양봉 2개 연속 상승 적용", value=False, help="두 캔들이 연속으로 상승(종가>시가)이며 종가가 연속 상승해야 함")
with sec2:
    allow_other_secondary = st.checkbox("다른 2차 조건 확장 허용", value=False, help="향후 다른 2차 필터 추가용 토글")

st.session_state["bb_cond"]  = bb_cond

# 구분선 추가
st.markdown("---")

# -----------------------------
# 데이터 수집
# -----------------------------
def estimate_calls(start_dt, end_dt, minutes_per_bar):
    mins = max(1, int((end_dt - start_dt).total_seconds() // 60))
    bars = max(1, mins // minutes_per_bar)
    return bars // 200 + 1

_session = requests.Session()
_retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])
_session.mount("https://", HTTPAdapter(max_retries=_retries))

@st.cache_data(ttl=120, show_spinner=False)
def fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar):
    if "minutes/" in interval_key:
        unit = interval_key.split("/")[1]
        url = f"https://api.upbit.com/v1/candles/minutes/{unit}"
    else:
        url = "https://api.upbit.com/v1/candles/days"
    calls_est = estimate_calls(start_dt, end_dt, minutes_per_bar)
    max_calls = min(calls_est + 2, 60)
    req_count = 200
    all_data, to_time = [], end_dt
    try:
        for _ in range(max_calls):
            params = {"market": market_code, "count": req_count, "to": to_time.strftime("%Y-%m-%d %H:%M:%S")}
            r = _session.get(url, params=params, headers={"Accept":"application/json"}, timeout=10)
            r.raise_for_status()
            batch = r.json()
            if not batch: break
            all_data.extend(batch)
            last_ts = pd.to_datetime(batch[-1]["candle_date_time_kst"])
            if last_ts <= start_dt: break
            to_time = last_ts - timedelta(seconds=1)
    except Exception:
        return pd.DataFrame()
    if not all_data: return pd.DataFrame()
    df = pd.DataFrame(all_data).rename(columns={
        "candle_date_time_kst":"time","opening_price":"open","high_price":"high",
        "low_price":"low","trade_price":"close","candle_acc_trade_volume":"volume"})
    df["time"] = pd.to_datetime(df["time"])
    df = df[["time","open","high","low","close","volume"]].sort_values("time").reset_index(drop=True)
    return df[(df["time"] >= start_dt) & (df["time"] <= end_dt)]

# -----------------------------
# 지표
# -----------------------------
def add_indicators(df, bb_window, bb_dev):
    out = df.copy()
    out["RSI13"] = ta.momentum.RSIIndicator(close=out["close"], window=13).rsi()
    bb = ta.volatility.BollingerBands(close=out["close"], window=bb_window, window_dev=bb_dev)
    out["BB_up"]  = bb.bollinger_hband().fillna(method="bfill").fillna(method="ffill")
    out["BB_low"] = bb.bollinger_lband().fillna(method="bfill").fillna(method="ffill")
    out["BB_mid"] = bb.bollinger_mavg().fillna(method="bfill").fillna(method="ffill")
    return out

# --------- 안전 스칼라 변환 헬퍼 ---------
def _to_float_safe(x):
    try:
        return float(x)
    except Exception:
        if hasattr(x, "iloc") and len(x) > 0:
            try:
                return float(x.iloc[0])
            except Exception:
                return np.nan
        return np.nan
# ----------------------------------------

# -----------------------------
# 시뮬레이션
# -----------------------------
def simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, thr_pct, bb_cond, dedup_mode,
             minutes_per_bar, market_code, bb_window, bb_dev, use_bull2=False, allow_other_secondary=False):
    res=[]
    n=len(df); thr=float(thr_pct)

    def bb_ok(i):
        close_i = float(df.at[i, "close"])
        up, lo, mid = df.at[i,"BB_up"], df.at[i,"BB_low"], df.at[i,"BB_mid"]

        if bb_cond == "상한선":
            # 가격이 상한선을 '초과'했을 때만 신호 (종가 기준)
            return pd.notna(up) and (close_i > float(up))

        if bb_cond == "하한선":
            # 가격이 하한선을 '초과'(아래로 벗어남)했을 때만 신호 (종가 기준)
            return pd.notna(lo) and (close_i < float(lo))

        if bb_cond == "중앙선":
            # 중앙선 '초과' 또는 '근처'에서 신호 판별
            if pd.isna(mid) or pd.isna(up) or pd.isna(lo):
                return False
            band_w = max(1e-9, float(up) - float(lo))
            near_eps = 0.1 * band_w  # 밴드폭의 10%를 근처로 간주
            return (close_i >= float(mid)) or (abs(close_i - float(mid)) <= near_eps)

        return False

    # --- RSI 판정: 모드에 따라 계산 ---
    if rsi_mode == "없음":
        rsi_idx = []
    elif rsi_mode == "현재(과매도/과매수 중 하나)":
        idx_low  = df.index[df["RSI13"] <= float(rsi_low)].tolist()
        idx_high = df.index[df["RSI13"] >= float(rsi_high)].tolist()
        rsi_idx = sorted(set(idx_low) | set(idx_high))  # OR
    elif rsi_mode == "과매도 기준":
        rsi_idx = df.index[df["RSI13"] <= float(rsi_low)].tolist()
    elif rsi_mode == "과매수 기준":
        rsi_idx = df.index[df["RSI13"] >= float(rsi_high)].tolist()
    else:
        rsi_idx = []

    # --- BB 판정 ---
    bb_idx = [i for i in df.index if bb_ok(i)] if bb_cond != "없음" else []

    # --- 1차 조건 결합 ---
    if rsi_mode != "없음" and bb_cond != "없음":
        base_sig_idx = sorted(set(rsi_idx) & set(bb_idx))  # AND
    elif rsi_mode != "없음":
        base_sig_idx = rsi_idx
    elif bb_cond != "없음":
        base_sig_idx = bb_idx
    else:
        base_sig_idx = []

    # --- 2차 조건: 양봉 2개 연속 상승 ---
    def bullish_two_in_a_row_anywhere(i, end_idx):
        """i+1 ~ end_idx 사이에 연속 양봉(두 캔들 모두 종가>시가이며, 종가가 연속 상승)이 한 번이라도 존재하는지"""
        if i+2 > end_idx:
            return False
        sub = df.loc[i+1:end_idx, ["open","close"]].reset_index(drop=True)
        for k in range(len(sub)-1):
            c0, o0 = float(sub.at[k,"close"]), float(sub.at[k,"open"])
            c1, o1 = float(sub.at[k+1,"close"]), float(sub.at[k+1,"open"])
            if (c0 > o0) and (c1 > o1) and (c1 > c0):
                return True
        return False

    # 최종 시그널 후보
    sig_idx = base_sig_idx

    i = 0
    while i < n:
        if i in sig_idx:
            end = i + lookahead
            if end >= n: break

            # --- 2차 조건 적용 (BB 여부와 무관하게 적용)
            if use_bull2:
                if not bullish_two_in_a_row_anywhere(i, end):
                    i += 1
                    continue

            base = (float(df.at[i,"open"]) + float(df.at[i,"low"])) / 2.0
            closes = df.loc[i+1:end, ["time","close"]]
            final_ret = (closes.iloc[-1]["close"]/base - 1) * 100 if not closes.empty else 0.0
            min_ret   = (closes["close"].min()/base - 1) * 100 if not closes.empty else 0.0
            max_ret   = (closes["close"].max()/base - 1) * 100 if not closes.empty else 0.0

            result = "중립"; reach_min = None
            end_time = df.at[end, "time"]
            end_close = float(df.at[end, "close"])

            # 성공: 기준 도달 첫 시점
            if max_ret >= thr and not closes.empty:
                target_price = base*(1+thr/100)
                first_hit = closes[closes["close"] >= target_price]
                if not first_hit.empty:
                    hit_time = first_hit.iloc[0]["time"]
                    reach_min = int((hit_time - df.at[i,"time"]).total_seconds()//60)
                    end_time = hit_time
                    idx_hit = df.index[df["time"] == hit_time]
                    if len(idx_hit) > 0:
                        end_close = float(df.at[int(idx_hit[0]), "close"])
                    else:
                        end_close = float(first_hit.iloc[0]["close"])
                result = "성공"
            elif final_ret < 0:
                result = "실패"

            bb_value = None
            if bb_cond=="상한선": bb_value = df.at[i,"BB_up"]
            elif bb_cond=="중앙선": bb_value = df.at[i,"BB_mid"]
            elif bb_cond=="하한선": bb_value = df.at[i,"BB_low"]

            res.append({
                "신호시간": df.at[i,"time"],
                "종료시간": end_time,
                "기준시가": int(round(base)),
                "종료가": end_close,
                "RSI(13)": round(float(df.at[i,"RSI13"]),1) if pd.notna(df.at[i,"RSI13"]) else None,
                "BB값": round(float(bb_value),1) if bb_value is not None else None,
                "성공기준(%)": round(thr,1),
                "결과": result,
                "도달분": reach_min,
                "최종수익률(%)": round(final_ret,2),
                "최저수익률(%)": round(min_ret,2),
                "최고수익률(%)": round(max_ret,2)
            })

            if dedup_mode.startswith("중복 제거"):
                i = end
            else:
                i += 1
        else:
            i += 1

    return pd.DataFrame(res)

# -----------------------------
# 실행
# -----------------------------
try:
    if start_date > end_date:
        st.error("시작 날짜가 종료 날짜보다 이후입니다.")
        st.stop()
    start_dt=datetime.combine(start_date, datetime.min.time())
    end_dt=datetime.combine(end_date, datetime.max.time())

    # RSI/BB 모두 없음이면 대기
    if (rsi_mode=="없음") and (bb_cond=="없음"):
        st.markdown('<div class="section-title">③ 요약 & 차트</div>', unsafe_allow_html=True)
        st.info("대기중..")
        st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)
        st.info("대기중..")
        st.stop()

    df=fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar)
    if df.empty:
        st.error("데이터가 없습니다.")
        st.stop()

    df=add_indicators(df, bb_window, bb_dev)
    bb_cond=st.session_state.get("bb_cond", bb_cond)

    # (변경) BB 미설정이어도 2차 조건 사용 가능 → 에러 팝업 제거

    # 조건 요약 (측정 구간 + 1차/2차 구분 표시)
    total_min = lookahead * minutes_per_bar
    hh, mm = divmod(int(total_min), 60)
    look_str = f"{lookahead}봉 / {hh:02d}:{mm:02d}"

    if rsi_mode == "없음":
        rsi_txt = "없음"
    elif rsi_mode == "현재(과매도/과매수 중 하나)":
        rsi_txt = f"현재: (과매도≤{int(rsi_low)}) 또는 (과매수≥{int(rsi_high)})"
    elif rsi_mode == "과매도 기준":
        rsi_txt = f"과매도≤{int(rsi_low)}"
    elif rsi_mode == "과매수 기준":
        rsi_txt = f"과매수≥{int(rsi_high)}"
    else:
        rsi_txt = "없음"

    bb_txt  = bb_cond if bb_cond!="없음" else "없음"
    sec_txt = f"양봉 2개 연속 상승: {'적용' if use_bull2 else '미적용'}"

    st.info(
        "설정 요약\n"
        f"- 측정 구간: {look_str}\n"
        f"- 1차 조건 · RSI: {rsi_txt} · BB: {bb_txt}\n"
        f"- 2차 조건 · {sec_txt}"
    )

    res_all=simulate(
        df, rsi_mode, rsi_low, rsi_high, lookahead, threshold_pct, bb_cond,
        "중복 포함 (연속 신호 모두)", minutes_per_bar, market_code, bb_window, bb_dev,
        use_bull2=use_bull2, allow_other_secondary=allow_other_secondary
    )
    res_dedup=simulate(
        df, rsi_mode, rsi_low, rsi_high, lookahead, threshold_pct, bb_cond,
        "중복 제거 (연속 동일 결과 1개)", minutes_per_bar, market_code, bb_window, bb_dev,
        use_bull2=use_bull2, allow_other_secondary=allow_other_secondary
    )
    res=res_all if dup_mode.startswith("중복 포함") else res_dedup

    # 요약 메트릭
    def _summarize(df_in):
        if df_in is None or df_in.empty: return 0,0,0,0,0.0,0.0
        total=len(df_in); succ=(df_in["결과"]=="성공").sum(); fail=(df_in["결과"]=="실패").sum(); neu=(df_in["결과"]=="중립").sum()
        win=succ/total*100 if total else 0.0; total_final=df_in["최종수익률(%)"].sum()
        return total,succ,fail,neu,win,total_final

    st.markdown('<div class="section-title">③ 요약 & 차트</div>', unsafe_allow_html=True)
    for label, data in [("중복 포함 (연속 신호 모두)",res_all),("중복 제거 (연속 동일 결과 1개)",res_dedup)]:
        total,succ,fail,neu,win,total_final=_summarize(data)
        st.markdown(f"**{label}**")
        m1,m2,m3,m4,m5,m6=st.columns(6)
        m1.metric("신호 수",f"{total}")
        m2.metric("성공",f"{succ}")
        m3.metric("실패",f"{fail}")
        m4.metric("중립",f"{neu}")
        m5.metric("승률",f"{win:.1f}%")
        col="red" if total_final>0 else "blue" if total_final<0 else "black"
        m6.markdown(f"<div style='font-weight:600;'>최종수익률 합계: <span style='color:{col}; font-size:1.1rem'>{total_final:.1f}%</span></div>",unsafe_allow_html=True)
        st.markdown("---")

    # 차트
    fig=make_subplots(rows=1,cols=1)

    # 캔들 & BB
    fig.add_trace(go.Candlestick(
        x=df["time"],open=df["open"],high=df["high"],low=df["low"],close=df["close"],
        name="가격",increasing_line_color="red",decreasing_line_color="blue",line=dict(width=1.1)
    ))
    fig.add_trace(go.Scatter(x=df["time"],y=df["BB_up"],mode="lines",line=dict(color="#FFB703",width=1.4),name="BB 상단"))
    fig.add_trace(go.Scatter(x=df["time"],y=df["BB_low"],mode="lines",line=dict(color="#219EBC",width=1.4),name="BB 하단"))
    fig.add_trace(go.Scatter(x=df["time"],y=df["BB_mid"],mode="lines",line=dict(color="#8D99AE",width=1.1,dash="dot"),name="BB 중앙"))

    if not res.empty:
        # 시작 마커 (기존 유지)
        for _label,_color in [("성공","red"),("실패","blue"),("중립","#FF9800")]:
            sub = res[res["결과"] == _label]
            if sub.empty: continue
            fig.add_trace(go.Scatter(
                x=sub["신호시간"], y=sub["기준시가"],
                mode="markers", name=f"신호({_label})",
                marker=dict(size=9, color=_color, symbol="circle", line=dict(width=1, color="black"))
            ))

        # 종료 마커 & 점선 — 그룹별 첫 점선만 범례 노출
        legend_emitted = {"성공": False, "실패": False, "중립": False}

        for _, row in res.iterrows():
            start_x = pd.to_datetime(row["신호시간"])
            start_y = float(row["기준시가"])
            end_x   = pd.to_datetime(row["종료시간"])
            end_close = float(row["종료가"])

            grp = row["결과"]
            color = "red" if grp=="성공" else ("blue" if grp=="실패" else "#FF9800")

            # 점선
            fig.add_trace(go.Scatter(
                x=[start_x, end_x], y=[start_y, end_close],
                mode="lines",
                line=dict(color=color, width=1.6 if grp=="성공" else 1.0, dash="dot"),
                opacity=0.9 if grp=="성공" else 0.5,
                showlegend=(not legend_emitted[grp]), name=f"신호(점선)-{grp}"
            ))
            legend_emitted[grp] = True

            if grp == "성공":
                # 성공 별마크: 캔들 막대(해당 봉의 high) "정확히 최상단"에 표시
                hit_row = df.loc[df["time"]==end_x]
                if not hit_row.empty:
                    star_y = float(hit_row.iloc[0]["high"])  # 정확히 high 위치
                else:
                    star_y = end_close  # 예외: 해당 시점 캔들을 못 찾은 경우 종가로 대체
                fig.add_trace(go.Scatter(
                    x=[end_x], y=[star_y],
                    mode="markers", name="목표 도달",
                    marker=dict(size=15, color="orange", symbol="star", line=dict(width=1, color="black")),
                    showlegend=False
                ))
            else:
                # 실패/중립: 도착 지점은 작은 X 마커
                fig.add_trace(go.Scatter(
                    x=[end_x], y=[end_close],
                    mode="markers", name=f"도착-{grp}",
                    marker=dict(size=8, color=color, symbol="x", line=dict(width=1, color="black")),
                    showlegend=False
                ))

    # RSI (네온 느낌 이중선 유지)
    fig.add_trace(go.Scatter(x=df["time"],y=df["RSI13"],mode="lines",
                             line=dict(color="rgba(42,157,143,0.3)",width=6),yaxis="y2",showlegend=False))
    fig.add_trace(go.Scatter(x=df["time"],y=df["RSI13"],mode="lines",
                             line=dict(color="#2A9D8F",width=2.4,dash="dot"),name="RSI(13)",yaxis="y2"))
    fig.add_hline(y=70,line_dash="dash",line_color="#E63946",line_width=1.1,yref="y2")
    fig.add_hline(y=30,line_dash="dash",line_color="#457B9D",line_width=1.1,yref="y2")

    fig.update_layout(
        title=f"{market_label.split(' — ')[0]} · {tf_label} · RSI(13) + BB 시뮬레이션",
        dragmode="zoom",xaxis_rangeslider_visible=False,height=600,legend_orientation="h",legend_y=1.05,
        margin=dict(l=60,r=40,t=60,b=40),yaxis=dict(title="가격"),
        yaxis2=dict(overlaying="y",side="right",showgrid=False,title="RSI(13)",range=[0,100])
    )
    st.plotly_chart(fig,use_container_width=True,config={"scrollZoom":True,"doubleClick":"reset"})

    # 표
    st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)
    if not res.empty:
        tbl=res.sort_values("신호시간",ascending=False).reset_index(drop=True).copy()
        tbl["신호시간"]=pd.to_datetime(tbl["신호시간"]).dt.strftime("%Y-%m-%d %H:%M")
        tbl["기준시가"]=tbl["기준시가"].map(lambda v:f"{int(v):,}")
        if "RSI(13)" in tbl: tbl["RSI(13)"]=tbl["RSI(13)"].map(lambda v:f"{v:.1f}" if pd.notna(v) else "")
        if "BB값" in tbl: tbl["BB값"]=tbl["BB값"].map(lambda v:f"{v:.1f}" if pd.notna(v) else "")
        for col in ["성공기준(%)","최종수익률(%)","최저수익률(%)","최고수익률(%)"]:
            if col in tbl: tbl[col]=tbl[col].map(lambda v:f"{v:.2f}%" if pd.notna(v) else "")
        def fmt_hhmm(start_str, end_str):
            if pd.isna(start_str) or pd.isna(end_str): return "-"
            try:
                s = pd.to_datetime(start_str); e = pd.to_datetime(end_str)
                m = int((e - s).total_seconds() // 60); h,mm = divmod(m,60)
                return f"{h:02d}:{mm:02d}"
            except Exception:
                return "-"
        tbl["도달시간"] = [
            fmt_hhmm(res.loc[i,"신호시간"], res.loc[i,"종료시간"]) 
            for i in range(len(res))
        ]
        if "도달분" in tbl: tbl=tbl.drop(columns=["도달분"])
        # 컬럼 구성
        tbl = tbl[["신호시간","기준시가","RSI(13)","성공기준(%)","결과","최종수익률(%)","최저수익률(%)","최고수익률(%)","도달시간"]]

        def style_result(val):
            if val == "성공":
                return "background-color: #FFF59D; color: #E53935;"
            elif val == "실패":
                return "color: #1E40AF;"
            elif val == "중립":
                return "color: #FF9800;"
            return ""

        styled_tbl = tbl.style.applymap(style_result, subset=["결과"])
        st.dataframe(styled_tbl, use_container_width=True)
    else:
        st.info("조건을 만족하는 신호가 없습니다.")
except Exception as e:
    st.error(f"오류: {e}")
