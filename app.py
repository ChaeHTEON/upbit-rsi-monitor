# ... (위쪽 동일) ...

# -----------------------------
# 섹션: 기본 설정
# -----------------------------
st.markdown('<div class="section-title">① 기본 설정</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    start_date = st.date_input("시작 날짜", value=datetime.today() - timedelta(days=7))
with c2:
    end_date = st.date_input("종료 날짜", value=datetime.today())
with c3:
    market_label, market_code = st.selectbox(
        "종목 선택",
        MARKET_LIST,
        index=default_idx,
        format_func=lambda x: x[0]
    )
tf_label = st.selectbox("봉 종류 선택", list(TF_MAP.keys()), index=2)

# -----------------------------
# 섹션: 조건 설정
# -----------------------------
st.markdown('<div class="section-title">② 조건 설정</div>', unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
with c4:
    lookahead = st.slider("측정 캔들 수 (기준 이후 N봉)", 1, 60, 10)
with c5:
    threshold_pct = st.slider("성공/실패 기준 값(%)", 0.1, 3.0, 1.0, step=0.1)
with c6:
    rsi_side = st.selectbox("RSI 조건", ["RSI ≤ 30 (급락)", "RSI ≥ 70 (급등)"], index=0)

c7, c8 = st.columns(2)
with c7:
    bb_cond = st.selectbox(
        "볼린저밴드 조건",
        ["없음", "하한선 하향돌파", "하한선 상향돌파", "상한선 하향돌파", "상한선 상향돌파"],
        index=0
    )

# -----------------------------
# 실행
# -----------------------------
try:
    # ... (데이터 fetch 동일) ...

    res = simulate(df, rsi_side, lookahead, threshold_pct, bb_cond, dup_mode)

    # -----------------------------
    # 섹션: 요약 & 차트
    # -----------------------------
    st.markdown('<div class="section-title">③ 요약 & 차트</div>', unsafe_allow_html=True)

    total = len(res)
    wins  = int((res["결과"] == "성공").sum()) if total else 0
    fails = int((res["결과"] == "실패").sum()) if total else 0
    neuts = int((res["결과"] == "중립").sum()) if total else 0
    # 중립도 성공으로 포함
    winrate = ((wins + neuts) / total * 100.0) if total else 0.0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("신호 수", f"{total}")
    m2.metric("성공", f"{wins}")
    m3.metric("실패", f"{fails}")
    m4.metric("중립", f"{neuts}")
    m5.metric("승률", f"{winrate:.1f}%")

    # 통합 차트 (캔들 + RSI + BB + 신호)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="가격"
    ))
    fig.add_trace(go.Scatter(x=df["time"], y=df["RSI13"], mode="lines", name="RSI(13)", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_up"], line=dict(color="red", dash="dot"), name="BB Upper"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_low"], line=dict(color="blue", dash="dot"), name="BB Lower"))

    if total > 0:
        for label, color, symbol in [("성공", "red", "triangle-up"),
                                     ("실패", "blue", "triangle-down"),
                                     ("중립", "green", "circle")]:
            sub = res[res["결과"] == label]
            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub["신호시간"], y=sub["기준시가"], mode="markers",
                    name=f"신호 ({label})",
                    marker=dict(size=9, color=color, symbol=symbol, line=dict(width=1, color="black"))
                ))

    fig.update_layout(
        title=f"{market_label.split(' — ')[0]} · {tf_label} · RSI(13) + BB 시뮬레이션",
        xaxis_title="시간", yaxis_title="가격/지표",
        xaxis_rangeslider_visible=False, height=600,
        legend_orientation="h", legend_y=-0.15
    )
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # 섹션: 신호 결과 표
    # -----------------------------
    st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)

    if total > 0:
        tbl = res.sort_values("신호시간", ascending=False).reset_index(drop=True).copy()
        tbl["기준시가"] = tbl["기준시가"].map(lambda v: f"{int(v):,}")
        tbl["RSI(13)"] = tbl["RSI(13)"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "")
        tbl["성공기준(%)"] = tbl["성공기준(%)"].map(lambda v: f"{v:.1f}%")
        tbl["최종수익률(%)"] = tbl["최종수익률(%)"].map(lambda v: f"{v:.1f}%")

        def color_result(val):
            if val == "성공":
                return 'color:red; font-weight:600;'
            if val == "실패":
                return 'color:blue; font-weight:600;'
            return 'color:green; font-weight:600;'

        styled = (tbl.style.applymap(color_result, subset=["결과"]))
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("조건을 만족하는 신호가 없습니다.")

except Exception as e:
    st.error(f"오류: {e}")
