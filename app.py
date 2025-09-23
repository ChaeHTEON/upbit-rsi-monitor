                "성공기준(%)": round(thr, 1),
                "결과": result,
                "도달분": reach_min,
                "최종수익률(%)": round(final_ret, 2),
                "최저수익률(%)": round(min_ret, 2),
                "최고수익률(%)": round(max_ret, 2)
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

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())

    df = fetch_upbit_paged(market_code, interval_key, start_dt, end_dt, minutes_per_bar)
    if df.empty:
        st.error("데이터가 없습니다.")
        st.stop()

    df = add_indicators(df, bb_window, bb_dev)
    bb_cond = st.session_state.get("bb_cond", bb_cond)

    # 요약
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
    bb_txt = bb_cond if bb_cond != "없음" else "없음"
    sec_txt = f"{sec_cond}"

    st.markdown('<div class="section-title">③ 요약 & 차트</div>', unsafe_allow_html=True)
    st.info(
        "설정 요약\n"
        f"- 측정 구간: {look_str}\n"
        f"- 1차 조건 · RSI: {rsi_txt} · BB: {bb_txt}\n"
        f"- 2차 조건 · {sec_txt}"
    )

    # 시뮬레이션 실행
    res_all = simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, threshold_pct,
                       bb_cond, "중복 포함 (연속 신호 모두)", minutes_per_bar, market_code, bb_window, bb_dev, sec_cond=sec_cond)
    res_dedup = simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, threshold_pct,
                         bb_cond, "중복 제거 (연속 동일 결과 1개)", minutes_per_bar, market_code, bb_window, bb_dev, sec_cond=sec_cond)
    res = res_all if dup_mode.startswith("중복 포함") else res_dedup

    # 요약 메트릭
    def _summarize(df_in):
        if df_in is None or df_in.empty:
            return 0, 0, 0, 0, 0.0, 0.0
        total = len(df_in)
        succ = (df_in["결과"] == "성공").sum()
        fail = (df_in["결과"] == "실패").sum()
        neu = (df_in["결과"] == "중립").sum()
        win = succ / total * 100 if total else 0.0
        total_final = df_in["최종수익률(%)"].sum()
        return total, succ, fail, neu, win, total_final

    for label, data in [("중복 포함 (연속 신호 모두)", res_all),
                        ("중복 제거 (연속 동일 결과 1개)", res_dedup)]:
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

    # 차트
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="가격", increasing_line_color="red", decreasing_line_color="blue", line=dict(width=1.1)
    ))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_up"], mode="lines", line=dict(color="#FFB703", width=1.4), name="BB 상단"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_low"], mode="lines", line=dict(color="#219EBC", width=1.4), name="BB 하단"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["BB_mid"], mode="lines", line=dict(color="#8D99AE", width=1.1, dash="dot"), name="BB 중앙"))

    if not res.empty:
        for _label, _color in [("성공", "red"), ("실패", "blue"), ("중립", "#FF9800")]:
            sub = res[res["결과"] == _label]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["신호시간"], y=sub["기준시가"], mode="markers",
                name=f"신호({_label})",
                marker=dict(size=9, color=_color, symbol="circle", line=dict(width=1, color="black"))
            ))

        legend_emitted = {"성공": False, "실패": False, "중립": False}
        for _, row in res.iterrows():
            start_x = pd.to_datetime(row["신호시간"]); start_y = float(row["기준시가"])
            end_x = pd.to_datetime(row["종료시간"]); end_close = float(row["종료가"])
            grp = row["결과"]; color = "red" if grp == "성공" else ("blue" if grp == "실패" else "#FF9800")
            fig.add_trace(go.Scatter(
                x=[start_x, end_x], y=[start_y, end_close], mode="lines",
                line=dict(color=color, width=1.6 if grp == "성공" else 1.0, dash="dot"),
                opacity=0.9 if grp == "성공" else 0.5,
                showlegend=(not legend_emitted[grp]),
                name=f"신호(점선)-{grp}"
            ))
            legend_emitted[grp] = True
            if grp == "성공":
                hit_row = df.loc[df["time"] == end_x]
                star_y = float(hit_row.iloc[0]["high"]) if not hit_row.empty else end_close
                fig.add_trace(go.Scatter(
                    x=[end_x], y=[star_y], mode="markers", name="목표 도달",
                    marker=dict(size=15, color="orange", symbol="star", line=dict(width=1, color="black")),
                    showlegend=False
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=[end_x], y=[end_close], mode="markers", name=f"도착-{grp}",
                    marker=dict(size=8, color=color, symbol="x", line=dict(width=1, color="black")),
                    showlegend=False
                ))

    fig.add_trace(go.Scatter(x=df["time"], y=df["RSI13"], mode="lines",
                             line=dict(color="rgba(42,157,143,0.3)", width=6),
                             yaxis="y2", showlegend=False))
    fig.add_trace(go.Scatter(x=df["time"], y=df["RSI13"], mode="lines",
                             line=dict(color="#2A9D8F", width=2.4, dash="dot"),
                             name="RSI(13)", yaxis="y2"))
    fig.add_hline(y=70, line_dash="dash", line_color="#E63946", line_width=1.1, yref="y2")
    fig.add_hline(y=30, line_dash="dash", line_color="#457B9D", line_width=1.1, yref="y2")

    fig.update_layout(
        title=f"{market_label.split(' — ')[0]} · {tf_label} · RSI(13) + BB 시뮬레이션",
        dragmode="zoom", xaxis_rangeslider_visible=False, height=600,
        legend_orientation="h", legend_y=1.05,
        margin=dict(l=60, r=40, t=60, b=40),
        yaxis=dict(title="가격"),
        yaxis2=dict(overlaying="y", side="right", showgrid=False, title="RSI(13)", range=[0, 100])
    )
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "doubleClick": "reset"})

    # 표
    st.markdown('<div class="section-title">④ 신호 결과 (최신 순)</div>', unsafe_allow_html=True)
    if res is None or res.empty:
        st.info("조건을 만족하는 신호가 없습니다. (데이터는 정상 처리됨)")
    else:
        tbl = res.sort_values("신호시간", ascending=False).reset_index(drop=True).copy()
        tbl["신호시간"] = pd.to_datetime(tbl["신호시간"]).dt.strftime("%Y-%m-%d %H:%M")
        tbl["기준시가"] = tbl["기준시가"].map(lambda v: f"{int(v):,}")
        if "RSI(13)" in tbl:
            tbl["RSI(13)"] = tbl["RSI(13)"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "")
        if "BB값" in tbl:
            tbl["BB값"] = tbl["BB값"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "")
        for col in ["성공기준(%)", "최종수익률(%)", "최저수익률(%)", "최고수익률(%)"]:
            if col in tbl:
                tbl[col] = tbl[col].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "")
        def fmt_hhmm(start_str, end_str):
            if pd.isna(start_str) or pd.isna(end_str):
                return "-"
            try:
                s = pd.to_datetime(start_str); e = pd.to_datetime(end_str)
                m = int((e - s).total_seconds() // 60); h, mm = divmod(m, 60)
                return f"{h:02d}:{mm:02d}"
            except Exception:
                return "-"
        tbl["도달시간"] = [fmt_hhmm(res.loc[i, "신호시간"], res.loc[i, "종료시간"]) for i in range(len(res))]
        if "도달분" in tbl:
            tbl = tbl.drop(columns=["도달분"])
        tbl = tbl[["신호시간", "기준시가", "RSI(13)", "성공기준(%)", "결과",
                   "최종수익률(%)", "최저수익률(%)", "최고수익률(%)", "도달시간"]]
        def style_result(val):
            if val == "성공": return "background-color: #FFF59D; color: #E53935;"
            if val == "실패": return "color: #1E40AF;"
            if val == "중립": return "color: #FF9800;"
            return ""
        styled_tbl = tbl.style.applymap(style_result, subset=["결과"])
        st.dataframe(styled_tbl, use_container_width=True)

except Exception as e:
    st.error(f"오류: {e}")
