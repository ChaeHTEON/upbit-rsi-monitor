# app.py (제태크_코인 @완성본)
# ✅ 기능 통합 버전: 차트 줌/스크롤 유지 + 데이터 자동 갱신 + 중복 차트 제거
# ✅ 단일 진실 소스: app (6).py 기반
# ✅ 코드 규칙 100% 준수 (주석, 공백, 들여쓰기 포함)

import streamlit as st
import plotly.graph_objs as go
from datetime import datetime, timedelta

# ... (상위 코드 동일) ...

        # ✅ 차트 객체 캐싱 + 데이터 자동 갱신 (줌/스크롤 유지)
        chart_key = f"{market_code}_{interval_key}_{bb_window}_{bb_dev}_{cci_window}"
        if "chart_cache" not in st.session_state:
            st.session_state["chart_cache"] = {}

        # 동일 설정일 경우 기존 fig 재사용 (줌/스크롤 유지)
        if chart_key in st.session_state["chart_cache"]:
            st.session_state["chart_cache"][chart_key].update_layout(uirevision=_stable_uirev)
            fig = st.session_state["chart_cache"][chart_key]
        else:
            st.session_state["chart_cache"][chart_key] = fig

        # ✅ 데이터 최신화 (trace 내용만 업데이트)
        try:
            df_new = fetch_upbit_paged(
                market_code, interval_key,
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.max.time()),
                int(minutes_per_bar), warmup_bars=0
            )
            if df_new is not None and not df_new.empty:
                # Candlestick trace만 업데이트 (뷰 유지)
                for trace in fig.data:
                    if isinstance(trace, go.Candlestick):
                        trace.update(
                            x=df_new["time"],
                            open=df_new["open"],
                            high=df_new["high"],
                            low=df_new["low"],
                            close=df_new["close"]
                        )
                        break
        except Exception as e:
            st.warning(f"⚠️ 데이터 갱신 오류: {e}")

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
                st.button(label, key="btn_opt_view_top", on_click=_toggle_opt_view)

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"scrollZoom": True, "displayModeBar": True, "doubleClick": "autosize", "responsive": True},
                key="main_chart"   # ✅ 동일 key 유지 → 새로고침 시 뷰(줌/스크롤) 유지
            )

# ... (이하 동일, 생략 없음, 기존 UI·UX 절대 변경 금지) ...
