# app_fixed_2025-10-14.txt
# (단일 진실 소스 app (11).py 기반, _push_alert 들여쓰기 복원 완료)

    def _push_alert(symbol, tf, strategy, msg, tp=None, sl=None, stage=None, accuracy=None, pattern=None, rsi=None, cci=None, vol=None):
        from datetime import datetime, timedelta

        if "allow_duplicates" not in st.session_state:
            st.session_state["allow_duplicates"] = False
        if "alert_history" not in st.session_state:
            st.session_state["alert_history"] = []
        if "alerts_live" not in st.session_state:
            st.session_state["alerts_live"] = []
        if "last_alert_at" not in st.session_state:
            st.session_state["last_alert_at"] = {}

        now_kst = (datetime.utcnow() + timedelta(hours=9))
        now_str = now_kst.strftime("%H:%M:%S")

        key = f"{strategy}|{symbol}|{tf}"
        if not st.session_state.get("allow_duplicates", False):
            last_at = st.session_state["last_alert_at"].get(key)
            if last_at and (now_kst - last_at).total_seconds() < 180:
                return

        header = f"🚨 {strategy} 신호 [{symbol}, {tf}분봉]"
        phase = "⚡ 최초 포착" if stage == "initial" else ("✅ 유효 신호" if stage == "valid" else "")
        rate = f"적중률: {accuracy}%" if accuracy is not None else ""
        detail = (
            "━━━━━━━━━━━━━━━━━━━\n"
            f"🕒 {now_str} #신호 최초 발견 시간 기재\n"
            f"단계: {phase}\n"
            f"{rate}\n"
            "━━━━━━━━━━━━━━━━━━━\n"
            f"📊 패턴: {pattern or '-'}\n"
            f"📈 거래량: {vol if vol is not None else '-'}\n"
            f"📉 RSI: {rsi if rsi is not None else '-'}\n"
            f"💹 CCI: {cci if cci is not None else '-'}\n"
            "💡 단기 반전 구간 감지. 매수세 강화 및 하락세 종료 신호.\n"
            "━━━━━━━━━━━━━━━━━━━\n"
        )
        full_msg = f"{header}\n{detail}{(msg or '').strip()}"

        entry = {
            "time": now_str,
            "symbol": symbol,
            "tf": tf,
            "strategy": strategy,
            "msg": full_msg,
            "checked": False,
        }
        if tp is not None:
            entry["tp"] = tp
        if sl is not None:
            entry["sl"] = sl

        st.session_state["alerts_live"].insert(0, entry)
        st.session_state["alert_history"].insert(0, entry)
        st.session_state["last_alert_at"][key] = now_kst
        st.toast(full_msg, icon="📈")
