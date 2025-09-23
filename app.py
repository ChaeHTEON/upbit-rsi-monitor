def simulate(df, rsi_mode, rsi_low, rsi_high, lookahead, thr_pct, bb_cond, dedup_mode,
             minutes_per_bar, market_code, bb_window, bb_dev, sec_cond="없음"):
    res = []
    n = len(df)
    thr = float(thr_pct)

    # -------- RSI 인덱스 --------
    if rsi_mode == "없음":
        rsi_idx = []
    elif rsi_mode == "현재(과매도/과매수 중 하나)":
        rsi_idx = sorted(set(df.index[df["RSI13"] <= float(rsi_low)].tolist()) |
                         set(df.index[df["RSI13"] >= float(rsi_high)].tolist()))
    elif rsi_mode == "과매도 기준":
        rsi_idx = df.index[df["RSI13"] <= float(rsi_low)].tolist()
    else:  # 과매수 기준
        rsi_idx = df.index[df["RSI13"] >= float(rsi_high)].tolist()

    # -------- BB 인덱스 --------
    def bb_ok(i):
        close_i = float(df.at[i, "close"])
        up, lo, mid = df.at[i, "BB_up"], df.at[i, "BB_low"], df.at[i, "BB_mid"]
        if bb_cond == "상한선":
            return pd.notna(up) and (close_i > float(up))
        if bb_cond == "하한선":
            return pd.notna(lo) and (close_i <= float(lo))
        if bb_cond == "중앙선":
            if pd.isna(mid) or pd.isna(up) or pd.isna(lo):
                return False
            band_w = max(1e-9, float(up) - float(lo))
            near_eps = 0.1 * band_w
            return (close_i >= float(mid)) or (abs(close_i - float(mid)) <= near_eps)
        return False

    bb_idx = [i for i in df.index if bb_cond != "없음" and bb_ok(i)]

    # -------- 1차 결합 --------
    if rsi_mode != "없음" and bb_cond != "없음":
        base_sig_idx = sorted(set(rsi_idx) & set(bb_idx))
    elif rsi_mode != "없음":
        base_sig_idx = rsi_idx
    elif bb_cond != "없음":
        base_sig_idx = bb_idx
    else:
        base_sig_idx = list(range(n)) if sec_cond != "없음" else []

    # -------- 보조 함수 --------
    def is_bull(idx):
        return float(df.at[idx, "close"]) > float(df.at[idx, "open"])

    def b1_pass(j):
        if not is_bull(j):
            return False
        if bb_cond == "상한선":
            ref = float(df.at[j, "BB_up"])
        elif bb_cond == "중앙선":
            ref = float(df.at[j, "BB_mid"])
        elif bb_cond == "하한선":
            ref = float(df.at[j, "BB_low"])
        else:
            return False
        if pd.isna(ref):
            return False
        o, c = float(df.at[j, "open"]), float(df.at[j, "close"])
        return (c >= o + 0.5 * (ref - o)) if (o < ref) else (c >= ref)

    # -------- 메인 루프 --------
    i = 0
    while i < n:
        if i not in base_sig_idx:
            i += 1
            continue

        # --- Anchor 기본값 (첫 신호봉 종가) ---
        anchor_idx = i
        signal_time = df.at[i, "time"]
        base_price = float(df.at[i, "close"])

        # --- 2차 조건 ---
        if sec_cond == "양봉 2개 연속 상승":
            if i + 2 < n:
                c0, o0 = float(df.at[i + 1, "close"]), float(df.at[i + 1, "open"])
                c1, o1 = float(df.at[i + 2, "close"]), float(df.at[i + 2, "open"])
                if not ((c0 > o0) and (c1 > o1) and (c1 > c0)):
                    i += 1
                    continue
            else:
                i += 1
                continue

        elif sec_cond == "BB 기반 첫 양봉 50% 진입":
            # B1 찾기
            B1_idx, B1_close = None, None
            for j in range(i + 1, n):
                if b1_pass(j):
                    v = df.at[j, "close"]
                    if pd.notna(v):
                        B1_idx, B1_close = j, float(v)
                        break
            if B1_idx is None:
                i += 1
                continue

            # B2, B3 찾기
            bull_cnt, B3_idx = 0, None
            for j in range(B1_idx + 1, min(B1_idx + lookahead, n)):
                if is_bull(j):
                    bull_cnt += 1
                    if bull_cnt == 2:
                        B3_idx = j
                        break
            if B3_idx is None:
                i += 1
                continue

            # T 찾기 (B3 이후 첫 B1_close 이상 돌파봉)
            T_idx = None
            for j in range(B3_idx + 1, n):
                cj = df.at[j, "close"]
                if pd.notna(cj) and float(cj) >= B1_close:
                    T_idx = j
                    break
            if T_idx is None:
                i += 1
                continue

            # ✅ Anchor를 T로 설정
            anchor_idx = T_idx
            signal_time = df.at[T_idx, "time"]
            base_price = float(df.at[T_idx, "close"])

        # --- 성과 측정 (성공 조기종료 / 중립·실패는 N봉 고정) ---
        end_idx = anchor_idx + lookahead
        if end_idx >= n:
            i += 1
            continue

        window = df.iloc[anchor_idx + 1:end_idx + 1]  # iloc으로 정확히 N개 캔들 확보
        end_time = df.at[end_idx, "time"]
        end_close = float(df.at[end_idx, "close"])
        final_ret = (end_close / base_price - 1) * 100

        min_ret = (window["close"].min() / base_price - 1) * 100 if not window.empty else 0.0
        max_ret = (window["close"].max() / base_price - 1) * 100 if not window.empty else 0.0

        result, reach_min = "중립", None
        target_price = base_price * (1 + thr / 100)
        hit_rows = window[window["close"] >= target_price]
        if not hit_rows.empty:
            hit_time = hit_rows.iloc[0]["time"]
            if pd.notna(hit_time) and pd.notna(signal_time):
                reach_min = int((hit_time - signal_time).total_seconds() // 60)
            end_time, end_close = hit_time, target_price
            final_ret, result = thr, "성공"
        else:
            if final_ret <= -thr:
                result = "실패"

        bb_value = None
        if bb_cond == "상한선": bb_value = df.at[i, "BB_up"]
        elif bb_cond == "중앙선": bb_value = df.at[i, "BB_mid"]
        elif bb_cond == "하한선": bb_value = df.at[i, "BB_low"]

        res.append({
            "신호시간": signal_time,
            "종료시간": end_time,
            "앵커idx": anchor_idx,   # ✅ 추가
            "끝idx": end_idx,        # ✅ 추가
            "기준시가": int(round(base_price)),
            "종료가": end_close,
            "RSI(13)": round(float(df.at[anchor_idx, "RSI13"]), 1) if pd.notna(df.at[anchor_idx, "RSI13"]) else None,
            "BB값": round(float(bb_value), 1) if bb_value is not None and pd.notna(bb_value) else None,
            "성공기준(%)": round(thr, 1),
            "결과": result,
            "도달분": reach_min,
            "최종수익률(%)": round(final_ret, 2),
            "최저수익률(%)": round(min_ret, 2),
            "최고수익률(%)": round(max_ret, 2)
        })

        i = end_idx if dedup_mode.startswith("중복 제거") else i + 1

    return pd.DataFrame(res)
