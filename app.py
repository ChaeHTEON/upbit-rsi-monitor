# app_23_triple_gc_fix.txt
# -------------------------------------------------
# ✅ Triple_GoldenCross 신호 감지 완전 복구 (app 23 기준)
# -------------------------------------------------

# ✅ Triple_GoldenCross 완전 복구 (동시 발생 → 즉시 신호)
            elif strategy == "Triple_GoldenCross":
                try:
                    # ❗ df 재할당/정렬/드롭 금지: 원본 좌표계(포지션) 유지가 핵심
                    n = len(df)

                    # ✅ 실제 발생 봉 기준 교차 감지 (경계값 포함)
                    rsi_gc  = (df["RSI13"].shift(1) <= 50) & (df["RSI13"] >= 50)
                    cci_gc  = (df["CCI"].shift(1)  <= 0)  & (df["CCI"]  >= 0)
                    macd_gc = (df["MACD"].shift(1) <= df["MACD_signal"].shift(1)) & (df["MACD"] >= df["MACD_signal"])

                    triple_gc = (rsi_gc & cci_gc & macd_gc)
                    triple_gc = triple_gc.fillna(False)

                    # ✅ 포지션 인덱스(0..n-1)로 변환하여 '다음 캔들' 계산을 안전하게
                    _tmp = []
                    for pos in range(n):
                        if bool(triple_gc.iloc[pos]):
                            next_pos = pos + 1 if (pos + 1) < n else (n - 1)
                            _tmp.append(pos if sec_cond != "없음" else next_pos)

                    # ✅ 기존 신호와 누적 결합 (덮어쓰기 방지)
                    if "base_sig_idx" in locals():
                        base_sig_idx = sorted(set(base_sig_idx) | set(_tmp))
                    else:
                        base_sig_idx = sorted(set(_tmp))
                except Exception:
                    base_sig_idx = []
