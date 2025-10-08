# flask_webhook.py
# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

# ✅ Streamlit Cloud Webhook 주소 (수정 필요)
STREAMLIT_WEBHOOK_URL = "https://upbit-rsi-monitor.streamlit.app/"  # 너의 Streamlit 앱 주소

@app.route("/", methods=["POST"])
def kakao_webhook():
    """카카오 오픈빌더 → Flask → Streamlit 중계"""
    try:
        data = request.get_json(force=True)
        print("📩 Received from Kakao:", data)

        # 카카오 메시지 본문 파싱
        user_msg = data.get("userRequest", {}).get("utterance", "").strip()
        print("🧾 utterance:", user_msg)

        # Streamlit 앱으로 메시지 전달
        try:
            payload = {"user_msg": user_msg}
            headers = {"Content-Type": "application/json"}
            res = requests.post(STREAMLIT_WEBHOOK_URL, json=payload, headers=headers, timeout=5)
            print("➡️ Forwarded to Streamlit:", res.status_code)
        except Exception as e:
            print("⚠️ Streamlit 전달 실패:", e)

        # 카카오 응답
        response = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {"simpleText": {"text": f"✅ Streamlit으로 '{user_msg}' 신호를 전달했습니다."}}
                ]
            }
        }
        return jsonify(response)

    except Exception as e:
        print("❌ Webhook 처리 중 오류:", e)
        return jsonify({
            "version": "2.0",
            "template": {
                "outputs": [
                    {"simpleText": {"text": "⚠️ 처리 중 오류가 발생했습니다."}}
                ]
            }
        })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Flask Webhook Server Started on port {port}")
    app.run(host="0.0.0.0", port=port)
