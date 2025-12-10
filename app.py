from flask import Flask, request, jsonify
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow all origins for all routes

with open("faq.json", "r", encoding="utf-8") as f:
    FAQ_DATA = json.load(f)

@app.route("/")
def health():
    return "Govardhan Chatbot is running"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    question = data.get("question", "")
    name = data.get("name", "").strip()
    mobile = data.get("mobile", "").strip()

    # Ask for name and mobile if not provided
    if not name or not mobile:
        return jsonify({
            "status": "need_user_details",
            "message": "Please share your name and mobile number so that our team can contact you.",
        })

    # Simple FAQ match (replace with your RAG logic)
    answer = None
    q_lower = question.lower()
    for qa in FAQ_DATA:
        if qa["question"].lower() in q_lower or q_lower in qa["question"].lower():
            answer = qa["answer"]
            break

    if not answer:
        # Human-like fallback, passing message to admin
        answer = (
            "Please connect to admin. "
            "We could not find an exact answer right now, "
            "but your query has been noted and our team will call you back soon."
        )

    return jsonify({
        "status": "success",
        "name": name,
        "mobile": mobile,
        "question": question,
        "answer": answer
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
