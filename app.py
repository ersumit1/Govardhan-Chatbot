import os
import json
from flask import Flask, request, jsonify

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

# Make sure GOOGLE_API_KEY is set in Render environment variables
# Settings -> Environment -> Add GOOGLE_API_KEY=your_key

app = Flask(__name__)

# ---- 1. Load FAQs from JSON ----
with open("faq.json", "r", encoding="utf-8") as f:
    FAQ_DATA = json.load(f)

# Convert QnA to LangChain Documents
docs = [
    Document(
        page_content=qa["question"] + " " + qa["answer"],
        metadata={"source": "faq", "index": i}
    )
    for i, qa in enumerate(FAQ_DATA)
]

# ---- 2. Build embeddings + vector store + retriever + QA chain ----
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(docs, embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,
)

@app.route("/")
def health():
    return "Govardhan Chatbot (LLM + RAG) is running"

# ---- 3. Chat endpoint ----
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    name = data.get("name", "").strip()
    mobile = data.get("mobile", "").strip()

    # Ask for user details if missing
    if not name or not mobile:
        return jsonify({
            "status": "need_user_details",
            "message": "Please share your name and mobile number so that our team can contact you."
        })

    if not question:
        return jsonify({
            "status": "error",
            "message": "Please enter your question."
        })

    try:
        # Always use LLM + RAG
        result = qa_chain.invoke({"query": question})
        answer_text = result["result"] if isinstance(result, dict) else str(result)

        # If LLM returns something too short or generic, treat as no answer
        if not answer_text or len(answer_text.strip()) < 5:
            raise ValueError("Empty/short answer from LLM")

    except Exception:
        # Human-like fallback, passes to admin
        answer_text = (
            "Please connect to admin. "
            "We could not find an exact answer right now, "
            "but your query has been noted and our team will call you back soon."
        )

    return jsonify({
        "status": "success",
        "name": name,
        "mobile": mobile,
        "question": question,
        "answer": answer_text
    })

if __name__ == "__main__":
    # Local development only
    app.run(host="0.0.0.0", port=5000, debug=True)
