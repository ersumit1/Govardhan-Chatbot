from flask import Flask, request, jsonify
import json

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document

app = Flask(__name__)

with open("faq.json", "r", encoding="utf-8") as f:
    FAQ_DATA = json.load(f)

docs = [
    Document(
        page_content=qa["question"] + " " + qa["answer"],
        metadata={"source": "faq", "index": i}
    )
    for i, qa in enumerate(FAQ_DATA)
]

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    name = data.get("name", "").strip()
    mobile = data.get("mobile", "").strip()

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
        # 1) retrieve similar FAQ chunks
        relevant_docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in relevant_docs])

        # 2) ask LLM using that context
        prompt = f"""
You are a helpful assistant for Govardhan Institute.
Use ONLY the information in the CONTEXT to answer.

CONTEXT:
{context}

QUESTION:
{question}

Answer in simple, clear language.
"""
        llm_response = llm.invoke(prompt)
        answer_text = llm_response.content if hasattr(llm_response, "content") else str(llm_response)

        if not answer_text or len(answer_text.strip()) < 5:
            raise ValueError("Empty/short answer from LLM")

    except Exception:
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
    app.run(host="0.0.0.0", port=5000, debug=True)
