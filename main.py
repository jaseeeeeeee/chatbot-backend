from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import nltk

nltk.download("stopwords")

# ---------------------------
# API Setup
# ---------------------------
app = FastAPI()

# Allow Vercel frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Load FAQ Data
# ---------------------------
with open("faq.json") as f:
    faq_data = json.load(f)

faq_questions = [item["question"] for item in faq_data]
faq_answers = [item["answer"] for item in faq_data]

vectorizer = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words("english"))
faq_vectors = vectorizer.fit_transform(faq_questions)

# ---------------------------
# Load HuggingFace Model
# ---------------------------
qa = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# ---------------------------
# Request Body Structure
# ---------------------------
class ChatRequest(BaseModel):
    message: str
    context: str | None = ""

# ---------------------------
# Core Chat Endpoint
# ---------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    user_input = req.message
    pdf_context = req.context or ""

    combined_context = (
        "This is an AI chatbot. It answers general questions and FAQs about courses, "
        "fees, duration, support, and technical topics. "
        + pdf_context
    )

    # Try transformer model
    try:
        answer = qa(question=user_input, context=combined_context)["answer"]
        if len(answer.strip()) > 3:
            return {"reply": answer}
    except:
        pass

    # Fallback to FAQ matching
    user_vec = vectorizer.transform([user_input])
    sim = cosine_similarity(user_vec, faq_vectors)
    idx = sim.argmax()
    return {"reply": faq_answers[idx]}
