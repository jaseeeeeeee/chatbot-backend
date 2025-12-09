from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import nltk
import numpy as np

nltk.download("stopwords")

# ----------------------
# FastAPI App Setup
# ----------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# Load FAQ Dataset
# ----------------------
with open("faq.json") as f:
    faq_data = json.load(f)

faq_questions = [item["question"] for item in faq_data]
faq_answers = [item["answer"] for item in faq_data]

# ----------------------
# Load Lightweight Model
# ----------------------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_embeddings = model.encode(faq_questions)

# ----------------------
# Request Body
# ----------------------
class ChatRequest(BaseModel):
    message: str
    context: str | None = ""

# ----------------------
# Helper Function
# ----------------------
def get_best_match(query, context_text=""):
    combined = faq_questions.copy()
    answers = faq_answers.copy()

    if context_text:
        combined.append(context_text)
        answers.append("Here is information from your uploaded PDF.")

    embeddings = model.encode(combined + [query])
    query_emb = embeddings[-1].reshape(1, -1)
    db_emb = embeddings[:-1]

    similarity_scores = cosine_similarity(query_emb, db_emb)[0]
    best_idx = np.argmax(similarity_scores)

    return answers[best_idx]

# ----------------------
# Chat Endpoint
# ----------------------
@app.post("/chat")
def chat(req: ChatRequest):
    reply = get_best_match(req.message, req.context)
    return {"reply": reply}
