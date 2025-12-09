from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import json

# HuggingFace QA model (FREE, NO KEY required)
HF_API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
HF_HEADERS = {"Authorization": ""}  # Empty because we are using free tier


# -----------------------------
# FASTAPI APP INITIALIZATION
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# LOAD FAQ FILE
# -----------------------------
with open("faq.json") as f:
    faq_data = json.load(f)

faq_questions = [item["question"] for item in faq_data]
faq_answers = [item["answer"] for item in faq_data]


# -----------------------------
# REQUEST BODY MODEL
# -----------------------------
class ChatRequest(BaseModel):
    message: str
    context: str | None = ""


# -----------------------------
# FIND BEST MATCH FROM FAQ
# -----------------------------
def get_faq_answer(user_input):
    user_input = user_input.lower()

    for q, a in zip(faq_questions, faq_answers):
        if q.lower() in user_input or user_input in q.lower():
            return a

    return None


# -----------------------------
# HUGGINGFACE API CALL
# -----------------------------
def ask_huggingface(question, context):
    payload = {"question": question, "context": context}

    try:
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload)
        data = response.json()

        # If the model returns an answer
        if isinstance(data, dict) and "answer" in data:
            return data["answer"]

        return "I'm not sure, but I'm still learning! 🤖"
    except:
        return "The AI model is currently unavailable. Try again later."


# -----------------------------
# MAIN CHAT ENDPOINT
# -----------------------------
@app.post("/chat")
def chat(request: ChatRequest):

    user_question = request.message
    user_context = request.context or ""

    # 1️⃣ Check FAQ first (fast!)
    faq_answer = get_faq_answer(user_question)
    if faq_answer:
        return {"reply": faq_answer}

    # 2️⃣ If not in FAQ → Ask HuggingFace model
    hf_answer = ask_huggingface(user_question, user_context)

    return {"reply": hf_answer}


# -----------------------------
# ROOT ROUTE FOR TESTING
# -----------------------------
@app.get("/")
def home():
    return {"status": "Chatbot backend is running successfully! 🚀"}
