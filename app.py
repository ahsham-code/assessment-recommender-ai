from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from retriever import search_catalog
from google import genai
import os
from dotenv import load_dotenv

# LOAD ENV VARIABLES
load_dotenv()

# GEMINI CLIENT
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# FASTAPI APP
app = FastAPI()

# ENABLE CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REQUEST MODELS
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


# ROOT ENDPOINT
@app.get("/")
def root():
    return {
        "message": "SHL Assessment Recommender API"
    }


# HEALTH ENDPOINT
@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# CHECK IF QUERY IS VAGUE
def is_vague(text):

    vague_terms = [
        "assessment",
        "test",
        "help",
        "hiring",
        "need assessment",
        "need test"
    ]

    text = text.lower().strip()

    if text in vague_terms:
        return True

    if len(text.split()) <= 3:
        return True

    return False


# CHECK OFF TOPIC
def is_off_topic(text):

    blocked_topics = [
        "salary",
        "politics",
        "legal",
        "movie",
        "recipe",
        "sports",
        "religion",
        "crypto"
    ]

    text = text.lower()

    for word in blocked_topics:
        if word in text:
            return True

    return False


# FORMAT RECOMMENDATIONS
def format_recommendations(items):

    recommendations = []
    seen = set()

    for item in items:

        # REMOVE DUPLICATES
        if item["name"] in seen:
            continue

        seen.add(item["name"])

        recommendations.append({
            "name": item["name"],
            "url": item["url"],
            "test_type": item["test_type"]
        })

    return recommendations


# CHAT ENDPOINT
@app.post("/chat")
def chat(req: ChatRequest):

    # SAFETY CHECK
    if not req.messages:
        return {
            "reply": "Please provide conversation messages.",
            "recommendations": [],
            "end_of_conversation": False
        }

    # BUILD HISTORY
    history = " ".join(
        [m.content for m in req.messages]
    )

    latest_message = req.messages[-1].content

    # OFF TOPIC REFUSAL
    if is_off_topic(latest_message):

        return {
            "reply": "I can only help with SHL assessments and hiring assessment recommendations.",
            "recommendations": [],
            "end_of_conversation": False
        }

    # VAGUE QUERY
    if is_vague(latest_message):

        return {
            "reply": "Can you share the role, seniority level, and important skills you are hiring for?",
            "recommendations": [],
            "end_of_conversation": False
        }

    # COMPARISON SUPPORT
    comparison_keywords = [
        "difference",
        "compare",
        "vs",
        "versus"
    ]

    if any(word in latest_message.lower() for word in comparison_keywords):

        retrieved = search_catalog(
            latest_message,
            top_k=2
        )

        recommendations = format_recommendations(retrieved)

        comparison_prompt = f"""
You are an SHL assessment expert.

Compare these SHL assessments ONLY using the provided information.

Assessment Data:
{retrieved}

Provide:
1. Main purpose
2. Skills measured
3. Key differences

Keep response concise and professional.
"""

        try:

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=comparison_prompt
            )

            reply_text = response.text

        except Exception:

            reply_text = (
                "These assessments evaluate different skills and capabilities."
            )

        return {
            "reply": reply_text,
            "recommendations": recommendations,
            "end_of_conversation": False
        }

    # RETRIEVE RECOMMENDATIONS
    retrieved = search_catalog(
        history,
        top_k=5
    )

    recommendations = format_recommendations(retrieved)

    # PROMPT
    recommendation_prompt = f"""
You are an SHL assessment recommendation assistant.

User hiring requirements:
{history}

Recommended SHL assessments:
{[r['name'] for r in retrieved]}

Write a concise recommendation response in 3-5 lines maximum.

Rules:
- Only discuss SHL assessments
- Do not hallucinate
- Keep response concise
- Mention why assessments fit the hiring needs
"""

    # GENERATE RESPONSE
    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=recommendation_prompt
        )

        reply_text = response.text

    except Exception:

        reply_text = (
            "Here are recommended SHL assessments based on your hiring requirements."
        )

    # FINAL RESPONSE
    return {
        "reply": reply_text,
        "recommendations": recommendations,
        "end_of_conversation": False
    }