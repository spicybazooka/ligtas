import os
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AsyncOpenAI

# --- 1. Configuration ---
load_dotenv()  # 1. Load the variables from the .env file
api_key = os.getenv("OPENAI_API_KEY")  # 2. Now check if the key exists

if not api_key:
    # This is line 13 where your code is crashing
    raise RuntimeError("❌ OPENAI_API_KEY not found in environment variables.")

# Initialize the Async client (more efficient for FastAPI than synchronous clients)
client = AsyncOpenAI(api_key=api_key)

app = FastAPI(title="Ligtas Chatbot API (OpenAI)")

# --- 2. System Instructions ---
LIGTAS_SYSTEM_PROMPT = """
You are the L.I.G.T.A.S. Advisory Engine (Location Intelligence & Geospatial Triage for Accelerated Support). 
Your role is to generate clear, concise, and actionable disaster safety advisories based strictly on structured input data.

STRICT RULES:
1. ONLY use provided structured data. Do NOT invent details.
2. Keep the response SHORT (maximum 2–3 sentences).
3. Use clear, simple language.
4. Prioritize LIFE-SAVING instructions (e.g., "Move to higher ground").
5. Do NOT mention AI, models, or that you are an assistant.
6. Return ONLY the final advisory message. No JSON, no formatting.
"""

# --- 3. Data Models ---
class Message(BaseModel):
    role: str  # Must be 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

# --- 4. API Endpoints ---
@app.get("/")
def root():
    return {"message": "Ligtas Chatbot API (OpenAI) is running!"}

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Prepare messages: Inject the System Prompt first, then follow with history
        openai_messages = [{"role": "system", "content": LIGTAS_SYSTEM_PROMPT}]
        
        for m in request.messages:
            openai_messages.append({"role": m.role, "content": m.content})

        if len(openai_messages) == 1:
            raise HTTPException(status_code=400, detail="No messages provided")

        # --- OpenAI API Call ---
        # No need for run_in_threadpool; AsyncOpenAI is non-blocking
        response = await client.chat.completions.create(
            model="gpt-4o-mini", # Fast and cost-effective for short instructions
            messages=openai_messages,
            temperature=0.7,
            max_tokens=150
        )

        reply = response.choices[0].message.content

        if not reply:
            return {"reply": "⚠️ OpenAI did not return a valid response."}

        return {"reply": reply}

    except Exception as e:
        print(f"🔥 OpenAI Error: {e}")
        # Graceful fallback for the user
        return {"reply": "The safety engine is currently busy. Please try again in a moment."}