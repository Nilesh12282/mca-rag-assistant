from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from pinecone import Pinecone
from dotenv import load_dotenv
import os

# ------------------ Load Environment Variables ------------------
load_dotenv()

# ------------------ App Setup ------------------
app = FastAPI(title="RAG MCA Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Configuration ------------------
OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434"
)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "hr-policies"
)

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"

if not PINECONE_API_KEY:
    raise ValueError("❌ Missing PINECONE_API_KEY in .env")


# ------------------ Pinecone Setup ------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(PINECONE_INDEX_NAME)


# ------------------ Request Schema ------------------
class AskRequest(BaseModel):
    question: str


# ------------------ Helper: Generate Embedding ------------------
def get_embedding(text: str):
    """Generate embedding using Ollama."""

    try:
        print("🔹 Generating embedding...")

        res = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={
                "model": EMBEDDING_MODEL,
                "prompt": text
            },
            timeout=60
        )

        res.raise_for_status()

        data = res.json()

        if "embedding" not in data:
            print(
                "❌ Ollama embedding response invalid:",
                data
            )

            raise HTTPException(
                status_code=500,
                detail="Invalid embedding response from Ollama"
            )

        print("✅ Embedding generated.")

        return data["embedding"]

    except requests.exceptions.RequestException as e:

        print("❌ Ollama embedding error:", e)

        raise HTTPException(
            status_code=500,
            detail=f"Ollama embedding error: {str(e)}"
        )


# ------------------ Main Ask Endpoint ------------------
@app.post("/ask")
def ask_question(request: AskRequest):

    question = request.question.strip()

    # ------------------ Validate Question ------------------
    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    print(f"\n🧠 Received question: {question}")


    # =========================================================
    # STEP 0: Handle Greetings Directly
    # =========================================================

    greetings = {
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "good morning",
        "good afternoon",
        "good evening"
    }

    if question.lower() in greetings:

        print("👋 Greeting detected. Skipping RAG.")

        return {
            "answer": (
                "Hi! 👋 How can I help you with MCA "
                "admissions, subjects, syllabus, exams, "
                "fees, placements, or college information?"
            )
        }


    # =========================================================
    # STEP 1: Generate Question Embedding
    # =========================================================

    embedding = get_embedding(question)


    # =========================================================
    # STEP 2: Query Pinecone
    # =========================================================

    try:

        print("🔹 Querying Pinecone...")

        search_results = index.query(
            vector=embedding,
            top_k=6,
            include_metadata=True
        )

        print("✅ Pinecone query complete.")

    except Exception as e:

        print("❌ Pinecone query failed:", e)

        raise HTTPException(
            status_code=500,
            detail=f"Pinecone query error: {str(e)}"
        )


    # =========================================================
    # STEP 3: Get Retrieved Matches
    # =========================================================

    matches = getattr(
        search_results,
        "matches",
        []
    )

    if not matches:

        print("⚠️ No matches found in Pinecone.")

        return {
            "answer": (
                "Sorry, I couldn't find this information "
                "in the available MCA documents."
            )
        }


    # =========================================================
    # STEP 4: Build Clean Context
    # =========================================================

    context_parts = []

    for match in matches:

        if not match.metadata:
            continue

        text = match.metadata.get(
            "text",
            ""
        )

        source = match.metadata.get(
            "source",
            "Unknown document"
        )

        if text.strip():

            context_parts.append(
                f"Source: {source}\n"
                f"{text.strip()}"
            )


    if not context_parts:

        print("⚠️ Retrieved matches contain no usable text.")

        return {
            "answer": (
                "Sorry, I couldn't find this information "
                "in the available MCA documents."
            )
        }


    context = "\n\n---\n\n".join(
        context_parts
    )

    print(
        f"📚 Context length: {len(context)}"
    )


    # =========================================================
    # STEP 5: Create RAG Prompt
    # =========================================================

    prompt = f"""
You are an MCA College Assistant.

Your job is to answer questions about the MCA course,
subjects, syllabus, examinations, fees, placements,
admissions, and other college information.

IMPORTANT RULES:

1. Answer ONLY using the information provided in the
   retrieved context.

2. Do NOT invent, assume, or guess information.

3. If the requested information is not available
   in the context, respond exactly with:

   "Sorry, I couldn't find this information
   in the available MCA documents."

4. Keep the answer direct, clear, and useful.

5. For subject or syllabus questions, provide the
   information available in the retrieved documents.

6. For questions asking for a list, provide the relevant
   items available in the context.

7. Do not use your general knowledge to answer questions
   that are outside the provided MCA documents.

8. If the question is unrelated to MCA or the available
   college documents, politely say that the information
   is not available in the MCA documents.

RETRIEVED CONTEXT:
{context}

STUDENT QUESTION:
{question}

ANSWER:
"""


    # =========================================================
    # STEP 6: Generate Answer Using Llama 3.2
    # =========================================================

    try:

        print(
            "🔹 Sending prompt to Ollama model..."
        )

        gen_res = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        gen_res.raise_for_status()

        data = gen_res.json()

        answer = data.get(
            "response",
            ""
        ).strip()


        if not answer:

            print(
                "⚠️ Ollama returned an empty response."
            )

            answer = (
                "Sorry, I couldn't find this information "
                "in the available MCA documents."
            )


        print(
            "✅ Final answer generated."
        )

        return {
            "answer": answer
        }


    except requests.exceptions.RequestException as e:

        print(
            "❌ Ollama generation failed:",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail=f"Ollama generation error: {str(e)}"
        )


# ------------------ Root Endpoint ------------------
@app.get("/")
def home():

    return {
        "message": (
            "✅ RAG MCA Assistant Backend is running!"
        )
    }