from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI
import os
import sqlite3
from datetime import datetime

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

app = FastAPI()

DB_PATH = os.getenv("CHAT_DB_PATH", "chat_history.db")


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
            """
        )


def now_iso() -> str:
    return datetime.utcnow().isoformat()


def tutor_system_prompt() -> str:
    return (
        "Eres MathBot, tutor de matemáticas para niñas y niños de primaria (6-12 años). "
        "Responde SIEMPRE en español con tono amable, motivador y claro. "
        "Reglas estrictas: "
        "1) Explica en pasos cortos y numerados. "
        "2) No resuelvas todo de una vez; guía una parte y espera al estudiante. "
        "3) Cada respuesta debe incluir al menos un ejemplo en bloque de código con formato ```txt```. "
        "4) Mantén respuestas breves (máximo 120 palabras). "
        "5) Si hay error del estudiante, corrige con cariño y explica por qué. "
        "6) Cierra SIEMPRE con una pregunta para continuar."
    )


def generate_math_title(first_message: str) -> str:
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Genera un título corto de tema matemático en español para una conversación escolar. "
                    "Debe tener entre 3 y 6 palabras, no llevar comillas ni puntuación final."
                ),
            },
            {
                "role": "user",
                "content": f"Mensaje inicial del estudiante: {first_message}",
            },
        ],
        temperature=0.3,
    )

    title = (completion.choices[0].message.content or "Tema de matemáticas").strip()
    return title[:80]


def build_memory_messages(conversation_id: int) -> List[dict]:
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()

    memory = [{"role": "system", "content": tutor_system_prompt()}]
    memory.extend([{"role": row["role"], "content": row["content"]} for row in rows])
    return memory


@app.on_event("startup")
def on_startup() -> None:
    init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- MODELOS CORRECTOS ----
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    conversation_id: int
    user_id: int
    message: str


class StartChatRequest(BaseModel):
    user_id: int
    first_message: str


class HistoryItem(BaseModel):
    id: int
    title: str
    created_at: str
    updated_at: str


class ConversationResponse(BaseModel):
    id: int
    title: str
    created_at: str
    messages: List[Message]


@app.post("/api/chat/start")
async def start_chat(payload: StartChatRequest):
    first = payload.first_message.strip()
    if not first:
        raise HTTPException(status_code=400, detail="El primer mensaje es obligatorio")

    try:
        title = generate_math_title(first)
        now = now_iso()

        with get_db_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO conversations (user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (payload.user_id, title, now, now),
            )
            conversation_id = cursor.lastrowid

        return {"ok": True, "conversationId": conversation_id, "title": title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest):

    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="El mensaje es obligatorio")

    try:
        with get_db_connection() as conn:
            convo = conn.execute(
                "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
                (payload.conversation_id, payload.user_id),
            ).fetchone()

        if not convo:
            raise HTTPException(status_code=404, detail="Conversación no encontrada")

        memory = build_memory_messages(payload.conversation_id)
        memory.append({"role": "user", "content": payload.message})

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=memory,
            temperature=0.4,
        )

        reply = response.choices[0].message.content or "Lo siento, no pude responder ahora."

        now = now_iso()
        with get_db_connection() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (payload.conversation_id, "user", payload.message, now),
            )
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (payload.conversation_id, "assistant", reply, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, payload.conversation_id),
            )

        return {
            "ok": True,
            "reply": reply,
            "conversationId": payload.conversation_id,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/api/history/{user_id}")
async def get_history(user_id: int, limit: int = 6):
    try:
        with get_db_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM conversations
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()

        return {
            "ok": True,
            "items": [dict(row) for row in rows],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history/{user_id}/{conversation_id}")
async def get_conversation(user_id: int, conversation_id: int):
    try:
        with get_db_connection() as conn:
            convo = conn.execute(
                "SELECT id, title, created_at FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            ).fetchone()
            if not convo:
                raise HTTPException(status_code=404, detail="Conversación no encontrada")

            rows = conn.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
                (conversation_id,),
            ).fetchall()

        return {
            "ok": True,
            "conversation": {
                "id": convo["id"],
                "title": convo["title"],
                "created_at": convo["created_at"],
                "messages": [dict(row) for row in rows],
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))