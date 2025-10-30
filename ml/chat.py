from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal
import ollama
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Ollama Qwen3 Chat Service",
    version="1.0.0",
)


class Message(BaseModel):
    role: Literal['system', 'user', 'assistant']
    content: str

class ChatHistoryRequest(BaseModel):
    messages: List[Message]
    model: str = 'qwen3:8b' 


@app.get("/")
def read_root():
    return {"message": "FastAPI сервис для Ollama qwen3:8b с поддержкой истории диалога"}

@app.post("/chat")
async def generate_chat_response(request: ChatHistoryRequest):
    """
    curl -X POST "http://127.0.0.1:8000/chat" -H "Content-Type: application/json" -d '{
    "messages": [
        {
        "role": "user",
        "content": "Объясни, что такое квантовая запутанность, простыми словами для пятиклассника."
        },
    ]
    }'
    """
    messages_for_ollama = [msg.dict() for msg in request.messages]

    response = ollama.chat(
        model=request.model,
        messages=messages_for_ollama,
        think=False
    )

    return {"response": response['message']['content']}

origins = [
    "*", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)