import torch
import soundfile as sf
import io
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Literal, Dict, Any
from num2words import num2words
from fastapi.middleware.cors import CORSMiddleware

app_state: Dict[str, Any] = {}
available_speakers = ['aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random']


def normalize_text(text: str) -> str:
    text = re.sub(r'\d+', lambda m: num2words(int(m.group(0)), lang='ru'), text)
    return text


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Запуск приложения: загрузка модели Silero TTS...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language='ru',
        speaker='v4_ru'
    )
    model.to(device)
    app_state["tts_model"] = model

    yield
    app_state.clear()


app = FastAPI(
    title="Russian Text-to-Speech API",
    description="Сервис для синтеза русской речи с поддержкой произношения чисел.",
    version="1.2.0",
    lifespan=lifespan
)



class TTSRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        json_schema_extra={'example': "..."}
    )
    speaker: Literal[tuple(available_speakers)] = Field(
        'baya',
        description=f"Выберите один из доступных голосов: {', '.join(available_speakers)}"
    )
    sample_rate: Literal[48000, 24000, 16000, 8000] = 48000


@app.get("/")
def read_root():
    return {
        "message": "Сервис для синтеза речи готов к работе.",
        "model_loaded": app_state.get("tts_model") is not None,
        "available_speakers": available_speakers
    }

@app.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    """
    curl -X POST "http://127.0.0.1:8080/synthesize" -H "Content-Type: application/json" -d '{
    "text": "Стоимость товара составляет 1999 рублей. Доставка займет 3 дня.",
    "speaker": "kseniya"
    }' --output speech.wav
    """
    model = app_state.get("tts_model")

    print(f"Исходный текст: {request.text}")
    normalized_text = normalize_text(request.text)
    print(f"Нормализованный текст: {normalized_text}")
    
    audio_tensor = model.apply_tts(
        text=normalized_text, 
        speaker=request.speaker,
        sample_rate=request.sample_rate
    )

    buffer = io.BytesIO()
    sf.write(buffer, audio_tensor.cpu().numpy(), request.sample_rate, format='WAV')
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav"}
    )


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

