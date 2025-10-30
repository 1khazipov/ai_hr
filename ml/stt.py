import whisper
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import tempfile
from fastapi.middleware.cors import CORSMiddleware

model = whisper.load_model("medium")

app = FastAPI()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    curl -X POST -F "file=@speech.wav" http://127.0.0.1:8080/transcribe
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    result = model.transcribe(tmp_path, language="ru")
    os.unlink(tmp_path)
    print("transcription", result["text"])
    return {"transcription": result["text"]}


@app.get("/")
def read_root():
    return {"message": "Сервис готов. Отправьте аудиофайл на /transcribe"}


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