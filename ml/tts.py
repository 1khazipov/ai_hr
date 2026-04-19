import io
import re
from contextlib import asynccontextmanager
from typing import Any, Dict, Literal, Optional

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from num2words import num2words
from pydantic import BaseModel, Field

try:
    import cmudict
    _cmudict = cmudict.dict()
    CMUDICT_AVAILABLE = True
except ImportError:
    _cmudict = {}
    CMUDICT_AVAILABLE = False

app_state: Dict[str, Any] = {}
available_speakers = ['aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random']


# Простейший словарь для часто встречающихся английских слов
EN_WORD_MAP = {
    "hello": "хеллоу",
    "hi": "хай",
    "world": "ворлд",
    "machine": "машин",
    "learning": "лёрнинг",
    "deep": "дип",
    "cloud": "клауд",
    "computing": "компьютинг",
    "data": "дата",
    "science": "сайенс",
    "digital": "диджитал",
    "transformation": "трансформейшн",
    "custom": "кастом",
    "software": "софтвер",
    "enterprise": "энтерпрайз",
    "team": "тим",
    "security": "секьюрити",
    "support": "саппорт",
    "service": "сервис",
    "api": "эй-пи-ай",
    "ai": "эй-ай",
    "it": "ай-ти",
    "gpu": "джи-пи-ю",
    "cpu": "си-пи-ю",
    "backend": "бекенд",
    "frontend": "фронтенд",
    "fullstack": "фуллстек",
}

EN_DIGRAPH_MAP = {
    "th": "с",
    "sh": "ш",
    "ch": "ч",
    "ph": "ф",
    "ng": "нг",
}

EN_CHAR_MAP = {
    "a": "а",
    "b": "б",
    "c": "к",
    "d": "д",
    "e": "е",
    "f": "ф",
    "g": "г",
    "h": "х",
    "i": "и",
    "j": "дж",
    "k": "к",
    "l": "л",
    "m": "м",
    "n": "н",
    "o": "о",
    "p": "п",
    "q": "к",
    "r": "р",
    "s": "с",
    "t": "т",
    "u": "у",
    "v": "в",
    "w": "в",
    "x": "кс",
    "y": "и",
    "z": "з",
}


# Маппинг ARPABET фонем в кириллицу для правильного произношения
# Основан на фонетическом соответствии английских и русских звуков
ARPABET_TO_CYRILLIC = {
    # Гласные
    "AA": "а",      # father -> фадэр
    "AE": "э",      # cat -> кэт
    "AH": "а",      # but -> бат
    "AO": "о",      # law -> ло
    "AW": "ау",     # cow -> кау
    "AY": "ай",     # hide -> хайд
    "EH": "э",      # bed -> бэд
    "ER": "эр",     # bird -> бэрд
    "EY": "ей",     # day -> дей
    "IH": "и",      # bit -> бит
    "IY": "и",      # beat -> бит
    "OW": "оу",     # go -> гоу
    "OY": "ой",     # boy -> бой
    "UH": "у",      # book -> бук
    "UW": "у",      # boot -> бут
    # Согласные
    "B": "б", "CH": "ч", "D": "д", "DH": "з", "F": "ф", "G": "г",
    "HH": "х", "JH": "дж", "K": "к", "L": "л", "M": "м", "N": "н",
    "NG": "нг", "P": "п", "R": "р", "S": "с", "SH": "ш", "T": "т",
    "TH": "с", "V": "в", "W": "в", "Y": "й", "Z": "з", "ZH": "ж",
}


def _arpabet_to_cyrillic(phonemes: list[str]) -> str:
    """
    Конвертирует ARPABET фонемы в кириллицу для русской TTS модели.
    Убирает ударения (цифры 0,1,2 в конце фонем).
    """
    result = []
    for ph in phonemes:
        # Убираем ударение (цифры в конце)
        ph_clean = re.sub(r'[0-2]$', '', ph)
        # Конвертируем в кириллицу
        cyr = ARPABET_TO_CYRILLIC.get(ph_clean, ph_clean.lower())
        result.append(cyr)
    return "".join(result)


def _transliterate_en_word(word: str) -> str:
    """
    Транслитерация английского слова в русское написание с использованием
    CMU Pronouncing Dictionary для правильного фонетического произношения.
    """
    lower = word.lower()

    # Сначала пытаемся по словарю часто встречающихся слов
    if lower in EN_WORD_MAP:
        return EN_WORD_MAP[lower]

    # Пытаемся использовать CMUdict для фонетической транскрипции
    if CMUDICT_AVAILABLE and lower in _cmudict:
        # Берем первую транскрипцию (самую частую)
        phonemes = _cmudict[lower][0]
        cyrillic = _arpabet_to_cyrillic(phonemes)
        if cyrillic:
            return cyrillic

    # Акронимы (IT, AI и т.п.) — по буквам
    if word.isupper() and 1 < len(word) <= 4:
        return " ".join(EN_CHAR_MAP.get(ch.lower(), ch.lower()) for ch in word)

    # Fallback: общий случай: биграммы + посимвольно
    out: list[str] = []
    i = 0
    while i < len(lower):
        dig = lower[i:i+2]
        if dig in EN_DIGRAPH_MAP:
            out.append(EN_DIGRAPH_MAP[dig])
            i += 2
            continue
        ch = lower[i]
        out.append(EN_CHAR_MAP.get(ch, ch))
        i += 1

    return "".join(out)


def _replace_english_words(text: str) -> str:
    """Заменяет английские слова на их русскую транслитерацию."""

    def repl(m: re.Match) -> str:
        word = m.group(0)
        return _transliterate_en_word(word)

    return re.sub(r"[A-Za-z][A-Za-z0-9']*", repl, text)


def normalize_text(text: str) -> str:
    """
    Нормализация текста:
    - транслитерация английских слов в русское написание
    - конвертация чисел в русские слова
    """
    text = _replace_english_words(text)
    text = re.sub(r'\d+', lambda m: num2words(int(m.group(0)), lang='ru'), text)
    return text


def normalize_ru_chunk(text: str) -> str:
    """
    Подготовка RU-куска для Silero RU TTS:
    - схлопываем пробелы/переводы строк
    - конвертируем числа
    - добавляем завершающую пунктуацию (иначе RU модель иногда падает на коротких сегментах)
    """
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return ""
    # Если в куске нет ни одной буквы/цифры, не отправляем его в RU модель
    # (иначе модель может падать на чистой пунктуации типа ',.')
    if not re.search(r"[А-Яа-яЁё0-9]", text):
        return ""
    text = normalize_text(text)
    if text and text[-1] not in ".!?":
        text += "."
    return text


def split_ru_en_segments(text: str):
    """Сейчас не используется (оставлено на будущее), вся строка идёт в RU модель."""
    return [('ru', text)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Запуск приложения: загрузка модели Silero TTS (ru)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Одна русская модель (как раньше)
    tts_ru, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language='ru',
        speaker='v4_ru',
    )
    tts_ru.to(device)

    app_state["device"] = device
    app_state["tts_model_ru"] = tts_ru

    yield
    app_state.clear()


app = FastAPI(
    title="Russian Text-to-Speech API",
    description=(
        "Сервис для синтеза русской речи с поддержкой чисел и вкраплений английских слов."
    ),
    version="1.3.0",
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
        "model_loaded": app_state.get("tts_model_ru") is not None,
        "available_speakers": available_speakers
    }

@app.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    """
    Пример:
    curl -X POST "http://127.0.0.1:8080/synthesize" \
         -H "Content-Type: application/json" \
         -d '{
               "text": "Стоимость товара 1999 рублей. Delivery will take 3 days.",
               "speaker": "kseniya"
             }' \
         --output speech.wav
    """
    model = app_state.get("tts_model_ru")
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Модель TTS не загружена.",
        )

    print(f"Исходный текст: {request.text}")
    normalized_text = normalize_text(request.text)
    print(f"Нормализованный текст: {normalized_text}")

    audio_tensor = model.apply_tts(
        text=normalized_text,
        speaker=request.speaker,
        sample_rate=request.sample_rate,
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