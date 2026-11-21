import os
import re
import time
import random
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# .env ko project root se load karo (core/ se ek level upar)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ---------------- Settings ----------------
# Agar Groq se model list na mile to yeh backup list use hogi
DEFAULT_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
# Jo naam in words se match karenge woh pehle try honge (best -> fast)
PREFERRED_KEYWORDS = ["gpt-oss-120b", "llama-3.3-70b", "llama-4", "qwen", "kimi",
                      "gpt-oss-20b", "llama-3.1-8b"]
# Yeh models text chat ke liye nahi hain, inhe skip karo
SKIP_KEYWORDS = ["whisper", "tts", "guard", "orpheus", "playai", "compound",
                 "embed", "safeguard", "distil"]

MAX_MODELS_TO_TRY = 4
ATTEMPTS_PER_MODEL = 3
MIN_GAP_SECONDS = 1.5
CHUNK_SIZE = 6000
CHUNK_OVERLAP = 200
MAX_COMBINED_CHARS = 12000

_models = None
_llms = {}
_bad_models = set()
_last_call_time = 0.0


# ---------------- Model discovery ----------------
def _discover_models() -> list:
    """Groq se aapki key ke liye available models ki list lo."""
    try:
        r = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
            timeout=20,
        )
        r.raise_for_status()
        ids = [m["id"] for m in r.json().get("data", [])]
    except Exception as e:
        print(f"Could not fetch Groq model list ({e}). Using defaults.")
        return DEFAULT_MODELS

    chat_models = [m for m in ids if not any(k in m.lower() for k in SKIP_KEYWORDS)]

    ordered = []
    for key in PREFERRED_KEYWORDS:
        for m in chat_models:
            if key in m.lower() and m not in ordered:
                ordered.append(m)
    for m in chat_models:
        if m not in ordered:
            ordered.append(m)

    if not ordered:
        return DEFAULT_MODELS

    print(f"Groq models available: {ordered[:MAX_MODELS_TO_TRY]}")
    return ordered[:MAX_MODELS_TO_TRY]


def _get_models() -> list:
    global _models
    if _models is None:
        _models = _discover_models()
    return _models


def _get_llm(model_name: str):
    if model_name not in _llms:
        _llms[model_name] = ChatGroq(
            model=model_name,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3,
            max_retries=1,
            timeout=90,
        )
    return _llms[model_name]


# ---------------- Throttle + retry ----------------
def _throttle():
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < MIN_GAP_SECONDS:
        time.sleep(MIN_GAP_SECONDS - elapsed)
    _last_call_time = time.time()


def _is_rate_limit(err: Exception) -> bool:
    text = str(err).lower()
    return "429" in text or "rate limit" in text or "rate_limit" in text


def _call_llm(system_prompt: str, text: str) -> str:
    """Available models ko order mein try karo. Fail model ko skip karo."""
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set in .env")

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}"),
    ])

    last_error = None
    for model_name in _get_models():
        if model_name in _bad_models:
            continue
        chain = prompt | _get_llm(model_name) | StrOutputParser()

        for attempt in range(ATTEMPTS_PER_MODEL):
            _throttle()
            try:
                return chain.invoke({"text": text})
            except Exception as err:
                last_error = err
                if _is_rate_limit(err) and attempt < ATTEMPTS_PER_MODEL - 1:
                    wait = 3 * (2 ** attempt) + random.random()
                    print(f"[{model_name}] rate limit. Waiting {wait:.0f}s...")
                    time.sleep(wait)
                    continue
                print(f"[{model_name}] failed: {type(err).__name__}. Trying next model.")
                _bad_models.add(model_name)
                break

    raise RuntimeError(f"Groq failed: {last_error}")


# ---------------- Local fallback (no API, never fails) ----------------
def _local_summary(transcript: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", transcript.strip())
    picked = [s for s in sentences if len(s) > 20][:6]
    if not picked:
        picked = [transcript[:300]]
    return "\n".join(f"- {s}" for s in picked)


def _local_title(transcript: str) -> str:
    words = transcript.strip().split()
    return " ".join(words[:8]) + ("..." if len(words) > 8 else "")


# ---------------- Public functions (same names as before) ----------------
def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_text(transcript)


FINAL_PROMPT = (
    "You are an expert summarizer. Write a clear, professional summary of this "
    "video transcript in bullet points. Reply in English."
)
MAP_PROMPT = "Summarize this portion of a video transcript concisely. Reply in English."
TITLE_PROMPT = (
    "Based on the video transcript, generate a short professional title "
    "(max 8 words) in English. Only return the title, nothing else."
)


def summarize(transcript: str) -> str:
    try:
        chunks = split_transcript(transcript)

        if len(chunks) <= 1:
            return _call_llm(FINAL_PROMPT, transcript)

        summaries = []
        for i, chunk in enumerate(chunks, 1):
            print(f"Summarizing part {i}/{len(chunks)}...")
            summaries.append(_call_llm(MAP_PROMPT, chunk))

        combined = "\n\n".join(summaries)

        while len(combined) > MAX_COMBINED_CHARS:
            parts = split_transcript(combined)
            if len(parts) <= 1:
                break
            print(f"Compressing {len(parts)} summaries...")
            combined = "\n\n".join(_call_llm(MAP_PROMPT, p) for p in parts)

        return _call_llm(FINAL_PROMPT, combined)

    except Exception as e:
        print(f"LLM unavailable ({e}). Using local fallback summary.")
        return _local_summary(transcript)


def generate_title(transcript: str) -> str:
    try:
        return _call_llm(TITLE_PROMPT, transcript[:2000]).strip().strip('"')
    except Exception as e:
        print(f"LLM unavailable ({e}). Using local fallback title.")
        return _local_title(transcript)