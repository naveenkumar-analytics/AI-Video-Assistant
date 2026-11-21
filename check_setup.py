"""
Project root (main.py ke saath) mein rakho aur chalao:
    python check_setup.py
Yeh poora setup check karta hai, bina video process kiye.
"""
import os
import sys
import importlib
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

problems = 0


def ok(msg):
    print(f"[OK]   {msg}")


def bad(msg):
    global problems
    problems += 1
    print(f"[FAIL] {msg}")


print("\n--- 1. Packages ---")
for pkg in ["whisper", "yt_dlp", "pydub", "requests", "langchain_groq",
            "langchain_chroma", "langchain_text_splitters", "sentence_transformers"]:
    try:
        importlib.import_module(pkg)
        ok(pkg)
    except Exception as e:
        bad(f"{pkg} install nahi hai ({type(e).__name__})")

print("\n--- 2. .env keys ---")
if os.getenv("GROQ_API_KEY"):
    ok("GROQ_API_KEY mili")
else:
    bad("GROQ_API_KEY .env mein nahi mili")
if os.getenv("SARVAM_API_KEY"):
    ok("SARVAM_API_KEY mili (hinglish ke liye)")
else:
    print("[INFO] SARVAM_API_KEY nahi mili (sirf hinglish mode ke liye chahiye)")

print("\n--- 3. Files ka version ---")
sys.path.insert(0, str(ROOT))
summarizer = None
try:
    from core import summarizer
    if hasattr(summarizer, "_get_models") and hasattr(summarizer, "_call_llm"):
        ok("core/summarizer.py NAYA version hai")
    else:
        bad("core/summarizer.py PURANA hai. Naya wala copy karke replace karo.")
        summarizer = None
except Exception as e:
    bad(f"core/summarizer.py import nahi hui: {e}")

for mod in ["core.extractor", "core.rag_engine", "core.vector_store",
            "core.transcriber", "utils.audio_processor"]:
    try:
        importlib.import_module(mod)
        ok(f"{mod} import ho gaya")
    except Exception as e:
        bad(f"{mod} import nahi hua: {type(e).__name__}: {e}")

print("\n--- 4. Groq models aur live test ---")
if summarizer is not None and os.getenv("GROQ_API_KEY"):
    try:
        models = summarizer._get_models()
        ok(f"Models jo try honge: {models}")
        reply = summarizer._call_llm("Reply with exactly one word.", "Say hello")
        ok(f"Groq ne jawab diya: {reply.strip()[:50]}")
    except Exception as e:
        bad(f"Groq call fail hui: {e}")
else:
    print("[SKIP] Groq test (upar ki problem pehle theek karo)")

print("\n" + "=" * 50)
if problems == 0:
    print("Sab theek hai. Ab main.py chala sakte ho.")
else:
    print(f"{problems} problem mili. Upar [FAIL] wali lines dekho.")