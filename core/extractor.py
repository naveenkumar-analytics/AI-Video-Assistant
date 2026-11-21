# Action items, decisions, questions

from core.summarizer import _call_llm

# Lamba transcript ho to sirf shuruaati hissa bhejo (token limit se bachne ke liye)
MAX_CHARS = 12000


def _extract(system_prompt: str, transcript: str, empty_message: str) -> str:
    try:
        return _call_llm(system_prompt, transcript[:MAX_CHARS])
    except Exception as e:
        print(f"LLM unavailable ({e}).")
        return empty_message


def extract_action_items(transcript: str) -> str:
    return _extract(
        "You are an expert analyst. From the transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'",
        transcript,
        "No action items found.",
    )


def extract_key_decisions(transcript: str) -> str:
    return _extract(
        "You are an expert analyst. From the transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'",
        transcript,
        "No key decisions found.",
    )


def extract_questions(transcript: str) -> str:
    return _extract(
        "From the transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'",
        transcript,
        "No open questions found.",
    )