from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.vector_store import build_vector_store, load_vector_store, get_retriever
from core import summarizer  # Groq setup (model discovery, key) yahin se reuse hoga


def get_llm():
    """Groq ka best available model, aur baaki models backup ke taur par."""
    models = [m for m in summarizer._get_models() if m not in summarizer._bad_models]
    if not models:
        models = summarizer._get_models()

    primary = summarizer._get_llm(models[0])
    fallbacks = [summarizer._get_llm(m) for m in models[1:]]
    return primary.with_fallbacks(fallbacks) if fallbacks else primary


SYSTEM_PROMPT = """You are an expert video assistant. Answer the user's question
based ONLY on the video transcript context provided below.

If the answer is not found in the context, say:
"I could not find this information in the video transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from video transcript:
{context}"""


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def _make_chain(retriever):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    # Full LCEL RAG pipeline
    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | get_llm()
        | StrOutputParser()
    )


def build_rag_chain(transcript: str):
    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k=4)
    return _make_chain(retriever)


def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=4)   # pehle yahan vector_store missing tha
    return _make_chain(retriever)


def ask_question(rag_chain, question: str) -> str:
    print(f"Question : {question}")
    try:
        answer = rag_chain.invoke(question)
    except Exception as e:
        print(f"LLM error: {e}")
        answer = "Sorry, I could not get an answer right now. Please try again in a moment."
    print(f"answer :{answer}")
    return answer