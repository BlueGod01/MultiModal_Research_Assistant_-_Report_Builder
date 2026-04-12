import os
from typing import Dict, Optional
from dotenv import load_dotenv

# LangChain LLMs
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# =========================
# GOOGLE GEMINI
# =========================
def google_llm(
    model: str = "gemini-1.5-pro",
    temperature: float = 0.0,
    streaming: bool = False,
):
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        streaming=streaming,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )



# =========================
# OPENAI
# =========================
def openai_llm(
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    streaming: bool = False,
):
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        streaming=streaming,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


# =========================
# ANTHROPIC (CLAUDE)
# =========================
def anthropic_llm(
    model: str = "claude-3-5-sonnet-20241022",
    temperature: float = 0.0,
    streaming: bool = False,
):
    return ChatAnthropic(
        model=model,
        temperature=temperature,
        streaming=streaming,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )


