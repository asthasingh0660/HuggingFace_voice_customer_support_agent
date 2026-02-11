import os
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import google.generativeai as genai


def get_llm(provider: str, model_name: str):
    if provider == "Groq":
        return ChatGroq(
            model=model_name,
            api_key=os.getenv("GROQ_API_KEY"),
        )

    elif provider == "OpenAI":
        return ChatOpenAI(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    elif provider == "Gemini":
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        return genai.GenerativeModel(model_name)

    else:
        raise ValueError("Unsupported provider")

