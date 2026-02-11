def get_llm_answer(llm, provider: str, prompt: str) -> str:
    if provider in ["Groq", "OpenAI"]:
        return llm.invoke(prompt).content

    elif provider == "Gemini":
        response = llm.generate_content(prompt)
        return response.text

    else:
        raise ValueError("Unsupported provider")

