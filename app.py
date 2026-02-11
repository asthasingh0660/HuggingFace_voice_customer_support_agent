import streamlit as st
from dotenv import load_dotenv
import os
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from elevenlabs.client import ElevenLabs

# LLM abstraction helpers
from llm_selector import get_llm
from llm_response import get_llm_answer


# ============================================================
# Environment setup
# ============================================================
load_dotenv()

eleven_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))


# ============================================================
# Embedding model (cached so it loads only once)
# ============================================================
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embed_model = load_embed_model()


# ============================================================
# Text chunking for RAG
# ============================================================
def chunk_text(text, chunk_size=300):
    paragraphs = text.split("\n\n")
    chunks = []

    current = ""
    for p in paragraphs:
        if len(current) + len(p) < chunk_size:
            current += p + "\n\n"
        else:
            chunks.append(current.strip())
            current = p + "\n\n"

    if current:
        chunks.append(current.strip())

    return chunks


# ============================================================
# Load support documentation
# ============================================================
def load_support_docs():
    with open("hf_docs.txt", "r", encoding="utf-8") as f:
        return f.read()


SUPPORT_TEXT = load_support_docs()
DOC_CHUNKS = chunk_text(SUPPORT_TEXT)
DOC_EMBEDDINGS = embed_model.encode(DOC_CHUNKS)


# ============================================================
# Hybrid retrieval: semantic similarity + keyword boosting
# ============================================================
def search_docs(query, top_k=3):
    query_lower = query.lower()

    # Semantic similarity
    query_emb = embed_model.encode([query])
    sims = cosine_similarity(query_emb, DOC_EMBEDDINGS)[0]

    # Keyword boosting for procedural queries
    keyword_scores = []
    for chunk in DOC_CHUNKS:
        score = 0
        if "pip" in query_lower and "pip" in chunk.lower():
            score += 0.2
        if "install" in query_lower and "install" in chunk.lower():
            score += 0.2
        if "transformers" in chunk.lower():
            score += 0.1
        keyword_scores.append(score)

    final_scores = sims + np.array(keyword_scores)

    top_idx = np.argsort(final_scores)[-top_k:]
    return "\n\n".join([DOC_CHUNKS[i] for i in top_idx])


# ============================================================
# ElevenLabs text-to-speech
# ============================================================
def speak_text(text, filename="response.mp3"):
    audio = eleven_client.text_to_speech.convert(
        voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel voice (free-tier supported)
        model_id="eleven_multilingual_v2",
        text=text,
    )

    with open(filename, "wb") as f:
        for chunk in audio:
            f.write(chunk)

    return filename


# ============================================================
# Local Whisper speech-to-text
# ============================================================
from faster_whisper import WhisperModel

@st.cache_resource
def load_whisper():
    # "base" is small, fast, and CPU friendly
    return WhisperModel("base", compute_type="int8")


whisper_model = load_whisper()


def transcribe_audio(file_path: str) -> str:
    segments, _ = whisper_model.transcribe(file_path)
    text = " ".join(segment.text for segment in segments)
    return text.strip()


# ============================================================
# Streamlit page configuration
# ============================================================
st.set_page_config(page_title="HF Voice Support Agent", layout="centered")

st.title("HuggingFace Voice Support Agent")


# ============================================================
# Conversation memory stored in session state
# ============================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Display previous chat messages
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)


# ============================================================
# LLM provider and model selection
# ============================================================
provider = st.selectbox(
    "Choose LLM Provider",
    ["Groq", "OpenAI", "Gemini"]
)

if provider == "Groq":
    model_name = st.selectbox(
        "Groq Model",
        ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
    )

elif provider == "OpenAI":
    model_name = st.selectbox(
        "OpenAI Model",
        ["gpt-4o-mini"]
    )

else:
    model_name = st.selectbox(
        "Gemini Model",
        ["gemini-2.0-flash", "gemini-2.5-flash"]
    )


st.divider()
st.subheader("Voice Input")

audio_file = st.file_uploader(
    "Upload a voice question (wav or mp3)",
    type=["wav", "mp3"]
)


# ============================================================
# Chat input submission
# ============================================================
typed_question = st.chat_input("Type your question")

voice_question = None

if audio_file is not None:
    if st.button("Transcribe audio"):
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_file.read())

        with st.spinner("Transcribing audio..."):
            voice_question = transcribe_audio("temp_audio.wav")

        st.success(f"Transcribed: {voice_question}")

# Final question selection
question = voice_question if voice_question else typed_question

if question:

    # 1. Save user message to memory
    st.session_state.chat_history.append(("user", question))

    # 2. Build recent conversation history for prompt
    history_text = ""
    for role, msg in st.session_state.chat_history[-6:]:
        history_text += f"{role.capitalize()}: {msg}\n"

    # 3. Retrieve relevant documentation context
    context = search_docs(question)

    # 4. Construct main support prompt
    prompt = f"""
You are a professional and friendly Hugging Face customer support assistant.

Conversation so far:
{history_text}

Guidelines:
- Be warm but concise
- Use a calm, confident tone
- Give clear, step-by-step instructions
- Prefer short paragraphs
- Avoid unnecessary explanations

Rules:
- Use ONLY the context below for technical facts
- If commands exist, include them clearly
- End the answer cleanly

Context:
{context}

User’s latest question:
{question}
"""

    # 5. Initialize selected LLM
    llm = get_llm(provider, model_name)

    # 6. Generate detailed text response
    text_answer_raw = get_llm_answer(llm, provider, prompt)

    opening = "I am happy to help you with that.\n\n"
    text_answer = opening + text_answer_raw[:500]

    # 7. Display assistant response
    with st.chat_message("assistant"):
        st.markdown(text_answer)

    # 8. Save assistant message to memory
    st.session_state.chat_history.append(("assistant", text_answer))

    # 9. Create concise voice-oriented summary prompt
    voice_prompt = f"""
You are a friendly technical support voice assistant.

Summarize the answer below for voice output:
- Be brief and conversational
- Do NOT read commands verbatim
- Refer to commands as "the command shown on screen"
- Focus on guidance, not details
- Sound reassuring and helpful
- Limit to 2 to 3 sentences

Detailed answer:
{text_answer}
"""

    # 10. Generate concise voice text
    voice_text = get_llm_answer(llm, provider, voice_prompt)

    # 11. Convert to speech and play audio
    audio_file = speak_text(voice_text)

    st.audio(audio_file)
