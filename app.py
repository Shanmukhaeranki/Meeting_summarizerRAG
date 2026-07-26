import streamlit as st
from groq import Groq, RateLimitError
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import time

load_dotenv()

st.set_page_config(page_title="Meeting Notes Summarizer", page_icon="📝")
st.title("📝 Meeting Notes Summarizer")
st.write(
    "Upload a meeting audio file to get a cleaned transcript, a structured "
    "summary, and ask follow-up questions grounded in the meeting content."
)

# Models: heavy model (70B) only for the final, once-per-meeting reduce step
# and Q&A; lighter model (8B) for repeated per-chunk calls to stay well
# within Groq's free-tier per-minute token limits.
FAST_MODEL = "llama-3.1-8b-instant"
STRONG_MODEL = "llama-3.3-70b-versatile"


# ============================================================
# CACHED MODEL LOADERS
# ============================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# GROQ CALL WRAPPER — retry with backoff on rate limits
# ============================================================
def call_groq_with_retry(client, max_retries=5, **kwargs):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError:
            wait_time = 10 * (attempt + 1)
            st.warning(f"Rate limit hit — waiting {wait_time}s before retrying...")
            time.sleep(wait_time)
    raise Exception("Groq API rate limit exceeded after multiple retries.")


# ============================================================
# HELPERS
# ============================================================
def chunk_text(pieces, max_words=800):
    """Group small text pieces into larger blocks of roughly max_words words."""
    groups, current, word_count = [], [], 0
    for p in pieces:
        current.append(p)
        word_count += len(p.split())
        if word_count >= max_words:
            groups.append(" ".join(current))
            current, word_count = [], 0
    if current:
        groups.append(" ".join(current))
    return groups


# ============================================================
# MODULE 1: AUDIO -> TEXT (Speech-to-Text)
# ============================================================
def transcribe_with_groq(filepath, client):
    with open(filepath, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
        )
    raw_segments = [seg["text"].strip() for seg in transcription.segments]
    raw_transcript = " ".join(raw_segments)
    return raw_segments, raw_transcript, transcription.language


# ============================================================
# MODULE 2: TRANSCRIPT CLEANUP (distinct step, per sir's feedback)
# ============================================================
def clean_transcript_block(raw_block, client):
    prompt = f"""Clean this raw meeting transcript: fix disfluencies ("um", "uh",
false starts), punctuation, and obvious ASR transcription errors, without
changing the meaning or removing any factual content.

Transcript:
{raw_block}
"""
    resp = call_groq_with_retry(
        client,
        model=FAST_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def clean_full_transcript(raw_segments, client):
    raw_blocks = chunk_text(raw_segments, max_words=800)
    cleaned_blocks = []
    progress = st.progress(0, text="Cleaning transcript...")
    for i, block in enumerate(raw_blocks):
        cleaned_blocks.append(clean_transcript_block(block, client))
        time.sleep(3)
        progress.progress((i + 1) / len(raw_blocks), text=f"Cleaning section {i+1}/{len(raw_blocks)}")
    progress.empty()
    return cleaned_blocks


# ============================================================
# MODULE 3: SUMMARY GENERATOR (map-reduce, chain-of-thought, Markdown)
# ============================================================
def map_step_summarize(cleaned_blocks, client):
    """Per-chunk extraction with explicit chain-of-thought reasoning for action items."""
    partial_summaries = []
    progress = st.progress(0, text="Summarizing meeting sections...")
    for i, block in enumerate(cleaned_blocks):
        map_prompt = f"""Summarize this portion of a meeting transcript.

First identify any sentences describing a task, commitment, or follow-up.
Then, for each one, determine who is responsible and whether a deadline
was stated. Finally, list only confirmed action items in the format:
- [Owner]: [Task] ([Deadline if any])

Also list any decisions made. Be concise — this is one part of a longer meeting.

Transcript portion:
{block}
"""
        resp = call_groq_with_retry(
            client,
            model=FAST_MODEL,
            messages=[{"role": "user", "content": map_prompt}],
            temperature=0.3,
        )
        partial_summaries.append(resp.choices[0].message.content)
        time.sleep(3)
        progress.progress((i + 1) / len(cleaned_blocks), text=f"Summarized section {i+1}/{len(cleaned_blocks)}")
    progress.empty()
    return partial_summaries


def reduce_step_summarize(partial_summaries, client):
    """Single call, strong model: combine into final Markdown-structured summary."""
    combined = "\n\n".join(f"Part {i+1}:\n{s}" for i, s in enumerate(partial_summaries))
    reduce_prompt = f"""You are given summaries of consecutive parts of one meeting.
Combine them into a single structured summary with exactly these sections:
1. Overall Summary (3-5 sentences)
2. Key Decisions Made
3. Action Items (with owner and deadline where mentioned)

Format your response in Markdown, using a header for each section and
bullet points for decisions and action items.

Remove duplicate points across parts. Keep it concise and non-repetitive.

Partial summaries:
{combined}
"""
    resp = call_groq_with_retry(
        client,
        model=STRONG_MODEL,
        messages=[{"role": "user", "content": reduce_prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content


# ============================================================
# MODULE 4: RAG Q&A (retrieval-augmented, grounded)
# ============================================================
def answer_question(question, rag_chunks, chunk_embeddings, embedder, client):
    question_embedding = embedder.encode([question])[0]
    similarities = np.dot(chunk_embeddings, question_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(question_embedding)
    )
    top_k = min(3, len(rag_chunks))
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    relevant_chunks = [rag_chunks[i] for i in top_indices]
    context = "\n".join(relevant_chunks)

    qa_prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "This wasn't discussed in the meeting" — do not guess.

Context:
{context}

Question: {question}
"""
    resp = call_groq_with_retry(
        client,
        model=STRONG_MODEL,
        messages=[{"role": "user", "content": qa_prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content, relevant_chunks


# ============================================================
# MAIN APP
# ============================================================
uploaded_file = st.file_uploader("Upload audio (mp3/wav/m4a)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    with open("temp_audio.mp3", "wb") as f:
        f.write(uploaded_file.read())

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # --- Module 1: Speech-to-Text ---
    with st.spinner("Transcribing audio..."):
        raw_segments, raw_transcript, language = transcribe_with_groq("temp_audio.mp3", client)
    st.subheader("Raw Transcript")
    with st.expander("Show raw transcript"):
        st.write(raw_transcript)
    st.caption(f"Detected language: {language} | {len(raw_segments)} raw segments")

    # --- Module 2: Transcript Cleanup ---
    with st.spinner("Cleaning transcript..."):
        cleaned_blocks = clean_full_transcript(raw_segments, client)
        cleaned_transcript = " ".join(cleaned_blocks)
    st.subheader("Cleaned Transcript")
    with st.expander("Show cleaned transcript"):
        st.write(cleaned_transcript)
    st.caption("Disfluencies, punctuation, and ASR errors corrected via a dedicated LLM cleanup pass.")

    # --- Module 3: Summary Generator (map-reduce) ---
    with st.spinner("Generating structured summary..."):
        partial_summaries = map_step_summarize(cleaned_blocks, client)
        summary = reduce_step_summarize(partial_summaries, client)
    st.caption(f"Processed transcript in {len(cleaned_blocks)} segment(s) using map-reduce summarization.")
    st.subheader("Summary")
    st.markdown(summary)

    # --- Module 4: RAG Q&A setup ---
    rag_chunks = chunk_text(cleaned_transcript.split(". "), max_words=150)
    embedder = load_embedder()
    chunk_embeddings = embedder.encode(rag_chunks)

    st.subheader("Ask a question about this meeting")
    question = st.text_input("e.g. What did we decide about the budget?")

    if question:
        with st.spinner("Finding the answer..."):
            answer, relevant_chunks = answer_question(question, rag_chunks, chunk_embeddings, embedder, client)
        st.write("**Answer:**", answer)
        with st.expander("Retrieved context used"):
            for chunk in relevant_chunks:
                st.write("-", chunk)

    os.remove("temp_audio.mp3")