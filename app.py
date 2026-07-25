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
st.write("Upload a meeting audio file to get a transcript, a structured summary, and ask follow-up questions.")


# ---------- CACHED MODEL LOADERS ----------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ---------- HELPERS ----------
def call_groq_with_retry(client, max_retries=5, **kwargs):
    """Calls Groq's API with automatic retry + exponential backoff on rate limits."""
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError:
            wait_time = 10 * (attempt + 1)  # 10s, 20s, 30s, 40s, 50s
            st.warning(f"Rate limit hit — waiting {wait_time}s before retrying...")
            time.sleep(wait_time)
    raise Exception("Groq API rate limit exceeded after multiple retries.")


def chunk_text(chunks, max_words=800):
    groups = []
    current = []
    word_count = 0
    for c in chunks:
        current.append(c)
        word_count += len(c.split())
        if word_count >= max_words:
            groups.append(" ".join(current))
            current = []
            word_count = 0
    if current:
        groups.append(" ".join(current))
    return groups


def transcribe_with_groq(filepath, client):
    with open(filepath, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
        )
    chunks = [seg["text"].strip() for seg in transcription.segments]
    transcript = " ".join(chunks)
    return chunks, transcript, transcription.language


def clean_transcript_chunk(raw_text, client):
    prompt = f"""Clean this raw meeting transcript: fix disfluencies, punctuation,
and obvious ASR transcription errors, without changing the meaning
or removing any factual content.

Transcript:
{raw_text}
"""
    resp = call_groq_with_retry(
        client,
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def clean_full_transcript(chunks, client):
    raw_blocks = chunk_text(chunks, max_words=800)
    cleaned_blocks = []
    progress = st.progress(0, text="Cleaning transcript...")
    for i, block in enumerate(raw_blocks):
        cleaned_blocks.append(clean_transcript_chunk(block, client))
        time.sleep(3)
        progress.progress((i + 1) / max(len(raw_blocks), 1), text=f"Cleaning section {i+1}/{len(raw_blocks)}")
    progress.empty()
    return cleaned_blocks


def summarize_long_transcript(cleaned_blocks, client):
    partial_summaries = []
    progress = st.progress(0, text="Summarizing meeting sections...")
    for i, segment in enumerate(cleaned_blocks):
        map_prompt = f"""Summarize this portion of a meeting transcript.

First identify any sentences describing a task, commitment, or
follow-up. Then, for each one, determine who is responsible and
whether a deadline was stated. Finally, list only confirmed
action items in the format: - [Owner]: [Task] ([Deadline if any]).

Also list any decisions made. Be concise — this is one part of a longer meeting.

Transcript portion:
{segment}
"""
        resp = call_groq_with_retry(
            client,
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": map_prompt}],
            temperature=0.3,
        )
        partial_summaries.append(resp.choices[0].message.content)
        time.sleep(3)
        progress.progress((i + 1) / max(len(cleaned_blocks), 1), text=f"Summarized section {i+1}/{len(cleaned_blocks)}")

    progress.empty()

    combined = "\n\n".join(f"Part {i+1}:\n{s}" for i, s in enumerate(partial_summaries))
    reduce_prompt = f"""You are given summaries of consecutive parts of one meeting.
Combine them into a single structured summary with exactly these sections:
1. Overall Summary (3-5 sentences)
2. Key Decisions Made
3. Action Items (with owner and deadline where mentioned)

Format your response in Markdown, using a header for each section
and bullet points for decisions and action items.

Remove duplicate points across parts. Keep it concise and non-repetitive.

Partial summaries:
{combined}
"""
    final = call_groq_with_retry(
        client,
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": reduce_prompt}],
        temperature=0.3,
    )
    return final.choices[0].message.content, len(cleaned_blocks)


# ---------- MAIN APP ----------
uploaded_file = st.file_uploader("Upload audio (mp3/wav/m4a)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    with open("temp_audio.mp3", "wb") as f:
        f.write(uploaded_file.read())

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    with st.spinner("Transcribing audio..."):
        chunks, raw_transcript, language = transcribe_with_groq("temp_audio.mp3", client)

    st.subheader("Raw Transcript")
    with st.expander("Show raw transcript"):
        st.write(raw_transcript)
    st.caption(f"Detected language: {language} | {len(chunks)} raw segments")

    with st.spinner("Cleaning transcript..."):
        cleaned_blocks = clean_full_transcript(chunks, client)
        cleaned_transcript = " ".join(cleaned_blocks)

    st.subheader("Cleaned Transcript")
    with st.expander("Show cleaned transcript"):
        st.write(cleaned_transcript)
    st.caption("Disfluencies, punctuation, and obvious ASR errors corrected via a dedicated LLM cleanup pass.")

    with st.spinner("Generating structured summary..."):
        summary, num_segments = summarize_long_transcript(cleaned_blocks, client)

    st.caption(f"Processed transcript in {num_segments} segment(s) using map-reduce summarization.")
    st.subheader("Summary")
    st.markdown(summary)

    rag_chunks = chunk_text([c for c in cleaned_transcript.split(". ")], max_words=150)
    embedder = load_embedder()
    chunk_embeddings = embedder.encode(rag_chunks)

    st.subheader("Ask a question about this meeting")
    question = st.text_input("e.g. What did we decide about the budget?")

    if question:
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
        qa_response = call_groq_with_retry(
            client,
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": qa_prompt}],
            temperature=0.2,
        )

        st.write("**Answer:**", qa_response.choices[0].message.content)

        with st.expander("Retrieved context used"):
            for chunk in relevant_chunks:
                st.write("-", chunk)

    os.remove("temp_audio.mp3")