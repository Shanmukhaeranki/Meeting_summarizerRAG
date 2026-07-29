import streamlit as st
from groq import Groq, RateLimitError
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import time
import subprocess

load_dotenv()

st.set_page_config(page_title="Meeting Notes Summarizer", page_icon="📝")
st.markdown("""
<style>
.block-container { padding-top: 2rem; }
h1, h2, h3 { font-weight: 700; }
.stTextInput > div > div > input { border-radius: 8px; }
[data-testid="stExpander"] { border-radius: 8px; border: 1px solid #2A2E37; }
</style>
""", unsafe_allow_html=True)
st.title("📝 Meeting Notes Summarizer")
st.write(
    "Upload a meeting audio file to get a cleaned transcript, a structured "
    "summary, and ask follow-up questions grounded in the meeting content."
)

FAST_MODEL = "llama-3.1-8b-instant"
STRONG_MODEL = "llama-3.3-70b-versatile"
GROQ_MAX_BYTES = 24 * 1024 * 1024  # stay safely under Groq's 25MB limit


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
# MODULE 1: SPEECH-TO-TEXT (with large-file chunking support)
# ============================================================
def split_audio_ffmpeg(filepath, segment_seconds=1200):
    """Splits a large audio file into ~20-minute, low-bitrate chunks using ffmpeg."""
    output_pattern = "chunk_%03d.mp3"
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", filepath,
            "-f", "segment", "-segment_time", str(segment_seconds),
            "-ar", "16000", "-ac", "1", "-b:a", "64k",
            output_pattern,
        ],
        check=True, capture_output=True,
    )
    chunk_files = sorted(f for f in os.listdir(".") if f.startswith("chunk_") and f.endswith(".mp3"))
    return chunk_files


def transcribe_with_groq(filepath, client):
    file_size = os.path.getsize(filepath)

    if file_size <= GROQ_MAX_BYTES:
        with open(filepath, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
            )
        raw_segments = [seg["text"].strip() for seg in transcription.segments]
        raw_transcript = " ".join(raw_segments)
        return raw_segments, raw_transcript, transcription.language

    # File too large for a single Groq request — split into chunks first
    st.info(f"File is {file_size / (1024*1024):.0f}MB — splitting into smaller pieces before transcription...")
    chunk_files = split_audio_ffmpeg(filepath)

    all_segments = []
    detected_language = "en"
    progress = st.progress(0, text="Transcribing large file in parts...")
    for i, chunk_file in enumerate(chunk_files):
        with open(chunk_file, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
            )
        chunk_segments = [seg["text"].strip() for seg in transcription.segments]
        all_segments.extend(chunk_segments)
        detected_language = transcription.language
        os.remove(chunk_file)
        time.sleep(1)
        progress.progress((i + 1) / len(chunk_files), text=f"Transcribed part {i+1}/{len(chunk_files)}")
    progress.empty()

    raw_transcript = " ".join(all_segments)
    return all_segments, raw_transcript, detected_language


# ============================================================
# MODULE 2: TRANSCRIPT CLEANUP
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
        time.sleep(1)
        progress.progress((i + 1) / len(raw_blocks), text=f"Cleaning section {i+1}/{len(raw_blocks)}")
    progress.empty()
    return cleaned_blocks


# ============================================================
# MODULE 3: SUMMARY GENERATOR (map-reduce, CoT, Markdown)
# ============================================================
def map_step_summarize(cleaned_blocks, client):
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
        time.sleep(1)
        progress.progress((i + 1) / len(cleaned_blocks), text=f"Summarized section {i+1}/{len(cleaned_blocks)}")
    progress.empty()
    return partial_summaries


def reduce_step_summarize(partial_summaries, client):
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
# MODULE 4: RAG Q&A
# ============================================================
def answer_question(question, rag_chunks, chunk_embeddings, embedder, client, top_k=5):
    question_embedding = embedder.encode([question])[0]
    similarities = np.dot(chunk_embeddings, question_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(question_embedding)
    )
    k = min(top_k, len(rag_chunks))
    top_indices = np.argsort(similarities)[-k:][::-1]
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
# PIPELINE RUNNER
# ============================================================
def run_pipeline(filepath, client):
    raw_segments, raw_transcript, language = transcribe_with_groq(filepath, client)

    st.subheader("Raw Transcript")
    with st.expander("Show raw transcript"):
        st.write(raw_transcript)
    st.caption(f"Detected language: {language} | {len(raw_segments)} raw segments")

    cleaned_blocks = clean_full_transcript(raw_segments, client)
    cleaned_transcript = " ".join(cleaned_blocks)

    st.subheader("Cleaned Transcript")
    with st.expander("Show cleaned transcript"):
        st.write(cleaned_transcript)
    st.caption("Disfluencies, punctuation, and ASR errors corrected via a dedicated LLM cleanup pass.")

    partial_summaries = map_step_summarize(cleaned_blocks, client)
    summary = reduce_step_summarize(partial_summaries, client)

    transcript_chunks = chunk_text(cleaned_transcript.split(". "), max_words=150)
    summary_chunks = chunk_text(summary.split(". "), max_words=150)
    rag_chunks = transcript_chunks + summary_chunks
    embedder = load_embedder()
    chunk_embeddings = embedder.encode(rag_chunks)

    return {
        "raw_transcript": raw_transcript,
        "cleaned_transcript": cleaned_transcript,
        "summary": summary,
        "num_segments": len(cleaned_blocks),
        "rag_chunks": rag_chunks,
        "chunk_embeddings": chunk_embeddings,
    }


# ============================================================
# MAIN APP
# ============================================================
uploaded_file = st.file_uploader("Upload audio (mp3/wav/m4a)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mp3")

    already_processed = (
        "processed_filename" in st.session_state
        and st.session_state.processed_filename == uploaded_file.name
    )

    if not already_processed:
        submitted = st.button("Submit")
    else:
        submitted = False

    if submitted:
        with open("temp_audio.mp3", "wb") as f:
            f.write(uploaded_file.read())

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        with st.spinner("Processing meeting (transcription, cleanup, summarization)..."):
            results = run_pipeline("temp_audio.mp3", client)

        st.session_state.processed_filename = uploaded_file.name
        st.session_state.results = results
        os.remove("temp_audio.mp3")
        already_processed = True

    if already_processed:
        results = st.session_state.results
        st.subheader("Raw Transcript")
        with st.expander("Show raw transcript"):
            st.write(results["raw_transcript"])
        st.subheader("Cleaned Transcript")
        with st.expander("Show cleaned transcript"):
            st.write(results["cleaned_transcript"])

        st.caption(f"Processed transcript in {results['num_segments']} segment(s) using map-reduce summarization.")
        st.subheader("Summary")
        st.markdown(results["summary"])

        st.subheader("Ask a question about this meeting")
        question = st.text_input("e.g. What did we decide about the budget?")

        if question:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            with st.spinner("Finding the answer..."):
                answer, relevant_chunks = answer_question(
                    question,
                    results["rag_chunks"],
                    results["chunk_embeddings"],
                    load_embedder(),
                    client,
                )
            st.write("**Answer:**", answer)
            with st.expander("Retrieved context used"):
                for chunk in relevant_chunks:
                    st.write("-", chunk)