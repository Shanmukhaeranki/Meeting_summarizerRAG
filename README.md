# Meeting Notes Summarizer

Upload a meeting audio file to get:
- A transcript (via Groq's hosted Whisper API)
- A structured summary — key points, decisions, action items (map-reduce for long meetings)
- A Q&A feature to ask follow-up questions, using RAG (retrieval-augmented generation)

## Setup
1. Create a virtual environment: `python -m venv venv`
2. Activate it: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
3. Install dependencies: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and add your free Groq API key from console.groq.com
5. Install ffmpeg (needed if extracting audio from video files)
6. Run: `streamlit run app.py`

## Tech stack
- Transcription: Groq-hosted Whisper (whisper-large-v3-turbo)
- Summarization: Llama 3.3 70B via Groq API
- Embeddings/RAG: sentence-transformers (all-MiniLM-L6-v2) + cosine similarity
- UI: Streamlit