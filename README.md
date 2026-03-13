## Citizen's Dashboard – FastAPI + RAG backend

This project turns long Indian parliamentary bills (PDFs) into a **structured, citizen‑friendly dashboard**.  
You upload any bill PDF, and the app builds a retrieval‑augmented generation (RAG) index and answers questions with **short, strictly grounded summaries and impact points.**

---

### 1. High‑level architecture

- **Backend (`FastAPI`)**
  - `main.py` – core **RAG engine** using:
    - `PyPDFLoader` to read PDFs.
    - `RecursiveCharacterTextSplitter` to chunk text.
    - `OpenAIEmbeddings` + `Chroma` for vector search.
    - `LangGraph` to orchestrate: **retrieve → (optional) compress via ScaleDown → generate structured answer**.
    - `DashboardResponse` (Pydantic) guarantees the model only returns:
      - `summary` – one‑line answer (max ~25 words).
      - `impact_points` – 2–3 short bullets.
      - `key_clauses` – exact sections/clauses from the bill or empty.
  - `backend.py` – **HTTP API** the frontend talks to:
    - `POST /upload` – upload and index a new PDF.
    - `GET /status` – check if indexing is ready.
    - `POST /ask` – ask questions about the indexed bill.
    - `GET /` – serves the interactive HTML UI.

- **Frontend (`static/index.html`)**
  - A **single, self‑contained HTML file** with:
    - Warm, editorial‑style UI (no framework, pure HTML/CSS/JS).
    - Left sidebar for **PDF upload, indexing status, and stats placeholders**.
    - Right panel is a **chat‑like assistant** that renders rich, card‑style answers.
  - JavaScript in the page talks directly to the FastAPI backend:
    - `POST /upload` – sends PDF via `FormData`.
    - Polls `GET /status` until `{"ready": true}`.
    - `POST /ask` with `{"question": "..."}` to get structured answers.

---

### 2. Backend API details

#### `POST /upload`

- **Request**: `multipart/form-data` with field `file` (a `.pdf` file).
- **Behavior**:
  - Saves the file under `uploads/`.
  - Builds a new `CitizenDashboardApp` instance on that PDF (loads + chunks + embeds).
  - Marks the global index as **ready**.
- **Response** (example):

```json
{
  "message": "PDF uploaded and indexed successfully.",
  "pdf_path": "uploads/FinanceBill2026.pdf",
  "stats": null
}
```

> The frontend already tolerates `stats: null`, but you can later extend this to include chunk counts or other graph metrics.

#### `GET /status`

- **Response**:

```json
{ "ready": true }
```

- `ready` is `true` only after a PDF has been successfully indexed.

#### `POST /ask`

- **Request body**:

```json
{
  "question": "What changes does this bill make for salaried taxpayers?",
  "thread_id": "citizens-dashboard" // optional
}
```

- **Response body** (shape tuned to the new UI):

```json
{
  "one_liner": "Short one-line answer, mapped from 'summary'.",
  "paragraph": "",
  "impact_points": ["..."],
  "key_sections": ["Section 2(44)", "Section 44AB"],
  "acts_referenced": []
}
```

- Internally, this calls `CitizenDashboardApp.ask_structured`, which returns a `DashboardResponse`. The backend then **maps**:
  - `summary` → `one_liner`
  - `impact_points` → `impact_points`
  - `key_clauses` → `key_sections`
  - `acts_referenced` is left empty for now (you can extend the model later).

---

### 3. RAG + LangGraph pipeline (`main.py`)

- **Loading / indexing**
  - For a given PDF path, `CitizenDashboardApp`:
    1. Loads the PDF pages via `PyPDFLoader`.
    2. Splits into overlapping chunks (`chunk_size=500`, `chunk_overlap=50`).
    3. Embeds chunks using `text-embedding-3-small` via OpenRouter.
    4. Stores embeddings in a `Chroma` collection named `"citizen_dashboard"`.
    5. Creates a retriever with `k=8` nearest chunks.

- **Graph steps**
  - `retrieve_node` – given the latest user message, pulls relevant chunks from Chroma.
  - `compress_node` – optionally compresses context using **ScaleDown API** (if `SCALEDOWN_API_KEY` is present).
  - `generate_node` – calls a `ChatOpenAI` model with a strict system prompt and structured output (`DashboardResponse`).
  - `LangGraph` handles state + memory so follow‑up questions stay in context via `thread_id`.

- **Hallucination‑prevention rules** baked into the system prompt:
  - Only answer from retrieved context; otherwise say:  
    `"The document does not provide information on this."`
  - Keep `summary` to a single short line.
  - Only include clauses/sections that actually appear in the context.

---

### 4. Frontend behavior (`static/index.html`)

- **Upload flow**
  - When a user selects a PDF:
    - The filename and size appear in the **Upload Document** card.
    - The **Index →** button becomes active.
  - On clicking **Index →**:
    - Sends the file to `POST /upload`.
    - Shows a smooth progress bar and status messages.
    - Polls `GET /status` until `{ "ready": true }`.
    - Once ready, enables the **Send** button and shows a welcome card.

- **Question/answer flow**
  - The message box supports:
    - `Enter` to send,
    - `Shift + Enter` for a new line.
  - On send:
    - Renders the user bubble.
    - Shows a “thinking” card with animated dots.
    - Calls `POST /ask` with the text.
    - Renders a rich **answer card**:
      - One‑liner TL;DR.
      - Plain‑language explanation (if you later populate `paragraph`).
      - “Who is affected” impacts list.
      - Key section tags.

---

### 5. Environment & configuration

Create a `.env` file (you already have `.env.example` as a reference) with:

```env
OPENAI_API_KEY=your_openrouter_key_here
SCALEDOWN_API_KEY=optional_scaledown_api_key_here
```

Notes:
- The app uses **OpenRouter** (`https://openrouter.ai/api/v1`) as the base for both embeddings and chat.
- If `SCALEDOWN_API_KEY` is not set, the compression step is skipped and raw context is used.

---

### 6. Running the project

From the project root (`c:\Users\hp\Desktop\index`):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m uvicorn backend:app --reload
```

Then open in your browser:

```text
http://127.0.0.1:8000/
```

Upload a bill PDF and start asking questions.

---

### 7. Extending the system

- **Better stats in the sidebar**
  - Compute real values for `chunks`, `nodes`, `edges`, and `refs` when building the index and return them from `/upload` as `stats`.
- **Richer answers**
  - Extend `DashboardResponse` with fields for `paragraph`, `acts_referenced`, etc., and wire them through `backend.py` so the UI cards can show cross‑references between Acts.
- **Authentication / multi‑user**
  - Add per‑user or per‑session identifiers and separate vector stores if you want isolated dashboards per user.
- **Persistent storage**
  - Swap out the in‑memory Chroma instance for a persistent DB directory so indices survive restarts.

