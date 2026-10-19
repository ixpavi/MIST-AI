# MIST AI - SRM Student Assistant

A purpose-built assistant for **SRM Institute of Science and Technology**. It is not a thin wrapper around a chatbot API:
a small intent model trained on SRM questions decides what the student needs, exact tools and database skills answer
what must be precise, a sourced knowledge base answers the rest, and a language model only phrases the final text.
Every reply reports how it was produced (route, intent, confidence, sources) and students can rate it.

## Architecture

```text
 question
    |
    v
 [1] Intent model  (TF-IDF + logistic regression, 20 SRM intents)      backend/intents.py
    |
    v
 [2] Rules         greeting / thanks / "who are you"                   -> route: rule
    |
    v
 [3] DB skills     faculty finder, PYQ links, portals, portal          -> route: skill
    |              how-tos, campus locations (PostgreSQL)              backend/skills.py
    v
 [4] Exact tools   SRMJEEE 2026 deadline checker, programme/fee table  -> route: tool
    |              (no language model, no hallucinated numbers)         backend/tools.py
    v
 [5] Retrieval     hybrid search over kb_chunks: embeddings + keywords -> route: kb / kb+llm
    |              + intent-topic boost; 77 sourced entries            backend/engine.py
    v
 [6] Wording layer Gemini (google-genai) rewrites ONLY the retrieved context; falls back
                   through several models, and to the raw retrieved text if no key/quota
 off-topic questions are declined by the intent model -> route: decline
```

Without a database or API key the app still runs in **KB-only mode**: intent model, rules, tools and keyword
retrieval over the bundled JSON knowledge base all work offline.

## What is trained

- **Intent classifier**: TF-IDF (word + character n-grams) and logistic regression over ~20 intents (admissions,
  exam, fees, hostel, placements, faculty, PYQ, portal, location, off-topic, ...). Training data and the held-out
  test set live in `backend/intents.py`; `python train_intent.py` trains and saves `backend/data/intent_model.joblib`.
  If the file is missing it is retrained in memory at start-up.
- **Embeddings** (optional): knowledge-base chunks are embedded with `gemini-embedding-001` by `ingest_kb.py` and
  stored in Postgres; queries are embedded at request time for hybrid retrieval.
- **Feedback loop**: thumbs up/down in the UI are stored in `chat_logs.rating`; `python ingest_kb.py --promote`
  adds thumbs-up grounded answers back into the knowledge base.

## Evaluation (`python evaluate.py`, runs offline)

| Metric | Result |
|--------|--------|
| Intent accuracy, 70 held-out questions never used for training | 95.7% (67/70) |
| Retrieval hit@3, keyword-only mode, 22 known-answer questions | 86% (hit@1 55%) |
| Routing checks (deadline/fee -> exact tool, off-topic -> decline, greeting -> rule) | 5/5 |

Embeddings (when `GEMINI_API_KEY` is set and `ingest_kb.py` has been run) raise retrieval quality further;
`evaluate.py` reports the mode it measured.

## API

| Endpoint | Purpose |
|----------|---------|
| `POST /chat` | `{message, history?, session_id?, action?}` -> `{reply, route, intent, intent_conf, sources, chat_id}` (message max 500 chars; `action: "faculty"` for the two-step faculty flow) |
| `POST /feedback` | `{chat_id, rating: 1 or -1}` |
| `GET /health` | `db_ok`, `kb_source` (`postgres`/`local`), `intent_model_loaded`, `llm_configured` |
| `GET /stats` | totals: chats, grounded answers, thumbs up/down |

## Project structure

```text
backend/
  app.py        Flask API + serves the frontend
  engine.py     the pipeline above
  intents.py    training data + intent model        tools.py   exact SRM tools
  skills.py     database skills                      config.py  environment-only configuration
  database.py   pooled psycopg2 access, KB/log helpers
  data/         sourced knowledge base JSON, programme table, QA seed, trained model
frontend/       "Signal Deck" UI (index.html, style.css, script.js)
database/       schema.sql (tables, seed data, kb_chunks, chat_logs)
train_intent.py  evaluate.py  ingest_kb.py
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # or export the variables below
python train_intent.py      # optional, retrains the intent model
python evaluate.py          # offline metrics
python ingest_kb.py --init  # once: create tables + load the knowledge base into Postgres
cd backend && python app.py # http://localhost:5000
```

### Environment variables (Render dashboard or a local, git-ignored `.env`)

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | wording layer + embeddings (optional; app degrades gracefully without it) |
| `DATABASE_URL` **or** `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASS`, `DB_NAME` | Postgres / Supabase |
| `DB_SSLMODE` | optional, defaults to `require` for remote hosts |
| `GEMINI_CHAT_MODEL`, `GEMINI_EMBED_MODEL` | optional model overrides |

Render start command is in `Procfile` (`gunicorn`, 1 worker, 4 threads). The free tier cold-starts, so open the
site a couple of minutes before a demo.

---

## 🗄 Database Schema & Relational Design

The system is built on a robust, normalized relational database.

### Core Tables

| Table | Description | Key Columns |
|-------|------------|-------------|
| `portals` | University systems (Academia, Student Portal) | `portal_id`, `portal_name`, `link` |
| `portal_features` | Features inside each portal | `feature_id`, `portal_id` (FK), `feature_name` |
| `pyq_resources` | Previous year question paper links | `pyq_id`, `subject_name`, `pyq_link` |
| `university_info` | General university Q&A knowledge base | `info_id`, `topic`, `question`, `answer` |
| `locations` | Campus locations and directions | `location_id`, `place_name`, `description` |
| `website_content` | Scraped web page content | `content_id`, `url` (UNIQUE), `content` |
| `chat_logs` | Conversation log + feedback | `chat_id`, `question`, `response`, `route`, `intent`, `intent_conf`, `grounded`, `rating`, `session_id` |
| `kb_chunks` | Sourced knowledge base (text + optional embedding) | `id`, `topic`, `text`, `source`, `embedding` |

### Key SQL Techniques Implemented
- **Full-Text Search Engine**: `to_tsvector('english', content) @@ plainto_tsquery('english', 'query')`
- **Performance Optimization**: GIN indexes applied to text vectors for lightning-fast retrieval.
- **Fuzzy Matching**: `pg_trgm` extension used for typo-tolerance in acronyms and searches.
- **Data Integrity**: Foreign keys with `ON DELETE CASCADE`.
- **Idempotent Updates**: `UPSERT` with `ON CONFLICT` constraints used extensively in the scraper pipeline.

---

## 🚀 Setup Instructions

### Prerequisites
- **Python 3.10+**
- **PostgreSQL** (running locally on port 5432)
- **pgAdmin 4** (optional, but recommended)
- **Gemini API Key** (optional, for LLM fallback)

### 1. Database Initialization
Create a new database named **`mist ai`**:
```sql
CREATE DATABASE "mist ai";
```

Initialize the schema using the provided script:
```bash
python apply_schema.py
```
*(Alternatively, run `database/schema.sql` manually in pgAdmin.)*

### 2. Environment Configuration
Create a `.env` file in the root directory:
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=mist ai
DB_USER=postgres
DB_PASS=your_password
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
You can start the backend using the provided batch script:
```bash
run_server.bat
```
Or manually:
```bash
cd backend
python app.py
```

The Flask server will start at **http://localhost:5000**. Navigate to this URL in your browser to access the MIST AI Signal Deck.

---

## 🕷 Data Pipeline (Scraping)

To keep the database up-to-date with the latest university information, run the scraper:

```bash
python srm_scraper.py
```
This fetches content from public SRM pages and intelligently `UPSERT`s it into the `website_content` table.

---

## 💬 Example questions

Try asking the AI these launch codes:
- *"Does SRM provide hostel?"* 
- *"Where can I check attendance?"* 
- *"How can I download my hall ticket?"* 
- *"Give me PYQ for DBMS"* 
- *"Where is the placement office?"* 

---

## 📊 Advanced SQL Analytics Examples

**Full-Text Search Execution:**
```sql
SELECT content
FROM website_content
WHERE to_tsvector('english', content) @@ plainto_tsquery('english', 'hostel facilities');
```

**Relational Join (Portals & Features):**
```sql
SELECT pf.feature_name, pf.description, p.portal_name, p.link
FROM portal_features pf
JOIN portals p ON p.portal_id = pf.portal_id
WHERE to_tsvector('english', pf.description) @@ plainto_tsquery('english', 'attendance');
```

---

*Engineered as an advanced Database Management Systems (DBMS) project.*
*MIST AI © 2026*
