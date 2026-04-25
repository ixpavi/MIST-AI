# 🌌 MIST AI – SRM Student Assistant

A full-stack, AI-powered chatbot designed to answer student questions about **SRM Institute of Science and Technology**. It uses a highly optimized **PostgreSQL** database as its primary knowledge source, and falls back to **Gemini AI (LLM)** for complex queries. Built with a focus on robust database design, SQL full-text search, relational schema logic, and a premium "Signal Deck" UI.

---

## 🎯 Features

- **Domain-Specific Knowledge**: Accurate answers about hostels, fees, attendance, scholarships, placements, and campus navigation.
- **Resource Finder**: Instantly retrieve previous year question paper (PYQ) links and portal routes (Academia, Student Portal).
- **Hybrid Search Engine**: 
  - Primary: Full-text search powered by PostgreSQL `to_tsvector`, `plainto_tsquery`, and `pg_trgm` (trigram matching).
  - Secondary: Fallback to Gemini AI for natural language understanding and broader context.
- **Web Scraping Pipeline**: Automated data population from public university pages directly into the database.
- **Premium UI**: "Signal Deck" interface featuring a modern, dark-themed glassmorphism aesthetic with responsive design.
- **Chat History Logging**: Conversation tracking stored securely in the database.

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | HTML5, CSS3 (Custom Properties, Grid/Flexbox), Vanilla JavaScript |
| **Backend** | Python 3.10+, Flask API |
| **Database** | PostgreSQL, `psycopg2` driver |
| **Search & AI**| PostgreSQL GIN Indexes, Gemini API |
| **Data Pipeline**| `requests`, `BeautifulSoup4` |

---

## 🗂 Project Structure

```text
mist-ai/
├── backend/
│   ├── app.py              # Flask API server & routing
│   ├── database.py         # PostgreSQL connection & transaction management
│   ├── llm.py              # Gemini AI fallback integration
│   ├── scraper.py          # Core scraping logic
│   └── search.py           # NLP processing & search retrieval engine
├── database/
│   └── schema.sql          # Relational DB schema
├── frontend/
│   ├── index.html          # Chat interface structure
│   ├── style.css           # Premium dark-themed UI styling
│   └── script.js           # Client-side API interactions
├── apply_schema.py         # Script to apply DB schema
├── generate_seed.py        # Seed data generation script
├── srm_scraper.py          # Specialized SRM site scraper
├── set_acronyms.py         # Dictionary mapping for common college terms
├── test_*.py               # Suite of test files (logic, queries, search)
├── run_server.bat          # Quick-start script for Windows
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

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
| `chat_logs` | Chatbot conversation history | `chat_id`, `question`, `response`, `timestamp` |

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

## 💬 System Capabilities

Try asking the AI these launch codes:
- *"Does SRM provide hostel?"* (Hits `university_info`)
- *"Where can I check attendance?"* (Hits `portal_features`)
- *"How can I download my hall ticket?"* (Hits `portal_features`)
- *"Give me PYQ for DBMS"* (Hits `pyq_resources`)
- *"Where is the placement office?"* (Hits `locations` + `university_info`)

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
