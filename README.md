# MIST AI – SRM Student Assistant

A full-stack chatbot that answers student questions about **SRM Institute of Science and Technology** by querying a **PostgreSQL** database. Built for a **DBMS course** with emphasis on database design, SQL queries, relational schema, and information retrieval.

---

## 🎯 Features

- Ask questions about hostel, fees, attendance, scholarships, placements, and more
- Retrieve previous year question paper (PYQ) links
- Search campus locations
- Full-text search powered by PostgreSQL `to_tsvector` and `plainto_tsquery`
- Web scraper to populate the database from university pages
- Chat history logging

---

## 🗂 Project Structure

```
mist-ai/
├── backend/
│   ├── app.py          # Flask API server
│   ├── database.py     # PostgreSQL connection helper
│   ├── scraper.py      # Web scraper for university pages
│   └── search.py       # Search & retrieval engine
├── database/
│   └── schema.sql      # Database schema + seed data
├── frontend/
│   ├── index.html      # Chat UI
│   ├── style.css       # Dark-themed styling
│   └── script.js       # Client-side chat logic
├── requirements.txt
└── README.md
```

---

## 🗄 Database Schema

### ER Diagram (Logical)

```
portals (1) ──── (N) portal_features
    portal_id PK         portal_id FK → portals

pyq_resources            university_info
    pyq_id PK                info_id PK

locations                website_content
    location_id PK           content_id PK

chat_logs
    chat_id PK
```

### Tables

| Table | Description | Key Columns |
|-------|------------|-------------|
| `portals` | University systems (Academia, Student Portal…) | `portal_id`, `portal_name`, `link` |
| `portal_features` | Features inside each portal | `feature_id`, `portal_id` (FK), `feature_name` |
| `pyq_resources` | Previous year question paper links | `pyq_id`, `subject_name`, `pyq_link` |
| `university_info` | General university Q&A | `info_id`, `topic`, `question`, `answer` |
| `locations` | Campus locations | `location_id`, `place_name`, `description` |
| `website_content` | Scraped web page content | `content_id`, `url` (UNIQUE), `content` |
| `chat_logs` | Chatbot conversation history | `chat_id`, `question`, `response`, `timestamp` |

### Key SQL Techniques

- **Full-text search**: `to_tsvector('english', content) @@ plainto_tsquery('english', 'hostel')`
- **GIN indexes** for fast text retrieval
- **Trigram extension** (`pg_trgm`) for fuzzy matching
- **Foreign keys** with `ON DELETE CASCADE`
- **UPSERT** with `ON CONFLICT` for web scraper

---

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.10+**
- **PostgreSQL** (running locally)
- **pgAdmin 4** (optional, for visual DB management)

### 1. Create the Database

Open pgAdmin 4 and create a database named **`mist ai`** (or use psql):

```sql
CREATE DATABASE "mist ai";
```

### 2. Run the Schema

Open pgAdmin → Query Tool → paste and run:

```
database/schema.sql
```

Or from terminal:

```bash
psql -U postgres -d "mist ai" -f database/schema.sql
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Server

```bash
cd backend
python app.py
```

The server starts at **http://localhost:5000**. It will also auto-initialize the schema on first run.

### 5. Open the Chatbot

Navigate to **http://localhost:5000** in your browser.

---

## 🕷 Running the Scraper (Optional)

The scraper fetches content from public SRM university pages and stores it in the `website_content` table:

```bash
cd backend
python scraper.py
```

---

## 💬 Example Queries

| Question | Source Table |
|----------|-------------|
| Does SRM provide hostel? | `university_info` |
| Where can I check attendance? | `portal_features` |
| Where do I pay fees? | `portal_features` |
| How can I download hall ticket? | `portal_features` |
| Give DBMS PYQ | `pyq_resources` |
| Where is the library? | `locations` + `university_info` |
| What scholarships are available? | `university_info` |
| Where is the placement office? | `locations` + `university_info` |

---

## ⚙️ Configuration

Database connection defaults (in `backend/database.py`):

| Parameter | Default | Env Variable |
|-----------|---------|--------------|
| Host | `localhost` | `DB_HOST` |
| Port | `5432` | `DB_PORT` |
| Database | `mist ai` | `DB_NAME` |
| User | `postgres` | `DB_USER` |
| Password | `1234` | `DB_PASS` |

---

## 📊 Key SQL Examples

### Full-Text Search
```sql
SELECT content
FROM website_content
WHERE to_tsvector('english', content) @@ plainto_tsquery('english', 'hostel');
```

### Join Query (Portal + Features)
```sql
SELECT pf.feature_name, pf.description, p.portal_name, p.link
FROM portal_features pf
JOIN portals p ON p.portal_id = pf.portal_id
WHERE to_tsvector('english', pf.description)
      @@ plainto_tsquery('english', 'attendance');
```

### Chat Logs
```sql
SELECT * FROM chat_logs ORDER BY timestamp DESC LIMIT 10;
```

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask |
| Database | PostgreSQL |
| Search | PostgreSQL full-text search (GIN indexes) |
| Scraper | requests, BeautifulSoup4 |
| Frontend | HTML, CSS, JavaScript |
| DB Driver | psycopg2 |

---

*Built as a DBMS course project — MIST AI © 2026*
