# TalentChat - AI-Powered Talent Acquisition Platform

An intelligent resume screening and interview preparation tool powered by RAG (Retrieval Augmented Generation) and LLM technology.

## Overview

TalentChat helps recruiters and hiring managers:
- Match candidates to job descriptions using semantic search
- Rank resumes with AI-powered scoring
- Extract candidate information automatically
- Generate personalized interview questions

## Architecture

TalentChat is built on the FarmerChat/Farmstack infrastructure, leveraging:
- RAG (Retrieval Augmented Generation) pipeline for content retrieval
- Vector embeddings for semantic similarity matching
- LLM-based ranking and question generation
- Multi-modal content processing

**Note:** This application reuses the robust content retrieval and AI infrastructure originally developed for agricultural Q&A (Farmer.Chat by Digital Green), adapted for HR use cases.

## Features

### 🎯 Candidate Matching
- Semantic job description analysis
- Resume parsing and vectorization
- Multi-factor ranking (skills, experience, relevance)
- Automatic candidate grouping and deduplication

### 📊 Intelligent Ranking
- **Deterministic Mode**: Score-based ranking using coverage, relevance, and bonus factors
- **LLM Mode**: GPT-powered ranking with reasoning
- Composite scoring combining multiple signals

### 💬 Interview Question Generation
- Tailored questions based on candidate profile + job description
- 5 aptitude MCQs
- 2 coding challenges
- 3 critical thinking scenarios
- Experience-level appropriate difficulty

### 🔍 Candidate Insights
- Automatic name extraction from resumes
- Contact information extraction (emails, phones)
- Skills and experience summarization
- Match reasoning and selection summaries

## Technology Stack

- **Language**: Python 3.9+
- **Framework**: Django + Django REST Framework
- **Database**: PostgreSQL 15.x
- **AI Services**:
  - OpenAI GPT-3.5/GPT-4 for embeddings, ranking, and question generation
  - Google Cloud (ASR, Translation, TTS) - optional for voice features
- **ORM**: Peewee
- **Vector Search**: Farmstack integration for resume storage and retrieval

## API Endpoints

### Candidate Retrieval
```
POST /api/chat/get_candidates_for_jd/
```
Submit a job description and get ranked candidates with match scores.

### Interview Questions
```
POST /api/chat/generate_interview_questions/
```
Generate personalized interview questions for a specific candidate.

### Health Check
```
GET /api/health/
```

For complete API specification, see [openapi.yaml](openapi.yaml)

## Setup Instructions

### Requirements
1. Linux or Mac OS
2. Python 3.9 or 3.10
3. PostgreSQL database (V 15.x)
4. OpenAI API key
5. Google Cloud credentials (optional - for voice features)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/M-Ayaan-21/TalentChat.git
cd TalentChat
```

2. **Create virtual environment**
```bash
python3 -m venv .myenv
source .myenv/bin/activate
pip install -r requirements.txt
```

3. **Configure environment variables**

Create a `.env` file in the project root (see [example_dot_env](example_dot_env) for reference):

```bash
# Django
SECRET_KEY=<Django-Project-Secret-Key>

# Database
DB_USER=<DB-Username>
DB_PASSWORD=<DB-Password>
DB_HOST=<DB-Host-IP>
DB_PORT=<DB-Port>
DB_NAME=<DB-Name>
DB_MAX_CONNECTIONS=<DB-Max-Connections-for-Connection-Pool>
DB_STALE_TIMEOUT=<DB-Stale-Timeout-Unused-Connections>

# OpenAI
OPENAI_API_KEY=<OpenAI-API-Key>

# Google Cloud (optional)
GOOGLE_APPLICATION_CREDENTIALS=<Path-to-Google-Application-Credentials-JSON>
```

4. **Start the Django development server**
```bash
python3 manage.py runserver
```

5. Once the development server is started, the APIs are accessible at `http://localhost:8000/api`

## Database Setup

Required only if database logging is enabled (`WITH_DB_CONFIG=True`):

```bash
cd database
pem migrate
psql -h <hostname> -p <port> -U <username> -d <database_name> -f ../multilingual_text_data.sql
```

## Configuration Reference

The following optional variables can be set to customize behaviour:

| Variable | Description |
|---|---|
| `WITH_DB_CONFIG` | `True/False` - enable conversation logging to database |
| `DJANGO_DEBUG_MODE` | `True/False` - run Django in debug mode |
| `CONTENT_DOMAIN_URL` | Farmstack base URL for content retrieval |
| `GPT_3_MODEL` | GPT-3 model version to use |
| `GPT_4_MODEL` | GPT-4 model version to use |
| `TEMPERATURE` | LLM temperature setting |
| `MAX_TOKENS` | Maximum tokens in LLM output |

## Contact

For queries or support, please open an issue in this repository.
