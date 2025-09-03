# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Start the application:**
```bash
./run.sh
```
Or manually:
```bash
cd backend && uv run uvicorn app:app --reload --port 8000
```

**Install dependencies:**
```bash
uv sync
```

**Install development dependencies:**
```bash
uv sync --group dev
```

**Code Quality Commands:**
```bash
# Format code (black + isort)
uv run black backend/ main.py
uv run isort backend/ main.py

# Run linting checks
uv run flake8 backend/ main.py --max-line-length=88 --extend-ignore=E203,W503
uv run mypy backend/ main.py --ignore-missing-imports

# Run all quality checks
./scripts/lint.sh

# Format and run full quality pipeline
./scripts/quality.sh
```

**Environment setup:**
- Copy `.env.example` to `.env` and add your `ANTHROPIC_API_KEY`
- The application requires Python 3.13+ and uses `uv` for package management

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for course materials with a tool-based approach where Claude decides when to search versus answer directly.

### Core Architecture Pattern

The system uses **intelligent routing** through Claude's decision-making rather than explicit orchestration:
- Simple queries get direct answers from Claude's knowledge
- Course-specific queries trigger the search tool to retrieve relevant content
- Claude synthesizes retrieved context into natural responses

### Key Components

**RAGSystem (`rag_system.py`)** - Main orchestrator that coordinates:
- Document processing and vector storage
- AI generation with tool access
- Session management for conversation history
- Query routing through Claude's system prompt

**AIGenerator (`ai_generator.py`)** - Claude API integration with:
- System prompt that defines search behavior and response guidelines
- Tool execution capability for course content retrieval
- Conversation history management

**VectorStore (`vector_store.py`)** - ChromaDB wrapper for:
- Semantic search using sentence transformers embeddings
- Course metadata and content chunk storage
- Duplicate prevention and analytics

**Tool System (`search_tools.py`)** - Provides:
- CourseSearchTool for semantic content retrieval
- ToolManager for registering and executing tools
- Source tracking for response attribution

### Data Flow

1. User query → FastAPI endpoint (`app.py`)
2. RAGSystem processes query with conversation context
3. Claude analyzes query via system prompt instructions
4. If course-specific: CourseSearchTool → VectorStore → ChromaDB
5. Claude generates response using retrieved context
6. SessionManager updates conversation history
7. Response returned with sources

### Configuration

**Config (`config.py`)** centralizes settings:
- Model selection: `claude-sonnet-4-20250514`
- Text chunking: 800 chars with 100 char overlap
- Search results: max 5 results
- Conversation history: 2 message limit
- ChromaDB path: `./chroma_db`

### Document Processing

Course documents are processed into structured chunks:
- Course metadata extracted with titles and lessons
- Text split into overlapping chunks for semantic search
- Embeddings generated using `all-MiniLM-L6-v2` model
- Stored in ChromaDB with metadata for retrieval

### Session Management

Stateless sessions with conversation history:
- Session IDs track individual conversations
- Limited history (2 exchanges) to manage context size
- History passed to Claude for coherent multi-turn conversations

## Application Startup

The system automatically loads documents from `../docs` folder on startup, processing any `.pdf`, `.docx`, or `.txt` files into the vector store. Duplicate courses are detected and skipped.

## API Endpoints

- `POST /api/query` - Main query endpoint with session support
- `GET /api/courses` - Course analytics and statistics
- Static files served from `../frontend` for the web interface
- please alway use nv instead of pip
- make sure to use uv to run all python files