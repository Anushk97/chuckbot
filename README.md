# ChuckBot - Automated Ticket Response System

An intelligent ticket processing system that uses semantic matching and keyword extraction to analyze support tickets and route them to relevant documentation.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         ChuckBot Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │  user_input  │───▶│  processing  │───▶│    evaluator     │   │
│  │   (main.py)  │    │   (llm.py)   │    │ (guardrails.py)  │   │
│  └──────────────┘    └──────────────┘    └──────────────────┘   │
│         │                   │                     │              │
│         │                   │                     ▼              │
│         │                   │            ┌──────────────────┐   │
│         │                   │            │     response     │   │
│         │                   │            │   (output.py)    │   │
│         │                   │            └──────────────────┘   │
│         │                   │                     │              │
│         ▼                   ▼                     ▼              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    db_layer (mongo.py)                     │ │
│  │                 Persistence & Analytics                     │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Modules

### user_input (`user_input/main.py`)
**Status: Implemented**

Processes incoming support tickets with:
- **Keyword Extraction**: Identifies known keywords (login, auth, refund, deployment, etc.)
- **Semantic Matching**: Uses Sentence Transformers (`all-MiniLM-L6-v2`) to find semantically similar document keys
- **Multi-Intent Support**: Handles complex queries with multiple topics, returning top-k matches above a similarity threshold

Key functions:
- `extract_keywords(ticket_content, custom_keywords)` - Extract matching keywords
- `semantic_match(ticket_content, document_keys, similarity_threshold, top_k)` - Semantic similarity search
- `process_ticket(ticket_data, document_keys)` - Full ticket processing pipeline

### processing (`processing/llm.py`)
**Status: Stub**

Will handle LLM integration for response generation:
- OpenAI/Anthropic API integration
- Context-aware response generation
- Token management

### evaluator (`evaluator/guardrails.py`)
**Status: Stub**

Response quality and safety validation:
- Content safety checks
- Response length validation (implemented)
- Relevance checking
- PII exposure detection

### response (`response/output.py`)
**Status: Stub**

Output formatting and delivery:
- Multi-channel support (email, ticket system, Slack, API)
- Template-based formatting
- HTML/markdown conversion

### db_layer (`db_layer/mongo.py`)
**Status: Stub**

MongoDB persistence layer:
- Ticket storage
- Response logging
- Analytics events

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd chuckbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from user_input.main import process_ticket, format_analysis

# Define your document keys (topics to match against)
document_keys = [
    "login failure",
    "authentication problem",
    "refund process",
    "deployment issues"
]

# Process a ticket
ticket = {
    "id": "001",
    "title": "Account Access",
    "content": "Having issues with login and authentication."
}

analysis = process_ticket(ticket, document_keys)
print(format_analysis(analysis))
```

### Complex Multi-Intent Queries

```python
# Handle tickets with multiple topics
complex_ticket = {
    "id": "002",
    "title": "Multiple Issues",
    "content": "I can't login, need a refund, and my API integration is broken."
}

# Lower threshold to catch more matches
analysis = process_ticket(
    complex_ticket,
    document_keys,
    similarity_threshold=0.2
)

# Access all semantic matches
for match in analysis.semantic_matches:
    print(f"{match.document_key}: {match.similarity_score:.3f}")
```

### Running the Demo

```bash
python -m user_input.main
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_user_input.py -v
```

## Project Structure

```
chuckbot/
├── user_input/
│   └── main.py          # Ticket processing (implemented)
├── processing/
│   └── llm.py           # LLM integration (stub)
├── evaluator/
│   └── guardrails.py    # Response validation (partial)
├── response/
│   └── output.py        # Output formatting (stub)
├── db_layer/
│   └── mongo.py         # Database layer (stub)
├── tests/
│   └── test_user_input.py  # Unit tests
├── requirements.txt
└── README.md
```

## Configuration

### Logging

Configure logging in your application:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Semantic Matching Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `similarity_threshold` | 0.3 | Minimum cosine similarity for a match |
| `top_k` | 3 | Maximum number of matches to return |

## Development Roadmap

### Phase 1 (Current)
- [x] Keyword extraction
- [x] Semantic matching with Sentence Transformers
- [x] Error handling and logging
- [x] Unit tests for core pipeline

### Phase 2 (Planned)
- [ ] LLM integration for response generation
- [ ] MongoDB persistence
- [ ] Basic guardrails implementation

### Phase 3 (Planned)
- [ ] Multi-channel output delivery
- [ ] Advanced guardrails (PII, safety)
- [ ] Analytics dashboard

## License

MIT License
