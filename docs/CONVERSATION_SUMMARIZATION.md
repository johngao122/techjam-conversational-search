# LLM-based Conversation Summarization

## Overview

The system uses DeepSeek API to automatically summarize conversation history after each turn. Summaries are stored in the ledger and cached across sessions by user ID, enabling the system to "remember" user preferences.

## Features

1. **Per-turn summarization**: After each `Agent.respond()` call, conversation is summarized
2. **Cross-session memory**: Summaries automatically transferred to new sessions with same `user_id`
3. **Structured output**: JSON-formatted summaries with preferences, topics, and intent
4. **Graceful degradation**: Works even if DeepSeek API unavailable (returns empty summary)

## Architecture

### Components

**LLMSummarizer** (`src/ledger/llm_summarizer.py`)
- OpenAI-compatible client wrapper for DeepSeek API
- Methods: `__init__()`, `summarize()`, `_format_history()`, `_parse_response()`
- Reads config from environment variables

**LedgerService** (`src/ledger/ledger.py`)
- Stores `conversation_summary` field per session
- Methods: `set_conversation_summary()`, `get_conversation_summary()`

**Agent** (`src/agent.py`)
- Calls summarizer after each turn
- Maintains `_summaries` dict for cross-session cache (keyed by user_id)
- Injects prior summary into new sessions in `reset()`

### Data Flow

```
User message
    ↓
Agent.respond()
    ↓
[1] Update history list
[2] Extract attributes
[3] ← Call LLMSummarizer here
[4] Store summary in ledger
[5] Store summary in cross-session cache (by user_id)
    ↓
New session with same user_id
    ↓
Agent.reset()
    ↓
Inject prior summary into new ledger
```

## Usage

### Basic Usage

```python
import os
from src.agent import Agent

# API key must be set
os.environ["API_KEY"] = "sk-..."

agent = Agent()  # Creates LLMSummarizer internally

# Start session
agent.reset("session_1", {"user_id": "alice"})

# Each turn summarizes automatically
result = agent.respond("session_1", "I want black leather boots under $80", turn=1)

# Access the summary
ledger = agent._ledger.read("session_1")
summary = ledger["conversation_summary"]
# {
#   "summary": "Customer seeking affordable black leather boots...",
#   "remembered_preferences": {"color": "black", "material": "leather", "budget": "80"},
#   "topics_covered": ["footwear", "pricing", "materials"],
#   "last_updated_turn": 1
# }
```

### Cross-Session Memory

```python
# Session 1: User provides preferences
agent.reset("session_1", {"user_id": "alice"})
agent.respond("session_1", "I want black leather boots under $80", turn=1)

# Later: Same user, new session
agent.reset("session_2", {"user_id": "alice"})
ledger2 = agent._ledger.read("session_2")
summary2 = ledger2["conversation_summary"]
# Automatically contains prior preferences!
```

### Direct Access

```python
# Get current session's summary
summary = agent._ledger.get_conversation_summary("session_1")

# Set custom summary
agent._ledger.set_conversation_summary("session_1", {
    "summary": "...",
    "remembered_preferences": {...},
    "topics_covered": [...],
    "last_updated_turn": 1
})
```

## Configuration

### Environment Variables

**Required:**
- `API_KEY` — DeepSeek API key. Read from `.env` file by default.

**Optional:**
- `DEEPSEEK_BASE_URL` — API endpoint. Defaults to `https://api.deepseek.com`
- `DEEPSEEK_MODEL` — Model name. Defaults to `deepseek-chat`

Example `.env`:
```
API_KEY=sk-ceb801ffcaef40b6b707911180723f1c
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

## Summary Structure

Each `conversation_summary` contains:

```python
{
    "summary": str,                              # One-line summary of user intent
    "remembered_preferences": dict[str, str],    # Key attributes (color, size, brand, etc.)
    "topics_covered": list[str],                 # Conversation topics discussed
    "last_updated_turn": int,                    # Turn when summary was last updated
}
```

Example:
```python
{
    "summary": "Looking for women's running shoes in size 9, Nike preferred, under $150",
    "remembered_preferences": {
        "category": "running shoes",
        "size": "9",
        "brand": "Nike",
        "budget": "150"
    },
    "topics_covered": ["athletic footwear", "brand preferences", "pricing", "sizing"],
    "last_updated_turn": 3
}
```

## Error Handling

The summarizer gracefully handles errors:

1. **Missing API_KEY**: Logs warning, returns empty summary
2. **Missing openai package**: Logs error, returns empty summary
3. **DeepSeek API call fails**: Logs error, returns empty summary with turn count

Example fallback:
```python
{
    "summary": "",
    "remembered_preferences": {},
    "topics_covered": [],
    "last_updated_turn": 0
}
```

## Testing

Run tests to verify integration:

```bash
# All tests (should pass 90/90)
python3 -m pytest tests/ -v

# Just ledger tests
python3 -m pytest tests/test_ledger.py -v
```

## Performance Considerations

- **API calls**: One DeepSeek API call per turn (after every message)
- **Latency**: ~200-500ms per call depending on history length
- **Token usage**: Depends on conversation length; ~100-300 tokens typical
- **Memory**: `_summaries` dict grows with unique user_ids (in-memory only)

## Limitations

- **No persistence**: Summaries lost on app restart (in-memory only)
- **User ID required**: Cross-session memory requires `user_id` in user_profile
- **Limited scope**: Currently only stores summary, not used in retrieval/ranking yet
- **No conversation history**: Full message history still stored only as raw text in history list

## Future Enhancements

1. **Database persistence**: Store summaries in PostgreSQL/Redis
2. **Use in retrieval**: Pass summary context to search/ranking functions
3. **Temporal memory**: Track summary evolution over time
4. **User preferences API**: Public endpoint to retrieve/update stored preferences
5. **Multi-language support**: Summarize conversations in user's native language

## Related Files

- `src/ledger/llm_summarizer.py` — LLM summarization logic
- `src/ledger/ledger.py` — Ledger storage (lines 32-37, 173-183)
- `src/agent.py` — Integration point (lines 28, 99-102, ~158-165, ~105-107)
- `.env` — API key configuration
