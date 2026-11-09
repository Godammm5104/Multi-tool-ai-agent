# Multi-Tool AI Agent

A small tool-using agent built with Python and the IO Intelligence API. Instead of only chatting, the agent can decide when to call local tools, read the results, and then produce a final answer.

## What it does

- Function calling loop with a hard step limit
- Four deterministic tools: summarize, extract keywords, generate ideas, and count words
- Tool trace so every decision is inspectable
- Graceful handling of invalid arguments and tool failures
- Small eval harness for checking that tool selection works

The tools are deterministic on purpose. The LLM handles language and tool selection; Python handles the actual text processing. That makes the behaviour easier to test and explain in an interview.

## Stack

- Python 3.10+
- OpenAI Python SDK
- IO Intelligence API
- python-dotenv
- Rich

## Setup

### Windows

```bat
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your IO Intelligence API key:

```text
IO_API_KEY=your_key_here
```

## Run

```bash
python main.py
```

Show the tool trace after every answer:

```bash
python main.py --show-trace
```

## Test

```bash
pytest
```

Tests use fake LLM responses, so they do not call the API or need a key.

## Eval

With a real API key set:

```bash
python eval.py
```

The eval checks whether the agent calls the expected tool for four clearly scoped tasks and whether any tool call errored.

## Tools

| Tool | Purpose |
| --- | --- |
| `summarize_text` | Extractive summary using word-frequency scoring |
| `extract_keywords` | Most frequent meaningful words |
| `generate_ideas` | Practical starter ideas for a topic |
| `word_count` | Words, characters, and sentence count |

## Project structure

```text
agent.py        tool-calling loop and retry handling
tools.py        deterministic implementations and JSON schemas
main.py         interactive CLI
eval.py         small tool-selection eval
tests/          unit tests with fake LLM responses
```

## Limitations

- Tool loop is capped at 4 steps by default.
- No persistent memory between runs.
- The tools are intentionally simple; this project demonstrates the agent loop, not production NLP.
- Tool arguments are validated by the tool functions themselves, not by a separate schema validator.

## CV bullet

```text
Multi-Tool AI Agent — Python, IO Intelligence API, function calling
Built a bounded tool-using agent that selects and invokes Python tools, records an inspectable trace, handles malformed tool calls, and includes an eval harness for tool selection.
```
