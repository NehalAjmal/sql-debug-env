---
title: SQL Debug Environment
emoji: 🛠️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# SQL Debug Environment

An OpenEnv reinforcement learning environment where an AI agent debugs and fixes broken SQL queries. This models a real-world task that developers face daily — identifying and correcting SQL errors given a schema and a broken query.

## Environment Description

The agent receives a broken SQL query along with the database schema and a hint about what's wrong. It must return a corrected SQL query that executes successfully and produces the expected result. The environment uses an in-memory SQLite database so no external database setup is needed.

## Motivation

SQL debugging is a task every software engineer and data analyst encounters regularly. Existing RL benchmarks focus heavily on games and synthetic puzzles, leaving a gap for environments grounded in real developer workflows. This environment fills that gap — it provides a structured, reproducible way to train and evaluate LLM agents on a skill that has immediate practical value. The programmatic grader means evaluation is fully automated with no human labeling required, making it ideal for large-scale agent training pipelines.

## Action Space
```json
{
  "fixed_query": "SELECT name, age FROM employees;"
}
```

| Field | Type | Description |
|---|---|---|
| `fixed_query` | string | The agent's corrected SQL query |

## Observation Space

| Field | Type | Description |
|---|---|---|
| `task_id` | string | Current task identifier |
| `broken_query` | string | The broken SQL query to fix |
| `db_schema` | string | Database schema as CREATE TABLE statements |
| `error_hint` | string | Hint about what's wrong |
| `task_description` | string | Natural language description of the task |
| `difficulty` | string | Task difficulty: easy / medium / hard / expert |
| `score` | float | Score for the last submitted query (0.0–1.0) |
| `feedback` | string | Feedback on the submitted query |
| `attempt` | int | Current attempt number |
| `max_attempts` | int | Maximum attempts allowed (3) |
| `reward` | float | Reward signal for this step |
| `done` | bool | Whether the episode is complete |

## Tasks

### Task 1 — Easy: Syntax Error Fix
The agent fixes a SELECT query with two keyword typos (`SELEC` and `FORM`).
Expected difficulty: any competent LLM should score 1.0.

### Task 2 — Medium: Wrong JOIN Condition
The agent fixes a JOIN query where the ON condition matches the wrong columns (`employees.id` instead of `employees.dept_id`).
Expected difficulty: requires understanding of relational schema structure.

### Task 3 — Hard: GROUP BY with Wrong HAVING Threshold
The agent fixes a GROUP BY query that finds departments with high earners, but the HAVING clause uses the wrong comparison operator and threshold.
Expected difficulty: requires understanding of aggregation filtering logic.

### Task 4 — Expert: Broken Window Functions
The agent fixes two bugs in a window function query — a wrong ORDER BY direction in RANK() and a wrong PARTITION BY column in MAX().
Expected difficulty: requires deep understanding of SQL window functions.

## Reward Function

- **1.0** — Query executes and returns exactly the expected result
- **0.3–0.8** — Partial credit for queries returning some correct rows
- **0.2–0.25** — Query executes but result partially or fully mismatches
- **0.1** — Query executes but returns no rows
- **0.0** — Query fails to execute (syntax/runtime error)
- **Penalty** — Each additional attempt after the first reduces reward by 0.05

## Baseline Scores

| Task | Difficulty | Model | Score |
|---|---|---|---|
| task_easy | Easy | llama-3.3-70b-versatile | 1.0 |
| task_medium | Medium | llama-3.3-70b-versatile | 1.0 |
| task_hard | Hard | llama-3.3-70b-versatile | 1.0 |
| task_expert | Expert | llama-3.3-70b-versatile | 1.0 |
| **Overall** | | | **1.0** |

## Setup & Usage

### Run locally with Docker
```bash
docker build -t sql-debug-env:latest .
docker run -p 7860:7860 -e API_BASE_URL="https://api.groq.com/openai/v1" -e MODEL_NAME="llama-3.3-70b-versatile" -e HF_TOKEN="your_key" sql-debug-env:latest
```

### Run baseline script
```bash
pip install openenv-core openai requests
export GROQ_API_KEY="your_key"
python baseline.py
```

### Run inference script
```bash
pip install openenv-core openai requests
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export API_KEY="your_key"
python inference.py --base-url https://nehubaby-sql-debug-env.hf.space
```

### Run against deployed Space
```bash
python inference.py --base-url https://nehubaby-sql-debug-env.hf.space
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/reset` | POST | Start a new episode |
| `/step` | POST | Submit a fixed query |
| `/state` | GET | Get current episode state |
| `/tasks` | GET | List all tasks and action schema |
| `/grader` | POST | Score a query against a specific task |
| `/baseline` | POST | Run baseline agent across all tasks |

## Example Interaction
```python
import requests

base = "http://localhost:7860"

# Start episode
obs = requests.post(f"{base}/reset").json()
print(obs["observation"]["broken_query"])
# SELEC name, age FORM employees;

# Submit fix
result = requests.post(f"{base}/step", json={
    "action": {"fixed_query": "SELECT name, age FROM employees;"}
}).json()
print(result["observation"]["score"])  # 1.0
print(result["observation"]["feedback"])  # Perfect!
```