---
title: SQL Data Detective
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
---

# 🔍 SQL Data Detective: A Multi-Turn RL Sandbox

Welcome to **SQL Data Detective**—a genuinely stateful, multi-turn reinforcement learning environment built for the Meta PyTorch OpenEnv ecosystem.

Instead of a simple one-shot trivia bot, this environment forces an AI agent to act like a real-world Data Analyst. The agent receives a high-level business question and must sequentially explore an SQLite database by executing actual SQL queries. It observes table returns, encounters runtime errors, refines its logic, and only submits a final answer when it successfully gathers the data.

### ✨ Why This Stands Out
* **True Multi-Step Reasoning**: The agent experiences a learning curve. Each query reveals new information that shapes the very next action.
* **Stateful HTTP Sessions**: Fully compliant `/env/*` APIs preserve the interaction state seamlessly across HTTP calls.
* **Dense Reward Function**: Agents receive a continuous reward `[0.0 to 1.0]` that factors in objective correctness *and* query efficiency. Proximal Policy models thrive on this!
* **Scalable Complexity**: Built-in tasks progress from *Easy* (simple counts) to *Expert* (complex JOINs, aggregations, and derived metrics).

## Environment Description

The agent receives a natural language business question about a company database (employees, projects, sales, expenses). At each step, the agent either:
1. **Runs a SQL query** to explore the database and receives actual query results
2. **Submits a final answer** when confident enough

The environment uses an in-memory SQLite database — no external setup needed.

## Action Space

```json
{
  "action_type": "query",
  "sql": "SELECT department, COUNT(*) FROM employees GROUP BY department",
  "answer": ""
}
```

or

```json
{
  "action_type": "answer",
  "sql": "",
  "answer": "Engineering: 4, Marketing: 3, HR: 3"
}
```

| Field | Type | Description |
|---|---|---|
| `action_type` | string | `"query"` to run SQL, `"answer"` to submit final answer |
| `sql` | string | SQL query to execute (when `action_type` is `"query"`) |
| `answer` | string | Final answer text (when `action_type` is `"answer"`) |

## Observation Space

| Field | Type | Description |
|---|---|---|
| `task_id` | string | Current task identifier |
| `question` | string | Business question to answer |
| `db_schema` | string | Database schema (CREATE TABLE statements) |
| `last_action_type` | string | Type of last action taken |
| `last_sql` | string | Last SQL query executed |
| `query_result` | string | Formatted result of last SQL query |
| `query_error` | string | Error message if last query failed |
| `query_count` | int | Number of SQL queries executed so far |
| `max_steps` | int | Maximum steps allowed (8) |
| `step` | int | Current step number |
| `score` | float | Score achieved (0.0–1.0, set on answer submission) |
| `feedback` | string | Feedback on the last action |
| `difficulty` | string | Task difficulty: easy / medium / hard / expert |
| `answer_submitted` | string | The submitted answer text |
| `reward` | float | Reward signal incorporating correctness + efficiency |
| `done` | bool | Whether the episode is complete |

## Database Schema

```sql
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT NOT NULL,
    salary REAL NOT NULL,
    hire_year INTEGER NOT NULL,
    manager_id INTEGER
);

CREATE TABLE projects (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT NOT NULL,
    budget REAL NOT NULL,
    status TEXT NOT NULL  -- 'active' or 'completed'
);

CREATE TABLE sales (
    id INTEGER PRIMARY KEY,
    employee_id INTEGER NOT NULL,
    product TEXT NOT NULL,
    amount REAL NOT NULL,
    sale_date TEXT NOT NULL,
    region TEXT NOT NULL
);

CREATE TABLE expenses (
    id INTEGER PRIMARY KEY,
    department TEXT NOT NULL,
    category TEXT NOT NULL,
    amount REAL NOT NULL,
    expense_date TEXT NOT NULL
);
```

## Tasks

### Task 1 — Easy: Department Employee Count
Count employees per department, ordered by headcount. Requires 1+ queries.
A warm-up task that any competent agent should solve in 1–2 queries.

### Task 2 — Medium: Top Sales in Q1 2024
Find the employee with highest total sales revenue in January–March 2024.
Requires joining employees and sales tables with date filtering. 2+ queries expected.

### Task 3 — Hard: Budget-to-Salary Ratio
For departments with active projects, calculate the ratio of total project budget to total salary.
Requires combining data from employees, projects tables with filtering and aggregation. 3+ queries expected.

### Task 4 — Expert: Sales Efficiency Score
Find the employee with the best "efficiency score" (total sales / salary) among those with 2+ sales.
Requires joining employees and sales, filtering, and computing derived metrics. 3+ queries expected.

## Reward Function

The reward combines **correctness** and **query efficiency**:

- **Correctness (70% weight)**:
  - `0.99` — All key values present in the answer
  - `0.5–0.7` — Partial credit for some correct values
  - `0.8` — Answer contains right info but in unexpected format
  - `0.01` — Incorrect answer

- **Efficiency bonus (30% weight)**:
  - Using `min_queries` or fewer: full 30% bonus
  - Each extra query beyond minimum: –8% penalty on bonus
  - Running out of steps without answering: 0.0 reward

- **Formula**: `reward = score × (0.7 + 0.3 × efficiency)`

## Setup & Usage

### Run locally
```bash
uv sync
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run with Docker
```bash
docker build -t sql-detective:latest .
docker run -p 7860:7860 sql-detective:latest
```

### Run inference (multi-turn agent)
```bash
export HF_TOKEN="your_key"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
python inference.py --base-url http://localhost:7860
```



## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Environment info |
| `/health` | GET | Health check |
| `/reset` | POST | Start a stateless episode (framework built-in) |
| `/step` | POST | Submit action to a stateless episode (framework built-in) |
| `/state` | GET | Get stateless episode state |
| `/env/reset` | POST | **Stateful:** Start a new multi-turn session (returns `session_id`) |
| `/env/step` | POST | **Stateful:** Submit action using `{"session_id": "...", "action": {...}}` |
| `/env/state/{id}` | GET | **Stateful:** Get current state of a specific session |
| `/tasks` | GET | List all tasks and action schema |
| `/grader` | POST | Score an answer against a specific task |

## Example Multi-Turn Interaction

```python
import requests

base = "http://localhost:7860"

# Start stateful episode
res = requests.post(f"{base}/env/reset", json={}).json()
session_id = res["session_id"]
obs = res["observation"]
print(obs["question"])
# "How many employees are in each department? ..."

# Step 1: Explore
result = requests.post(f"{base}/env/step", json={
    "session_id": session_id,
    "action": {
        "action_type": "query",
        "sql": "SELECT department, COUNT(*) as cnt FROM employees GROUP BY department ORDER BY cnt DESC",
        "answer": ""
    }
}).json()
print(result["observation"]["query_result"])
# Engineering | 4
# Marketing | 3
# HR | 3

# Step 2: Submit answer
result = requests.post(f"{base}/env/step", json={
    "session_id": session_id,
    "action": {
        "action_type": "answer",
        "sql": "",
        "answer": "Engineering: 4, Marketing: 3, HR: 3"
    }
}).json()
print(result["observation"]["score"])   # 1.0
print(result["observation"]["reward"])  # 1.0
# Environment automatically advances to next task
```