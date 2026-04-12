# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the SQL Data Detective Environment.

Endpoints:
    - POST /reset       : Reset the environment (start new episode)
    - POST /step        : Execute an action (query or answer)
    - GET  /state       : Get current environment state
    - GET  /schema      : Get action/observation schemas
    - WS   /ws          : WebSocket endpoint
    - GET  /tasks       : List all tasks and action schema
    - POST /grader      : Grade a submitted answer for a task
    - POST /baseline    : Run baseline multi-turn agent across all tasks
"""

import os
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Run: uv sync") from e

try:
    from models import SqlDetectiveAction, SqlDetectiveObservation
    from server.sql_debug_env_environment import (
        TASKS,
        SqlDebugEnvironment,
        _grade_answer,
        _run_query,
        _format_result,
        DB_SCHEMA,
    )
except ImportError:
    from sql_debug_env.models import SqlDetectiveAction, SqlDetectiveObservation
    from sql_debug_env.server.sql_debug_env_environment import (
        TASKS,
        SqlDebugEnvironment,
        _grade_answer,
        _run_query,
        _format_result,
        DB_SCHEMA,
    )


# ---------------------------------------------------------------------------
# Base app from OpenEnv
# ---------------------------------------------------------------------------

app = create_app(
    SqlDebugEnvironment,
    SqlDetectiveAction,
    SqlDetectiveObservation,
    env_name="sql_debug_env",
    max_concurrent_envs=10,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Stateful session management
# ---------------------------------------------------------------------------
# The OpenEnv framework's /reset and /step HTTP endpoints are stateless —
# they create a NEW environment instance per request and destroy it after.
# For multi-turn interaction, we need persistent state across calls.
# These /env/* endpoints manage session-based environment instances.

import uuid
from threading import Lock

_sessions: dict[str, SqlDebugEnvironment] = {}
_session_lock = Lock()


class EnvResetRequest(BaseModel):
    session_id: str = ""


class EnvStepRequest(BaseModel):
    session_id: str
    action: dict


@app.post("/env/reset")
def env_reset(request: EnvResetRequest = None) -> dict[str, Any]:
    """Reset the environment and return a new session with initial observation."""
    if request and request.session_id:
        # Reuse existing session ID (allows re-reset)
        session_id = request.session_id
    else:
        session_id = str(uuid.uuid4())

    env = SqlDebugEnvironment()
    obs = env.reset()

    with _session_lock:
        _sessions[session_id] = env

    return {
        "session_id": session_id,
        "observation": obs.model_dump(),
        "reward": 0.0,
        "done": False,
    }


@app.post("/env/step")
def env_step(request: EnvStepRequest) -> dict[str, Any]:
    """Execute an action in an existing session's environment."""
    with _session_lock:
        env = _sessions.get(request.session_id)

    if env is None:
        return {"error": f"Session '{request.session_id}' not found. Call /env/reset first."}

    action_data = request.action
    action = SqlDetectiveAction(**action_data)
    obs = env.step(action)

    return {
        "session_id": request.session_id,
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }


@app.get("/env/state/{session_id}")
def env_state(session_id: str) -> dict[str, Any]:
    """Get the current state of a session's environment."""
    with _session_lock:
        env = _sessions.get(session_id)

    if env is None:
        return {"error": f"Session '{session_id}' not found."}

    return {
        "session_id": session_id,
        "state": env.state.model_dump() if hasattr(env.state, "model_dump") else {"episode_id": env.state.episode_id, "step_count": env.state.step_count},
    }


# ---------------------------------------------------------------------------
# /tasks endpoint
# ---------------------------------------------------------------------------

@app.get("/tasks")
def get_tasks() -> dict[str, Any]:
    """Return all tasks and the action schema for the multi-turn environment."""
    return {
        "tasks": [
            {
                "task_id": t["task_id"],
                "difficulty": t["difficulty"],
                "question": t["question"],
                "db_schema": DB_SCHEMA,
                "min_queries": t["min_queries"],
            }
            for t in TASKS
        ],
        "action_schema": {
            "type": "object",
            "properties": {
                "action_type": {
                    "type": "string",
                    "enum": ["query", "answer"],
                    "description": "Action type: 'query' to execute SQL, 'answer' to submit final answer",
                },
                "sql": {
                    "type": "string",
                    "description": "SQL query to execute (when action_type is 'query')",
                },
                "answer": {
                    "type": "string",
                    "description": "Final answer to the business question (when action_type is 'answer')",
                },
            },
            "required": ["action_type"],
        },
    }


# ---------------------------------------------------------------------------
# /grader endpoint
# ---------------------------------------------------------------------------

class GraderRequest(BaseModel):
    task_id: str
    answer: str


@app.post("/grader")
def grade_answer(request: GraderRequest) -> dict[str, Any]:
    """Grade a submitted answer against a specific task."""
    task = next((t for t in TASKS if t["task_id"] == request.task_id), None)
    if task is None:
        return {
            "error": f"Task '{request.task_id}' not found.",
            "valid_task_ids": [t["task_id"] for t in TASKS],
        }

    score, feedback = _grade_answer(task, request.answer)
    return {
        "task_id": request.task_id,
        "difficulty": task["difficulty"],
        "question": task["question"],
        "score": score,
        "feedback": feedback,
        "submitted_answer": request.answer,
        "expected_answer": task["expected_answer"],
    }


# ---------------------------------------------------------------------------
# /baseline endpoint — multi-turn agent
# ---------------------------------------------------------------------------

def _run_baseline_for_task(task: dict) -> dict[str, Any]:
    """Run a multi-turn LLM agent on a single task and return the result."""
    from openai import OpenAI

    # Support both OpenAI and Groq (OpenAI-compatible)
    groq_key = os.environ.get("GROQ_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if groq_key:
        client = OpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
        )
        model = "llama-3.3-70b-versatile"
    elif openai_key:
        client = OpenAI(api_key=openai_key)
        model = "gpt-4o-mini"
    else:
        return {
            "task_id": task["task_id"],
            "difficulty": task["difficulty"],
            "score": 0.0,
            "feedback": "No API key found. Set GROQ_API_KEY or OPENAI_API_KEY.",
            "queries_run": 0,
            "answer": "",
        }

    max_queries = 6
    messages = [
        {
            "role": "system",
            "content": (
                "You are a data analyst investigating a company database. "
                "You will be given a business question. Explore the database by writing SQL queries. "
                "When you have enough information, provide your final answer.\n\n"
                "IMPORTANT: Respond with EXACTLY one of these two formats:\n"
                "1. To run a SQL query: SQL: <your query here>\n"
                "2. To submit your answer: ANSWER: <your answer here>\n\n"
                "Do not include any other text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Business Question: {task['question']}\n\n"
                f"Database Schema:\n{DB_SCHEMA}\n\n"
                "Start by exploring the relevant tables, then submit your answer."
            ),
        },
    ]

    queries_run = 0
    final_answer = ""

    try:
        for step in range(max_queries + 1):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=500,
                temperature=0.0,
            )
            reply = response.choices[0].message.content.strip()
            messages.append({"role": "assistant", "content": reply})

            if reply.upper().startswith("ANSWER:"):
                final_answer = reply[7:].strip()
                break
            elif reply.upper().startswith("SQL:"):
                sql = reply[4:].strip()
                queries_run += 1
                rows, cols, error = _run_query(sql)
                if error:
                    result_msg = f"Query error: {error}"
                else:
                    result_msg = _format_result(rows, cols)
                messages.append({"role": "user", "content": f"Query result:\n{result_msg}"})
            else:
                # LLM didn't follow format, try to extract answer
                final_answer = reply
                break

        if not final_answer:
            final_answer = "(no answer submitted)"

        score, feedback = _grade_answer(task, final_answer)
        return {
            "task_id": task["task_id"],
            "difficulty": task["difficulty"],
            "score": score,
            "feedback": feedback,
            "queries_run": queries_run,
            "answer": final_answer,
        }

    except Exception as e:
        return {
            "task_id": task["task_id"],
            "difficulty": task["difficulty"],
            "score": 0.0,
            "feedback": f"API error: {str(e)}",
            "queries_run": queries_run,
            "answer": "",
        }


@app.post("/baseline")
def run_baseline() -> dict[str, Any]:
    """Run the baseline multi-turn agent across all tasks and return scores."""
    results = [_run_baseline_for_task(task) for task in TASKS]
    total_score = sum(r["score"] for r in results) / len(results)
    return {
        "baseline_model": "llama-3.3-70b-versatile (Groq) or gpt-4o-mini (OpenAI)",
        "environment_type": "multi-turn SQL Data Detective",
        "total_score": round(total_score, 4),
        "results": results,
    }


@app.get("/")
def root():
    return {
        "message": "SQL Data Detective Environment is running",
        "type": "multi-turn RL environment",
        "description": (
            "Agent receives business questions, explores a database via SQL queries, "
            "and submits final answers. Genuinely multi-turn with state changes each step."
        ),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()