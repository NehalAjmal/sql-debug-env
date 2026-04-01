# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the SQL Debug Environment.

Endpoints:
    - POST /reset       : Reset the environment
    - POST /step        : Execute an action
    - GET  /state       : Get current environment state
    - GET  /schema      : Get action/observation schemas
    - WS   /ws          : WebSocket endpoint
    - GET  /tasks       : List all tasks and action schema
    - POST /grader      : Score a query against a task
    - POST /baseline    : Run baseline agent across all 4 tasks
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
    from models import SqlDebugAction, SqlDebugObservation
    from server.sql_debug_env_environment import TASKS, SqlDebugEnvironment, _score_query
except ImportError:
    from sql_debug_env.models import SqlDebugAction, SqlDebugObservation
    from sql_debug_env.server.sql_debug_env_environment import TASKS, SqlDebugEnvironment, _score_query


# ---------------------------------------------------------------------------
# Base app from OpenEnv
# ---------------------------------------------------------------------------

app = create_app(
    SqlDebugEnvironment,
    SqlDebugAction,
    SqlDebugObservation,
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
# /tasks endpoint
# ---------------------------------------------------------------------------

@app.get("/tasks")
def get_tasks() -> dict[str, Any]:
    """Return all tasks and the action schema."""
    return {
        "tasks": [
            {
                "task_id": t["task_id"],
                "difficulty": t["difficulty"],
                "description": t["description"],
                "db_schema": t["db_schema"],
                "broken_query": t["broken_query"],
                "error_hint": t["error_hint"],
            }
            for t in TASKS
        ],
        "action_schema": {
            "type": "object",
            "properties": {
                "fixed_query": {
                    "type": "string",
                    "description": "The corrected SQL query",
                }
            },
            "required": ["fixed_query"],
        },
    }


# ---------------------------------------------------------------------------
# /grader endpoint
# ---------------------------------------------------------------------------

class GraderRequest(BaseModel):
    task_id: str
    fixed_query: str


@app.post("/grader")
def grade_query(request: GraderRequest) -> dict[str, Any]:
    """Score a fixed query against a specific task."""
    task = next((t for t in TASKS if t["task_id"] == request.task_id), None)
    if task is None:
        return {
            "error": f"Task '{request.task_id}' not found.",
            "valid_task_ids": [t["task_id"] for t in TASKS],
        }

    score, feedback = _score_query(task, request.fixed_query)
    return {
        "task_id": request.task_id,
        "difficulty": task["difficulty"],
        "score": score,
        "feedback": feedback,
        "fixed_query": request.fixed_query,
    }


# ---------------------------------------------------------------------------
# /baseline endpoint
# ---------------------------------------------------------------------------

def _run_baseline_for_task(task: dict) -> dict[str, Any]:
    """Run the Groq/OpenAI model on a single task and return score."""
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
            "fixed_query": "",
        }

    prompt = f"""You are an expert SQL debugger.

Database Schema:
{task['db_schema']}

Task: {task['description']}

Broken SQL Query:
{task['broken_query']}

Hint: {task['error_hint']}

Return ONLY the corrected SQL query with no explanation, no markdown, no backticks."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.0,
        )
        fixed_query = response.choices[0].message.content.strip()
        score, feedback = _score_query(task, fixed_query)
        return {
            "task_id": task["task_id"],
            "difficulty": task["difficulty"],
            "score": score,
            "feedback": feedback,
            "fixed_query": fixed_query,
        }
    except Exception as e:
        return {
            "task_id": task["task_id"],
            "difficulty": task["difficulty"],
            "score": 0.0,
            "feedback": f"API error: {str(e)}",
            "fixed_query": "",
        }


@app.post("/baseline")
def run_baseline() -> dict[str, Any]:
    """Run the baseline agent across all 4 tasks and return scores."""
    results = [_run_baseline_for_task(task) for task in TASKS]
    total_score = sum(r["score"] for r in results) / len(results)
    return {
        "baseline_model": "llama-3.3-70b-versatile (Groq) or gpt-4o-mini (OpenAI)",
        "total_score": round(total_score, 4),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)