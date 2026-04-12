"""
Baseline inference script for SQL Data Detective Environment.

Runs a multi-turn baseline LLM agent (Groq or OpenAI) against all 4 tasks.
The agent explores the database via SQL queries, then submits a final answer.

Usage:
    export GROQ_API_KEY="your_key"      # recommended (free)
    export OPENAI_API_KEY="your_key"    # alternative
    python3.11 baseline.py

    # Against a deployed HF Space:
    python3.11 baseline.py --base-url https://your-space.hf.space
"""

import argparse
import json
import os
import sys

import requests
from openai import OpenAI


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://nehubaby-sql-debug-env.hf.space"
MAX_EXPLORE_STEPS = 6


def get_llm_client():
    """Return an OpenAI-compatible client (Groq preferred, OpenAI fallback)."""
    groq_key = os.environ.get("GROQ_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if groq_key:
        print("Using Groq (llama-3.3-70b-versatile)")
        return OpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
        ), "llama-3.3-70b-versatile"
    elif openai_key:
        print("Using OpenAI (gpt-4o-mini)")
        return OpenAI(api_key=openai_key), "gpt-4o-mini"
    else:
        print("ERROR: Set GROQ_API_KEY or OPENAI_API_KEY")
        sys.exit(1)


def run_task_multiturn(client, model: str, task: dict, base_url: str, session_id: str) -> dict:
    """
    Run a single task using multi-turn SQL exploration.

    The agent:
    1. Reads the question and schema
    2. Runs SQL queries to explore the database
    3. Submits a final answer when confident
    """
    db_schema = task.get("db_schema", "")
    question = task.get("question", "")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a data analyst investigating a company database. "
                "You receive a business question and must answer it by exploring the database.\n\n"
                "At each turn you MUST respond with EXACTLY one of these two formats:\n"
                "1. To run a SQL query:   SQL: <your query>\n"
                "2. To submit your final answer:   ANSWER: <your answer>\n\n"
                "Rules:\n"
                "- Start by exploring relevant tables to understand the data\n"
                "- Build up your understanding step by step\n"
                "- When confident, submit your answer\n"
                "- Your answer should be concise and include the key values asked for\n"
                "- Do NOT include any other text outside the SQL: or ANSWER: format"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Business Question: {question}\n\n"
                f"Database Schema:\n{db_schema}\n\n"
                "Explore the database with SQL queries, then submit your answer."
            ),
        },
    ]

    queries_run = 0
    query_log = []
    final_answer = ""

    for step in range(MAX_EXPLORE_STEPS + 1):
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

            # Execute query via grader endpoint or directly
            # For baseline, we run queries directly against the server
            try:
                step_resp = requests.post(
                    f"{base_url}/env/step",
                    json={"session_id": session_id, "action": {"action_type": "query", "sql": sql, "answer": ""}},
                    timeout=30,
                )
                step_data = step_resp.json()
                obs = step_data.get("observation", step_data)
                query_result = obs.get("query_result", "(no result)")
                query_error = obs.get("query_error", "")

                if query_error:
                    result_msg = f"Query error: {query_error}"
                else:
                    result_msg = f"Query result:\n{query_result}"
            except Exception as e:
                result_msg = f"Query execution failed: {str(e)}"

            query_log.append({"sql": sql, "result": result_msg[:200]})
            messages.append({"role": "user", "content": result_msg})

        else:
            # LLM didn't follow format — treat as answer
            final_answer = reply
            break

    if not final_answer:
        final_answer = "(no answer submitted)"

    # Grade the answer
    grade_resp = requests.post(
        f"{base_url}/grader",
        json={"task_id": task["task_id"], "answer": final_answer},
        timeout=30,
    )
    grade = grade_resp.json()

    return {
        "task_id": task["task_id"],
        "difficulty": task["difficulty"],
        "question": question,
        "queries_run": queries_run,
        "query_log": query_log,
        "answer": final_answer,
        "score": grade.get("score", 0.0),
        "feedback": grade.get("feedback", ""),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_baseline(base_url: str):
    """Run the multi-turn baseline agent against all tasks and print results."""
    print(f"\n{'='*60}")
    print("SQL Data Detective — Multi-Turn Baseline Evaluation")
    print(f"Server: {base_url}")
    print(f"{'='*60}\n")

    client, model = get_llm_client()

    # Fetch tasks
    resp = requests.get(f"{base_url}/tasks")
    resp.raise_for_status()
    tasks = resp.json()["tasks"]
    print(f"Found {len(tasks)} tasks\n")

    results = []

    # Create a single stateful session for all tasks
    reset_resp = requests.post(f"{base_url}/env/reset", json={}, timeout=30)
    reset_resp.raise_for_status()
    session_id = reset_resp.json()["session_id"]
    print(f"Session: {session_id[:8]}...\n")

    for task in tasks:
        print(f"Task: {task['task_id']} [{task['difficulty'].upper()}]")
        print(f"  Question: {task['question'][:80]}...")

        result = run_task_multiturn(client, model, task, base_url, session_id)

        print(f"  Queries Run : {result['queries_run']}")
        print(f"  Answer      : {result['answer'][:80]}")
        print(f"  Score       : {result['score']}")
        print(f"  Feedback    : {result['feedback']}")
        print()

        results.append(result)

    # Summary
    total = sum(r["score"] for r in results) / len(results)
    print(f"{'='*60}")
    print(f"TOTAL BASELINE SCORE: {total:.4f}")
    print(f"Model: {model}")
    print(f"Avg queries per task: {sum(r['queries_run'] for r in results) / len(results):.1f}")
    print(f"{'='*60}\n")

    # Save results
    output = {
        "model": model,
        "base_url": base_url,
        "environment_type": "multi-turn SQL Data Detective",
        "total_score": round(total, 4),
        "results": results,
    }
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Results saved to baseline_results.json")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SQL Data Detective Baseline")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL of the environment server",
    )
    args = parser.parse_args()
    run_baseline(args.base_url)