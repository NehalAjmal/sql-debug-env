"""
Baseline inference script for SQL Debug Environment.

Runs a baseline LLM agent (Groq or OpenAI) against all 4 tasks.

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

DEFAULT_BASE_URL = "http://localhost:8000"


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


def fix_query_with_llm(client, model: str, task: dict) -> str:
    """Ask the LLM to fix the broken SQL query."""
    prompt = f"""You are an expert SQL debugger.

Database Schema:
{task['db_schema']}

Task: {task['description']}

Broken SQL Query:
{task['broken_query']}

Hint: {task['error_hint']}

Return ONLY the corrected SQL query with no explanation, no markdown, no backticks."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_baseline(base_url: str):
    """Run the baseline agent against all tasks and print results."""
    print(f"\n{'='*60}")
    print("SQL Debug Environment — Baseline Evaluation")
    print(f"Server: {base_url}")
    print(f"{'='*60}\n")

    client, model = get_llm_client()

    # Fetch tasks
    resp = requests.get(f"{base_url}/tasks")
    resp.raise_for_status()
    tasks = resp.json()["tasks"]
    print(f"Found {len(tasks)} tasks\n")

    results = []

    for task in tasks:
        print(f"Task: {task['task_id']} [{task['difficulty'].upper()}]")
        print(f"  Description : {task['description']}")
        print(f"  Broken Query: {task['broken_query']}")

        # Get LLM fix
        fixed_query = fix_query_with_llm(client, model, task)
        print(f"  Fixed Query : {fixed_query}")

        # Grade it
        grade_resp = requests.post(
            f"{base_url}/grader",
            json={"task_id": task["task_id"], "fixed_query": fixed_query},
        )
        grade_resp.raise_for_status()
        grade = grade_resp.json()

        score = grade["score"]
        feedback = grade["feedback"]
        print(f"  Score       : {score}")
        print(f"  Feedback    : {feedback}")
        print()

        results.append({
            "task_id": task["task_id"],
            "difficulty": task["difficulty"],
            "fixed_query": fixed_query,
            "score": score,
            "feedback": feedback,
        })

    # Summary
    total = sum(r["score"] for r in results) / len(results)
    print(f"{'='*60}")
    print(f"TOTAL BASELINE SCORE: {total:.4f}")
    print(f"Model: {model}")
    print(f"{'='*60}\n")

    # Save results
    output = {
        "model": model,
        "base_url": base_url,
        "total_score": round(total, 4),
        "results": results,
    }
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Results saved to baseline_results.json")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SQL Debug Env Baseline")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL of the environment server",
    )
    args = parser.parse_args()
    run_baseline(args.base_url)