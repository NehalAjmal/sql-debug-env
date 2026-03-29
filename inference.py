"""
Inference Script for SQL Debug Environment
===================================
MANDATORY environment variables:
  API_BASE_URL  - The API endpoint for the LLM (e.g. https://router.huggingface.co/v1)
  MODEL_NAME    - The model identifier to use for inference
  HF_TOKEN      - Your Hugging Face / API key

Usage:
    export API_BASE_URL="https://api.groq.com/openai/v1"
    export MODEL_NAME="llama-3.3-70b-versatile"
    export HF_TOKEN="your_groq_or_hf_token"
    python inference.py

    # Against a deployed HF Space:
    python inference.py --base-url https://nehubaby-sql-debug-env.hf.space
"""

import argparse
import json
import os
import sys

import requests
from openai import OpenAI


# ---------------------------------------------------------------------------
# Mandatory environment variables (per hackathon spec)
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")

DEFAULT_BASE_URL = "http://localhost:7860"

MAX_STEPS = 3       # max attempts per task
TEMPERATURE = 0.0   # deterministic output
MAX_TOKENS = 300    # well within memory limits


def get_llm_client() -> OpenAI:
    """Return an OpenAI-compatible client using mandatory env variables."""
    if not HF_TOKEN:
        print("ERROR: Set HF_TOKEN (or GROQ_API_KEY / OPENAI_API_KEY)")
        sys.exit(1)

    print(f"Using model  : {MODEL_NAME}")
    print(f"API base URL : {API_BASE_URL}")

    return OpenAI(
        api_key=HF_TOKEN,
        base_url=API_BASE_URL,
    )


def fix_query_with_llm(client: OpenAI, task: dict) -> str:
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
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


def run_inference(base_url: str) -> dict:
    """Run the inference agent against all tasks and return results."""
    print(f"\n{'='*60}")
    print("SQL Debug Environment — Inference Evaluation")
    print(f"Server : {base_url}")
    print(f"{'='*60}\n")

    client = get_llm_client()

    # Fetch all tasks from the environment
    resp = requests.get(f"{base_url}/tasks", timeout=30)
    resp.raise_for_status()
    tasks = resp.json()["tasks"]
    print(f"Found {len(tasks)} tasks\n")

    results = []

    for task in tasks:
        print(f"Task: {task['task_id']} [{task['difficulty'].upper()}]")
        print(f"  Description : {task['description']}")
        print(f"  Broken Query: {task['broken_query']}")

        # Get LLM fix
        fixed_query = fix_query_with_llm(client, task)
        print(f"  Fixed Query : {fixed_query}")

        # Grade it via /grader endpoint
        grade_resp = requests.post(
            f"{base_url}/grader",
            json={"task_id": task["task_id"], "fixed_query": fixed_query},
            timeout=30,
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
    print(f"TOTAL SCORE : {total:.4f}")
    print(f"MODEL       : {MODEL_NAME}")
    print(f"{'='*60}\n")

    output = {
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "base_url": base_url,
        "total_score": round(total, 4),
        "results": results,
    }

    with open("inference_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Results saved to inference_results.json")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SQL Debug Env Inference")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL of the environment server",
    )
    args = parser.parse_args()
    run_inference(args.base_url)