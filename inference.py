"""
Inference Script for SQL Debug Environment
===================================
MANDATORY environment variables:
  API_BASE_URL  The API endpoint for the LLM.
  MODEL_NAME    The model identifier to use for inference.
  HF_TOKEN      Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
"""

import argparse
import json
import os
import sys

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Mandatory env variables (per hackathon spec)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_STEPS = 3
TEMPERATURE = 0.0
MAX_TOKENS = 300
DEFAULT_BASE_URL = "http://localhost:7860"


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
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    return response.choices[0].message.content or ""


def main(base_url: str = DEFAULT_BASE_URL) -> None:
    if not API_KEY:
        print("ERROR: Set HF_TOKEN or API_KEY environment variable.")
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"\n{'='*60}")
    print("SQL Debug Environment — Inference Evaluation")
    print(f"Server    : {base_url}")
    print(f"Model     : {MODEL_NAME}")
    print(f"API URL   : {API_BASE_URL}")
    print(f"{'='*60}\n")

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

        # Get LLM fix (up to MAX_STEPS attempts)
        fixed_query = ""
        score = 0.0
        feedback = ""

        for step in range(1, MAX_STEPS + 1):
            try:
                fixed_query = fix_query_with_llm(client, task)
            except Exception as exc:
                print(f"  Model request failed ({exc}). Skipping.")
                break

            print(f"  Step {step} Fixed Query: {fixed_query}")

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

            if score == 1.0:
                print("  ✓ Solved!")
                break

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SQL Debug Env Inference")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL of the environment server",
    )
    args = parser.parse_args()
    main(args.base_url)