"""
Inference Script for SQL Debug Environment
===================================
MANDATORY environment variables:
  API_BASE_URL  The API endpoint for the LLM.
  MODEL_NAME    The model identifier to use for inference.
  HF_TOKEN      Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory
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
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_STEPS = 3        # max attempts per task
TEMPERATURE = 0.0    # deterministic for reproducibility
MAX_TOKENS = 300
DEFAULT_BASE_URL = "http://localhost:7860"


def get_llm_response(client: OpenAI, task: dict, feedback: str = "", attempt: int = 0) -> str:
    """
    Given a task observation, ask the LLM to fix the SQL query.
    Includes previous feedback on retries.
    """
    feedback_section = f"\nPrevious attempt feedback: {feedback}" if feedback and attempt > 0 else ""

    prompt = f"""You are an expert SQL debugger.

Database Schema:
{task.get('db_schema', '')}

Task: {task.get('description', '')}

Broken SQL Query:
{task.get('broken_query', '')}

Hint: {task.get('error_hint', '')}{feedback_section}

Return ONLY the corrected SQL query with no explanation, no markdown, no backticks."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    return response.choices[0].message.content.strip() or ""


def run_episode(base_url: str, client: OpenAI) -> dict:
    """
    Run a full RL episode across all tasks:
      - Fetch tasks from /tasks
      - For each task: attempt up to MAX_STEPS fixes using the LLM
      - Score each attempt via /grader
      - Track rewards and results
    """
    # Get all tasks
    tasks_resp = requests.get(f"{base_url}/tasks", timeout=30)
    tasks_resp.raise_for_status()
    tasks = tasks_resp.json()["tasks"]

    print(f"\n  Found {len(tasks)} tasks in environment")

    total_reward = 0.0
    total_steps = 0
    task_results = []

    for task in tasks:
        task_id = task["task_id"]
        difficulty = task["difficulty"]
        print(f"\n  Task: {task_id} [{difficulty.upper()}]")
        print(f"  Broken: {task['broken_query']}")

        best_score = 0.0
        best_query = ""
        best_feedback = ""
        feedback = ""

        for attempt in range(1, MAX_STEPS + 1):
            # LLM generates a fix
            fixed_query = get_llm_response(client, task, feedback, attempt)
            print(f"  Attempt {attempt}: {fixed_query[:80]}...")

            # Grade the fix via /grader
            grade_resp = requests.post(
                f"{base_url}/grader",
                json={"task_id": task_id, "fixed_query": fixed_query},
                timeout=30,
            )
            grade_resp.raise_for_status()
            grade = grade_resp.json()

            score = grade["score"]
            feedback = grade["feedback"]

            # Reward with attempt penalty (matches environment reward logic)
            attempt_penalty = (attempt - 1) * 0.05
            reward = max(0.0, score - attempt_penalty)
            total_reward += reward
            total_steps += 1

            print(f"           Score: {score} | Reward: {reward:.2f} | {feedback[:60]}")

            if score > best_score:
                best_score = score
                best_query = fixed_query
                best_feedback = feedback

            # Stop early if solved
            if score == 1.0:
                print(f"           ✓ Solved in {attempt} attempt(s)!")
                break

        task_results.append({
            "task_id": task_id,
            "difficulty": difficulty,
            "fixed_query": best_query,
            "score": best_score,
            "feedback": best_feedback,
            "attempts": attempt,
        })

    return {
        "total_reward": total_reward,
        "steps": total_steps,
        "task_results": task_results,
    }


def main(base_url: str = DEFAULT_BASE_URL) -> None:
    if not API_KEY:
        print("ERROR: Set HF_TOKEN or API_KEY environment variable.")
        sys.exit(1)

    # Initialize OpenAI-compatible client with mandatory env variables
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"\n{'='*60}")
    print("SQL Debug Environment — Inference Evaluation")
    print(f"Server    : {base_url}")
    print(f"Model     : {MODEL_NAME}")
    print(f"API URL   : {API_BASE_URL}")
    print(f"Max Steps : {MAX_STEPS} per task")
    print(f"{'='*60}")

    # Verify environment is healthy
    health = requests.get(f"{base_url}/health", timeout=10)
    health.raise_for_status()
    print(f"\nEnvironment health: {health.json()}")

    # Run the RL episode
    print("\nStarting RL episode...")
    episode = run_episode(base_url, client)

    # Summary
    task_results = episode["task_results"]
    total_score = sum(r["score"] for r in task_results) / len(task_results) if task_results else 0.0

    print(f"\n{'='*60}")
    print("EPISODE COMPLETE")
    print(f"{'='*60}")
    print(f"Total reward : {episode['total_reward']:.4f}")
    print(f"Total steps  : {episode['steps']}")
    print(f"Tasks solved : {sum(1 for r in task_results if r['score'] == 1.0)}/{len(task_results)}")
    print(f"Mean score   : {total_score:.4f}")
    print(f"Model        : {MODEL_NAME}")
    print(f"{'='*60}\n")

    # Per-task breakdown
    for r in task_results:
        status = "✓" if r["score"] == 1.0 else "✗"
        print(f"  {status} {r['task_id']:15s} [{r['difficulty']:6s}] score={r['score']} attempts={r['attempts']}")

    # Save results
    output = {
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "base_url": base_url,
        "total_reward": episode["total_reward"],
        "total_score": round(total_score, 4),
        "steps": episode["steps"],
        "task_results": task_results,
    }

    with open("inference_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to inference_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SQL Debug Env Inference")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL of the environment server",
    )
    args = parser.parse_args()
    main(args.base_url)