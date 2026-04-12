"""
Inference Script for SQL Data Detective Environment
====================================================
Multi-turn RL agent that explores a database by running SQL queries,
then submits a final answer to business questions.

MANDATORY environment variables:
  API_BASE_URL  The API endpoint for the LLM.
  MODEL_NAME    The model identifier to use for inference.
  HF_TOKEN      Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory
- Participants must use OpenAI Client for all LLM calls using above variables
"""

import os
import sys
import json
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Mandatory env variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_EXPLORE_STEPS = 6          # max SQL queries per task before forcing answer
TEMPERATURE = 0.0
MAX_TOKENS = 500
DEFAULT_BASE_URL = "https://nehubaby-sql-debug-env.hf.space"
BENCHMARK = "sql-debug-env"

SYSTEM_PROMPT = """You are a data analyst investigating a company database.
You receive a business question and must answer it by exploring the database.

At each turn you MUST respond with EXACTLY one of these two formats:
1. To run a SQL query:   SQL: <your query>
2. To submit your final answer:   ANSWER: <your answer>

Rules:
- Start by exploring relevant tables to understand the data
- Build up your understanding step by step
- When confident, submit your answer
- Your answer should be concise and include the key values asked for
- Do NOT include any other text outside the SQL: or ANSWER: format"""


def build_initial_prompt(observation: dict) -> str:
    """Build the initial user prompt from the reset observation."""
    return (
        f"Business Question: {observation.get('question', '')}\n\n"
        f"Database Schema:\n{observation.get('db_schema', '')}\n\n"
        f"Difficulty: {observation.get('difficulty', 'unknown')}\n\n"
        "Explore the database with SQL queries, then submit your answer."
    )


def parse_llm_response(reply: str) -> tuple[str, str]:
    """
    Parse LLM response into (action_type, content).
    Returns ('query', sql) or ('answer', answer_text).
    """
    stripped = reply.strip()

    # Check for SQL: prefix
    for prefix in ["SQL:", "sql:", "Sql:"]:
        if stripped.startswith(prefix):
            return "query", stripped[len(prefix):].strip()

    # Check for ANSWER: prefix
    for prefix in ["ANSWER:", "answer:", "Answer:"]:
        if stripped.startswith(prefix):
            return "answer", stripped[len(prefix):].strip()

    # Check for ```sql code blocks
    if "```sql" in stripped:
        sql_start = stripped.index("```sql") + 6
        sql_end = stripped.index("```", sql_start) if "```" in stripped[sql_start:] else len(stripped)
        return "query", stripped[sql_start:sql_end].strip()

    # Fallback: if it looks like SQL, treat as query
    sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "WITH"]
    upper = stripped.upper()
    if any(upper.startswith(kw) for kw in sql_keywords):
        return "query", stripped

    # Otherwise treat as answer
    return "answer", stripped


def run_episode(base_url: str, client: OpenAI) -> None:
    """Run a full episode: reset → multi-turn explore → answer for each task."""

    # Reset the environment — use stateful /env/reset for session persistence
    reset_resp = requests.post(f"{base_url}/env/reset", json={}, timeout=30)
    reset_resp.raise_for_status()
    reset_data = reset_resp.json()
    session_id = reset_data.get("session_id", "")
    obs = reset_data.get("observation", reset_data)

    while True:
        task_id = obs.get("task_id", "unknown")
        question = obs.get("question", "")
        difficulty = obs.get("difficulty", "")

        # [START]
        print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

        # Build conversation for this task
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_initial_prompt(obs)},
        ]

        all_rewards = []
        success = False
        step_num = 0
        final_answer = ""
        final_score = 0.0

        for explore_step in range(1, MAX_EXPLORE_STEPS + 2):  # +1 for final answer step
            step_num = explore_step

            try:
                # Get LLM response
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                reply = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
                messages.append({"role": "assistant", "content": reply})

                # Parse LLM response
                action_type, content = parse_llm_response(reply)

                # Force answer on last step
                if explore_step >= MAX_EXPLORE_STEPS and action_type == "query":
                    action_type = "answer"
                    content = content if content else "Unable to determine answer"

                # Build action payload
                if action_type == "query":
                    action_payload = {
                        "action": {
                            "action_type": "query",
                            "sql": content,
                            "answer": "",
                        }
                    }
                else:
                    action_payload = {
                        "action": {
                            "action_type": "answer",
                            "sql": "",
                            "answer": content,
                        }
                    }
                    final_answer = content

                # Step the environment — use stateful /env/step with session_id
                step_payload = {
                    "session_id": session_id,
                    "action": action_payload["action"],
                }
                step_resp = requests.post(
                    f"{base_url}/env/step",
                    json=step_payload,
                    timeout=30,
                )
                step_resp.raise_for_status()
                step_data = step_resp.json()
                obs = step_data.get("observation", step_data)

                reward = step_data.get("reward", obs.get("reward", 0.0))
                done = step_data.get("done", obs.get("done", False))
                score = obs.get("score", 0.0)
                final_score = score

                all_rewards.append(reward)

                # Build action string for logging (MUST have no newlines)
                clean_content = content[:70].replace('\n', ' ').replace('\r', '')
                if action_type == "query":
                    action_str = f"SQL: {clean_content}"
                else:
                    action_str = f"ANSWER: {clean_content}"

                # [STEP]
                print(
                    f"[STEP] step={step_num} action={action_str} "
                    f"reward={reward:.2f} done={str(done).lower()} error=null",
                    flush=True,
                )

                if action_type == "query":
                    # Feed query result back to LLM
                    query_result = obs.get("query_result", "")
                    query_error = obs.get("query_error", "")
                    feedback = obs.get("feedback", "")

                    if query_error:
                        result_msg = f"Query error: {query_error}\n{feedback}"
                    else:
                        result_msg = f"Query result:\n{query_result}\n\n{feedback}"

                    messages.append({"role": "user", "content": result_msg})

                if action_type == "answer":
                    success = score >= 0.8
                    break

                if done:
                    break

            except Exception as e:
                all_rewards.append(0.0)
                clean_err = str(e)[:80].replace('\n', ' ').replace('\r', '')
                print(
                    f"[STEP] step={step_num} action=null "
                    f"reward=0.00 done=false error={clean_err}",
                    flush=True,
                )
                break

        # [END]
        rewards_str = ",".join(f"{r:.2f}" for r in all_rewards) if all_rewards else "0.00"
        print(
            f"[END] success={str(success).lower()} steps={len(all_rewards)} score={final_score:.2f} rewards={rewards_str}",
            flush=True,
        )

        # Check if episode is done (all tasks completed)
        done = obs.get("done", False)
        if done:
            break

        # If not done, obs already contains the next task info
        # (the environment advances to next task after an answer)


def main(base_url: str = DEFAULT_BASE_URL) -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    run_episode(base_url, client)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="SQL Data Detective — Multi-turn inference agent"
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args()
    main(args.base_url)