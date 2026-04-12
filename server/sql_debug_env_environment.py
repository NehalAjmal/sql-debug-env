# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SQL Data Detective Environment.

The agent receives a business question about a company database.
It must explore the database by running SQL queries, then submit
a final answer. This models the real-world workflow of a data analyst.

Multi-turn design:
- Each step: agent either runs a SQL query or submits final answer
- Agent sees query results each step and builds understanding
- Reward is based on correctness AND query efficiency
- Tasks require 2-6 queries for a competent agent
"""

import sqlite3
import json
from uuid import uuid4
from typing import Any

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import SqlDetectiveAction, SqlDetectiveObservation
except ImportError:
    from sql_debug_env.models import SqlDetectiveAction, SqlDetectiveObservation


# ---------------------------------------------------------------------------
# Shared database seed — same schema across all tasks
# ---------------------------------------------------------------------------

DB_SCHEMA = """CREATE TABLE employees (
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
    status TEXT NOT NULL
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
);"""

SEED_DATA = [
    # Employees
    "INSERT INTO employees VALUES (1, 'Alice Johnson', 'Engineering', 95000, 2019, NULL);",
    "INSERT INTO employees VALUES (2, 'Bob Smith', 'Engineering', 85000, 2021, 1);",
    "INSERT INTO employees VALUES (3, 'Carol White', 'Marketing', 75000, 2018, NULL);",
    "INSERT INTO employees VALUES (4, 'Dave Brown', 'Engineering', 90000, 2020, 1);",
    "INSERT INTO employees VALUES (5, 'Eve Davis', 'Marketing', 72000, 2022, 3);",
    "INSERT INTO employees VALUES (6, 'Frank Miller', 'HR', 65000, 2017, NULL);",
    "INSERT INTO employees VALUES (7, 'Grace Wilson', 'HR', 68000, 2021, 6);",
    "INSERT INTO employees VALUES (8, 'Henry Moore', 'Engineering', 92000, 2019, 1);",
    "INSERT INTO employees VALUES (9, 'Iris Taylor', 'Marketing', 78000, 2020, 3);",
    "INSERT INTO employees VALUES (10, 'Jack Anderson', 'HR', 61000, 2022, 6);",

    # Projects
    "INSERT INTO projects VALUES (1, 'Phoenix Platform', 'Engineering', 500000, 'active');",
    "INSERT INTO projects VALUES (2, 'Brand Refresh', 'Marketing', 150000, 'completed');",
    "INSERT INTO projects VALUES (3, 'HR Automation', 'HR', 80000, 'active');",
    "INSERT INTO projects VALUES (4, 'Data Pipeline', 'Engineering', 320000, 'active');",
    "INSERT INTO projects VALUES (5, 'Customer Portal', 'Engineering', 420000, 'completed');",
    "INSERT INTO projects VALUES (6, 'Social Campaign', 'Marketing', 95000, 'active');",

    # Sales
    "INSERT INTO sales VALUES (1, 1, 'Enterprise License', 45000, '2024-01-15', 'North');",
    "INSERT INTO sales VALUES (2, 2, 'Support Package', 12000, '2024-01-22', 'South');",
    "INSERT INTO sales VALUES (3, 4, 'Enterprise License', 48000, '2024-02-10', 'North');",
    "INSERT INTO sales VALUES (4, 8, 'Consulting', 25000, '2024-02-14', 'East');",
    "INSERT INTO sales VALUES (5, 1, 'Enterprise License', 52000, '2024-02-28', 'West');",
    "INSERT INTO sales VALUES (6, 4, 'Support Package', 15000, '2024-03-05', 'North');",
    "INSERT INTO sales VALUES (7, 8, 'Enterprise License', 49000, '2024-03-12', 'East');",
    "INSERT INTO sales VALUES (8, 2, 'Consulting', 18000, '2024-03-20', 'South');",
    "INSERT INTO sales VALUES (9, 1, 'Support Package', 11000, '2024-04-02', 'North');",
    "INSERT INTO sales VALUES (10, 8, 'Enterprise License', 55000, '2024-04-18', 'West');",

    # Expenses
    "INSERT INTO expenses VALUES (1, 'Engineering', 'Software', 12000, '2024-01-10');",
    "INSERT INTO expenses VALUES (2, 'Marketing', 'Advertising', 25000, '2024-01-15');",
    "INSERT INTO expenses VALUES (3, 'HR', 'Training', 8000, '2024-01-20');",
    "INSERT INTO expenses VALUES (4, 'Engineering', 'Hardware', 35000, '2024-02-05');",
    "INSERT INTO expenses VALUES (5, 'Marketing', 'Events', 18000, '2024-02-12');",
    "INSERT INTO expenses VALUES (6, 'Engineering', 'Software', 9000, '2024-03-01');",
    "INSERT INTO expenses VALUES (7, 'HR', 'Recruitment', 15000, '2024-03-10');",
    "INSERT INTO expenses VALUES (8, 'Marketing', 'Advertising', 22000, '2024-03-20');",
    "INSERT INTO expenses VALUES (9, 'Engineering', 'Consulting', 45000, '2024-04-05');",
    "INSERT INTO expenses VALUES (10, 'Marketing', 'Events', 12000, '2024-04-15');",
]


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

TASKS = [
    {
        "task_id": "task_easy",
        "difficulty": "easy",
        "question": (
            "How many employees are in each department? "
            "List all departments with their employee count, "
            "ordered from most to fewest employees."
        ),
        "expected_answer": "Engineering: 4, Marketing: 3, HR: 3",
        "answer_variants": [
            "engineering: 4",
            "engineering 4",
            "4 engineering",
            "engineering has 4",
        ],
        "key_values": ["4", "3", "engineering"],
        "min_queries": 1,
    },
    {
        "task_id": "task_medium",
        "difficulty": "medium",
        "question": (
            "Which employee generated the highest total sales revenue in Q1 2024 "
            "(January through March)? Provide their full name and total revenue."
        ),
        "expected_answer": "Alice Johnson, 97000",
        "answer_variants": [
            "alice johnson",
            "alice",
            "97000",
            "alice johnson, 97000",
            "alice johnson: 97000",
        ],
        "key_values": ["alice", "97000"],
        "min_queries": 2,
    },
    {
        "task_id": "task_hard",
        "difficulty": "hard",
        "question": (
            "For each department that has active projects, calculate the ratio of "
            "total project budget to total annual salary expenditure. "
            "Which department has the highest ratio? Provide the department name "
            "and the ratio rounded to 2 decimal places."
        ),
        # Engineering: budget = 500000+320000 = 820000, salary = 95000+85000+90000+92000 = 362000
        # ratio = 820000/362000 = 2.27
        # HR: budget = 80000, salary = 65000+68000+61000 = 194000
        # ratio = 80000/194000 = 0.41
        # Marketing has no active projects with budget check:
        # Social Campaign is active: 95000, salary = 75000+72000+78000 = 225000
        # ratio = 95000/225000 = 0.42
        # Engineering wins with 2.27
        "expected_answer": "Engineering, 2.27",
        "answer_variants": [
            "engineering",
            "engineering, 2.27",
            "engineering: 2.27",
            "engineering 2.27",
        ],
        "key_values": ["engineering", "2.27"],
        "min_queries": 3,
    },
    {
        "task_id": "task_expert",
        "difficulty": "expert",
        "question": (
            "Identify the employee who has the best 'efficiency score', defined as: "
            "total sales revenue divided by annual salary. "
            "Only consider employees who made at least 2 sales. "
            "What is their name and efficiency score rounded to 2 decimal places?"
        ),
        # Alice: sales = 45000+52000+11000 = 108000, salary = 95000, score = 1.14, count = 3
        # Bob: sales = 12000+18000 = 30000, salary = 85000, score = 0.35, count = 2
        # Dave: sales = 48000+15000 = 63000, salary = 90000, score = 0.70, count = 2
        # Henry: sales = 25000+49000+55000 = 129000, salary = 92000, score = 1.40, count = 3
        # Winner: Henry Moore, 1.40
        "expected_answer": "Henry Moore, 1.40",
        "answer_variants": [
            "henry moore",
            "henry",
            "1.40",
            "henry moore, 1.40",
            "henry moore: 1.40",
        ],
        "key_values": ["henry", "1.40"],
        "min_queries": 3,
    },
]


# ---------------------------------------------------------------------------
# Database helper
# ---------------------------------------------------------------------------

def _run_query(query: str) -> tuple[list, str | None]:
    """Run a query against the shared in-memory database. Returns (rows, error)."""
    conn = sqlite3.connect(":memory:")
    try:
        schema_statements = [s.strip() for s in DB_SCHEMA.split(";") if s.strip()]
        for stmt in schema_statements:
            conn.execute(stmt)
        conn.commit()
        for stmt in SEED_DATA:
            conn.execute(stmt)
        conn.commit()
        cursor = conn.execute(query)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description] if cursor.description else []
        return rows, cols, None
    except Exception as e:
        return [], [], str(e)
    finally:
        conn.close()


def _format_result(rows: list, cols: list, max_rows: int = 20) -> str:
    """Format query results as a readable string."""
    if not rows:
        return "(no rows returned)"
    if cols:
        header = " | ".join(cols)
        separator = "-" * len(header)
        lines = [header, separator]
    else:
        lines = []
    for row in rows[:max_rows]:
        lines.append(" | ".join(str(v) for v in row))
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows)")
    return "\n".join(lines)


def _grade_answer(task: dict, answer: str) -> tuple[float, str]:
    """Grade the submitted answer. Returns (score, feedback)."""
    answer_lower = answer.strip().lower()

    # Check key values — all must be present for full credit
    key_values = task.get("key_values", [])
    hits = sum(1 for kv in key_values if kv.lower() in answer_lower)

    if hits == len(key_values):
        return 1.0, f"Correct! The answer '{task['expected_answer']}' is right."

    if hits > 0:
        partial = hits / len(key_values)
        return round(0.3 + 0.4 * partial, 2), (
            f"Partially correct: {hits}/{len(key_values)} key values found. "
            f"Expected answer relates to: {task['expected_answer']}"
        )

    # Check answer variants
    for variant in task.get("answer_variants", []):
        if variant.lower() in answer_lower:
            return 0.8, "Close! Answer contains the right information but may be incomplete."

    return 0.0, (
        f"Incorrect. The answer should relate to: {task['expected_answer']}. "
        "Try running more queries to explore the data."
    )


def _compute_reward(score: float, query_count: int, min_queries: int) -> float:
    """
    Reward function that incentivises correctness AND efficiency.
    More queries = lower reward for same score.
    """
    if score == 0.0:
        return 0.0
    # Efficiency bonus: fewer queries relative to min needed = higher reward
    efficiency = max(0.0, 1.0 - max(0, query_count - min_queries) * 0.08)
    return round(score * (0.7 + 0.3 * efficiency), 3)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SqlDebugEnvironment(Environment):
    """
    SQL Data Detective Environment.

    The agent receives a business question and must explore a company
    database through multi-turn SQL queries, then submit a final answer.

    This models the real-world workflow of a data analyst investigating
    business metrics.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task_index = 0
        self._query_count = 0
        self._max_steps = 8
        self._done = False

    def reset(self) -> SqlDetectiveObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task_index = 0
        self._query_count = 0
        self._done = False

        task = TASKS[self._current_task_index]
        return SqlDetectiveObservation(
            task_id=task["task_id"],
            question=task["question"],
            db_schema=DB_SCHEMA,
            last_action_type="",
            last_sql="",
            query_result="",
            query_error="",
            query_count=0,
            max_steps=self._max_steps,
            step=0,
            score=0.0,
            feedback=(
                "New episode started. You have a database with tables: "
                "employees, projects, sales, expenses. "
                "Run SQL queries to explore the data, then submit your answer."
            ),
            difficulty=task["difficulty"],
            answer_submitted="",
            done=False,
            reward=0.0,
        )

    def step(self, action: SqlDetectiveAction) -> SqlDetectiveObservation:
        self._state.step_count += 1
        task = TASKS[self._current_task_index]
        is_last_task = self._current_task_index == len(TASKS) - 1

        # --- Handle QUERY action ---
        if action.action_type == "query":
            self._query_count += 1
            rows, cols, error = _run_query(action.sql)

            if error:
                result_str = f"ERROR: {error}"
                feedback = f"Query failed: {error}. Check your SQL syntax."
            else:
                result_str = _format_result(rows, cols)
                feedback = f"Query returned {len(rows)} row(s). Keep exploring or submit your answer."

            steps_left = self._max_steps - self._state.step_count
            out_of_steps = steps_left <= 0

            if out_of_steps:
                return SqlDetectiveObservation(
                    task_id=task["task_id"],
                    question=task["question"],
                    db_schema=DB_SCHEMA,
                    last_action_type="query",
                    last_sql=action.sql,
                    query_result=result_str,
                    query_error=error or "",
                    query_count=self._query_count,
                    max_steps=self._max_steps,
                    step=self._state.step_count,
                    score=0.0,
                    feedback="Out of steps without submitting an answer. Score: 0.",
                    difficulty=task["difficulty"],
                    answer_submitted="",
                    done=True,
                    reward=0.0,
                )

            return SqlDetectiveObservation(
                task_id=task["task_id"],
                question=task["question"],
                db_schema=DB_SCHEMA,
                last_action_type="query",
                last_sql=action.sql,
                query_result=result_str,
                query_error=error or "",
                query_count=self._query_count,
                max_steps=self._max_steps,
                step=self._state.step_count,
                score=0.0,
                feedback=feedback + f" ({steps_left} steps remaining)",
                difficulty=task["difficulty"],
                answer_submitted="",
                done=False,
                reward=0.0,
            )

        # --- Handle ANSWER action ---
        elif action.action_type == "answer":
            score, grade_feedback = _grade_answer(task, action.answer)
            reward = _compute_reward(score, self._query_count, task["min_queries"])

            if is_last_task:
                return SqlDetectiveObservation(
                    task_id=task["task_id"],
                    question=task["question"],
                    db_schema=DB_SCHEMA,
                    last_action_type="answer",
                    last_sql="",
                    query_result="",
                    query_error="",
                    query_count=self._query_count,
                    max_steps=self._max_steps,
                    step=self._state.step_count,
                    score=score,
                    feedback=grade_feedback,
                    difficulty=task["difficulty"],
                    answer_submitted=action.answer,
                    done=True,
                    reward=reward,
                )
            else:
                # Move to next task
                self._current_task_index += 1
                self._query_count = 0
                self._state.step_count = 0  # Fresh step budget for new task
                next_task = TASKS[self._current_task_index]
                return SqlDetectiveObservation(
                    task_id=next_task["task_id"],
                    question=next_task["question"],
                    db_schema=DB_SCHEMA,
                    last_action_type="answer",
                    last_sql="",
                    query_result="",
                    query_error="",
                    query_count=0,
                    max_steps=self._max_steps,
                    step=0,
                    score=score,
                    feedback=grade_feedback + " Moving to next task.",
                    difficulty=next_task["difficulty"],
                    answer_submitted=action.answer,
                    done=False,
                    reward=reward,
                )

        # --- Invalid action type ---
        else:
            return SqlDetectiveObservation(
                task_id=task["task_id"],
                question=task["question"],
                db_schema=DB_SCHEMA,
                last_action_type=action.action_type,
                last_sql="",
                query_result="",
                query_error=f"Unknown action_type: {action.action_type}. Use 'query' or 'answer'.",
                query_count=self._query_count,
                max_steps=self._max_steps,
                step=self._state.step_count,
                score=0.0,
                feedback="Invalid action. Use action_type='query' with sql, or action_type='answer' with answer.",
                difficulty=task["difficulty"],
                answer_submitted="",
                done=False,
                reward=0.0,
            )

    @property
    def state(self) -> State:
        return self._state