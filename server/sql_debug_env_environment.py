# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sqlite3
from uuid import uuid4
from typing import Any

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import SqlDebugAction, SqlDebugObservation
except ImportError:
    from sql_debug_env.models import SqlDebugAction, SqlDebugObservation

TASKS = [
    {
        "task_id": "task_easy",
        "difficulty": "easy",
        "description": (
            "Fix the broken SELECT query. The query is trying to fetch the "
            "names and ages of all employees from the 'employees' table, "
            "but it contains a syntax error."
        ),
        "db_schema": (
            "CREATE TABLE employees (\n"
            "    id INTEGER PRIMARY KEY,\n"
            "    name TEXT NOT NULL,\n"
            "    age INTEGER,\n"
            "    department TEXT\n"
            ");"
        ),
        "seed_data": [
            "INSERT INTO employees VALUES (1, 'Alice', 30, 'Engineering');",
            "INSERT INTO employees VALUES (2, 'Bob', 25, 'Marketing');",
            "INSERT INTO employees VALUES (3, 'Carol', 35, 'Engineering');",
        ],
        "broken_query": "SELEC name, age FORM employees;",
        "error_hint": "There are two typos in the SQL keywords.",
        "correct_query": "SELECT name, age FROM employees;",
        "expected_result": [("Alice", 30), ("Bob", 25), ("Carol", 35)],
    },
    {
        "task_id": "task_medium",
        "difficulty": "medium",
        "description": (
            "Fix the broken JOIN query. The query should return the name of "
            "each employee and the name of their department manager, but the "
            "JOIN condition is wrong."
        ),
        "db_schema": (
            "CREATE TABLE employees (\n"
            "    id INTEGER PRIMARY KEY,\n"
            "    name TEXT NOT NULL,\n"
            "    dept_id INTEGER\n"
            ");\n"
            "CREATE TABLE departments (\n"
            "    id INTEGER PRIMARY KEY,\n"
            "    dept_name TEXT,\n"
            "    manager TEXT\n"
            ");"
        ),
        "seed_data": [
            "INSERT INTO employees VALUES (1, 'Alice', 1);",
            "INSERT INTO employees VALUES (2, 'Bob', 2);",
            "INSERT INTO employees VALUES (3, 'Carol', 1);",
            "INSERT INTO departments VALUES (1, 'Engineering', 'Dave');",
            "INSERT INTO departments VALUES (2, 'Marketing', 'Eve');",
        ],
        "broken_query": (
            "SELECT employees.name, departments.manager "
            "FROM employees "
            "JOIN departments ON employees.id = departments.id;"
        ),
        "error_hint": "The JOIN condition is matching the wrong columns.",
        "correct_query": (
            "SELECT employees.name, departments.manager "
            "FROM employees "
            "JOIN departments ON employees.dept_id = departments.id;"
        ),
        "expected_result": [
            ("Alice", "Dave"),
            ("Bob", "Eve"),
            ("Carol", "Dave"),
        ],
    },
    {
        "task_id": "task_hard",
        "difficulty": "hard",
        "description": (
            "Fix the broken query. The query should return each department name "
            "along with the number of employees in that department who earn more "
            "than the company-wide average salary, but only for departments that "
            "have at least 2 such employees. The query has multiple bugs."
        ),
        "db_schema": (
            "CREATE TABLE employees (\n"
            "    id INTEGER PRIMARY KEY,\n"
            "    name TEXT NOT NULL,\n"
            "    department TEXT,\n"
            "    salary REAL\n"
            ");"
        ),
        "seed_data": [
            "INSERT INTO employees VALUES (1, 'Alice', 'Engineering', 90000);",
            "INSERT INTO employees VALUES (2, 'Bob', 'Engineering', 70000);",
            "INSERT INTO employees VALUES (3, 'Carol', 'Marketing', 60000);",
            "INSERT INTO employees VALUES (4, 'Dave', 'Marketing', 80000);",
            "INSERT INTO employees VALUES (5, 'Eve', 'Engineering', 95000);",
            "INSERT INTO employees VALUES (6, 'Frank', 'Engineering', 72000);",
            "INSERT INTO employees VALUES (7, 'Grace', 'Marketing', 55000);",
        ],
        "broken_query": (
            "SELECT department, COUNT(*) as high_earners "
            "FROM employees "
            "WHERE salary > (SELECT AVG(salary) FROM employees) "
            "GROUP BY department "
            "HAVING COUNT(*) > 3;"
        ),
        "error_hint": "The HAVING clause threshold is wrong. Review what count qualifies as 'at least 2'.",
        "correct_query": (
            "SELECT department, COUNT(*) as high_earners "
            "FROM employees "
            "WHERE salary > (SELECT AVG(salary) FROM employees) "
            "GROUP BY department "
            "HAVING COUNT(*) >= 2;"
        ),
        "expected_result": [("Engineering", 2)],
    },
    {
        "task_id": "task_expert",
        "difficulty": "expert",
        "description": (
            "Fix the broken window function query. The query should return "
            "each employee's name, salary, and their salary rank within their "
            "department (1 = highest paid). It should also return the difference "
            "between their salary and the highest salary in their department. "
            "The query has multiple bugs in the window functions."
        ),
        "db_schema": (
            "CREATE TABLE employees (\n"
            "    id INTEGER PRIMARY KEY,\n"
            "    name TEXT NOT NULL,\n"
            "    department TEXT,\n"
            "    salary REAL\n"
            ");"
        ),
        "seed_data": [
            "INSERT INTO employees VALUES (1, 'Alice', 'Engineering', 90000);",
            "INSERT INTO employees VALUES (2, 'Bob', 'Engineering', 70000);",
            "INSERT INTO employees VALUES (3, 'Carol', 'Marketing', 60000);",
            "INSERT INTO employees VALUES (4, 'Dave', 'Marketing', 80000);",
            "INSERT INTO employees VALUES (5, 'Eve', 'Engineering', 95000);",
        ],
        "broken_query": (
            "SELECT name, salary, "
            "RANK() OVER (PARTITION BY department ORDER BY salary ASC) as dept_rank, "
            "salary - MAX(salary) OVER (PARTITION BY id) as diff_from_top "
            "FROM employees;"
        ),
        "error_hint": "There are two bugs: the ranking order is wrong, and the MAX window partition is wrong.",
        "correct_query": (
            "SELECT name, salary, "
            "RANK() OVER (PARTITION BY department ORDER BY salary DESC) as dept_rank, "
            "salary - MAX(salary) OVER (PARTITION BY department) as diff_from_top "
            "FROM employees;"
        ),
        "expected_result": [
            ("Alice", 90000.0, 2, -5000.0),
            ("Bob", 70000.0, 3, -25000.0),
            ("Carol", 60000.0, 2, -20000.0),
            ("Dave", 80000.0, 1, 0.0),
            ("Eve", 95000.0, 1, 0.0),
        ],
    },
]


def _run_query(db_schema: str, seed_data: list[str], query: str):
    conn = sqlite3.connect(":memory:")
    try:
        # executescript needs each statement to end with semicolon
        # Normalize and execute schema statements one by one for reliability
        schema_statements = [s.strip() for s in db_schema.split(";") if s.strip()]
        for stmt in schema_statements:
            conn.execute(stmt)
        conn.commit()
        for stmt in seed_data:
            conn.execute(stmt)
        conn.commit()
        cursor = conn.execute(query)
        rows = cursor.fetchall()
        return rows, None
    except Exception as e:
        return None, str(e)
    finally:
        conn.close()


def _score_query(task: dict[str, Any], query: str) -> tuple[float, str]:
    rows, error = _run_query(task["db_schema"], task["seed_data"], query)

    if error:
        return 0.0, f"Query failed to execute: {error}"

    expected = set(task["expected_result"])
    got = set(rows)

    if got == expected:
        return 1.0, "Perfect! Query is correct and returns the expected result."

    if not got:
        return 0.1, "Query executed but returned no rows. Check your conditions."

    overlap = len(got & expected)
    total = len(expected)
    partial = overlap / total if total > 0 else 0.0

    if overlap > 0:
        return round(0.3 + 0.5 * partial, 2), (
            f"Partially correct: {overlap}/{total} expected rows matched. "
            f"Got: {list(rows)}"
        )

    return 0.2, f"Query executed but result doesn't match. Got: {list(rows)}"


class SqlDebugEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task_index = 0
        self._attempt = 0
        self._max_attempts = 3
        self._last_score = 0.0

    def reset(self) -> SqlDebugObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task_index = 0
        self._attempt = 0
        self._last_score = 0.0
        task = TASKS[self._current_task_index]
        return SqlDebugObservation(
            task_id=task["task_id"],
            broken_query=task["broken_query"],
            db_schema=task["db_schema"],
            error_hint=task["error_hint"],
            task_description=task["description"],
            difficulty=task["difficulty"],
            score=0.0,
            feedback="New episode started. Fix the broken SQL query.",
            attempt=0,
            max_attempts=self._max_attempts,
            done=False,
            reward=0.0,
        )

    def step(self, action: SqlDebugAction) -> SqlDebugObservation:
        self._state.step_count += 1
        self._attempt += 1
        task = TASKS[self._current_task_index]
        score, feedback = _score_query(task, action.fixed_query)
        self._last_score = score

        attempt_penalty = (self._attempt - 1) * 0.05
        reward = max(0.0, score - attempt_penalty)

        task_solved = score == 1.0
        out_of_attempts = self._attempt >= self._max_attempts
        is_last_task = self._current_task_index == len(TASKS) - 1

        if task_solved or out_of_attempts:
            if is_last_task:
                done = True
            else:
                self._current_task_index += 1
                self._attempt = 0
                next_task = TASKS[self._current_task_index]
                return SqlDebugObservation(
                    task_id=next_task["task_id"],
                    broken_query=next_task["broken_query"],
                    db_schema=next_task["db_schema"],
                    error_hint=next_task["error_hint"],
                    task_description=next_task["description"],
                    difficulty=next_task["difficulty"],
                    score=score,
                    feedback=feedback + " Moving to next task.",
                    attempt=0,
                    max_attempts=self._max_attempts,
                    done=False,
                    reward=reward,
                )
        else:
            done = False

        return SqlDebugObservation(
            task_id=task["task_id"],
            broken_query=task["broken_query"],
            db_schema=task["db_schema"],
            error_hint=task["error_hint"],
            task_description=task["description"],
            difficulty=task["difficulty"],
            score=score,
            feedback=feedback,
            attempt=self._attempt,
            max_attempts=self._max_attempts,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> State:
        return self._state