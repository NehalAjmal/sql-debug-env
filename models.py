# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the SQL Data Detective Environment.

The agent receives a business question and must explore a database
by running SQL queries, then submit a final answer.
"""

from typing import Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class SqlDetectiveAction(Action):
    """
    Action for the SQL Data Detective environment.

    The agent can either run a SQL query to explore the database,
    or submit a final answer when confident.

    action_type: "query" to run SQL, "answer" to submit final answer
    sql: SQL query string (used when action_type == "query")
    answer: final answer string (used when action_type == "answer")
    """
    action_type: str = Field(
        ...,
        description="Action type: 'query' to execute SQL, 'answer' to submit final answer"
    )
    sql: str = Field(
        default="",
        description="SQL query to execute against the database (when action_type is 'query')"
    )
    answer: str = Field(
        default="",
        description="Final answer to the business question (when action_type is 'answer')"
    )


class SqlDetectiveObservation(Observation):
    """Observation returned after each step in the SQL Data Detective environment."""

    task_id: str = Field(default="", description="Current task identifier")
    question: str = Field(default="", description="Business question the agent must answer")
    db_schema: str = Field(default="", description="Database schema (table definitions)")
    last_action_type: str = Field(default="", description="Type of last action taken")
    last_sql: str = Field(default="", description="Last SQL query executed")
    query_result: str = Field(default="", description="Result of last SQL query as formatted string")
    query_error: str = Field(default="", description="Error message if last query failed")
    query_count: int = Field(default=0, description="Number of SQL queries executed so far")
    max_steps: int = Field(default=8, description="Maximum steps allowed per task")
    step: int = Field(default=0, description="Current step number")
    score: float = Field(default=0.0, description="Score achieved (set when episode ends)")
    feedback: str = Field(default="", description="Feedback on the last action")
    difficulty: str = Field(default="", description="Task difficulty: easy/medium/hard/expert")
    answer_submitted: str = Field(default="", description="The answer that was submitted")