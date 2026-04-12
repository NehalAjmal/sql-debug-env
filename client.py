# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SQL Data Detective Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SqlDetectiveAction, SqlDetectiveObservation


class SqlDetectiveEnv(
    EnvClient[SqlDetectiveAction, SqlDetectiveObservation, State]
):
    """
    Client for the SQL Data Detective Environment.

    The agent receives a business question and a database schema,
    then explores the database through SQL queries before submitting
    a final answer.

    Example:
        >>> with SqlDetectiveEnv(base_url="http://localhost:7860") as client:
        ...     result = client.reset()
        ...     print(result.observation.question)
        ...
        ...     # Explore with SQL
        ...     result = client.step(SqlDetectiveAction(
        ...         action_type="query",
        ...         sql="SELECT * FROM employees LIMIT 5"
        ...     ))
        ...     print(result.observation.query_result)
        ...
        ...     # Submit answer
        ...     result = client.step(SqlDetectiveAction(
        ...         action_type="answer",
        ...         answer="Engineering: 4, Marketing: 3, HR: 3"
        ...     ))
        ...     print(result.observation.score)
    """

    def _step_payload(self, action: SqlDetectiveAction) -> Dict:
        """Convert SqlDetectiveAction to JSON payload for step message."""
        return {
            "action_type": action.action_type,
            "sql": action.sql,
            "answer": action.answer,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SqlDetectiveObservation]:
        """Parse server response into StepResult[SqlDetectiveObservation]."""
        obs_data = payload.get("observation", {})
        observation = SqlDetectiveObservation(
            task_id=obs_data.get("task_id", ""),
            question=obs_data.get("question", ""),
            db_schema=obs_data.get("db_schema", ""),
            last_action_type=obs_data.get("last_action_type", ""),
            last_sql=obs_data.get("last_sql", ""),
            query_result=obs_data.get("query_result", ""),
            query_error=obs_data.get("query_error", ""),
            query_count=obs_data.get("query_count", 0),
            max_steps=obs_data.get("max_steps", 8),
            step=obs_data.get("step", 0),
            score=obs_data.get("score", 0.0),
            feedback=obs_data.get("feedback", ""),
            difficulty=obs_data.get("difficulty", ""),
            answer_submitted=obs_data.get("answer_submitted", ""),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
