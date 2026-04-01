# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SQL Debug Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SqlDebugAction, SqlDebugObservation


class SqlDebugEnv(
    EnvClient[SqlDebugAction, SqlDebugObservation, State]
):
    """
    Client for the SQL Debug Environment.

    The agent receives a broken SQL query and database schema,
    and must return a corrected query that produces the expected result.

    Example:
        >>> with SqlDebugEnv(base_url="http://localhost:7860") as client:
        ...     result = client.reset()
        ...     print(result.observation.broken_query)
        ...
        ...     result = client.step(SqlDebugAction(fixed_query="SELECT name FROM employees;"))
        ...     print(result.observation.score)
        ...     print(result.observation.feedback)
    """

    def _step_payload(self, action: SqlDebugAction) -> Dict:
        """Convert SqlDebugAction to JSON payload for step message."""
        return {
            "fixed_query": action.fixed_query,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SqlDebugObservation]:
        """Parse server response into StepResult[SqlDebugObservation]."""
        obs_data = payload.get("observation", {})
        observation = SqlDebugObservation(
            task_id=obs_data.get("task_id", ""),
            broken_query=obs_data.get("broken_query", ""),
            db_schema=obs_data.get("db_schema", ""),
            error_hint=obs_data.get("error_hint", ""),
            task_description=obs_data.get("task_description", ""),
            difficulty=obs_data.get("difficulty", ""),
            score=obs_data.get("score", 0.0),
            feedback=obs_data.get("feedback", ""),
            attempt=obs_data.get("attempt", 0),
            max_attempts=obs_data.get("max_attempts", 3),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
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
