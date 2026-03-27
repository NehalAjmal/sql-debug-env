# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class SqlDebugAction(Action):
    """Action: the agent submits a fixed SQL query."""
    fixed_query: str = Field(..., description="The agent's corrected SQL query")


class SqlDebugObservation(Observation):
    """Observation returned after each step."""
    task_id: str = Field(default="", description="Current task identifier")
    broken_query: str = Field(default="", description="The broken SQL query to fix")
    db_schema: str = Field(default="", description="Database schema as CREATE TABLE statements")
    error_hint: str = Field(default="", description="Error message or hint about what's wrong")
    score: float = Field(default=0.0, description="Score for the last submitted query (0.0-1.0)")
    feedback: str = Field(default="", description="Feedback on the submitted query")
    task_description: str = Field(default="", description="Natural language description of the task")
    difficulty: str = Field(default="", description="Task difficulty: easy / medium / hard")
    attempt: int = Field(default=0, description="Current attempt number")
    max_attempts: int = Field(default=3, description="Maximum attempts allowed")