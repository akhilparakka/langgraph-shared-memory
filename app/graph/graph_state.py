from typing import TypedDict, Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class UpdateMemory(TypedDict):
    update_type: Literal["user", "todo", "instructions"]


class Profile(BaseModel):
    name: Optional[str] = Field(description="The users name", default=None)
    location: Optional[str] = Field(description="The users Location", default=None)
    job: Optional[str] = Field(description="the users job", default=None)
    connections: list[str] = Field(
        description="Personal connections of the user, such as family, friends etc.",
        default_factory=list,
    )
    interests: list[str] = Field(
        description="The users interests", default_factory=list
    )


class todo(BaseModel):
    task: str = Field(description="The task to be completed.")
    time_to_complete: Optional[int] = Field(
        description="Estimated time to complete the task (minutes)."
    )
    deadline: Optional[datetime] = Field(
        description="When the task needs to be completed by (if applicable)",
        default=None,
    )
    solutions: list[str] = Field(
        description="List of specific, actionable solutions (e.g., specific ideas, service providers, or concrete options relevant to completing the task)",
        min_items=1,
        default_factory=list,
    )
    status: Literal["not started", "in progress", "done", "archived"] = Field(
        description="Current status of the task", default="not started"
    )
