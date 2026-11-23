from pydantic import BaseModel, Field
from typing import Literal

class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=3000)

class ReviewAction(BaseModel):
    action: Literal["approve", "reject"]
    note: str = Field(default="", max_length=2000)

class BookCreate(BaseModel):
    project_id: str
    asset_ids: list[str] = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=180)
    subtitle: str = Field(default="", max_length=300)


class ProcessRequest(BaseModel):
    params: dict = Field(default_factory=dict)
    selected_engine: Literal["A", "B", "C", "D"] = "D"
    run_stability: bool = False
