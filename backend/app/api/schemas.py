from typing import Literal
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict, field_validator

class ProblemRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    goal: str=Field(default='',max_length=4000)
    task: Literal['auto','classification','regression','clustering']='auto'
    target: str|None=Field(default=None,max_length=200)
    cv_folds: int=Field(default=3,ge=2,le=5)
    seed: int=Field(default=42,ge=1,le=999999)

class AnalysisRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    dataset_id: UUID
    goal: str=Field(min_length=5,max_length=4000)
    task: Literal['auto','classification','regression','clustering']='auto'
    target: str|None=Field(default=None,max_length=200)
    features: list[str]|None=Field(default=None,max_length=100)
    include_dl: bool=True
    use_llm: bool=False
    seed: int=Field(default=42,ge=1,le=999999)
    cv_folds: int=Field(default=3,ge=2,le=5)
    tuning_iterations: int=Field(default=3,ge=1,le=10)
    time_budget_seconds: int=Field(default=600,ge=10,le=7200)
    quick: bool=False

class PredictionRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    rows: list[dict[str,str|float|int|bool|None]]=Field(min_length=1,max_length=200)
    @field_validator('rows')
    @classmethod
    def bounded(cls,rows):
        if any(len(row)>100 or any(len(str(v))>20000 for v in row.values()) for row in rows):raise ValueError('Prediction inputs exceed the column or cell length limit.')
        return rows

class ChatRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    question: str=Field(min_length=2,max_length=2000)
    use_llm: bool=False
    provider: Literal['groq','configured','local']|None=None
    @field_validator('question')
    @classmethod
    def meaningful_question(cls,value):
        value=value.strip()
        if len(value)<2:raise ValueError('Enter a question with at least two characters.')
        return value
