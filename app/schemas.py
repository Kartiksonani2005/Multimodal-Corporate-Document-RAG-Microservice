from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 8
    rerank_top_n: Optional[int] = 6

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[str]
    images_sent: List[str]
    latency: float

class HealthResponse(BaseModel):
    status: str
    service: str