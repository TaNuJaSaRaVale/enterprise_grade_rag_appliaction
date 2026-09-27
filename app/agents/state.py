from typing import TypedDict,List,Annotated
import operator


class AgentState(TypedDict):
    messages: Annotated[List[dict],operator.add]
    current_query : str
    documents: List[str]
    retrieval_scores: List[float]  # cross-encoder score per kept document (drives the confidence badge)
    plan: List[str]
    status:str
    final_answer: str 