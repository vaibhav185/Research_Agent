# models/research_state.py
from typing import TypedDict, List, Dict, Optional, Annotated
from langgraph.graph import add_messages

class ResearchState(TypedDict):
    research_question: str
    search_queries: List[str]
    raw_papers: List[Dict]
    filtered_papers: List[Dict]
    paper_summaries: Dict[str, Dict]  # paper_id -> summary_data
    key_themes: List[Dict]
    literature_gaps: List[Dict]
    final_report: str
    current_step: str
    error: Optional[str]
    
    # LangGraph message history
    messages: Annotated[List[Dict], add_messages]