# graph/research_graph.py
from langgraph.graph import StateGraph, END
from models.research_state import ResearchState
from agents.query_agent import query_analysis_node
from agents.search_agent import search_node
from agents.filter_agent import filtering_node
from agents.summary_agent import summarization_node
from agents.theme_agent import theme_analysis_node
from agents.report_agent import report_generation_node

def create_research_graph():
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("search", search_node)
    workflow.add_node("filter", filtering_node)
    workflow.add_node("summarize", summarization_node)
    workflow.add_node("theme_analysis", theme_analysis_node)
    workflow.add_node("report_generation", report_generation_node)
    
    # Define edges
    workflow.set_entry_point("query_analysis")
    
    workflow.add_edge("query_analysis", "search")
    workflow.add_edge("search", "filter")
    workflow.add_conditional_edges(
        "filter",
        should_continue,
        {
            "summarize": "summarize",
            "end": END
        }
    )
    workflow.add_edge("summarize", "theme_analysis")
    workflow.add_edge("theme_analysis", "report_generation")
    workflow.add_edge("report_generation", END)
    
    return workflow.compile()

def should_continue(state: ResearchState) -> str:
    """Check if we found enough papers to continue"""
    if len(state.get("filtered_papers", [])) >= 3:
        return "summarize"
    return "end"