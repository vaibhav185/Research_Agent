# agents/query_agent.py
from models.research_state import ResearchState
import re
from typing import Dict
from typing import Dict, List, Optional, TypedDict

def query_analysis_node(state: ResearchState) -> ResearchState:
    research_question = state["research_question"]
    
    # Expand query using keyword extraction and synonyms
    expanded_queries = expand_research_query(research_question)
    
    return {
        **state,
        "search_queries": expanded_queries,
        "current_step": "query_analysis"
    }

def expand_research_query(question: str) -> List[str]:
    """Expand a research question into multiple search queries"""
    # Extract key terms
    key_terms = extract_key_terms(question)
    
    queries = []
    
    # Create different query combinations
    for i, term in enumerate(key_terms):
        # Broad search
        queries.append(question)
        
        # Specific term searches
        queries.append(f'"{term}"')
        
        # Combination searches
        if i < len(key_terms) - 1:
            queries.append(f"{term} AND {key_terms[i+1]}")
    
    return list(set(queries))  # Remove duplicates

def extract_key_terms(text: str) -> List[str]:
    """Extract meaningful terms from research question"""
    # Remove common words and extract nouns/adjectives
    stop_words = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "but"}
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    return [word for word in words if word not in stop_words][:5]