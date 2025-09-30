# agents/search_agent.py
from models.research_state import ResearchState
from tools.arxiv_client import search_arxiv
from tools.semantic_scholar import search_semantic_scholar
import asyncio
from typing import Dict, List, Optional, TypedDict

def search_node(state: ResearchState) -> ResearchState:
    queries = state["search_queries"]
    all_papers = []
    
    for query in queries:
        # Search multiple sources
        arxiv_results = search_arxiv(query, max_results=10)
        semantic_results = search_semantic_scholar(query, max_results=10)
        
        all_papers.extend(arxiv_results)
        all_papers.extend(semantic_results)
    
    # Remove duplicates by paper title
    unique_papers = remove_duplicate_papers(all_papers)
    
    return {
        **state,
        "raw_papers": unique_papers,
        "current_step": "search"
    }

def remove_duplicate_papers(papers: List[Dict]) -> List[Dict]:
    """Remove duplicate papers based on title similarity"""
    seen_titles = set()
    unique_papers = []
    
    for paper in papers:
        title = paper.get('title', '').lower().strip()
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_papers.append(paper)
    
    return unique_papers