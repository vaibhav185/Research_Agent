# agents/filter_agent.py
from models.research_state import ResearchState
from datetime import datetime
from typing import Dict, List, Optional, TypedDict

def filtering_node(state: ResearchState) -> ResearchState:
    papers = state["raw_papers"]
    
    if not papers:
        return {**state, "filtered_papers": [], "current_step": "filter"}
    
    scored_papers = []
    
    for paper in papers:
        score = calculate_paper_score(paper)
        paper['relevance_score'] = score
        scored_papers.append(paper)
    
    # Sort by score and take top 20
    filtered_papers = sorted(scored_papers, key=lambda x: x['relevance_score'], reverse=True)[:20]
    
    return {
        **state,
        "filtered_papers": filtered_papers,
        "current_step": "filter"
    }

def calculate_paper_score(paper: Dict) -> float:
    """Calculate relevance score for a paper"""
    score = 0.0
    
    # Recency (more recent = higher score)
    if 'year' in paper:
        year = int(paper['year'])
        current_year = datetime.now().year
        recency = max(0, 1 - (current_year - year) / 20)  # 20-year window
        score += recency * 0.3
    
    # Citation count (if available)
    if 'citation_count' in paper:
        citations = min(paper['citation_count'] / 100, 1.0)  # Normalize
        score += citations * 0.4
    
    # Abstract length (proxy for quality)
    if 'abstract' in paper:
        abstract_len = len(paper['abstract'])
        abstract_score = min(abstract_len / 500, 1.0)
        score += abstract_score * 0.2
    
    # Journal/conference prestige (simplified)
    if 'venue' in paper:
        prestige_venues = ['nature', 'science', 'neurips', 'icml', 'acl', 'cvpr']
        venue = paper['venue'].lower()
        if any(pv in venue for pv in prestige_venues):
            score += 0.1
    
    return min(score, 1.0)