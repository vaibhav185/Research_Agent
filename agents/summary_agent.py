# agents/summary_agent.py
from models.research_state import ResearchState
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import asyncio
from typing import Dict
from typing import Dict, List, Optional, TypedDict

# Initialize LLM
llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")

def summarization_node(state: ResearchState) -> ResearchState:
    papers = state["filtered_papers"]
    summaries = {}
    
    # Process papers in batches to avoid rate limits
    batch_size = 5
    for i in range(0, len(papers), batch_size):
        batch = papers[i:i + batch_size]
        batch_summaries = process_paper_batch(batch)
        summaries.update(batch_summaries)
    
    return {
        **state,
        "paper_summaries": summaries,
        "current_step": "summarize"
    }

def process_paper_batch(papers: List[Dict]) -> Dict[str, Dict]:
    """Process a batch of papers for summarization"""
    summaries = {}
    
    for paper in papers:
        paper_id = paper.get('id', paper.get('title', 'unknown'))
        summary = summarize_paper(paper)
        summaries[paper_id] = summary
    
    return summaries

def summarize_paper(paper: Dict) -> Dict:
    """Generate structured summary for a single paper"""
    system_prompt = """You are an expert academic researcher. Summarize research papers in a structured format focusing on:
    - Key contributions
    - Methodology used
    - Main findings
    - Limitations
    - Relation to broader field"""
    
    paper_text = f"""
    Title: {paper.get('title', 'N/A')}
    Authors: {', '.join(paper.get('authors', [])) if paper.get('authors') else 'N/A'}
    Abstract: {paper.get('abstract', 'N/A')}
    Year: {paper.get('year', 'N/A')}
    """
    
    human_message = f"Please summarize this paper:\n\n{paper_text}"
    
    response = llm([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_message)
    ])
    
    return {
        "summary": response.content,
        "key_contributions": extract_key_points(response.content),
        "methodology": extract_methodology(response.content),
        "year": paper.get('year'),
        "citation_count": paper.get('citation_count', 0)
    }

def extract_key_points(summary: str) -> List[str]:
    """Extract key contributions from summary"""
    # Simple extraction - can be enhanced with more sophisticated NLP
    sentences = summary.split('. ')
    return [s.strip() for s in sentences if any(keyword in s.lower() for keyword in 
            ['contribution', 'propose', 'introduce', 'develop', 'present'])][:3]

def extract_methodology(summary: str) -> List[str]:
    """Extract methodology mentions"""
    methodologies = ['neural network', 'machine learning', 'statistical', 'experiment', 
                    'survey', 'case study', 'simulation', 'analysis']
    found_methods = []
    for method in methodologies:
        if method in summary.lower():
            found_methods.append(method)
    return found_methods