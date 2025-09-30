# tools/arxiv_client.py
import arxiv
import asyncio
from typing import List, Dict

def search_arxiv(query: str, max_results: int = 10) -> List[Dict]:
    """Search arXiv for papers"""
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        for result in client.results(search):
            paper = {
                'id': result.entry_id,
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'abstract': result.summary,
                'year': result.published.year,
                'pdf_url': result.pdf_url,
                'published': result.published.isoformat(),
                'source': 'arxiv'
            }
            papers.append(paper)
        
        return papers
    except Exception as e:
        print(f"Error searching arXiv: {e}")
        return []