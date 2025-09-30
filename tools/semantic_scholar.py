# tools/semantic_scholar.py
import requests
from typing import List, Dict

def search_semantic_scholar(query: str, max_results: int = 10) -> List[Dict]:
    """Search Semantic Scholar for papers"""
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            'query': query,
            'limit': max_results,
            'fields': 'paperId,title,authors,abstract,year,citationCount,venue'
        }
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return []
        
        data = response.json()
        papers = []
        
        for item in data.get('data', []):
            paper = {
                'id': item.get('paperId'),
                'title': item.get('title', ''),
                'authors': [author.get('name', '') for author in item.get('authors', [])],
                'abstract': item.get('abstract', ''),
                'year': item.get('year'),
                'citation_count': item.get('citationCount', 0),
                'venue': item.get('venue', ''),
                'source': 'semantic_scholar'
            }
            papers.append(paper)
        
        return papers
    except Exception as e:
        print(f"Error searching Semantic Scholar: {e}")
        return []