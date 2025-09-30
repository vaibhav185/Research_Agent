# tools/__init__.py
from .arxiv_client import search_arxiv
from .semantic_scholar import search_semantic_scholar
from .citation_analyzer import CitationAnalyzer, normalize_paper_id, batch_process_papers

__all__ = [
    'search_arxiv',
    'search_semantic_scholar', 
    'CitationAnalyzer',
    'normalize_paper_id',
    'batch_process_papers'
]