# agents/theme_agent.py
from models.research_state import ResearchState
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from typing import Dict, List, Optional, TypedDict

llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")

def theme_analysis_node(state: ResearchState) -> ResearchState:
    summaries = state["paper_summaries"]
    
    if not summaries:
        return {**state, "key_themes": [], "current_step": "theme_analysis"}
    
    # Extract text for clustering
    all_texts = [summary_data["summary"] for summary_data in summaries.values()]
    
    # Perform thematic analysis
    themes = identify_themes(all_texts)
    gaps = identify_research_gaps(summaries)
    
    return {
        **state,
        "key_themes": themes,
        "literature_gaps": gaps,
        "current_step": "theme_analysis"
    }

def identify_themes(texts: List[str]) -> List[Dict]:
    """Identify key themes across papers using clustering"""
    if len(texts) < 3:
        return [{"theme": "Insufficient papers for thematic analysis", "papers_count": len(texts)}]
    
    # Use TF-IDF and clustering
    vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Cluster papers
    n_clusters = min(5, len(texts))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(tfidf_matrix)
    
    themes = []
    for cluster_id in range(n_clusters):
        cluster_texts = [texts[i] for i in range(len(texts)) if clusters[i] == cluster_id]
        theme_name = extract_theme_name(cluster_texts)
        themes.append({
            "theme": theme_name,
            "papers_count": len(cluster_texts),
            "cluster_id": cluster_id
        })
    
    return themes

def extract_theme_name(texts: List[str]) -> str:
    """Extract theme name using LLM"""
    combined_text = " ".join(texts)[:2000]  # Limit context length
    
    prompt = f"""Based on the following research paper summaries, identify the main theme or topic in 3-5 words:

    {combined_text}
    
    Theme:"""
    
    response = llm([HumanMessage(content=prompt)])
    return response.content.strip()

def identify_research_gaps(summaries: Dict[str, Dict]) -> List[Dict]:
    """Identify research gaps across the literature"""
    all_limitations = []
    
    for paper_id, summary_data in summaries.items():
        limitations = extract_limitations(summary_data["summary"])
        all_limitations.extend(limitations)
    
    # Cluster limitations to find common gaps
    gap_categories = categorize_limitations(all_limitations)
    
    return gap_categories

def extract_limitations(summary: str) -> List[str]:
    """Extract limitation mentions from summaries"""
    limitation_keywords = ['limitation', 'future work', 'challenge', 'drawback', 'constraint']
    sentences = summary.split('. ')
    limitation_sentences = [
        s for s in sentences 
        if any(keyword in s.lower() for keyword in limitation_keywords)
    ]
    return limitation_sentences[:2]  # Limit to top 2 per paper

def categorize_limitations(limitations: List[str]) -> List[Dict]:
    """Categorize limitations into research gaps"""
    if not limitations:
        return [{"gap": "No explicit limitations identified in summaries", "frequency": 1}]
    
    # Simple frequency-based categorization
    gap_counter = Counter()
    for limitation in limitations:
        # Simple keyword-based categorization
        if any(word in limitation.lower() for word in ['data', 'dataset', 'sample']):
            gap_counter['data_limitations'] += 1
        elif any(word in limitation.lower() for word in ['method', 'approach', 'technique']):
            gap_counter['methodological_limitations'] += 1
        elif any(word in limitation.lower() for word in ['scale', 'computation', 'efficiency']):
            gap_counter['scalability_issues'] += 1
        elif any(word in limitation.lower() for word in ['general', 'applicability', 'domain']):
            gap_counter['generalization_issues'] += 1
        else:
            gap_counter['other_limitations'] += 1
    
    return [{"gap": gap, "frequency": count} for gap, count in gap_counter.items()]

# agents/theme_agent.py - Add this function
def analyze_citation_patterns(state: ResearchState) -> Dict:
    """Enhanced theme analysis with citation patterns"""
    from tools.citation_analyzer import CitationAnalyzer
    
    papers = state["filtered_papers"]
    analyzer = CitationAnalyzer()
    
    citation_report = analyzer.generate_citation_report(papers)
    
    # Add citation insights to themes
    themes = state["key_themes"]
    for theme in themes:
        theme['citation_metrics'] = {
            'avg_citations_in_theme': calculate_theme_citations(theme, citation_report),
            'most_influential_paper': find_theme_influential_paper(theme, citation_report)
        }
    
    return {**state, "citation_analysis": citation_report}