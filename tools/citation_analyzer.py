# tools/citation_analyzer.py
import requests
from typing import List, Dict, Optional
from collections import defaultdict, Counter
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
import json
import numpy as np


class CitationAnalyzer:
    """
    Analyzes citation patterns and networks for academic papers
    """
    
    def __init__(self):
        self.semantic_scholar_base_url = "https://api.semanticscholar.org/graph/v1"
        
    def get_citation_data(self, paper_id: str) -> Optional[Dict]:
        """
        Get citation data for a paper from Semantic Scholar
        """
        try:
            url = f"{self.semantic_scholar_base_url}/paper/{paper_id}"
            params = {
                'fields': 'citationCount,referenceCount,citations,references,title,authors,year'
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error fetching citation data for {paper_id}: {response.status_code}")
                return None
        except Exception as e:
            print(f"Exception in get_citation_data: {e}")
            return None
    
    def analyze_citation_network(self, papers: List[Dict]) -> Dict:
        """
        Analyze citation relationships between papers
        """
        citation_graph = nx.DiGraph()
        paper_stats = {}
        
        for paper in papers:
            paper_id = paper.get('id')
            if not paper_id:
                continue
                
            citation_data = self.get_citation_data(paper_id)
            if citation_data:
                # Add paper to graph
                citation_graph.add_node(
                    paper_id,
                    title=paper.get('title', ''),
                    year=paper.get('year'),
                    citation_count=citation_data.get('citationCount', 0)
                )
                
                # Add citation edges
                if 'citations' in citation_data:
                    for cited_paper in citation_data['citations']:
                        if cited_paper.get('paperId'):
                            citation_graph.add_edge(paper_id, cited_paper['paperId'])
                
                # Add reference edges (this paper cites others)
                if 'references' in citation_data:
                    for ref_paper in citation_data['references']:
                        if ref_paper.get('paperId'):
                            citation_graph.add_edge(ref_paper['paperId'], paper_id)
                
                # Store paper statistics
                paper_stats[paper_id] = {
                    'citation_count': citation_data.get('citationCount', 0),
                    'reference_count': citation_data.get('referenceCount', 0),
                    'title': paper.get('title', ''),
                    'year': paper.get('year')
                }
        
        return {
            'graph': citation_graph,
            'paper_stats': paper_stats,
            'network_metrics': self.calculate_network_metrics(citation_graph, paper_stats)
        }
    
    def calculate_network_metrics(self, graph: nx.DiGraph, paper_stats: Dict) -> Dict:
        """
        Calculate various network metrics for the citation graph
        """
        if len(graph.nodes) == 0:
            return {}
        
        metrics = {}
        
        # Basic graph metrics
        metrics['num_nodes'] = len(graph.nodes)
        metrics['num_edges'] = len(graph.edges)
        metrics['density'] = nx.density(graph)
        
        # Degree centrality
        in_degree_centrality = nx.in_degree_centrality(graph)
        out_degree_centrality = nx.out_degree_centrality(graph)
        
        # Find most influential papers (high in-degree = highly cited)
        if in_degree_centrality:
            most_cited = max(in_degree_centrality.items(), key=lambda x: x[1])
            metrics['most_cited_paper'] = {
                'paper_id': most_cited[0],
                'centrality': most_cited[1],
                'title': paper_stats.get(most_cited[0], {}).get('title', 'Unknown'),
                'citation_count': paper_stats.get(most_cited[0], {}).get('citation_count', 0)
            }
        
        # Find most citing papers (high out-degree = cite many others)
        if out_degree_centrality:
            most_citing = max(out_degree_centrality.items(), key=lambda x: x[1])
            metrics['most_citing_paper'] = {
                'paper_id': most_citing[0],
                'centrality': most_citing[1],
                'title': paper_stats.get(most_citing[0], {}).get('title', 'Unknown')
            }
        
        # Citation statistics
        citation_counts = [stats.get('citation_count', 0) for stats in paper_stats.values()]
        if citation_counts:
            metrics['avg_citations'] = sum(citation_counts) / len(citation_counts)
            metrics['max_citations'] = max(citation_counts)
            metrics['min_citations'] = min(citation_counts)
        
        # Temporal analysis
        years = [stats.get('year') for stats in paper_stats.values() if stats.get('year')]
        if years:
            metrics['publication_range'] = {
                'oldest': min(years),
                'newest': max(years),
                'span': max(years) - min(years) if years else 0
            }
        
        return metrics
    
    def identify_influential_papers(self, papers: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Identify the most influential papers based on citation metrics
        """
        citation_data = self.analyze_citation_network(papers)
        graph = citation_data['graph']
        paper_stats = citation_data['paper_stats']
        
        if len(graph.nodes) == 0:
            return []
        
        # Calculate multiple influence metrics
        papers_with_scores = []
        
        for paper_id, stats in paper_stats.items():
            if paper_id in graph:
                # Combine multiple metrics for overall influence score
                in_degree = graph.in_degree(paper_id)
                page_rank = nx.pagerank(graph).get(paper_id, 0)
                citation_count = stats.get('citation_count', 0)
                
                # Weighted influence score
                influence_score = (
                    citation_count * 0.5 +  # Raw citation count
                    in_degree * 0.3 +       # Network position
                    page_rank * 0.2         # PageRank importance
                )
                
                papers_with_scores.append({
                    'paper_id': paper_id,
                    'title': stats.get('title', ''),
                    'year': stats.get('year'),
                    'citation_count': citation_count,
                    'in_degree': in_degree,
                    'page_rank': page_rank,
                    'influence_score': influence_score,
                    'authors': stats.get('authors', [])
                })
        
        # Sort by influence score and return top k
        influential_papers = sorted(
            papers_with_scores, 
            key=lambda x: x['influence_score'], 
            reverse=True
        )[:top_k]
        
        return influential_papers
    
    def detect_research_clusters(self, papers: List[Dict]) -> List[Dict]:
        """
        Detect research clusters or communities in the citation network
        """
        citation_data = self.analyze_citation_network(papers)
        graph = citation_data['graph']
        
        if len(graph.nodes) < 3:
            return [{'cluster_id': 0, 'papers': list(graph.nodes), 'size': len(graph.nodes)}]
        
        # Convert to undirected graph for community detection
        undirected_graph = graph.to_undirected()
        
        # Use Louvain method for community detection
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(undirected_graph)
            
            # Group papers by cluster
            clusters = defaultdict(list)
            for paper_id, cluster_id in partition.items():
                clusters[cluster_id].append(paper_id)
            
            # Format cluster information
            research_clusters = []
            for cluster_id, paper_ids in clusters.items():
                cluster_papers = [
                    {
                        'paper_id': pid,
                        'title': citation_data['paper_stats'].get(pid, {}).get('title', ''),
                        'citation_count': citation_data['paper_stats'].get(pid, {}).get('citation_count', 0)
                    }
                    for pid in paper_ids
                ]
                
                research_clusters.append({
                    'cluster_id': cluster_id,
                    'size': len(paper_ids),
                    'papers': cluster_papers,
                    'avg_citations': sum(p['citation_count'] for p in cluster_papers) / len(cluster_papers) if cluster_papers else 0
                })
            
            return research_clusters
            
        except ImportError:
            # Fallback to simple connected components
            clusters = []
            for i, component in enumerate(nx.connected_components(undirected_graph)):
                cluster_papers = [
                    {
                        'paper_id': pid,
                        'title': citation_data['paper_stats'].get(pid, {}).get('title', ''),
                        'citation_count': citation_data['paper_stats'].get(pid, {}).get('citation_count', 0)
                    }
                    for pid in component
                ]
                
                clusters.append({
                    'cluster_id': i,
                    'size': len(component),
                    'papers': cluster_papers,
                    'avg_citations': sum(p['citation_count'] for p in cluster_papers) / len(cluster_papers) if cluster_papers else 0
                })
            
            return clusters
    
    def generate_citation_report(self, papers: List[Dict]) -> Dict:
        """
        Generate a comprehensive citation analysis report
        """
        citation_data = self.analyze_citation_network(papers)
        influential_papers = self.identify_influential_papers(papers)
        research_clusters = self.detect_research_clusters(papers)
        
        report = {
            'summary': {
                'total_papers_analyzed': len(papers),
                'papers_with_citation_data': len(citation_data['paper_stats']),
                'analysis_timestamp': datetime.now().isoformat()
            },
            'network_metrics': citation_data['network_metrics'],
            'influential_papers': influential_papers,
            'research_clusters': research_clusters,
            'temporal_analysis': self.analyze_temporal_trends(papers),
            'collaboration_analysis': self.analyze_collaboration_patterns(papers)
        }
        
        return report
    
    def analyze_temporal_trends(self, papers: List[Dict]) -> Dict:
        """
        Analyze citation trends over time
        """
        years = [p.get('year') for p in papers if p.get('year')]
        if not years:
            return {}
        
        year_counts = Counter(years)
        
        return {
            'publication_years': dict(year_counts),
            'trend_analysis': {
                'oldest_publication': min(years),
                'newest_publication': max(years),
                'most_productive_year': max(year_counts.items(), key=lambda x: x[1])[0] if year_counts else None
            }
        }
    
    def analyze_collaboration_patterns(self, papers: List[Dict]) -> Dict:
        """
        Analyze author collaboration patterns
        """
        author_network = defaultdict(set)
        paper_count_per_author = Counter()
        
        for paper in papers:
            authors = paper.get('authors', [])
            for i, author in enumerate(authors):
                paper_count_per_author[author] += 1
                # Connect authors who co-authored this paper
                for j, other_author in enumerate(authors):
                    if i != j:
                        author_network[author].add(other_author)
        
        return {
            'total_unique_authors': len(paper_count_per_author),
            'most_prolific_authors': paper_count_per_author.most_common(10),
            'collaboration_network_size': len(author_network)
        }
    
    def visualize_citation_network(self, papers: List[Dict], save_path: Optional[str] = None):
        """
        Generate a visualization of the citation network
        """
        try:
            citation_data = self.analyze_citation_network(papers)
            graph = citation_data['graph']
            
            if len(graph.nodes) == 0:
                print("No citation data available for visualization")
                return
            
            plt.figure(figsize=(12, 10))
            
            # Use spring layout for better visualization
            pos = nx.spring_layout(graph, k=1, iterations=50)
            
            # Draw the network
            nx.draw_networkx_nodes(graph, pos, node_size=50, node_color='lightblue')
            nx.draw_networkx_edges(graph, pos, edge_color='gray', arrows=True, arrowsize=10)
            
            # Add labels for important nodes (high citation count)
            important_nodes = [
                node for node, data in graph.nodes(data=True) 
                if data.get('citation_count', 0) > np.percentile(
                    [d.get('citation_count', 0) for d in graph.nodes(data=True)], 75
                )
            ]
            
            labels = {node: f"{data.get('title', '')[:30]}..." 
                     for node, data in graph.nodes(data=True) if node in important_nodes}
            nx.draw_networkx_labels(graph, pos, labels, font_size=8)
            
            plt.title("Citation Network Analysis")
            plt.axis('off')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Network visualization saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            print(f"Error generating visualization: {e}")

# Utility functions
def normalize_paper_id(paper_id: str) -> str:
    """
    Normalize paper ID for consistent API calls
    """
    # Remove arXiv version suffix if present
    if 'v' in paper_id and paper_id.startswith('arXiv:'):
        return paper_id.split('v')[0]
    return paper_id

def batch_process_papers(paper_ids: List[str], batch_size: int = 10) -> List[Dict]:
    """
    Process papers in batches to avoid API rate limits
    """
    analyzer = CitationAnalyzer()
    results = []
    
    for i in range(0, len(paper_ids), batch_size):
        batch = paper_ids[i:i + batch_size]
        batch_results = []
        
        for paper_id in batch:
            data = analyzer.get_citation_data(paper_id)
            if data:
                batch_results.append(data)
        
        results.extend(batch_results)
        
        # Be nice to the API
        import time
        time.sleep(1)
    
    return results

# Example usage
if __name__ == "__main__":
    # Example paper IDs (arXiv IDs or Semantic Scholar IDs)
    example_papers = [
        {"id": "1706.03762", "title": "Attention Is All You Need", "year": 2017},
        {"id": "1810.04805", "title": "BERT: Pre-training of Deep Bidirectional Transformers", "year": 2018}
    ]
    
    analyzer = CitationAnalyzer()
    report = analyzer.generate_citation_report(example_papers)
    
    print("Citation Analysis Report:")
    print(json.dumps(report, indent=2, ensure_ascii=False))