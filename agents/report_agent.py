# agents/report_agent.py
from models.research_state import ResearchState
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from datetime import datetime

llm = ChatOpenAI(temperature=0.3, model="gpt-4")  # Use better model for final report

def report_generation_node(state: ResearchState) -> ResearchState:
    research_question = state["research_question"]
    summaries = state["paper_summaries"]
    themes = state["key_themes"]
    gaps = state["literature_gaps"]
    
    report = generate_comprehensive_report(research_question, summaries, themes, gaps)
    
    return {
        **state,
        "final_report": report,
        "current_step": "report_generation"
    }

def generate_comprehensive_report(question: str, summaries: Dict, themes: List, gaps: List) -> str:
    """Generate a structured literature review report"""
    
    system_prompt = """You are an expert academic writer. Generate a comprehensive literature review report with the following structure:
    
    1. Introduction and Research Question
    2. Methodology (Search Strategy, Inclusion Criteria)
    3. Thematic Analysis of Literature
    4. Critical Analysis of Key Papers
    5. Identification of Research Gaps
    6. Conclusion and Future Research Directions
    
    Write in formal academic style with clear section headings."""
    
    # Prepare context for the report
    context = f"""
    RESEARCH QUESTION: {question}
    
    NUMBER OF PAPERS REVIEWED: {len(summaries)}
    
    KEY THEMES IDENTIFIED:
    {chr(10).join([f"- {theme['theme']} ({theme['papers_count']} papers)" for theme in themes])}
    
    RESEARCH GAPS IDENTIFIED:
    {chr(10).join([f"- {gap['gap']} (mentioned {gap['frequency']} times)" for gap in gaps])}
    
    PAPER SUMMARIES:
    """
    
    # Add key paper summaries (limit to top 5 by citation count)
    top_papers = sorted(summaries.items(), 
                       key=lambda x: x[1].get('citation_count', 0), 
                       reverse=True)[:5]
    
    for i, (paper_id, summary_data) in enumerate(top_papers, 1):
        context += f"\n\nPAPER {i}:\n{summary_data['summary']}\n"
    
    human_message = f"Please generate a literature review report based on the following research context:\n\n{context}"
    
    response = llm([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_message)
    ])
    
    return response.content