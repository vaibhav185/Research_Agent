# main.py
from graph.research_graph import create_research_graph
from models.research_state import ResearchState
import asyncio

def main():
    # Initialize the graph
    research_graph = create_research_graph()
    
    # Example research question
    research_question = "What are the recent advancements in transformer architectures for natural language processing?"
    
    # Initial state
    initial_state = ResearchState(
        research_question=research_question,
        search_queries=[],
        raw_papers=[],
        filtered_papers=[],
        paper_summaries={},
        key_themes=[],
        literature_gaps=[],
        final_report="",
        current_step="start",
        error=None,
        messages=[]
    )
    
    # Run the graph
    print("Starting literature review analysis...")
    final_state = research_graph.invoke(initial_state)
    
    # Display results
    print("\n" + "="*50)
    print("LITERATURE REVIEW COMPLETE")
    print("="*50)
    print(f"Research Question: {final_state['research_question']}")
    print(f"Papers Found: {len(final_state['raw_papers'])}")
    print(f"Papers Analyzed: {len(final_state['filtered_papers'])}")
    print(f"Key Themes Identified: {len(final_state['key_themes'])}")
    print(f"Research Gaps Found: {len(final_state['literature_gaps'])}")
    print("\nFINAL REPORT:")
    print("="*50)
    print(final_state['final_report'])

if __name__ == "__main__":
    main()