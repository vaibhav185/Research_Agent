# Research_Agent
📚 Academic Literature Review Assistant
A powerful, AI-powered literature review assistant built with LangGraph that automates the process of academic research, analysis, and report generation.

🌟 Features
🔍 Multi-Source Search: Simultaneously search arXiv, Semantic Scholar, and other academic databases

📊 Smart Filtering: Automatic relevance scoring and paper ranking based on citations, recency, and quality

🤖 AI-Powered Analysis: Automated summarization, theme extraction, and gap identification

📝 Structured Reporting: Generate comprehensive literature reviews with proper academic formatting

🔄 Workflow Management: Stateful, multi-step research pipeline with error handling and conditional logic

🏗️ Architecture
This project uses LangGraph to create a stateful, multi-agent workflow for academic research:

text
Research Question → Query Expansion → Multi-Database Search → 
Relevance Filtering → Parallel Summarization → Theme Analysis → 
Gap Identification → Report Generation
📦 Installation
Prerequisites
Python 3.8+

OpenAI API key

(Optional) Semantic Scholar API key

Step 1: Clone the Repository
bash
git clone https://github.com/yourusername/academic-literature-assistant.git
cd academic-literature-assistant
Step 2: Install Dependencies
bash
pip install -r requirements.txt
Step 3: Environment Configuration
Create a .env file in the root directory:

env
OPENAI_API_KEY=your_openai_api_key_here
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key_optional
🚀 Quick Start
Basic Usage
python
from main import main

# Run with default research question
main()
Custom Research Question
python
from graph.research_graph import create_research_graph
from models.research_state import ResearchState

# Initialize the graph
research_graph = create_research_graph()

# Define your research question
research_question = "What are the recent advancements in quantum machine learning?"

# Create initial state
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

# Run the analysis
results = research_graph.invoke(initial_state)
print(results['final_report'])
📁 Project Structure
text
academic-literature-assistant/
│
├── agents/                 # LangGraph node implementations
│   ├── query_agent.py     # Query analysis and expansion
│   ├── search_agent.py    # Multi-database search
│   ├── filter_agent.py    # Relevance filtering
│   ├── summary_agent.py   # Paper summarization
│   ├── theme_agent.py     # Thematic analysis
│   └── report_agent.py    # Report generation
│
├── graph/                 # LangGraph workflow definition
│   └── research_graph.py  # Main graph construction
│
├── models/               # Data models
│   └── research_state.py # State management
│
├── tools/               # External API integrations
│   ├── arxiv_client.py  # arXiv API wrapper
│   └── semantic_scholar.py # Semantic Scholar API
│
├── tests/              # Unit tests
├── examples/           # Usage examples
├── requirements.txt    # Python dependencies
└── main.py            # Entry point
🔧 Configuration
API Keys
The system requires the following API keys:

OpenAI API Key: Required for AI-powered analysis and summarization

Semantic Scholar API Key: Optional, for enhanced search capabilities

Search Parameters
Modify search behavior in agents/search_agent.py:

python
# Adjust these parameters
MAX_RESULTS_PER_QUERY = 10
SEARCH_SOURCES = ['arxiv', 'semantic_scholar']
Filtering Criteria
Customize paper scoring in agents/filter_agent.py:

python
# Weighting factors for relevance scoring
RECENCY_WEIGHT = 0.3
CITATION_WEIGHT = 0.4
QUALITY_WEIGHT = 0.2
PRESTIGE_WEIGHT = 0.1
📊 Output Formats
The system generates multiple outputs:

Structured Literature Review: Comprehensive academic report

Paper Summaries: Individual analysis of each relevant paper

Thematic Analysis: Clustered research themes and trends

Research Gaps: Identified limitations and future directions

Example Output Structure
json
{
  "research_question": "Your research question here",
  "papers_analyzed": 25,
  "key_themes": [
    {"theme": "Theme 1", "papers_count": 8},
    {"theme": "Theme 2", "papers_count": 12}
  ],
  "literature_gaps": [
    {"gap": "Methodological limitation", "frequency": 5}
  ],
  "final_report": "Full literature review text..."
}
🧪 Testing
Run the test suite to ensure everything works correctly:

bash
python -m pytest tests/ -v
Testing Individual Components
python
# Test query expansion
from agents.query_agent import expand_research_query
queries = expand_research_query("machine learning in healthcare")
print(queries)

# Test paper filtering
from agents.filter_agent import calculate_paper_score
sample_paper = {"title": "Test", "year": 2023, "citation_count": 50}
score = calculate_paper_score(sample_paper)
print(f"Paper score: {score}")
🔍 Advanced Usage
Custom Research Domains
Extend the system for specific research domains by modifying the query expansion and filtering logic:

python
# Add domain-specific terminology
DOMAIN_TERMS = {
    "bioinformatics": ["genomic", "proteomic", "sequence alignment"],
    "computer_vision": ["object detection", "semantic segmentation", "CNN"]
}
Adding New Data Sources
Implement new search connectors in the tools/ directory:

python
# tools/new_source.py
def search_new_source(query: str, max_results: int) -> List[Dict]:
    # Implement your custom search logic
    pass
🤝 Contributing
We welcome contributions! Please see our Contributing Guidelines for details.

Development Setup
Fork the repository

Create a feature branch: git checkout -b feature/amazing-feature

Commit changes: git commit -m 'Add amazing feature'

Push to branch: git push origin feature/amazing-feature

Open a Pull Request
