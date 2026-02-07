# LangGraph DeepSearch

Reproducing deep search functionality using LangGraph framework.

## 📚 Why LangGraph?

- **Stateful Workflows**: Built-in state management for complex search pipelines
- **Cyclic Graphs**: Support for iterative refinement and multi-step reasoning
- **Conditional Edges**: Dynamic workflow branching based on search quality
- **Checkpointing**: Resume interrupted searches and inspect intermediate states
- **Human-in-the-Loop**: Easy integration of human feedback during search process
- **Streaming Support**: Real-time results as the search progresses
- **Composability**: Modular nodes that can be reused and recombined

## Quick Start

### Prerequisites

- Python 3.11+
- LangGraph CLI (`pip install langgraph-cli`)

### Installation

```bash
# Clone the repository
git clone https://github.com/EricKing22/LangGraph_DeepSearch.git
cd LangGraph_DeepSearch

# Install dependencies
pip install -e .

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Environment Configuration

Create a `.env` file in the project root with the following variables:

```env
# LLM Configuration (choose one or more)
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o
OPENAI_TEMPERATURE=0.7

# Search API
TAVILY_API_KEY=your_tavily_api_key

# Search Configuration
MAX_SUB_QUESTIONS=5
MAX_SEARCH_RESULTS=5
SEARCH_TIMEOUT=10
```

## Deployment & Usage

### Local Development with LangGraph Dev

The easiest way to run and test this project is using `langgraph dev`:

```bash
# Start the LangGraph development server
langgraph dev
```

This command will:
- Start a local API server (default: http://localhost:2024)
- Enable the LangGraph Studio UI for visual debugging
- Provide hot-reload for code changes
- Set up automatic checkpointing for conversation state

### How It Works

1. **Query Extraction**: Extracts the user's query from messages
2. **Query Analysis**: Decides whether to break down the query into sub-questions
3. **Question Generation** (Optional): Creates focused sub-questions for comprehensive search
4. **Web Search**: Executes Tavily searches for each question
5. **Relevance Filtering**: Uses LLM to filter out irrelevant results
6. **Synthesis**: Generates a comprehensive answer with citations

### Features

- **Automatic State Management**: LangGraph Cloud handles checkpointing automatically
- **Multi-turn Conversations**: Each thread maintains conversation history
- **Human-in-the-Loop**: Support for human feedback during question generation
- **Iterative Refinement**: Can regenerate sub-questions based on feedback
- **Source Attribution**: Includes citations to original sources


## Troubleshooting

### Command Not Found: `langgraph`

Install the LangGraph CLI:

```bash
pip install langgraph-cli
```

### Port Already in Use

If port 2024 is already in use, specify a different port:

```bash
langgraph dev --port 8080
```

### API Key Issues

Ensure your `.env` file is properly configured

## Project Structure

```
LangGraph_DeepSearch/
├── src/
│   ├── graphs/
│   │   └── web_search_graph.py    # Main graph definition
│   ├── nodes/
│   │   ├── question_nodes.py      # Query processing nodes
│   │   └── search_nodes.py        # Search execution nodes
│   ├── state/
│   │   └── states.py              # State schemas
│   ├── tools/
│   │   └── search_tool.py         # Tavily search integration
│   ├── prompts/
│   │   └── search_prompts.py      # LLM prompts
│   ├── config.py                  # Configuration management
│   └── llm.py                     # LLM initialization
├── langgraph.json                 # LangGraph configuration
├── .env                           # Environment variables
├── .env.example                   # Environment template
├── pyproject.toml                 # Project dependencies
└── README.md                      # This file
```
=
