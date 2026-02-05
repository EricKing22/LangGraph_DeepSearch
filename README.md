# LangGraph DeepSearch

Reproducing deep search functionality using LangGraph framework.

## Why LangGraph?

- **Stateful Workflows**: Built-in state management for complex search pipelines
- **Cyclic Graphs**: Support for iterative refinement and multi-step reasoning
- **Conditional Edges**: Dynamic workflow branching based on search quality
- **Checkpointing**: Resume interrupted searches and inspect intermediate states
- **Human-in-the-Loop**: Easy integration of human feedback during search process
- **Streaming Support**: Real-time results as the search progresses
- **Composability**: Modular nodes that can be reused and recombined

## Quick Start

```bash
# Install dependencies
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run examples
python examples/standalone_search.py
python examples/gradio_demo.py
```

## Repository

https://github.com/EricKing22/LangGraph_DeepSearch
