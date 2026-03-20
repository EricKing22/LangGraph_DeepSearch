# LangGraph DeepSearch

A powerful **AI-powered deep research agent** built with LangGraph that combines **human-in-the-loop** control, **automatic quality review**, and **self-learning capabilities** to deliver high-quality research results across multiple platforms.

> **⚠️ Development Status**: This project is currently under active development. More features and improvements will be added over time.

## 🔄 Graph Architecture

![DeepSearch Graph](assets/DeepSearch_Graph.png)

*The LangGraph workflow showing the complete search pipeline with conditional edges, human-in-the-loop feedback, and iterative refinement.*

---

## ✨ Key Features

### 1. 🔄 Human-in-the-Loop Control
Full control over the research process with interruptible workflows:
- **Plan Review**: Review and modify AI-generated sub-questions before execution
- **Iterative Refinement**: Provide feedback to improve search strategy
- **Transparent Decision-Making**: See exactly what the agent is planning to search for

### 2. 📊 Automatic Quality Review
Built-in quality assurance with automatic scoring:
- **Auto-Scoring**: Every summary receives a score (1-10) based on completeness and accuracy
- **Strengths/Weaknesses Analysis**: Detailed feedback on what's good and what needs improvement
- **Iterative Improvement**: Low scores trigger automatic re-generation with specific improvement focus

### 3. 🧠 Self-Learning System (Closed-loop Learning)
The agent learns from every interaction to improve over time:
- **Recall Past Experiences**: Before planning, the agent recalls relevant lessons from previous tasks
- **Compare Plan A & B**: Analyzes the difference between AI's initial plan and human-modified plan
- **Extract Actionable Lessons**: LLM distills meaningful insights from human corrections
- **Persistent Memory**: Stores lessons in LangGraph Store for cross-session learning

> **Key Benefit**: Over time, the agent generates better initial plans that require less human correction.

### 4. 🌐 Multi-Platform Search
Search across multiple platforms in a single query:

| Platform | Flag | What it searches |
|---|---|---|
| **Web** (Tavily) | `--sources web` | General web, news, documentation |
| **HuggingFace** | `--sources hf` | ML models, datasets, spaces, papers |
| **arXiv** | `--sources arxiv` | Academic research papers |

Combine platforms freely: `--sources web hf arxiv`

---

## ⚙️ How the Three Core Mechanisms Work Together

```
Before Search          After Search           After Completion
─────────────          ────────────           ────────────────
Agent → Plan A    →    Summary generated  →   Compare Plan A vs B
User modifies          Auto-scored (1-10)      Extract lesson
     ↓                      ↓                  Save to memory
   Plan B             score > 7 → done
     ↓                score ≤ 7 → improve
  Execute
```

---

## 📚 Why LangGraph?

- **Stateful Workflows**: Built-in state management for complex search pipelines
- **Cyclic Graphs**: Support for iterative refinement and multi-step reasoning
- **Conditional Edges**: Dynamic workflow branching based on search quality
- **Checkpointing**: Resume interrupted searches and inspect intermediate states
- **Human-in-the-Loop**: Easy integration of human feedback during search process
- **Streaming Support**: Real-time results as the search progresses

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- An OpenAI-compatible API key
- Tavily API key (for web search)

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

### Minimal `.env` Configuration

```env
# LLM
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o

# Web search (required for --sources web)
TAVILY_API_KEY=your_tavily_api_key

# HuggingFace search (optional, no key needed for public content)
HUGGINGFACE_SEARCH_ENABLED=false
# HUGGINGFACE_TOKEN=hf_...   # only needed for private/gated models

# arXiv search (optional, no API key required)
ARXIV_SEARCH_ENABLED=false
```

---

## 💻 CLI Usage

### Check Active Configuration

```bash
deepsearch --show-config
```

```
⚙️  Active Configuration
============================================================
  LLM Provider      : OpenAI
  OpenAI Model      : gpt-4o
  Search Sources    : web, hf, arxiv
  Tavily API Key    : ✓ set
  HuggingFace Search: enabled
  HF Search Types   : models, datasets, spaces, papers
  HF Token          : ✗ not set (public only)
  arXiv Search      : enabled
  Max Sub-Questions : 5
  Max Search Results: 5
  Max Review Loops  : 1
  Self-Learning     : enabled
============================================================
```

---

### 🌐 Web Research (Tavily)

General web research — news, documentation, blog posts, official sites.

```bash
# Basic web research
deepsearch --query "What is the difference between LangSmith and LangGraph?"

# Web-only research with custom thread for continuity
deepsearch --query "Latest developments in AI agents" \
           --sources web \
           --thread-id ai-agents-research

# Skip human review step (auto-approve sub-questions)
deepsearch --query "How does RAG work in production?" \
           --sources web \
           --no-feedback

# Verbose output with source details and node execution trace
deepsearch --query "Transformer architecture explained" \
           --sources web \
           --verbose
```

**Example session:**

```
🔍 Processing query ... [WEB]

🤖 [recall] Recalled 1 past experience(s):
- When researching architecture topics, include both theory and practical examples

🤖 [plan] I'm now going to search for these topics:
**1**. What is the Transformer architecture and its key components?
**2**. How does self-attention mechanism work in Transformers?
**3**. What are practical applications of Transformer models?

🤖 [feedback] I've generated the following sub-questions:

1. What is the Transformer architecture and its key components?
2. How does self-attention mechanism work in Transformers?
3. What are practical applications of Transformer models?

Please provide feedback (or press Enter to approve):

Your feedback: ↵

🤖 [search_web] [WEB] Search for: **Transformer architecture** (Found 4 relevant results)
🤖 [search_web] [WEB] Search for: **self-attention mechanism** (Found 5 relevant results)
🤖 [search_web] [WEB] Search for: **Transformer applications** (Found 3 relevant results)

🤖 [summarise] [Generating comprehensive answer...]

============================================================
🎯 FINAL SEARCH RESULTS
============================================================

📄 Summary:
# Transformer Architecture

...

⭐ Review Score: 8/10  [████████░░]
💪 Strengths  : Comprehensive coverage of attention mechanism...
📚 Sources consulted: 12 (12 web)
```

---

### 🤗 HuggingFace Research

Research ML models, datasets, demo spaces, and HuggingFace papers.

```bash
# Enable HuggingFace search via flag (overrides .env setting)
deepsearch --query "Best open-source LLMs for code generation" \
           --sources hf

# Combine web + HuggingFace for richer results
deepsearch --query "Multimodal vision-language models" \
           --sources web hf

# Research a specific model type — HuggingFace only, more focused
deepsearch --query "Text-to-image diffusion models" \
           --sources hf \
           --max-questions 3 \
           --no-feedback
```

**Example output (HuggingFace sources):**

```
🔍 Processing query ... [HF]

🤖 [plan] I'm now going to search for these topics:
**1**. Top HuggingFace models for code generation with benchmark scores
**2**. Available code generation datasets on HuggingFace Hub
**3**. Interactive code generation demos on HuggingFace Spaces

[After search...]

📄 Summary:
## Best Open-Source LLMs for Code Generation

### Top Models
- **[HF Model] Qwen/Qwen2.5-Coder-32B-Instruct** — ...
- **[HF Model] deepseek-ai/DeepSeek-Coder-V2** — ...

### Key Datasets
- **[HF Dataset] bigcode/the-stack-v2** — ...

...

⭐ Review Score: 9/10  [█████████░]
📚 Sources consulted: 8 (8 HuggingFace)
```

**Persist a HuggingFace research session:**

```bash
# Start a research thread
deepsearch --query "RLHF training techniques" \
           --sources web hf \
           --thread-id rlhf-research

# Continue the same thread later with a follow-up
deepsearch --query "RLHF vs DPO comparison" \
           --sources web hf \
           --continue rlhf-research
```

---

### 📄 arXiv Academic Research

Deep-dive into peer-reviewed research papers. No API key required.

```bash
# Pure academic research via arXiv
deepsearch --query "Mixture of Experts scaling laws" \
           --sources arxiv

# Combine web + arXiv for both theory and practical coverage
deepsearch --query "Mechanistic interpretability in large language models" \
           --sources web arxiv

# All three platforms: web context + HuggingFace implementations + arXiv papers
deepsearch --query "State space models vs Transformers" \
           --sources web hf arxiv \
           --max-questions 4
```

**Example output (arXiv sources):**

```
🔍 Processing query ... [ARXIV]

🤖 [plan] I'm now going to search for these topics:
**1**. Mixture of Experts architecture and theoretical foundations
**2**. Scaling laws for MoE models compared to dense transformers
**3**. Recent MoE implementations and benchmark results

[After search...]

📄 Summary:
## Mixture of Experts: Scaling Laws

### Key Papers
- **[arXiv] Mixtral of Experts** (Jiang et al., 2024)
  arXiv:2401.04088 — Introduces sparse MoE with 8 experts per layer...

- **[arXiv] Scaling Laws for Fine-Grained Mixture of Experts**
  arXiv:2402.07871 — Derives optimal expert count vs. compute tradeoffs...

...

⭐ Review Score: 9/10  [█████████░]
📚 Sources consulted: 10 (10 arXiv)
```

---

### 🔀 Combined Multi-Platform Research

Use all sources together for the most comprehensive results.

```bash
# Full research: web news + HuggingFace models + arXiv papers
deepsearch --query "Retrieval-Augmented Generation best practices" \
           --sources web hf arxiv

# Academic + implementations (great for replicating papers)
deepsearch --query "LoRA fine-tuning for vision models" \
           --sources hf arxiv \
           --max-questions 5 \
           --verbose

# Quick scan with no human review
deepsearch --query "Flash Attention optimizations" \
           --sources web arxiv \
           --no-feedback \
           --max-questions 3
```

**Source breakdown in output:**

```
📚 Sources consulted: 18 (6 web, 7 HuggingFace, 5 arXiv)
```

---

### 📋 All CLI Flags

```
usage: deepsearch [-q QUERY] [--sources SOURCE [SOURCE ...]]
                  [--max-questions N] [--no-feedback] [--verbose]
                  [--thread-id ID] [--continue THREAD_ID]
                  [--show-config] [--list-threads] [--show-memory]

Core:
  -q, --query TEXT          The search query to process
  --thread-id ID            Custom thread ID (auto-generated if omitted)

Search Control:
  --sources SOURCE ...      Backends: web | hf | arxiv  (default: env config)
  --max-questions N         Override max sub-questions for this run
  --no-feedback             Auto-approve generated sub-questions

Output:
  -v, --verbose             Show node execution trace and full source list

Utilities:
  --show-config             Display active configuration and exit
  --list-threads            List conversation threads (requires LangGraph Studio)
  --show-memory             Show memory store info
  --continue THREAD_ID      Resume a previous conversation thread
```

---

## 🛠️ Local Development with LangGraph Studio

For visual debugging and graph inspection:

```bash
langgraph dev
```

This starts a local API server at `http://localhost:2024` with:
- Visual graph execution in LangGraph Studio
- Hot-reload on code changes
- Automatic checkpointing
- Memory store inspection

If port 2024 is in use: `langgraph dev --port 8080`

---

## 📁 Project Structure

```
LangGraph_DeepSearch/
├── src/
│   ├── graphs/
│   │   └── web_search_graph.py    # Main graph definition and edge routing
│   ├── nodes/
│   │   ├── question_nodes.py      # Planning, summarise, review routing
│   │   ├── search_nodes.py        # Multi-platform search orchestration
│   │   ├── review_nodes.py        # Quality scoring (1-10)
│   │   └── learning_nodes.py      # Recall, compare, learn nodes
│   ├── state/
│   │   └── states.py              # WebSearchState, Search, Plan, etc.
│   ├── tools/
│   │   ├── search_tool.py         # Tavily web search
│   │   ├── huggingface_tool.py    # HuggingFace Hub search
│   │   ├── arxiv_tool.py          # arXiv paper search
│   │   └── consult_note.py        # LangGraph Store (lessons memory)
│   ├── prompts/
│   │   └── search_prompts.py      # LLM prompts for all nodes
│   ├── cli.py                     # Command-line interface
│   ├── config.py                  # Configuration & logging
│   └── llm.py                     # LLM provider initialization
├── tests/
│   ├── test_graphs.py
│   ├── test_nodes.py
│   └── test_tools.py
├── langgraph.json                 # LangGraph config (Store, graph entrypoint)
├── .env.example                   # Environment template
├── pyproject.toml                 # Dependencies & CLI entrypoint
└── README.md
```

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `command not found: deepsearch` | Run `pip install -e .` in the project root |
| `TAVILY_API_KEY not set` | Add key to `.env` or use `--sources hf arxiv` to skip web |
| `huggingface_hub not found` | Run `pip install huggingface_hub` |
| Port 2024 in use | `langgraph dev --port 8080` |
| Low review scores | Provide feedback at the sub-question prompt to guide the search |
