import argparse
import sys
import uuid
import asyncio
from datetime import datetime
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from .graphs.web_search_graph import builder
from src import config as app_config
from src.tools.memory_store import list_all_memories, MEMORY_FILE


# ── Utility commands ──────────────────────────────────────────────────────────


def show_config():
    """Display the active runtime configuration."""
    print("\n⚙️  Active Configuration")
    print("=" * 60)

    # LLM
    llm_provider = "OpenAI"
    if app_config.QWEN_API_KEY:
        llm_provider = "Qwen"
    elif app_config.MINMAX_API_KEY:
        llm_provider = "MiniMax"
    print(f"  LLM Provider      : {llm_provider}")
    print(f"  OpenAI Model      : {app_config.OPENAI_MODEL}")

    # Search sources
    sources = ["web"]
    if app_config.HUGGINGFACE_SEARCH_ENABLED:
        sources.append("hf")
    if app_config.ARXIV_SEARCH_ENABLED:
        sources.append("arxiv")
    print(f"  Search Sources    : {', '.join(sources)}")
    print(
        f"  Tavily API Key    : {'✓ set' if app_config.TAVILY_API_KEY else '✗ missing'}"
    )
    print(
        f"  HuggingFace Search: {'enabled' if app_config.HUGGINGFACE_SEARCH_ENABLED else 'disabled'}"
    )
    if app_config.HUGGINGFACE_SEARCH_ENABLED:
        print(f"  HF Search Types   : {', '.join(app_config.HUGGINGFACE_SEARCH_TYPES)}")
        print(
            f"  HF Token          : {'✓ set' if app_config.HUGGINGFACE_TOKEN else '✗ not set (public only)'}"
        )
    print(
        f"  arXiv Search      : {'enabled' if app_config.ARXIV_SEARCH_ENABLED else 'disabled'}"
    )
    print(
        f"  GitHub Search     : {'enabled' if app_config.GITHUB_SEARCH_ENABLED else 'disabled'}"
    )
    if app_config.GITHUB_SEARCH_ENABLED:
        print(f"  GitHub Types      : {', '.join(app_config.GITHUB_SEARCH_TYPES)}")
        print(
            f"  GitHub Token      : {'✓ set (code search enabled)' if app_config.GITHUB_TOKEN else '✗ not set (repos + issues only)'}"
        )

    # Search params
    print(f"  Max Sub-Questions : {app_config.MAX_SUB_QUESTIONS}")
    print(f"  Max Search Results: {app_config.MAX_SEARCH_RESULTS}")
    print(f"  Max Review Loops  : {app_config.MAX_SUMMARISE_ITERATIONS}")

    # Learning
    print(
        f"  Self-Learning     : {'enabled' if app_config.ENABLE_LEARNING else 'disabled'}"
    )
    print(f"  Memory File       : {MEMORY_FILE}")
    memories = list_all_memories()
    print(f"  Lessons Stored    : {len(memories)}")
    print(f"  Debug Mode        : {'enabled' if app_config.DEBUG else 'disabled'}")
    print("=" * 60 + "\n")


def list_threads():
    """List all conversation threads (placeholder - requires checkpointer access)"""
    print("\n📋 Thread Management")
    print("=" * 60)
    print("⚠️  Thread listing requires LangGraph Studio or custom checkpointer.")
    print("💡 Tip: Use 'langgraph dev' and access the Studio UI for thread management.")
    print("=" * 60 + "\n")


def show_memory():
    """Display all lessons stored in the persistent memory file."""
    memories = list_all_memories()
    print("\n🧠 Long-term Memory Store")
    print(f"   File: {MEMORY_FILE}")
    print("=" * 60)
    if not memories:
        print("  No lessons stored yet.")
        print("  Lessons are saved automatically after each research session")
        print("  where you modified the AI's suggested sub-questions.")
    else:
        print(f"  {len(memories)} lesson(s) stored:\n")
        for m in memories:
            print(f"  [{m['id']}] {m['timestamp'][:10]}")
            print(f"  Task   : {m['task_query']}")
            print(f"  Lesson : {m['lesson']}")
            print()
    print("=" * 60 + "\n")


# ── Core search runner ────────────────────────────────────────────────────────


async def run_search(args, thread_id):
    """Async function to run the search graph"""
    # Compile graph — no Store needed; memory is file-based (memory_store.py)
    graph = builder.compile(checkpointer=MemorySaver())

    # Resolve active search sources (CLI flag > env config)
    if args.sources:
        search_sources = args.sources
    else:
        search_sources = ["web"]
        if app_config.HUGGINGFACE_SEARCH_ENABLED:
            search_sources.append("hf")
        if app_config.ARXIV_SEARCH_ENABLED:
            search_sources.append("arxiv")
        if app_config.GITHUB_SEARCH_ENABLED:
            search_sources.append("github")

    # Resolve max sub-questions (CLI flag > env config)
    max_questions = (
        args.max_questions if args.max_questions else app_config.MAX_SUB_QUESTIONS
    )

    thread = {
        "configurable": {
            "thread_id": thread_id,
            "search_sources": search_sources,
            "max_sub_questions": max_questions,
        }
    }

    # Print startup banner
    sources_label = " + ".join(s.upper() for s in search_sources)
    if args.verbose:
        print(f"🔍 Query      : {args.query}")
        print(f"🆔 Thread ID  : {thread_id}")
        print(f"🌐 Sources    : {sources_label}")
        print(f"❓ Max Qs     : {max_questions}")
        print()
    else:
        sys.stdout.write(f"\r🔍 Processing query ... [{sources_label}]\n")
        sys.stdout.flush()

    initial_state = {"query": args.query} if args.query else None
    auto_approve = args.no_feedback

    # Nodes whose AIMessage output is shown in the final section — skip during streaming
    _FINAL_OUTPUT_NODES = {"summarise"}

    async for update in graph.astream(initial_state, thread, stream_mode="updates"):
        for node_name, node_update in update.items():
            if args.verbose:
                print(f"🔄 Executing node: {node_name}")
            if "messages" in node_update and node_update["messages"]:
                for msg in node_update["messages"]:
                    if hasattr(msg, "content") and msg.content:
                        if hasattr(msg, "type") and msg.type == "ai":
                            if node_name not in _FINAL_OUTPUT_NODES:
                                print(f"\n🤖 [{node_name}] {msg.content}")
            if args.verbose and "recalled_notes" in node_update:
                notes = node_update["recalled_notes"]
                if notes:
                    print(f"💭 Recalled {len(notes)} past experience(s)")

    # Check for interrupt
    state = await graph.aget_state(thread)

    # Handle interrupt loop — interrupt() pauses the graph, resume with Command(resume=)
    while state.next:
        interrupt_prompt = None
        for task in state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                interrupt_prompt = task.interrupts[0].value
                break

        if interrupt_prompt is None:
            break

        if auto_approve:
            if args.verbose:
                print("\n⚡ Auto-feedback mode: Proceeding with generated questions\n")
            feedback = ""
        else:
            print(f"\n{interrupt_prompt}")
            loop = asyncio.get_event_loop()
            feedback = await loop.run_in_executor(
                None, lambda: input("\nYour feedback: ").strip()
            )

        async for update in graph.astream(
            Command(resume=feedback), thread, stream_mode="updates"
        ):
            for node_name, node_update in update.items():
                if args.verbose:
                    print(f"🔄 Executing node: {node_name}")
                if "messages" in node_update and node_update["messages"]:
                    for msg in node_update["messages"]:
                        if hasattr(msg, "content") and msg.content:
                            if hasattr(msg, "type") and msg.type == "ai":
                                if node_name not in _FINAL_OUTPUT_NODES:
                                    print(f"\n🤖 [{node_name}] {msg.content}")

        state = await graph.aget_state(thread)

    result = state.values

    # ── Final output ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🎯 FINAL SEARCH RESULTS")
    print("=" * 60)

    if "summary" in result and result["summary"]:
        print("\n📄 Summary:")
        summary_content = result["summary"]
        if hasattr(summary_content, "content"):
            print(summary_content.content)
        else:
            print(summary_content)

    # Always show review score and trust analysis when available
    if "score" in result and result["score"]:
        score = result["score"]
        bar = "█" * score + "░" * (10 - score)
        print(f"\n⭐ Reliability Score: {score}/10  [{bar}]")
        if result.get("strengths"):
            print(f"\n✅ Trusted aspects:\n{result['strengths']}")
        if result.get("weaknesses"):
            print(f"\n⚠️  Limitations:\n{result['weaknesses']}")

    if "sources" in result and result["sources"]:
        sources_list = result["sources"]
        hf_count = sum(1 for s in sources_list if "[HF" in s.get("title", ""))
        arxiv_count = sum(1 for s in sources_list if "[arXiv]" in s.get("title", ""))
        github_count = sum(1 for s in sources_list if "[GitHub" in s.get("title", ""))
        web_count = len(sources_list) - hf_count - arxiv_count - github_count

        print(f"\n📚 Sources consulted: {len(sources_list)}", end="")
        breakdown = []
        if web_count:
            breakdown.append(f"{web_count} web")
        if hf_count:
            breakdown.append(f"{hf_count} HuggingFace")
        if arxiv_count:
            breakdown.append(f"{arxiv_count} arXiv")
        if github_count:
            breakdown.append(f"{github_count} GitHub")
        if breakdown:
            print(f" ({', '.join(breakdown)})", end="")
        print()

        if args.verbose:
            print("\nSource details:")
            for i, source in enumerate(sources_list[:10], 1):
                print(
                    f"  {i}. {source.get('title', 'Untitled')} — {source.get('url', 'No URL')}"
                )

    if args.verbose and result.get("lesson_learned"):
        print("📝 New lesson learned and saved to memory")

    print(f"\n💾 Thread ID: {thread_id}")
    print("💡 Use --continue {thread_id} to continue this conversation")
    print("\n" + "=" * 60 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="DeepSearch - AI-powered deep research with human-in-the-loop and self-learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  deepsearch --query "What is LangGraph?"
  deepsearch --query "Best image generation models" --sources hf
  deepsearch --query "Compare BERT and GPT" --sources web hf
  deepsearch --query "Transformer attention mechanisms" --sources web arxiv
  deepsearch --query "LLM benchmarks" --sources web hf arxiv
  deepsearch --query "Explain quantum computing" --no-feedback
  deepsearch --query "AI safety concerns" --verbose
  deepsearch --query "LLM fine-tuning" --max-questions 3
  deepsearch --show-config
  deepsearch --list-threads
  deepsearch --show-memory
        """,
    )

    # Core
    parser.add_argument("-q", "--query", type=str, help="The search query to process")
    parser.add_argument(
        "--thread-id",
        type=str,
        help="Thread ID for conversation tracking (auto-generated if not provided)",
    )

    # Search control
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["web", "hf", "arxiv", "github"],
        metavar="SOURCE",
        help="Search backends: web (Tavily), hf (HuggingFace), arxiv, github. "
        "Defaults to env config. Example: --sources web github arxiv",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        metavar="N",
        help=f"Override max sub-questions per query (default: {app_config.MAX_SUB_QUESTIONS})",
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="Skip human feedback step and auto-approve generated questions",
    )

    # Output
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed execution information including node steps and sources",
    )

    # Utility
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Display active configuration and exit",
    )
    parser.add_argument(
        "--list-threads",
        action="store_true",
        help="List all conversation threads",
    )
    parser.add_argument(
        "--show-memory",
        action="store_true",
        help="Show learned lessons from memory store",
    )
    parser.add_argument(
        "--continue",
        dest="continue_thread",
        type=str,
        metavar="THREAD_ID",
        help="Continue an existing conversation thread",
    )

    args = parser.parse_args()

    # Handle utility commands
    if args.show_config:
        show_config()
        return 0

    if args.list_threads:
        list_threads()
        return 0

    if args.show_memory:
        show_memory()
        return 0

    # Validate that query is provided for search operations
    if not args.query and not args.continue_thread:
        parser.error(
            "--query is required unless using --show-config, --list-threads, --show-memory, or --continue"
        )

    # Resolve thread ID
    if args.continue_thread:
        thread_id = args.continue_thread
        print(f"📝 Continuing thread: {thread_id}")
    elif args.thread_id:
        thread_id = args.thread_id
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        thread_id = f"search_{timestamp}_{str(uuid.uuid4())[:8]}"
        if args.verbose:
            print(f"🆔 Generated thread ID: {thread_id}")

    try:
        asyncio.run(run_search(args, thread_id))
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Search interrupted by user.")
        print(f"💾 Thread saved: {thread_id}")
        return 1
    except Exception as e:
        print(f"\n\n❌ Error occurred: {str(e)}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
