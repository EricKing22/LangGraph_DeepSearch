import argparse
import sys
import uuid
import asyncio
from datetime import datetime
from .graphs.web_search_graph import graph


def list_threads():
    """List all conversation threads (placeholder - requires checkpointer access)"""
    print("\n📋 Thread Management")
    print("=" * 60)
    print("⚠️  Thread listing requires LangGraph Studio or custom checkpointer.")
    print("💡 Tip: Use 'langgraph dev' and access the Studio UI for thread management.")
    print("=" * 60 + "\n")


def show_memory():
    """Show learned lessons from memory store (placeholder - requires store access)"""
    print("\n🧠 Memory Store")
    print("=" * 60)
    print("⚠️  Direct memory access requires running within LangGraph context.")
    print("💡 Tip: Use 'langgraph dev' to inspect the store via LangGraph Studio.")
    print("📚 Lessons are automatically recalled at the start of each search.")
    print("=" * 60 + "\n")


async def run_search(args, thread_id):
    """Async function to run the search graph"""
    thread = {"configurable": {"thread_id": thread_id}}

    if args.verbose:
        print(f"🔍 Processing query: {args.query}")
        print(f"🆔 Thread ID: {thread_id}\n")
    else:
        sys.stdout.write("\r🔍 Processing query ...\n")
        sys.stdout.flush()

    # Initial invocation - will stop at human_feedback interrupt unless --no-feedback
    initial_state = {"query": args.query} if args.query else None

    async for update in graph.astream(initial_state, thread, stream_mode="updates"):
        # update is a dict with node_name as key and state updates as value
        for node_name, node_update in update.items():
            if args.verbose:
                print(f"🔄 Executing node: {node_name}")

            # Print AI messages as they come from node updates
            if "messages" in node_update and node_update["messages"]:
                for msg in node_update["messages"]:
                    if hasattr(msg, "content") and msg.content:
                        # Only print if it's a new message
                        if hasattr(msg, "type"):
                            if msg.type == "ai":
                                if not args.verbose:
                                    sys.stdout.write("\r✓ Query processed!     \n")
                                    sys.stdout.flush()
                                print(f"\n🤖 [{node_name}] {msg.content}")

            # Show recalled notes if verbose
            if args.verbose and "recalled_notes" in node_update:
                notes = node_update["recalled_notes"]
                if notes:
                    print(f"💭 Recalled {len(notes)} past experience(s)")

    # Check if we're at an interrupt point
    state = await graph.aget_state(thread)

    # Handle human feedback loop
    if args.no_feedback:
        # Skip human feedback and auto-approve
        if state.next and "human_feedback" in state.next:
            if args.verbose:
                print("\n⚡ Auto-feedback mode: Proceeding with generated questions\n")
            from langchain_core.messages import HumanMessage

            await graph.aupdate_state(
                thread,
                {
                    "messages": [
                        HumanMessage(content="The questions look good, please proceed.")
                    ]
                },
            )

            # Continue execution
            async for update in graph.astream(None, thread, stream_mode="updates"):
                for node_name, node_update in update.items():
                    if args.verbose:
                        print(f"🔄 Executing node: {node_name}")
                    if "messages" in node_update and node_update["messages"]:
                        for msg in node_update["messages"]:
                            if hasattr(msg, "content") and msg.content:
                                if hasattr(msg, "type") and msg.type == "ai":
                                    print(f"\n🤖 [{node_name}] {msg.content}")
            state = await graph.aget_state(thread)
    else:
        # Normal interactive feedback loop
        while state.next and "human_feedback" in state.next:
            # Get human feedback
            print("\n💬 Please provide feedback on the sub-questions:")
            print("(Press Enter with no input to proceed as-is)")

            # Run input in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            feedback = await loop.run_in_executor(
                None, lambda: input("\nYour feedback: ").strip()
            )

            if not feedback:
                feedback = "The questions look good, please proceed."

            print(f"\n✓ Received feedback: {feedback}\n")

            # Update state with human feedback message and resume
            from langchain_core.messages import HumanMessage

            # First, update the state with the human feedback message
            await graph.aupdate_state(
                thread, {"messages": [HumanMessage(content=feedback)]}
            )

            # Then resume from the interrupt (pass None to continue)
            async for update in graph.astream(None, thread, stream_mode="updates"):
                # update is a dict with node_name as key and state updates as value
                for node_name, node_update in update.items():
                    if args.verbose:
                        print(f"🔄 Executing node: {node_name}")
                    # Print AI messages as they come from node updates
                    if "messages" in node_update and node_update["messages"]:
                        for msg in node_update["messages"]:
                            if hasattr(msg, "content") and msg.content:
                                if hasattr(msg, "type") and msg.type == "ai":
                                    print(f"\n🤖 [{node_name}] {msg.content}")

            # Check state again for more interrupts
            state = await graph.aget_state(thread)

    # Get final result
    result = state.values

    # Print final results
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

    # Show review score if available
    if args.verbose and "score" in result and result["score"]:
        print(f"\n⭐ Review Score: {result['score']}/10")
        if result.get("strengths"):
            print(f"💪 Strengths: {result['strengths']}")
        if result.get("weaknesses"):
            print(f"⚠️  Weaknesses: {result['weaknesses']}")

    if "sources" in result and result["sources"]:
        print(f"\n📚 Sources consulted: {len(result['sources'])}")
        if args.verbose:
            print("\nSource details:")
            for i, source in enumerate(result["sources"][:5], 1):  # Show first 5
                print(
                    f"  {i}. {source.get('title', 'Untitled')} - {source.get('url', 'No URL')}"
                )

    # Show learning info if verbose
    if args.verbose:
        if result.get("recalled_notes"):
            print(f"\n💭 Used {len(result['recalled_notes'])} past experience(s)")
        if result.get("lesson_learned"):
            print("📝 New lesson learned and saved to memory")

    print(f"\n💾 Thread ID: {thread_id}")
    print("💡 Use --continue {thread_id} to continue this conversation")
    print("\n" + "=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="DeepSearch - AI-powered deep web search with closed-loop learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  deepsearch --query "What is LangGraph?"
  deepsearch --query "Compare Python async frameworks" --thread-id my-research
  deepsearch --query "Explain quantum computing" --no-feedback
  deepsearch --query "AI safety concerns" --verbose
  deepsearch --list-threads
  deepsearch --show-memory
        """,
    )
    parser.add_argument("-q", "--query", type=str, help="The search query to process")
    parser.add_argument(
        "--thread-id",
        type=str,
        help="Thread ID for conversation tracking (auto-generated if not provided)",
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="Skip human feedback step and use auto-generated questions",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed execution information",
    )
    parser.add_argument(
        "--list-threads", action="store_true", help="List all conversation threads"
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
    if args.list_threads:
        list_threads()
        return 0

    if args.show_memory:
        show_memory()
        return 0

    # Validate that query is provided for search operations
    if not args.query and not args.continue_thread:
        parser.error(
            "--query is required unless using --list-threads, --show-memory, or --continue"
        )

    # Generate or use provided thread ID
    if args.continue_thread:
        thread_id = args.continue_thread
        print(f"📝 Continuing thread: {thread_id}")
    elif args.thread_id:
        thread_id = args.thread_id
    else:
        # Auto-generate a meaningful thread ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        thread_id = f"search_{timestamp}_{str(uuid.uuid4())[:8]}"
        if args.verbose:
            print(f"🆔 Generated thread ID: {thread_id}")

    try:
        # Run the async search function
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
