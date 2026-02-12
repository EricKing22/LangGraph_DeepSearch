import argparse
import sys
from .graphs.web_search_graph import graph


def main():
    parser = argparse.ArgumentParser(
        description="DeepSearch - AI-powered deep web search"
    )
    parser.add_argument(
        "-query", "--query", type=str, required=True, help="The search query to process"
    )
    parser.add_argument(
        "--thread-id", type=str, default="1", help="Thread ID for conversation tracking"
    )

    args = parser.parse_args()

    # Set up thread configuration
    thread = {"configurable": {"thread_id": args.thread_id}}

    # Initialize and run your graph
    sys.stdout.write("\r🔍 Processing query ...\n")
    sys.stdout.flush()
    try:
        # Initial invocation - will stop at human_feedback interrupt
        for update in graph.stream(
            {"query": args.query}, thread, stream_mode="updates"
        ):
            # update is a dict with node_name as key and state updates as value
            for node_name, node_update in update.items():
                # Print AI messages as they come from node updates
                if "messages" in node_update and node_update["messages"]:
                    for msg in node_update["messages"]:
                        if hasattr(msg, "content") and msg.content:
                            # Only print if it's a new message
                            if hasattr(msg, "type"):
                                if msg.type == "ai":
                                    sys.stdout.write("\r✓ Query processed!     \n")
                                    sys.stdout.flush()
                                    print(f"\n🤖 [{node_name}] {msg.content}")

        # Check if we're at an interrupt point
        state = graph.get_state(thread)

        # Loop to handle multiple human feedback iterations
        while state.next and "human_feedback" in state.next:
            # Display current sub-questions
            # if state.values.get("questions"):
            #     print("\n" + "="*60)
            #     print("📋 GENERATED SUB-QUESTIONS:")
            #     print("="*60)
            #     for i, q in enumerate(state.values["questions"], 1):
            #         print(f"  {i}. {q}")
            #     print("="*60)

            # Get human feedback
            print("\n💬 Please provide feedback on the sub-questions:")
            print("(Press Enter with no input to proceed as-is)")
            feedback = input("\nYour feedback: ").strip()

            if not feedback:
                feedback = "The questions look good, please proceed."

            print(f"\n✓ Received feedback: {feedback}\n")

            # Update state with human feedback message and resume
            from langchain_core.messages import HumanMessage

            # First, update the state with the human feedback message
            graph.update_state(thread, {"messages": [HumanMessage(content=feedback)]})

            # Then resume from the interrupt (pass None to continue)
            for update in graph.stream(None, thread, stream_mode="updates"):
                # update is a dict with node_name as key and state updates as value
                for node_name, node_update in update.items():
                    # Print AI messages as they come from node updates
                    if "messages" in node_update and node_update["messages"]:
                        for msg in node_update["messages"]:
                            if hasattr(msg, "content") and msg.content:
                                if hasattr(msg, "type") and msg.type == "ai":
                                    print(f"\n🤖 [{node_name}] {msg.content}")

            # Check state again for more interrupts
            state = graph.get_state(thread)

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

        if "sources" in result and result["sources"]:
            print(f"\n📚 Sources consulted: {len(result['sources'])}")

        print("\n" + "=" * 60 + "\n")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Search interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n\n❌ Error occurred: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
