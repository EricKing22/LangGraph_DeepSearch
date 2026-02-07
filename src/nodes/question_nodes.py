from pydantic import BaseModel, Field
from state import Question
from typing import List
from llm import ollama_llm as llm
from langgraph.types import Send
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from prompts import BREAK_QUESTIONS_PROMPT, SYNTHESIS_PROMPT
from src import config


def extract_query(state: Question):
    def extract_text_content(content):
        """Safely extract text from message content (handles both str and list formats)"""
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            # Extract text from content blocks
            texts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        texts.append(block.get("text", ""))
                elif isinstance(block, str):
                    texts.append(block)
            return " ".join(texts).strip()

        # Fallback
        return str(content)

    """Extract query from the most recent HumanMessage"""
    # Extract from message history (overrides old query)
    messages = state.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            query = extract_text_content(message.content)
            if query:
                return {"query": query}

    # If no HumanMessage found, use existing query
    existing_query = state.get("query", "")
    if existing_query:
        return {"query": existing_query}

    raise ValueError("No query found in state")


def create_questions(state: Question):
    """
    Generate a list of sub-questions based on user's original query.
    Uses LLM to generate multiple sub-queries for separate search and analysis in subsequent steps.
    """
    print("Creating sub-questions")

    query = state["query"]
    questions = state.get("questions", [])

    class Sub_Questions(BaseModel):
        questions: List[str] = Field(
            description="Sub queries generated from the original query to explore different angles and aspects of the topic.",
        )

    structured_llm = llm.with_structured_output(Sub_Questions)
    human_feedback = state.get("human_feedback", "")

    messages = [
        SystemMessage(
            content=BREAK_QUESTIONS_PROMPT.format(query=query)
            + f"\nRemember the max number of sub questions shouldn't exceed {config.MAX_SUB_QUESTIONS}."
        )
    ]

    if questions:
        messages.append(
            SystemMessage(
                content=f"Current sub questions: {questions}"
                f"Improve reasons: {state.get('next_step_reason')}"
            )
        )
    if human_feedback:
        messages.append(
            HumanMessage(
                content=f"Human Feedback: {human_feedback}."
                "Please reproduce the sub questions based on feedback"
            )
        )

    questions = structured_llm.invoke(messages)

    # Update state
    return {
        "query": query,
        "break_questions_iterations_count": state.get(
            "break_questions_iterations_count", 0
        )
        + 1,
        "questions": questions.questions,
        "messages": [
            AIMessage(
                content=f"Breaking down the query into sub-questions: {questions.questions}"
            )
        ],
    }


def should_break_query(state: Question):
    """
    Decide the next step based on human feedback.
    """
    human_feedback = state.get("human_feedback", "")

    class Router(BaseModel):
        """
        Router node that decides the next step based on human feedback.
        """

        next_step: str = Field(
            description="The next step to execute. Possible values: 'search_web' or 'create_questions'.",
        )
        reason: str = Field(
            description="The reasoning behind the decision.",
        )

    structured_router = llm.with_structured_output(Router)

    # Extract query from state or messages
    query = state.get("query", "")
    if not query:
        for message in reversed(state.get("messages", [])):
            if isinstance(message, HumanMessage):
                query = message.content
                break

    if not query:
        raise ValueError("No query found in state or message history")

    questions = state.get("questions", [])

    messages = [HumanMessage(content=query)]
    if questions:
        questions_str = "\n".join(f"{idx+1}. {q}" for idx, q in enumerate(questions))
        messages.append(
            SystemMessage(
                content=f"These are the previous generated questions:\n{questions_str}"
            )
        )
    if human_feedback:
        messages.append(HumanMessage(content=f"Human Feedback: {human_feedback}"))

    messages.append(
        SystemMessage(
            content="Based on the above information, decide the next step. "
            "If you are happy with the current sub questions return 'search_web' in 'next_step'. "
            "If you need to create more or rewrite sub-questions based on human feedback, return 'create_questions' in 'next_step'."
            "You should put your reasoning in the 'reason' field, use question index to help understanding."
            f"Remember the max number of sub questions shouldn't exceed {config.MAX_SUB_QUESTIONS}."
            "Your response should be in JSON format with 'next_step' and 'reason' fields."
        )
    )

    result = structured_router.invoke(messages)
    next_step = result.next_step
    next_step_reason = result.reason

    state["next_step_reason"] = (
        next_step_reason if next_step == "create_questions" else None
    )

    # Check the number of iterations to prevent infinite loops
    break_iteration = state.get("break_questions_iterations_count", 0)
    if break_iteration >= 3:
        print(
            f"Reached maximum break iterations ({break_iteration}). Forcing next step to 'search_web'."
        )
        next_step = "search_web"

    # Print current sub questions
    print("Current sub questions are:")
    for idx, question in enumerate(questions, 1):
        print(f"{idx}. {question}")

    # Note: Conditional edges that return strings don't update state directly
    # The routing decision is implicitly tracked by which node gets executed next
    print(f"Router decision: {next_step}")
    print(f"Router reasoning: {next_step_reason}")

    # state["messages"] = state.get("messages", []) + [AIMessage(content=f"Router decision: {next_step}. Reason: {next_step_reason}")]
    if next_step == "create_questions":
        return next_step
    else:
        return map_search(state)


def map_search(state: Question):
    """
    Use Send to dispatch each sub-question to the search_web node for searching.
    """
    questions = state.get(
        "questions", [state["query"]]
    )  # If no sub-questions, search the original query directly
    return [Send("search_web", {"query": question}) for question in questions]


def answer_directly(state: Question):
    """
    Generate an answer directly based on the original query and existing information without further search.
    Uses LLM to generate a direct response based on the information in the current state.
    """
    query = state["query"]
    context = state.get("search_results", [])
    prompt = f"Based on the following context, answer the question: {query}\nContext: {context}"

    messages = [SystemMessage(content=prompt)]

    answer = llm.invoke(messages)

    # Track the direct answer with both user query and AI response
    return {
        "context": answer,
        "messages": [
            HumanMessage(content=query),
            answer,  # answer is already an AIMessage
        ],
    }


def summarise(state: Question):
    """
    Summarize the search results and extract key information and insights.
    Uses LLM to analyze each search result and generate a comprehensive summary.
    """

    prompt = SYNTHESIS_PROMPT.format(
        query=state["query"],
        context=state["search_results"],
        sources=state.get("sources", []),
    )

    messages = [SystemMessage(content=prompt)]

    summary = llm.invoke(messages)

    # Track the summarization with the actual summary content
    return {"summary": summary, "messages": [summary]}
