from pydantic import BaseModel, Field
from src.state import Plan, WebSearchState
from typing import List, Literal
from src.llm import question_llm as llm
from src.llm import report_llm as summarize_llm
from langgraph.types import Send
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import END
from src.prompts import BREAK_QUESTIONS_PROMPT, SYNTHESIS_PROMPT
from src import config
import logging

logger = logging.getLogger("LangGraph_DeepSearch.question_nodes")


def extract_query(state: Plan):
    """Extract query from the most recent HumanMessage. Initialise all other state fields to default values."""

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

    query = state.get("query", "")
    if query:
        return query
    else:
        # Extract from message history (overrides old query)
        messages = state.get("messages", [])
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                query = extract_text_content(message.content)
                if query:
                    return query

    raise ValueError("No query found in state")


def plan(state: Plan):
    """
    Generate a list of sub-questions based on user's original query.
    Uses LLM to generate multiple sub-queries for separate search and analysis in subsequent steps.
    """
    logger.debug("Creating sub-questions")

    query = extract_query(state)
    questions = state.get("questions", [])
    human_feedback = state.get("human_feedback", "")
    score = state.get("score", None)

    class Sub_Questions(BaseModel):
        questions: List[str] = Field(
            description="Sub queries generated from the original query to explore different angles and aspects of the topic.",
        )
        reason: str = Field(
            description="The reasoning behind the generated sub-questions, explaining how they relate to the original query and cover different aspects of the topic.",
        )

    structured_llm = llm.with_structured_output(Sub_Questions)

    messages = [
        SystemMessage(
            content=BREAK_QUESTIONS_PROMPT.format(query=query)
            + f"\nRemember the max number of sub questions shouldn't exceed {config.MAX_SUB_QUESTIONS}."
        )
    ]

    if questions:
        messages.append(SystemMessage(content=f"Current sub questions: {questions}"))

    # Prioritise on either human feedback or review feedback
    if score:
        messages.append(
            SystemMessage(
                content=f"Previous summary received a score of {score}."
                f"Received feedback indicates that the summary has weaknesses in {state.get('weaknesses', '')} "
                f"and strengths in {state.get('strengths', '')}."
                f"Please reproduce the sub questions based on this feedback, improving the weaknesses and maintaining the strengths."
            )
        )
    elif human_feedback:
        messages.append(
            HumanMessage(
                content=f"Human Feedback: {human_feedback}."
                "Please reproduce the sub questions based on feedback"
            )
        )

    results = structured_llm.invoke(messages)
    questions = results.questions
    reason = results.reason

    # Update state
    return {
        "query": query,
        "break_questions_iterations_count": state.get(
            "break_questions_iterations_count", 0
        )
        + 1,
        "questions": questions,
        "messages": [
            AIMessage(
                content="I'm now going to search for these topics:\n"
                + "\n".join(f"**{i + 1}**. **{q}**" for i, q in enumerate(questions))
                + f"\n\n**Reason for these sub-questions:**\n{reason}"
            )
        ],
    }


def should_skip_human_feedback(state: Plan):
    """
    Decide whether to skip human feedback
    """
    summarise_iterations = state.get("summarise_iterations", 0)
    if summarise_iterations > 0:
        return map_search(state)
    else:
        return "human_feedback"


def should_break_query(state: Plan):
    """
    Decide the next step based on human feedback.
    """

    class Router(BaseModel):
        """
        Router node that decides the next step based on human feedback.
        """

        next_step: Literal["search_web", "plan"] = Field(
            description="The next step to execute. Possible values: 'search_web' or 'plan'.",
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
        questions_str = "\n".join(f"{idx + 1}. {q}" for idx, q in enumerate(questions))
        messages.append(
            SystemMessage(
                content=f"These are the previous generated questions:\n{questions_str}"
            )
        )

    human_feedback = state.get("human_feedback", "")
    if human_feedback:
        logger.debug("Considering human feedback ...")
        messages.append(HumanMessage(content=f"Human Feedback: {human_feedback}"))

    messages.append(
        SystemMessage(
            content="Based on the above information, decide the next step. "
            "If you are happy with the current sub questions return 'search_web' in 'next_step'. "
            "If you need to create more or rewrite sub-questions based on human feedback, return 'plan' in 'next_step'."
            "You should put your reasoning in the 'reason' field, use question index to help understanding."
            f"Remember the max number of sub questions shouldn't exceed {config.MAX_SUB_QUESTIONS}."
            "Your response should be in JSON format with 'next_step' and 'reason' fields."
        )
    )

    result = structured_router.invoke(messages)
    next_step = result.next_step
    next_step_reason = result.reason

    # Check the number of iterations to prevent infinite loops
    break_iteration = state.get("break_questions_iterations_count", 0)
    if break_iteration >= 3:
        logger.debug(
            f"Reached maximum break iterations ({break_iteration}). Forcing next step to 'search_web'."
        )
        next_step = "search_web"

    # Log the router decision and reasoning for debugging and transparency
    logger.debug(f"Router decision: {next_step}")
    logger.debug(f"Router reasoning: {next_step_reason}")

    if next_step == "plan":
        return next_step
    else:
        return map_search(state)


def human_feedback(state: Plan):
    """
    Collect human feedback on the generated sub-questions to improve them iteratively.
    This node can be used to capture human feedback and update the state accordingly for the next iteration of question planning.
    """
    feedback = ""
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            feedback = message.content
            break
    logger.debug("Feedback Received")
    return {"human_feedback": feedback}


def map_search(state: Plan):
    """
    Use Send to dispatch each sub-question to the search_web node for searching.
    """
    questions = state.get(
        "questions", [state["query"]]
    )  # If no sub-questions, search the original query directly
    return [Send("search_web", {"query": question}) for question in questions]


def answer_directly(state: Plan):
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


def summarise(state: WebSearchState):
    """
    Summarize the search results and extract key information and insights.
    Uses LLM to analyze each search result and generate a comprehensive summary.
    """

    prompt = SYNTHESIS_PROMPT.format(
        query=state["query"],
        context=state["search_results"],
        sources=state.get("sources", []),
    )

    score = state.get("score", None)
    if score:
        prompt += (
            f"\nPrevious summary received a score of {score} "
            f"with strengths: {state.get('strengths', '')} and "
            f"weaknesses: {state.get('weaknesses', '')}. "
            f"Please improve the summary based on this feedback."
        )

    messages = [SystemMessage(content=prompt)]

    summary = summarize_llm.invoke(messages)

    # Track the summarization with the actual summary content
    return {
        "summary": summary,
        "messages": [summary],
        "summarise_iterations": state.get("summarise_iterations", 0) + 1,
    }


def is_finished(state: WebSearchState):
    """
    Decide whether the agent has gathered enough information to answer the original query.
    Uses LLM to evaluate the current state and determine if it's sufficient to generate a final answer.
    """
    # This is a placeholder implementation. You can design your own logic or LLM prompt to make this decision.
    query = state.get("query", "")
    score = state.get("score", "5")
    strengths = state.get("strengths", "")
    weaknesses = state.get("weaknesses", "")
    summarise_iterations = state.get("summarise_iterations", 0)

    if score > 7 or summarise_iterations == config.MAX_REVIEW_IMPROVE_ITERATIONS:
        return END

    class Router(BaseModel):
        """
        Router node that decides the next step based on human feedback.
        """

        next_step: Literal["summarise", "plan"] = Field(
            description="The next step to execute. Possible values: 'summarise' or 'plan'.",
        )
        reason: str = Field(
            description="The reasoning behind the decision.",
        )

    prompt = (
        f"Based on the original query is {query},"
        f"the reviewer produced a score of {score} with strengths: {strengths} and weaknesses: {weaknesses}.\n "
        f"The resources gathered so far are {state.get('search_results', [])}.\n"
        f"How do you think the report can be improved?\n"
        f"if you think the report contains all information only structure should be improved, return 'summarise' in next_step"
        f"If you think the report is missing critical information to answer the query, return 'plan' in next_step"
    )
    structured_router = llm.with_structured_output(Router)
    result = structured_router.invoke([SystemMessage(content=prompt)])
    next_step = result.next_step
    reason = result.reason

    logger.debug(f"Verifier Router decision: {next_step}")
    logger.debug(f"Verifier Router reasoning: {reason}")

    return next_step
