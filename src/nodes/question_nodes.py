from pydantic import BaseModel, Field
from state import Question
from typing import List
from llm import qwen_llm as llm
from langgraph.types import Send
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from prompts import BREAK_QUESTIONS_PROMPT, SYNTHESIS_PROMPT
from src import config


def create_questions(state: Question):
    """
    根据用户输入的原始查询，生成分解后的查询列表。
    使用LLM生成多个子查询，以便在后续步骤中分别搜索和分析。
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
    根据人工反馈决定下一步：
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
    query = state["query"]
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

    messages.extend(
        [
            HumanMessage(
                content="Based on the above information, decide the next step. "
                "If you are happy with the current sub questions return 'search_web' in 'next_step'. "
                "If you need to create more or rewrite sub-questions based on human feedback, return 'create_questions' in 'next_step'."
                "You should put your reasoning in the 'reason' field, use question index to help understanding."
                f"Remember the max number of sub questions shouldn't exceed {config.MAX_SUB_QUESTIONS}."
            ),
            SystemMessage(
                content="Your response should be in JSON format with 'next_step' and 'reason' fields."
            ),
        ]
    )

    result = structured_router.invoke(messages)
    next_step = result.next_step
    next_step_reason = result.reason

    state["next_step_reason"] = (
        next_step_reason if next_step == "create_questions" else None
    )

    # check the number of iterations to prevent infinite loops
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
    用Send把每一sub question发给search_web节点，进行搜索。
    """
    questions = state.get(
        "questions", [state["query"]]
    )  # 如果没有分解问题，就直接搜索原始查询
    return [Send("search_web", {"query": question}) for question in questions]


def answer_directly(state: Question):
    """
    直接根据原始查询和现有信息生成答案，而不进行进一步搜索。
    使用LLM根据当前状态中的信息生成一个直接的回答。
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
    对搜索结果进行总结，提取关键信息和洞见。
    使用LLM对每个搜索结果进行分析，并生成一个综合的总结。
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
