from pydantic import BaseModel, Field
from state import Question
from typing import List
from llm import ollama_llm as llm
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from tools.search_tool import search_tavily
from prompts import SEARCH_QUERY_PROMPT, SYNTHESIS_PROMPT


class Sub_Questions(BaseModel):
    questions: List[str] = Field(
        description="Sub queries generated from the original query to explore different angles and aspects of the topic.",
    )


class Router(BaseModel):
    """
    Router node that decides the next step based on human feedback.
    """

    next_step: str = Field(
        description="The next step to execute. Possible values: 'search_web', 'answer_directly' or 'create_questions'.",
    )


def create_questions(state: Question):
    """
    根据用户输入的原始查询，生成分解后的查询列表。
    使用LLM生成多个子查询，以便在后续步骤中分别搜索和分析。
    """
    query = state["query"]
    questions = state.get("questions", [])

    structured_llm = llm.with_structured_output(Sub_Questions)
    human_feedback = state.get("human_feedback", "")

    messages = [SystemMessage(content=SEARCH_QUERY_PROMPT), HumanMessage(content=query)]

    if questions:
        messages.append(SystemMessage(content=f"Current questions: {questions}"))
    if human_feedback:
        messages.append(HumanMessage(content=f"Human Feedback: {human_feedback}"))

    questions = structured_llm.invoke(messages)

    # Track the AI's decision to create sub-questions
    return {
        "questions": questions.questions,
        "messages": [
            AIMessage(
                content=f"Breaking down the query into sub-questions: {questions.questions}"
            )
        ],
    }


def should_continue(state: Question) -> str:
    """
    根据人工反馈决定下一步：
    """
    human_feedback = state.get("human_feedback", "")

    structured_router = llm.with_structured_output(Router)
    query = state["query"]
    questions = state.get("questions", [])

    messages = [HumanMessage(content=query)]
    if questions:
        messages.append(
            SystemMessage(
                content=f"In your answer think about these sub questions: {questions}"
            )
        )
    if human_feedback:
        messages.append(HumanMessage(content=f"Human Feedback: {human_feedback}"))

    messages.append(
        HumanMessage(
            content="Based on the above information, decide the next step. "
            "If you need to search for more information, return 'search_web'. "
            "If you have enough information to answer directly, return 'answer_directly'. "
            "If you need to create more  or rewrite sub-questions based on human feedback, return 'create_questions'."
        )
    )

    result = structured_router.invoke(messages).next_step

    # Note: Conditional edges that return strings don't update state directly
    # The routing decision is implicitly tracked by which node gets executed next
    print(f"Router decision: {result}")

    return result


def should_break_query(state: Question) -> str:
    """
    根据人工反馈决定是否生成新的子问题或直接回答原始查询。
    """

    messages = [
        SystemMessage(
            content="Given the query, check if it is too complex to answer directly. "
            "Be generous in breaking down the query into sub-questions "
            "Only answer directly if you think the query is tool simple like 'Hello' and doesn't require any information gathering. "
            "Your answer should either be 'create_questions' or 'answer_directly'"
        ),
        HumanMessage(content=state["query"]),
    ]

    class Router(BaseModel):
        should_break: str = Field(
            description="Return 'create_questions' to break the query, or 'answer_directly' to answer the question.",
        )

    structured_router = llm.with_structured_output(Router)

    result = structured_router.invoke(messages).should_break

    # Debug output
    print(f"should_break_query returned: '{result}'")

    # Note: Conditional edges that return strings don't update state directly
    # The routing decision is implicitly tracked by which node gets executed next

    return result


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


def search_web(state: Question):
    """
    对每个子问题执行 Tavily 搜索

    Args:
        state: 包含 questions 列表的状态

    Returns:
        包含搜索结果的字典
    """
    questions = state.get("questions", [])
    search_results = []

    for question in questions:
        try:
            results = search_tavily(query=question)
            search_results.append({"question": question, "results": results})
        except Exception as e:
            print(f"搜索失败 '{question}': {str(e)}")
            search_results.append(
                {"question": question, "results": [], "error": str(e)}
            )

    # Track search action with summary of what was searched
    search_summary = f"Searched web for {len(questions)} sub-questions: {', '.join(questions[:3])}{'...' if len(questions) > 3 else ''}"

    return {
        "messages": [AIMessage(content=search_summary)],
        "context": search_results,
        "sources": [result for result in search_results if "results" in result],
    }


def summarise(state: Question):
    """
    对搜索结果进行总结，提取关键信息和洞见。
    使用LLM对每个搜索结果进行分析，并生成一个综合的总结。
    """

    prompt = SYNTHESIS_PROMPT.format(
        query=state["query"], context=state["context"], sources=state.get("sources", [])
    )

    messages = [SystemMessage(content=prompt)]

    summary = llm.invoke(messages)

    # Track the summarization with the actual summary content
    return {"summary": summary, "messages": [summary]}
