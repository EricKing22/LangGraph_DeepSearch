from pydantic import BaseModel, Field
from langchain.messages import AIMessage
from src.prompts import REVIEW_REPORT_PROMPT
from src.state import Review
from src.llm import question_llm as llm


def review(state: Review):
    """
    Review the generated summary and provide feedback for improvement.
    This node can be used to capture human feedback on the summary and update the state accordingly for further refinement.
    """
    report = state.get("summary", "")

    sources = ""
    for source in state.get("sources", []):
        content = (
            source.get("title", "")
            + ":\n"
            + source.get("url", "")
            + ":\n"
            + source.get("content", "")
        )
        sources += content + "\n\n"

    query = state.get("query", "")

    # Generate review report using the prompt
    prompt = REVIEW_REPORT_PROMPT.format(query=query, sources=sources, report=report)

    class Review(BaseModel):
        score: int = Field(description="Overall score for the summary (1-10)")
        strengths: str = Field(description="Strengths of the summary")
        weaknesses: str = Field(description="Weaknesses of the summary")

    structured_llm = llm.with_structured_output(Review)
    try:
        feedback = structured_llm.invoke(prompt)
        message = f"Review feedback:\n\n**Score**={feedback.score},\n**Strengths**={feedback.strengths},\n**Weaknesses**={feedback.weaknesses}"
    except Exception as e:
        message = f"Error during review: {str(e)}"
        feedback = Review(score=0, strengths="N/A", weaknesses="Error during review")

    return {
        "score": feedback.score,
        "strengths": feedback.strengths,
        "weaknesses": feedback.weaknesses,
        "messages": [AIMessage(content=message)],
    }
