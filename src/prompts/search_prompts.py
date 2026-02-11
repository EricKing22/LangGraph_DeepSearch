"""Search-related prompt templates."""

BREAK_QUESTIONS_PROMPT = """
###
You are an expert Research Librarian specializing in query decomposition. Your goal is to break down a complex user query into a logical sequence of atomic, search-engine-friendly sub-questions.

### Constraints
1. **Direct Relevance:** Every sub-question must serve as a necessary building block to answer the original query. Eliminate "how-to" or peripheral questions (e.g., if the topic is "benefits," do not ask about "history" or "preparation").
2. **Search Optimization:** Phrase questions as standalone search queries. Avoid using pronouns like "it" or "they"; use the specific subject names.
3. **Efficiency:** Do not generate "burden" questions. If three questions cover the scope, do not provide six.
4. **Simplicity Check:** If the original query is already atomic and specific, return a single-item list containing a refined version of that query.

### Logic for Decomposition
- **Identify Entities:** What are the core subjects?
- **Identify Parameters:** Are there specific timeframes, locations, or conditions?
- **Identify Mechanisms:** Does the query ask "why" or "how"? Break those into structural components.

### Format
Return a clean, valid Python-style list of strings. No conversational filler.

Original Question: {query}

### Output Format
Provide your assessment in **JSON format** with the following keys:
- "questions": List[str] (list of sub-questions, each as a string)

"""

SYNTHESIS_PROMPT = """You are an AI research assistant tasked with synthesizing information from web sources.

Query: {query}

Context from web sources:
{context}

### Instructions
1. **Comprehensive Answer:** Provide a well-structured answer based *only* on the context provided. Include specific facts, data, and details.
2. **Attribution:** You must cite your sources. Use [number] format (e.g., [1], [2]) inline with the text where the information is used.
3. **Reference List:** At the end of your response, list the sources in IEEE style. Ensure there are no duplicate references.
4. **Limitations:** If the provided context is insufficient to answer the query, acknowledge this limitation.
5. **Language:** **Always answer in the same language as the user's query.** (e.g., if the query is in Chinese, answer in Chinese, even if the sources are in English).

### Format
- **Introduction (summarise the question and answer)**
- **Detailed Analysis** (with inline citations)
- **References**

Sources:
{sources}

Reference your sources professionally in IEEE reference style for example:

[1] ... \n
[2] ... \n

Don't have repeated references. For example this is wrong:
[1] HU Han, GU Wentao, "Adult attachment style and emotional speech acoustic characteristics: A study on the role of attachment style in emotional speech acoustic characteristics," J. Psychol. Sci., vol. 2023, no. 5, pp. 11-20, 2023.
[2] HU Han, GU Wentao, "Adult attachment style and emotional speech acoustic characteristics: A study on the role of attachment style in emotional speech acoustic characteristics," J. Psychol. Sci., vol. 2023, no. 5, pp. 21-30, 2023.

Make sure to include all relevant sources in your answer

**Always answer in the same language as the query.**
Answer:
"""


RELEVANCE_CHECK_PROMPT = """You are an expert content evaluator. Assess whether the following search result is relevant and useful for answering the query.

Query: {query}

Search Result:
Title: {title}
Content: {content}

### Evaluation Criteria
1. **Direct Relation:** Is the content directly related to the query?
2. **Information Value:** Does it contain factual information, insights, or data useful for the answer?
3. **Quality:** Is it free from being spam, purely promotional, or irrelevant filler?

### Output Format
Provide your assessment in **JSON format** with the following keys:
- "is_relevant": boolean (true or false)
- "reason": string (brief explanation of your decision)

Ensure the output is valid JSON and nothing else.
"""

REVIEW_REPORT_PROMPT = """You are an expert reviewer tasked with evaluating the quality of the following report based on the original query and the sources used.

Original Query:
{query}

Given the sources below, evaluate the quality of the report in terms of how well it answers the original query and utilizes the provided sources.
{sources}

Report:
{report}

Generate a comprehensive review of the report considering the following criteria:
1. **Accuracy:** Does the report accurately reflect the information from the sources?
2. **Completeness:** Does the report cover all relevant aspects of the query?
3. **Clarity:** Is the report well-structured and easy to understand?
4. **Source Utilization:** Are the sources appropriately cited and utilized in the report?


Provide your review in **JSON format** with the following keys:
- "score": integer (1-10, where 10 is the best)
- "strengths": string (briefly describe the strengths of the report)
- "weaknesses": string (briefly describe the weaknesses of the report)
Ensure the output is valid JSON and nothing else.
"""
