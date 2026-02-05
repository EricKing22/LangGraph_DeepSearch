"""Search-related prompt templates."""

SYNTHESIS_PROMPT = """You are an AI research assistant tasked with synthesizing information from web sources.

Query: {query}

Context from web sources:
{context}

Please provide a comprehensive, well-structured answer to the query based on the context above.
Include specific information, facts, and data from the sources. If the sources don't contain
enough information to fully answer the query, acknowledge this limitation.

Format your response in a clear, readable manner with:
- Direct answer to the query
- Supporting details and evidence
- Source attribution where relevant
- Any caveats or limitations

Answer:"""


SEARCH_QUERY_PROMPT = """Given the user's question, generate an optimized search query that will
find the most relevant information.

User Question: {question}

Generate a concise, focused search query (1-2 sentences max):"""


RELEVANCE_CHECK_PROMPT = """Assess whether the following content is relevant to the query.

Query: {query}

Content: {content}

Is this content relevant? Respond with YES or NO, followed by a brief explanation.

Assessment:"""
