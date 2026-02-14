WRITE_NOTES_PROMPT = """You are a learning analysis expert. Analyze the differences between the two plans below and distill a reusable lesson.

## Original Task
{query}

## Human Feedback
{human_feedback}

## Agent's Initial Plan (Plan A)
{plan_a}

## Final-Modified Plan (Plan B)
{plan_b}

## Analysis Requirements
1. Identify key differences between Plan A and Plan B
2. Analyze why the human made these modifications
3. Distill one concise, actionable lesson
4. This lesson should help the agent produce better plans for similar tasks in the future

## Output Requirements
- has_lesson: Set to true if the difference is meaningful and worth learning from
- lesson: A one-sentence summary of the lesson (e.g., "When plotting sine waves, label the extrema")
- reasoning: A brief explanation of why this lesson matters
"""
