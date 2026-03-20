# TODO - Future Development Priorities

## High Priority

### 1. Human Feedback After Review
- **Description**: Allow human to provide additional feedback after reviewing the generated summary
- **Rationale**: Currently only AI provides review feedback; human can provide more nuanced guidance
- **Impact**: Improves result quality by allowing human to point out missing aspects

### 2. Learning from Review Feedback
- **Description**: The self-learning system should also learn from review scores and feedback
- **Rationale**: Current learning only captures Plan A vs Plan B differences, not review feedback
- **Impact**: Agent can learn what makes a good vs bad summary

### 3. Multi-Source Search Support
- **Description**: Support multiple search APIs (Tavily, DuckDuckGo, SerpAPI, etc.)
- **Rationale**: Different search providers have different strengths and coverage
- **Impact**: More comprehensive results, better fallback options

## Medium Priority

### 4. Web UI / Dashboard
- **Description**: Build a web interface for easier interaction
- **Rationale**: CLI is powerful but a UI would be more accessible
- **Impact**: Better user experience, visualization of research process

### 5. Export Formats
- **Description**: Support exporting results in multiple formats (Markdown, PDF, HTML)
- **Rationale**: Users may want to share or publish research results
- **Impact**: More practical output options

### 6. Citation Management
- **Description**: Enhanced citation handling with bibliography generation
- **Rationale**: Academic/professional use requires proper citations
- **Impact**: Better credibility and usability of outputs

### 7. Configurable Review Thresholds
- **Description**: Allow users to configure what score threshold triggers re-generation
- **Rationale**: Different tasks may have different quality requirements
- **Impact**: More flexible workflow control

## Low Priority / Exploratory

### 8. Multi-Agent Collaboration
- **Description**: Support multiple specialized agents working on different aspects
- **Rationale**: Complex topics may benefit from domain-specific agents
- **Impact**: Better handling of complex multi-faceted queries

### 9. Knowledge Graph Integration
- **Description**: Build and visualize knowledge graphs from research
- **Rationale**: Visual representation can help understand relationships
- **Impact**: Better understanding of complex topics

### 10. Active Learning
- **Description**: Agent proactively asks clarifying questions when query is ambiguous
- **Rationale**: Reduces wasted effort on wrong questions
- **Impact**: Better precision in research results

---

## Implementation Notes

### Quick Wins (1-2 days)
- Items 1, 2, 7 - These leverage existing infrastructure

### Medium Effort (1 week)
- Items 3, 5, 6 - These require new integrations but are well-scoped

### Larger Projects (2+ weeks)
- Items 4, 8, 9 - These are significant feature additions
