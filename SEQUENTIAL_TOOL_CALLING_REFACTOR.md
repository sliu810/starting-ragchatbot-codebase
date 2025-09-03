# Sequential Tool Calling Refactor Plan

## Overview

Refactor `backend/ai_generator.py` to support sequential tool calling where Claude can make up to 2 tool calls in separate API rounds. This enables complex queries requiring multiple searches, comparisons, or multi-part questions.

## Current vs Desired Behavior

### Current Behavior
```
User Query → Claude + Tools → Tool Execution → Tools Removed → Final Response
```
- Claude makes 1 tool call
- Tools are removed from subsequent API calls
- If Claude wants another tool call after seeing results, it can't (gets empty response)

### Desired Behavior
```
User Query → Round 1: Claude + Tools → Tool Execution → Round 2: Claude + Tools → Final Response
```
- Each tool call is a separate API request where Claude can reason about previous results
- Support for complex queries requiring multiple searches

## Example Use Case

**Query:** "Search for a course that discusses the same topic as lesson 4 of course X"

**Flow:**
1. **Round 1:** Claude calls `get_course_outline("course X")` → gets lesson 4 title
2. **Round 2:** Claude uses lesson 4 title to call `search_course_content(lesson_4_topic)` → finds similar courses
3. **Final:** Claude synthesizes complete answer

## Two Architectural Approaches Analyzed

### Approach 1: State Management Pattern
- **Philosophy:** Heavy state management with explicit tracking
- **Key:** ConversationState class manages all state across rounds
- **Pros:** Very explicit control, detailed logging
- **Cons:** Complex state management, harder to test

### Approach 2: Functional/Immutable Pattern ⭐ **RECOMMENDED**
- **Philosophy:** Minimal state, functional programming, Claude-centric
- **Key:** Immutable ConversationContext + recursive rounds
- **Pros:** Simpler, functional style, easier to test, leverages Claude's conversation memory
- **Cons:** Less explicit control

## Recommended Implementation: Functional Approach

### Core Design Principles
1. **Immutable data structures** - no mutable state
2. **Pure functions** - predictable inputs/outputs
3. **Recursive conversation rounds** - natural flow control
4. **Claude-centric** - leverage Claude's conversation memory
5. **Return values for errors** - no exceptions breaking flow

### Key Components

#### 1. Immutable Conversation Context
```python
@dataclass(frozen=True)
class ConversationContext:
    """Immutable context for conversation rounds"""
    original_query: str
    history: Optional[str]
    rounds_remaining: int
    tool_conversations: List[Dict] = field(default_factory=list)
    
    def with_tool_round(self, tool_results: List[Dict]) -> 'ConversationContext':
        """Create new context with tool round added - immutable pattern"""
        return ConversationContext(
            original_query=self.original_query,
            history=self.history,
            rounds_remaining=self.rounds_remaining - 1,
            tool_conversations=self.tool_conversations + [tool_results]
        )
```

#### 2. Main Sequential Method
```python
def generate_response_with_rounds(self, query: str, 
                                conversation_history: Optional[str] = None,
                                tools: Optional[List] = None,
                                tool_manager=None,
                                max_rounds: int = 2) -> Tuple[str, List[Dict]]:
    """Generate response with up to max_rounds of tool calls."""
    
    context = ConversationContext(
        original_query=query,
        history=conversation_history,
        rounds_remaining=max_rounds
    )
    
    return self._process_conversation_round(context, tools, tool_manager)
```

#### 3. Recursive Round Processing
```python
def _process_conversation_round(self, context: ConversationContext, 
                              tools: Optional[List], 
                              tool_manager) -> Tuple[str, List[Dict]]:
    """Process a single conversation round - pure function approach."""
    
    # Build current conversation state for Claude
    current_messages = self._build_messages_for_claude(context)
    
    # Make API call
    response = self._make_claude_api_call(current_messages, tools)
    
    # Handle response based on type
    if response.stop_reason != "tool_use":
        # Final answer - return immediately
        return response.content[0].text, []
        
    if context.rounds_remaining <= 0:
        # Max rounds reached - force final answer
        return self._force_final_answer(current_messages), []
        
    # Execute tools and prepare next round
    tool_results = self._execute_tools_immutably(response, tool_manager)
    
    # Create new context for next round
    next_context = context.with_tool_round(tool_results)
    
    # Recurse for next round
    return self._process_conversation_round(next_context, tools, tool_manager)
```

#### 4. Pure Message Building Function
```python
def _build_messages_for_claude(self, context: ConversationContext) -> List[Dict]:
    """Build message history leveraging Claude's native conversation handling."""
    messages = []
    
    # Add conversation history if exists
    if context.history:
        messages.extend(self._parse_history_to_messages(context.history))
    
    # Add original query
    messages.append({"role": "user", "content": context.original_query})
    
    # Add each tool conversation round
    for tool_conversation in context.tool_conversations:
        messages.append({"role": "assistant", "content": tool_conversation["assistant_content"]})
        messages.append({"role": "user", "content": tool_conversation["tool_results"]})
    
    return messages
```

#### 5. Immutable Tool Execution
```python
def _execute_tools_immutably(self, response, tool_manager) -> List[Dict]:
    """Execute tools and return results immutably. No exceptions."""
    tool_results = []
    
    for content_block in response.content:
        if content_block.type == "tool_use":
            try:
                result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": result
                })
                
            except Exception as e:
                # Handle error gracefully through return values
                tool_results.append({
                    "type": "tool_result", 
                    "tool_use_id": content_block.id,
                    "content": f"Tool execution failed: {str(e)}"
                })
    
    return {
        "assistant_content": response.content,
        "tool_results": tool_results
    }
```

### System Prompt Updates

```python
SYSTEM_PROMPT_SEQUENTIAL = """You are an AI assistant specialized in course materials with access to comprehensive tools.

Available Tools:
- **search_course_content**: Search for specific course content and materials
- **get_course_outline**: Get course structure, titles, and lesson lists

Sequential Tool Usage:
- You may use tools up to 2 times to answer complex questions
- First tool call: Gather initial information (e.g., get course outline)  
- Second tool call: Search for specific content based on first results
- After tool usage, provide a comprehensive final answer

Multi-Step Query Examples:
- "Compare lesson X of course A with similar topics" → Get outline first, then search
- "Find prerequisites for topic Y" → Get course structure, then search prerequisites
- "What courses cover similar content to lesson Z?" → Get lesson details, then find similar

Response Guidelines:
- For simple questions: Answer directly without tools
- For complex questions: Use tools sequentially as needed
- Synthesize all information into clear, educational responses
- Include relevant examples when helpful

Current conversation allows for sequential tool usage to fully answer your question.
"""
```

## Termination Conditions

The system terminates when:
1. **Natural completion**: Claude provides a text response (not tool_use)
2. **Round limit**: Maximum 2 tool calling rounds reached  
3. **Error handling**: Tool execution fails gracefully with error messages

## Integration with Existing System

### RAGSystem Changes (Minimal)
```python
# In rag_system.py
def query(self, query: str, session_id: Optional[str] = None) -> Tuple[str, List[str]]:
    """Process query with sequential tool calling support."""
    
    # Get conversation history if session exists
    history = None
    if session_id:
        history = self.session_manager.get_conversation_history(session_id)
    
    # Use new sequential method
    response, tool_log = self.ai_generator.generate_response_with_rounds(
        query=f"Answer this question about course materials: {query}",
        conversation_history=history,
        tools=self.tool_manager.get_tool_definitions(),
        tool_manager=self.tool_manager,
        max_rounds=2
    )
    
    # Extract sources from tool log instead of last_sources pattern
    sources = self._extract_sources_from_tool_log(tool_log)
    
    # Update conversation history
    if session_id:
        self.session_manager.add_exchange(session_id, query, response)
    
    return response, sources
```

## Testing Strategy

### Test Categories
1. **Single Round Tests**: Simple queries completing in one tool call
2. **Sequential Tool Tests**: Complex queries requiring 2 rounds
3. **Termination Tests**: Max rounds reached, natural completion
4. **Error Handling Tests**: Tool failures, invalid inputs
5. **Integration Tests**: End-to-end with real RAG system

### Example Test Structure
```python
class TestSequentialToolCalling:
    def test_single_round_completion(self):
        """Test queries that complete in one tool round"""
        
    def test_two_round_sequence(self):
        """Test queries requiring two sequential tool calls"""
        
    def test_max_rounds_termination(self):
        """Test termination when max rounds reached"""
        
    def test_immutable_context(self):
        """Test that contexts are properly immutable"""
        
    def test_complex_query_patterns(self):
        """Test real-world complex query scenarios"""
```

## Implementation Phases

### Phase 1: Core Infrastructure
1. Create `ConversationContext` dataclass
2. Add `generate_response_with_rounds` method 
3. Implement recursive round processing
4. Update system prompt

### Phase 2: Integration & Testing
1. Integrate with existing RAGSystem
2. Create comprehensive test suite
3. Test with various query patterns
4. Handle edge cases and errors

### Phase 3: Validation & Optimization
1. Test complex multi-step queries
2. Performance optimization
3. Production deployment
4. Monitor and refine

## Benefits of This Approach

1. **Enhanced Query Capabilities**: Support complex multi-step questions
2. **Maintains Simplicity**: Functional approach keeps code clean
3. **Easy Testing**: Pure functions with immutable data
4. **Leverages Claude**: Uses Claude's conversation strengths
5. **Backward Compatible**: Existing single-tool queries still work
6. **Error Resilient**: Graceful handling through return values

## Files to Modify

1. **`backend/ai_generator.py`** - Core sequential tool calling logic
2. **`backend/rag_system.py`** - Integration updates (minimal)
3. **`backend/tests/test_ai_generator.py`** - Updated tests for sequential calling
4. **Add: `backend/tests/test_sequential_tool_calling.py`** - New comprehensive tests

This refactor enables sophisticated multi-step reasoning while maintaining the existing system's simplicity and reliability.