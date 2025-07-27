# LiteLLM OpenAI-Compatible API Guide - Essential Edition

**The Complete Guide to Using LiteLLM with 2025's Latest Models**

*A streamlined, production-ready guide covering GPT-4.1, O3, O4-Mini, Claude-4, Gemini-2.5, and Grok-4*

---

## Quick Start

### Installation

```bash
pip install litellm
```

### First Completion with 2025 Models

```python
from litellm import completion
import os

# Set your API keys
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key" 
os.environ["GOOGLE_API_KEY"] = "your-google-key"
os.environ["XAI_API_KEY"] = "your-xai-key"

# Latest 2025 models - same interface, different capabilities
messages = [{"role": "user", "content": "Explain quantum computing in simple terms"}]

# OpenAI's latest
gpt_response = completion(model="gpt-4.1", messages=messages)

# Anthropic's most capable
claude_response = completion(model="claude-4-opus-20250514", messages=messages)

# Google's flagship
gemini_response = completion(model="gemini-2.5-pro", messages=messages)

# xAI's reasoning model
grok_response = completion(model="xai/grok-4", messages=messages)

print(gpt_response.choices[0].message.content)
```

---

## Latest 2025 Models

### OpenAI 2025 Flagship Models

**GPT-4.1** - The Evolution of GPT-4
```python
response = completion(
    model="gpt-4.1",                    # Latest GPT model
    messages=messages,
    max_tokens=4096,
    temperature=0.7
)
# Context: 1,047,576 tokens (1.05M) | Cost: $2/$8 per 1M tokens
# Features: Vision, JSON Mode, Multimodal
```

**O3** - Advanced Reasoning Model
```python
response = completion(
    model="o3-2025-04-16",              # Latest reasoning model
    messages=messages,
    reasoning_effort="medium",           # low/medium/high
    max_tokens=2048
)
# Context: 200K tokens | Cost: $2/$8 per 1M tokens  
# Features: Advanced Reasoning, Vision
```

**O4-Mini** - Efficient Reasoning
```python
response = completion(
    model="o4-mini-2025-04-16",         # Cost-effective reasoning
    messages=messages,
    reasoning_effort="high",
    temperature=1.0                     # ⚠️ ONLY temperature=1.0 is supported!
)
# Context: 200K tokens | Cost: $1.1/$4.4 per 1M tokens
# Features: Reasoning, Vision, Fast
```

### Anthropic 2025 Claude-4 Series

**Claude-4 Sonnet** - Balanced Performance
```python
response = completion(
    model="claude-4-sonnet-20250514",   # Latest Sonnet
    messages=messages,
    max_tokens=4096,
    temperature=0.7
)
# Context: 200K tokens | Cost: $3/$15 per 1M tokens
# Features: Vision, Reasoning, Computer Use
```

**Claude-4 Opus** - Most Capable
```python
response = completion(
    model="claude-4-opus-20250514",     # Most capable Claude
    messages=messages,
    max_tokens=4096,
    temperature=0.5
)
# Context: 200K tokens | Cost: $15/$75 per 1M tokens
# Features: Superior Reasoning, Vision, Computer Use, Complex Tasks
```

### Google 2025 Gemini-2.5 Series

**Gemini-2.5 Pro** - Flagship Multimodal
```python
response = completion(
    model="gemini-2.5-pro",             # Google's flagship
    messages=messages,
    max_tokens=8192,
    temperature=0.8
)
# Context: 1,048,576 tokens (1M) | Cost: $1.25/$10 per 1M tokens
# Features: 1M Context, Vision, Audio, Video, Reasoning
```

**Gemini-2.5 Flash** - Fast & Efficient
```python
response = completion(
    model="gemini-2.5-flash",           # Fast inference
    messages=messages,
    max_tokens=4096,
    temperature=0.7
)
# Context: 1,048,576 tokens (1M) | Cost: $0.30/$2.50 per 1M tokens
# Features: Ultra-fast, 1M Context, Vision, Multimodal
```

### xAI 2025 Grok-4

**Grok-4** - Latest Reasoning
```python
response = completion(
    model="xai/grok-4",                 # Latest Grok
    messages=messages,
    max_tokens=4096,
    temperature=0.6
)
# Context: 256K tokens | Cost: $3/$15 per 1M tokens
# Features: Advanced Reasoning, Real-time Data
```

### 2025 Model Comparison

**Note**: Context Window = Input context limit. Output limits vary by model (~4K-32K tokens).

| Model | Context Window | Cost (Input/Output) | Key Strengths |
|-------|---------------|---------------------|---------------|
| **GPT-4.1** | 1.05M input | $2/$8 | Largest input context, multimodal |
| **O3** | 200K tokens | $2/$8 | Advanced reasoning |
| **O4-Mini** | 200K tokens | $1.1/$4.4 | Efficient reasoning |
| **Claude-4-Opus** | 200K tokens | $15/$75 | Most capable overall |
| **Claude-4-Sonnet** | 200K tokens | $3/$15 | Balanced performance |
| **Gemini-2.5-Pro** | 1M input | $1.25/$10 | Best value for long context |
| **Gemini-2.5-Flash** | 1M input | $0.30/$2.50 | Fastest + cheapest |
| **Grok-4** | 256K tokens | $3/$15 | Real-time reasoning |

### Model Selection Guide

**For Maximum Context:**
- `gpt-4.1` (1.05M tokens) - Largest context window
- `gemini-2.5-pro` (1M tokens) - Best value for long documents
- `gemini-2.5-flash` (1M tokens) - Fastest with large context

**For Advanced Reasoning:**
- `claude-4-opus-20250514` - Most capable reasoning
- `o3-2025-04-16` - OpenAI's reasoning specialist
- `xai/grok-4` - Real-time reasoning with current data

**For Cost Efficiency:**
- `gemini-2.5-flash` - Best price/performance ratio
- `o4-mini-2025-04-16` - Efficient reasoning model
- `claude-4-sonnet-20250514` - Balanced cost/capability

**For Computer Use:**
- `claude-4-opus-20250514` - Best computer interaction
- `claude-4-sonnet-20250514` - Computer use + good value

---

## Essential Parameters & Context Management

### Core Parameters

#### Temperature (0.0 - 2.0)
Controls response creativity and randomness.

```python
# Conservative/Factual (0.0-0.3)
response = completion(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    temperature=0.1        # Consistent, factual responses
)

# Balanced (0.5-0.8)  
response = completion(
    model="claude-4-sonnet-20250514",
    messages=[{"role": "user", "content": "Write a creative story"}],
    temperature=0.7        # Creative but coherent
)

# Highly Creative (1.0-2.0)
response = completion(
    model="gemini-2.5-pro",
    messages=[{"role": "user", "content": "Brainstorm innovative ideas"}],
    temperature=1.2        # Maximum creativity
)
```

#### Max Tokens / Max Completion Tokens
Controls response length limits.

```python
# Short responses
response = completion(
    model="o4-mini-2025-04-16",
    messages=messages,
    max_tokens=100         # Brief responses
)

# Long-form content
response = completion(
    model="gpt-4.1",
    messages=messages,
    max_completion_tokens=8192  # Extended responses (newer parameter)
)

# Context-aware limits using built-in functions
def adaptive_max_tokens(model, input_length):
    max_context = litellm.get_max_tokens(model)
    if not max_context:
        max_context = 200000  # Fallback for unknown models
    
    available = max_context - input_length
    return min(available - 100, 4096)  # Leave buffer

max_tokens = adaptive_max_tokens("gpt-4.1", len(input_text))
```

#### Advanced Sampling Parameters

```python
response = completion(
    model="claude-4-opus-20250514",
    messages=messages,
    temperature=0.7,
    top_p=0.9,             # Nucleus sampling - alternative to temperature
    frequency_penalty=0.3,  # Reduce repetition (OpenAI models)
    presence_penalty=0.2,   # Encourage new topics (OpenAI models)
    seed=12345             # Reproducible outputs
)
```

#### Reasoning Parameters (O3, O4, Claude-4, Gemini-2.5)

```python
# OpenAI O3/O4 Reasoning
response = completion(
    model="o3-2025-04-16",
    messages=messages,
    reasoning_effort="high",    # low/medium/high
    max_reasoning_tokens=10000, # Tokens for internal reasoning
    temperature=1.0             # ⚠️ ONLY temperature=1.0 is supported!
)

# ❌ This will FAIL - O-series models only accept temperature=1.0
# response = completion(
#     model="o4-mini-2025-04-16",
#     messages=messages,
#     reasoning_effort="medium",
#     temperature=0.7  # ❌ UnsupportedParamsError
# )

# Claude-4 Reasoning
response = completion(
    model="claude-4-opus-20250514", 
    messages=messages,
    reasoning_content=True,     # Show reasoning process
    temperature=0.5
)

# Gemini-2.5 Reasoning (OpenAI-compatible)
response = completion(
    model="gemini-2.5-pro",
    messages=messages,
    reasoning_effort="high",    # low/medium/high (maps to thinking budget)
    temperature=0.4
)

# Gemini-2.5 Reasoning (Anthropic-compatible)
response = completion(
    model="gemini-2.5-flash",
    messages=messages,
    thinking={"type": "enabled", "budget_tokens": 4096},  # Direct token control
    temperature=0.4
)

# Gemini Reasoning Parameter Mapping
# LiteLLM converts to Google's native thinkingConfig:
# reasoning_effort="low"     → thinkingBudget: 1024 tokens
# reasoning_effort="medium"  → thinkingBudget: 2048 tokens  
# reasoning_effort="high"    → thinkingBudget: 4096 tokens
# reasoning_effort="disable" → thinkingBudget: 0 tokens
# thinking={"budget_tokens": X} → thinkingBudget: X tokens
```

### Context Window Management

#### Token Counting and Estimation

```python
import tiktoken

def count_tokens(text, model="gpt-4.1"):
    """Estimate token count for different models"""
    
    # OpenAI models
    if model.startswith(("gpt", "o3", "o4")):
        encoding = tiktoken.encoding_for_model("gpt-4")
        return len(encoding.encode(text))
    
    # Rough estimation for other providers (4 chars ≈ 1 token)
    return len(text) // 4

def check_context_limit(messages, model):
    """Check if messages fit within context window using built-in functions"""
    
    # Get actual context limit from LiteLLM
    limit = litellm.get_max_tokens(model)
    if not limit:
        limit = 200000  # Fallback for unknown models
    
    total_tokens = sum(count_tokens(msg["content"]) for msg in messages)
    
    return total_tokens <= limit * 0.9  # 90% safety margin

# Usage
if check_context_limit(messages, "gpt-4.1"):
    response = completion(model="gpt-4.1", messages=messages)
else:
    print("Messages exceed context limit, consider summarization")
```

#### Context Window Strategies

```python
def summarize_conversation(messages, model="gpt-4.1"):
    """Summarize old messages to fit context window"""
    
    if len(messages) <= 10:
        return messages
    
    # Keep system message and recent messages
    system_msgs = [m for m in messages if m["role"] == "system"]
    recent_msgs = messages[-6:]  # Keep last 6 messages
    
    # Summarize middle conversation
    middle_msgs = messages[len(system_msgs):-6]
    if middle_msgs:
        summary_prompt = {
            "role": "user",
            "content": f"Summarize this conversation history: {middle_msgs}"
        }
        
        summary_response = completion(
            model=model,
            messages=[summary_prompt],
            max_tokens=500
        )
        
        summary_msg = {
            "role": "assistant", 
            "content": f"[Summary]: {summary_response.choices[0].message.content}"
        }
        
        return system_msgs + [summary_msg] + recent_msgs
    
    return messages

# Smart context management
def smart_completion(model, messages, **kwargs):
    """Completion with automatic context management"""
    
    # Check context limits
    if not check_context_limit(messages, model):
        messages = summarize_conversation(messages, model)
    
    return completion(model=model, messages=messages, **kwargs)

# Usage
response = smart_completion("gpt-4.1", long_conversation_messages)
```

### Streaming and Real-time Responses

```python
# Basic streaming
def stream_response(model, messages):
    """Stream responses in real-time"""
    
    response = completion(
        model=model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True}  # Get token usage
    )
    
    full_response = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content
    
    return full_response

# Advanced streaming with usage tracking
def advanced_stream(model, messages):
    """Stream with usage monitoring"""
    
    response = completion(
        model=model,
        messages=messages,
        stream=True,
        stream_options={
            "include_usage": True,
            "include_tokens": True
        }
    )
    
    chunks = []
    for chunk in response:
        chunks.append(chunk)
        
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
        
        # Final chunk contains usage info
        if chunk.usage:
            print(f"\n\nTokens used: {chunk.usage.total_tokens}")
    
    return chunks

# Usage
print("Streaming response:")
result = stream_response("claude-4-sonnet-20250514", messages)
```

### Parameter Compatibility Matrix

| Parameter | OpenAI | Anthropic | Google | xAI | Notes |
|-----------|--------|-----------|--------|-----|-------|
| **temperature** | ⚠️ | ✅ | ✅ | ✅ | O3/O4: Only 1.0 allowed |
| **max_tokens** | ✅ | ✅ | ✅ | ✅ | Universal |
| **top_p** | ⚠️ | ✅ | ✅ | ✅ | Not supported on O3/O4 |
| **stream** | ✅ | ✅ | ✅ | ✅ | Universal |
| **seed** | ⚠️ | ❌ | ❌ | ❌ | Not supported on O3/O4 |
| **frequency_penalty** | ⚠️ | ❌ | ❌ | ❌ | Not supported on O3/O4 |
| **reasoning_effort** | ✅ (O3/O4) | ❌ | ✅ | ❌ | OpenAI + Gemini reasoning |
| **cache_control** | ❌ | ✅ | ✅ | ❌ | Explicit caching |
| **reasoning_content** | ❌ | ✅ | ❌ | ❌ | Claude reasoning |

**⚠️ Critical O3/O4 Limitations:**
- **Temperature**: Only `temperature=1.0` is supported
- **Unsupported**: `top_p`, `frequency_penalty`, `presence_penalty`, `logprobs`
- **Workaround**: Use `drop_params=True` to ignore unsupported parameters

---

## Typed Message Interfaces

LiteLLM provides comprehensive **TypedDict** interfaces instead of raw dictionaries for type-safe message handling across all providers.

### Basic Typed Messages

```python
from litellm.types.completion import (
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionMessageParam  # Union of all message types
)
from typing import List

# ✅ Type-safe messages instead of raw dictionaries
user_message: ChatCompletionUserMessageParam = {
    "role": "user",
    "content": "Write a creative story about quantum computing",
    "name": "storyteller"  # Optional participant name
}

system_message: ChatCompletionSystemMessageParam = {
    "role": "system", 
    "content": "You are a creative writing assistant specializing in science fiction.",
    "name": "assistant_config"
}

# Type-safe message list
messages: List[ChatCompletionMessageParam] = [system_message, user_message]

response = completion(
    model="claude-4-sonnet-20250514",
    messages=messages,
    max_tokens=1000,
    temperature=0.8
)
```

### Multimodal Content Types

```python
from litellm.types.completion import (
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionUserMessageParam
)

# ✅ Structured multimodal message with types
multimodal_message: ChatCompletionUserMessageParam = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "Analyze this architectural design and suggest improvements"
        },
        {
            "type": "image_url", 
            "image_url": {
                "url": "data:image/jpeg;base64,/9j/4AAQ...",
                "detail": "high"  # "auto", "low", "high"
            }
        }
    ]
}

response = completion(
    model="gpt-4.1",
    messages=[multimodal_message],
    max_tokens=1500
)
```


### Message Builder Pattern

```python
from litellm.types.completion import ChatCompletionMessageParam
from typing import List

class TypedMessageBuilder:
    """Type-safe message builder for complex conversations"""
    
    def __init__(self):
        self.messages: List[ChatCompletionMessageParam] = []
    
    def add_system(self, content: str, name: str = None) -> 'TypedMessageBuilder':
        message: ChatCompletionMessageParam = {
            "role": "system",
            "content": content
        }
        if name:
            message["name"] = name
        self.messages.append(message)
        return self
    
    def add_user_text(self, content: str, name: str = None) -> 'TypedMessageBuilder':
        message: ChatCompletionMessageParam = {
            "role": "user", 
            "content": content
        }
        if name:
            message["name"] = name
        self.messages.append(message)
        return self
    
    def add_user_multimodal(
        self, 
        text: str, 
        image_url: str, 
        image_detail: str = "auto",
        name: str = None
    ) -> 'TypedMessageBuilder':
        message: ChatCompletionMessageParam = {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": image_detail
                    }
                }
            ]
        }
        if name:
            message["name"] = name
        self.messages.append(message)
        return self
    
    def add_assistant(self, content: str, name: str = None) -> 'TypedMessageBuilder':
        message: ChatCompletionMessageParam = {
            "role": "assistant",
            "content": content
        }
        if name:
            message["name"] = name
        self.messages.append(message)
        return self
    
    def build(self) -> List[ChatCompletionMessageParam]:
        return self.messages

# ✅ Usage with full type safety and IDE support
conversation = (
    TypedMessageBuilder()
    .add_system("You are an expert technical writer", "technical_assistant")
    .add_user_multimodal(
        text="Explain the architecture shown in this diagram",
        image_url="https://example.com/architecture.png",
        image_detail="high",
        name="developer"
    )
    .add_assistant("I'll analyze the architecture diagram for you.", "analyst")
    .build()
)

response = completion(
    model="gpt-4.1",
    messages=conversation,
    max_tokens=2000
)
```

### Provider-Specific Typed Extensions

```python
# Anthropic-specific features with types
def create_anthropic_conversation():
    """Create conversation optimized for Anthropic Claude"""
    
    messages: List[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are Claude, an AI assistant created by Anthropic.",
                    "cache_control": {"type": "ephemeral"}  # Anthropic caching
                }
            ]
        },
        {
            "role": "user",
            "content": "Help me understand quantum entanglement"
        }
    ]
    
    return messages

# OpenAI O3/O4 with reasoning
def create_reasoning_conversation():
    """Create conversation for OpenAI reasoning models"""
    
    messages: List[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": "You are an expert problem solver. Show your reasoning clearly."
        },
        {
            "role": "user", 
            "content": "Solve this step by step: A train travels 240km in 3 hours, then 160km in 2 hours. What's the average speed?"
        }
    ]
    
    return messages

# Usage with model-specific parameters
claude_response = completion(
    model="claude-4-opus-20250514",
    messages=create_anthropic_conversation(),
    reasoning_content=True  # Show reasoning process
)

o3_response = completion(
    model="o3-2025-04-16", 
    messages=create_reasoning_conversation(),
    reasoning_effort="high",  # High reasoning effort
    temperature=1.0         # Only 1.0 allowed for O3/O4
)
```

### Type Safety Benefits

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # ✅ Full IDE support and type checking
    from litellm.types.completion import ChatCompletionMessageParam

def create_safe_conversation(user_input: str) -> List['ChatCompletionMessageParam']:
    """Type-safe function with full IDE support"""
    
    # ✅ IDE autocomplete and error detection
    messages: List[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
            # ✅ IDE will catch missing comma or typos
        },
        {
            "role": "user",
            "content": user_input
            # ✅ IDE validates required fields
        }
    ]
    
    return messages

# ✅ Type checking catches errors at development time
def process_response(response) -> str:
    message: Message = response.choices[0].message
    
    # ✅ IDE knows message.content is Optional[str]
    if message.content:
        return message.content
    else:
        return "No content received"
```

### Available Type Definitions

| Type | Purpose | Location |
|------|---------|----------|
| **ChatCompletionMessageParam** | Union of all message types | `litellm.types.completion` |
| **ChatCompletionUserMessageParam** | User message structure | `litellm.types.completion` |
| **ChatCompletionAssistantMessageParam** | Assistant message structure | `litellm.types.completion` |
| **ChatCompletionSystemMessageParam** | System message structure | `litellm.types.completion` |
| **ChatCompletionToolMessageParam** | Tool response structure | `litellm.types.completion` |
| **ChatCompletionContentPartTextParam** | Text content part | `litellm.types.completion` |
| **ChatCompletionContentPartImageParam** | Image content part | `litellm.types.completion` |
| **Message** | Response message class | `litellm.types.utils` |
| **Delta** | Streaming delta class | `litellm.types.utils` |

### Benefits of Typed Interfaces

✅ **Type Safety**: Catch errors at development time, not runtime  
✅ **IDE Support**: Full autocomplete, IntelliSense, and error detection  
✅ **Documentation**: Self-documenting code with clear interfaces  
✅ **Refactoring**: Safe code refactoring with type checking  
✅ **Team Development**: Consistent interfaces across team members  
✅ **Multimodal**: Proper types for text, images, audio, video content  
✅ **Type Safety**: Complete type coverage for all features  
✅ **Provider Compatibility**: Works across all LiteLLM providers  

**Use LiteLLM's typed interfaces instead of raw dictionaries for professional, maintainable code with full IDE support!**

---

## Provider-Specific Properties

### Anthropic-Specific Features

#### Cache Control for Cost Optimization
```python
# Explicit caching for expensive content
messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are an expert data analyst with extensive knowledge...",
                "cache_control": {"type": "ephemeral"}  # Cache system prompt
            }
        ]
    },
    {
        "role": "user", 
        "content": "What insights can you provide?"
    }
]

response = completion(
    model="claude-4-sonnet-20250514",
    messages=messages
)
```

#### Computer Use Capabilities
```python
# Claude computer use setup
response = completion(
    model="claude-4-sonnet-20250514",
    messages=[
        {
            "role": "user",
            "content": "Take a screenshot and help me organize my desktop"
        }
    ],
    tools=[
        {
            "type": "computer_20241022",
            "name": "computer",
            "display_width_px": 1920,
            "display_height_px": 1080,
            "display_number": 1
        }
    ],
    tool_choice="auto"
)

# Check for computer actions
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        print(f"Computer action: {tool_call.function.name}")
        print(f"Parameters: {tool_call.function.arguments}")
```

#### Assistant Message Prefilling
```python
# Start Claude's response with specific text
messages = [
    {"role": "user", "content": "Write a Python function to calculate fibonacci"},
    {"role": "assistant", "content": "```python\ndef fibonacci("}  # Prefill response
]

response = completion(
    model="claude-4-opus-20250514",
    messages=messages,
    max_tokens=500
)
```

### OpenAI-Specific Features

#### JSON Mode and Structured Outputs
```python
# Force JSON responses
response = completion(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
        {"role": "user", "content": "Generate user profile data for John Doe"}
    ],
    response_format={"type": "json_object"},
    temperature=0.3
)

# Structured outputs with schema
response = completion(
    model="gpt-4.1",
    messages=messages,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "user_profile",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "skills": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["name", "age"]
            }
        }
    }
)
```


#### Reasoning Tokens (O3/O4 Models)
```python
# Access reasoning process
response = completion(
    model="o3-2025-04-16",
    messages=[{"role": "user", "content": "Solve this complex math problem step by step"}],
    reasoning_effort="high",
    max_reasoning_tokens=15000
)

# Check reasoning usage
if hasattr(response.usage, 'reasoning_tokens'):
    print(f"Reasoning tokens used: {response.usage.reasoning_tokens}")
    print(f"Regular tokens used: {response.usage.completion_tokens}")
```

### Google-Specific Features

#### Multimodal Capabilities
```python
# Audio processing
response = completion(
    model="gemini-2.5-pro",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe and analyze this audio"},
                {
                    "type": "audio",
                    "audio": {
                        "data": audio_base64,
                        "mime_type": "audio/mp3"
                    }
                }
            ]
        }
    ]
)

# Video analysis
response = completion(
    model="gemini-2.5-pro",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe what happens in this video"},
                {
                    "type": "video",
                    "video": {
                        "data": video_base64,
                        "mime_type": "video/mp4"
                    }
                }
            ]
        }
    ]
)
```

#### Vertex AI Configuration
```python
# Using Vertex AI endpoints
response = completion(
    model="vertex_ai/gemini-2.5-pro",
    messages=messages,
    vertex_ai_project="your-project-id",
    vertex_ai_location="us-central1",
    vertex_ai_credentials="path/to/credentials.json"
)
```

### xAI-Specific Features

#### Real-time Data Access
```python
# Grok with current information
response = completion(
    model="xai/grok-4",
    messages=[
        {
            "role": "user", 
            "content": "What are the latest developments in AI announced today?"
        }
    ],
    temperature=0.7
)

# Grok's real-time capabilities work automatically - no special configuration needed
```

---

## Universal Caching Strategies

### Understanding Caching Types

**Implicit Caching (Automatic):**
- OpenAI, Azure, Cohere, Deepseek
- Always enabled, no configuration needed
- Provider manages caching automatically

**Explicit Caching (Manual Control):**
- Anthropic, Google/Gemini
- Requires `cache_control` parameter
- You control what gets cached

### Universal Cache Control Approach

```python
def add_universal_caching(messages):
    """Add cache_control to all messages - works everywhere"""
    
    cached_messages = []
    for msg in messages:
        if isinstance(msg.get("content"), str):
            # Convert string to structured format with caching
            cached_msg = {
                "role": msg["role"],
                "content": [
                    {
                        "type": "text",
                        "text": msg["content"],
                        "cache_control": {"type": "ephemeral"}  # Add everywhere
                    }
                ]
            }
        else:
            # Already structured - add cache_control
            cached_msg = msg.copy()
            if isinstance(msg.get("content"), list):
                for item in cached_msg["content"]:
                    if item.get("type") == "text":
                        item["cache_control"] = {"type": "ephemeral"}
            
        cached_messages.append(cached_msg)
    
    return cached_messages

# Universal caching function
def cached_completion(model, messages, **kwargs):
    """Completion with automatic caching optimization"""
    
    # Add cache_control to all messages
    cached_messages = add_universal_caching(messages)
    
    # This works with ALL providers:
    # ✅ Anthropic - Uses cache_control for explicit caching
    # ✅ Google - Uses cache_control for context caching  
    # ✅ OpenAI - Ignores cache_control, uses implicit caching
    # ✅ xAI - Ignores cache_control, uses implicit caching
    
    return completion(model=model, messages=cached_messages, **kwargs)

# Usage - works optimally with any provider
response = cached_completion("claude-4-sonnet-20250514", messages)
response = cached_completion("gpt-4.1", messages)  
response = cached_completion("gemini-2.5-pro", messages)
response = cached_completion("xai/grok-4", messages)
```

### Smart Caching for Different Use Cases

```python
class SmartCachingChat:
    """Conversation manager with optimal caching"""
    
    def __init__(self, model, system_prompt=None):
        self.model = model
        self.messages = []
        
        # Cache system prompt automatically
        if system_prompt:
            self.messages.append({
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            })
    
    def add_message(self, role, content, cache=None):
        """Add message with smart caching"""
        
        # Auto-cache long content or important messages
        should_cache = (
            cache or 
            len(content) > 1000 or  # Long content
            role == "system" or     # System messages
            len(self.messages) % 5 == 0  # Every 5th message as checkpoint
        )
        
        if should_cache:
            message = {
                "role": role,
                "content": [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            }
        else:
            message = {"role": role, "content": content}
        
        self.messages.append(message)
    
    def get_response(self, user_message, **kwargs):
        """Get response with optimal caching"""
        
        # Add user message with smart caching
        self.add_message("user", user_message)
        
        # Get completion
        response = completion(model=self.model, messages=self.messages, **kwargs)
        
        # Add assistant response
        assistant_content = response.choices[0].message.content
        self.add_message("assistant", assistant_content)
        
        return response

# Usage with any 2025 model
chat = SmartCachingChat("claude-4-opus-20250514", "You are a helpful AI assistant")
response1 = chat.get_response("What is machine learning?")
response2 = chat.get_response("Can you give me examples?")  # Benefits from cached context
```

### Cost Optimization with Caching

```python
def monitor_cache_savings(model, messages):
    """Track caching effectiveness and cost savings"""
    
    response = cached_completion(model, messages)
    
    cache_stats = {
        "total_tokens": 0,
        "cached_tokens": 0,
        "cache_hit_rate": 0,
        "estimated_savings": 0
    }
    
    if response.usage:
        cache_stats["total_tokens"] = response.usage.total_tokens
        
        # Check for cached tokens (Anthropic format)
        if hasattr(response.usage, 'prompt_tokens_details'):
            details = response.usage.prompt_tokens_details
            if isinstance(details, dict) and 'cached_tokens' in details:
                cached = details['cached_tokens']
                cache_stats["cached_tokens"] = cached
                cache_stats["cache_hit_rate"] = (
                    cached / response.usage.prompt_tokens * 100
                    if response.usage.prompt_tokens > 0 else 0
                )
                
                # Estimate cost savings (cached tokens cost ~75% less)
                token_cost = {
                    "gpt-4.1": 0.000002,
                    "claude-4-sonnet-20250514": 0.000003,
                    "gemini-2.5-pro": 0.00000125
                }.get(model, 0.000002)
                
                cache_stats["estimated_savings"] = cached * token_cost * 0.75
    
    return response, cache_stats

# Monitor caching across providers
models = ["gpt-4.1", "claude-4-sonnet-20250514", "gemini-2.5-pro"]
for model in models:
    response, stats = monitor_cache_savings(model, messages)
    print(f"{model}: {stats['cache_hit_rate']:.1f}% cached, ${stats['estimated_savings']:.4f} saved")
```

### Caching Best Practices

```python
# ✅ GOOD: Cache stable, reusable content
system_prompt = {
    "role": "system",
    "content": [
        {
            "type": "text",
            "text": "Large, stable system instructions that won't change...",
            "cache_control": {"type": "ephemeral"}  # Cache this
        }
    ]
}

# ✅ GOOD: Cache long documents or context
document_message = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "Large document content that will be referenced multiple times...",
            "cache_control": {"type": "ephemeral"}  # Cache this
        }
    ]
}

# ❌ BAD: Don't cache dynamic content
dynamic_message = {
    "role": "user",
    "content": f"Current time: {datetime.now()}",  # Changes every call
    # Don't add cache_control here
}

# ✅ GOOD: Cache at strategic conversation points
def cache_conversation_checkpoints(messages):
    """Add caching at strategic points"""
    
    enhanced = []
    for i, msg in enumerate(messages):
        # Cache every 10th message as checkpoint
        if i > 0 and i % 10 == 0:
            enhanced.append(add_cache_control(msg))
        else:
            enhanced.append(msg)
    
    return enhanced
```

---

## File Input Support

### Image Processing with 2025 Models

```python
import base64
from pathlib import Path

def encode_image(image_path):
    """Encode image for API consumption"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Image analysis with different providers
def analyze_image(image_path, question="What do you see in this image?"):
    """Universal image analysis across providers"""
    
    image_data = encode_image(image_path)
    
    # Universal message format for images
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}",
                        "detail": "high"  # OpenAI: "low", "high", "auto"
                    }
                }
            ]
        }
    ]
    
    # Try multiple models for comparison
    models = [
        "gpt-4.1",                    # OpenAI's best vision
        "claude-4-opus-20250514",     # Anthropic's best vision  
        "gemini-2.5-pro",            # Google's best vision
    ]
    
    results = {}
    for model in models:
        try:
            response = completion(model=model, messages=messages, max_tokens=1000)
            results[model] = response.choices[0].message.content
        except Exception as e:
            results[model] = f"Error: {str(e)}"
    
    return results

# Usage
image_analysis = analyze_image("screenshot.png", "Analyze this UI design")
for model, analysis in image_analysis.items():
    print(f"\n{model}:")
    print(analysis)
```

### PDF Document Processing

```python
import PyPDF2
from io import BytesIO

def process_pdf_document(pdf_path, question="Summarize this document"):
    """Extract and analyze PDF content"""
    
    # Extract text from PDF
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    
    # Check if text fits in context window
    if len(text) > 50000:  # Rough token estimate
        # Use models with large context windows
        models = ["gpt-4.1", "gemini-2.5-pro", "gemini-2.5-flash"]
    else:
        models = ["claude-4-opus-20250514", "gpt-4.1", "gemini-2.5-pro"]
    
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a document analysis expert. Provide detailed insights.",
                    "cache_control": {"type": "ephemeral"}  # Cache system prompt
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Document content:\n\n{text}\n\nQuestion: {question}",
                    "cache_control": {"type": "ephemeral"}  # Cache document
                }
            ]
        }
    ]
    
    # Use model with best context/cost ratio
    response = completion(
        model="gemini-2.5-flash",  # Best value for large documents
        messages=messages,
        max_tokens=2000,
        temperature=0.3
    )
    
    return response.choices[0].message.content

# Usage
pdf_analysis = process_pdf_document("research-paper.pdf", "What are the key findings?")
print(pdf_analysis)
```

### Multi-File Processing

```python
def process_multiple_files(file_paths, task="Analyze these files"):
    """Process multiple files simultaneously"""
    
    content_parts = [{"type": "text", "text": task}]
    
    for file_path in file_paths:
        path = Path(file_path)
        
        if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
            # Image file
            image_data = encode_image(file_path)
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}",
                    "detail": "high"
                }
            })
            
        elif path.suffix.lower() == '.pdf':
            # PDF file
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                
                content_parts.append({
                    "type": "text",
                    "text": f"\n\n--- Content from {path.name} ---\n{text}",
                    "cache_control": {"type": "ephemeral"}  # Cache file content
                })
        
        elif path.suffix.lower() in ['.txt', '.md', '.py', '.js']:
            # Text files
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                content_parts.append({
                    "type": "text", 
                    "text": f"\n\n--- Content from {path.name} ---\n{text}",
                    "cache_control": {"type": "ephemeral"}
                })
    
    messages = [{"role": "user", "content": content_parts}]
    
    # Use model with largest context for multi-file processing
    response = completion(
        model="gpt-4.1",  # Largest context window
        messages=messages,
        max_tokens=4000,
        temperature=0.5
    )
    
    return response.choices[0].message.content

# Process mixed file types
files = ["image1.png", "document.pdf", "code.py", "notes.txt"]
analysis = process_multiple_files(files, "Compare and synthesize insights from these files")
print(analysis)
```

### File Input Capabilities by Provider

| Provider | Images | PDFs | Audio | Video | Documents | Notes |
|----------|--------|------|-------|-------|-----------|-------|
| **GPT-4.1** | ✅ High | ✅ Text | ❌ | ❌ | ✅ Text | Best image analysis |
| **Claude-4** | ✅ High | ✅ Text | ❌ | ❌ | ✅ Text | Great document analysis |
| **Gemini-2.5-Pro** | ✅ High | ✅ Text | ✅ | ✅ | ✅ Text | Only with audio/video |
| **Gemini-2.5-Flash** | ✅ Good | ✅ Text | ✅ | ✅ | ✅ Text | Fast multimodal |
| **Grok-4** | ❌ | ❌ | ❌ | ❌ | ❌ | Text only currently |

### Advanced File Processing Patterns

```python
class FileProcessor:
    """Advanced file processing with caching and optimization"""
    
    def __init__(self, preferred_model="gpt-4.1"):
        self.preferred_model = preferred_model
        self.file_cache = {}
    
    def process_file(self, file_path, task, use_cache=True):
        """Process file with caching"""
        
        cache_key = f"{file_path}:{hash(task)}"
        if use_cache and cache_key in self.file_cache:
            return self.file_cache[cache_key]
        
        # Process based on file type
        result = self._process_by_type(file_path, task)
        
        if use_cache:
            self.file_cache[cache_key] = result
        
        return result
    
    def _process_by_type(self, file_path, task):
        """Route processing by file type"""
        
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        if suffix in ['.jpg', '.jpeg', '.png', '.gif']:
            return self._process_image(file_path, task)
        elif suffix == '.pdf':
            return self._process_pdf(file_path, task)
        elif suffix in ['.mp3', '.wav', '.m4a']:
            return self._process_audio(file_path, task)
        else:
            return self._process_text(file_path, task)
    
    def _process_image(self, image_path, task):
        """Optimized image processing"""
        image_data = encode_image(image_path)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
        
        response = completion(
            model=self.preferred_model,
            messages=messages,
            max_tokens=1500
        )
        
        return response.choices[0].message.content

# Usage
processor = FileProcessor("claude-4-opus-20250514")
result = processor.process_file("chart.png", "Extract data from this chart")
```

---

## Vision & Multimodal

### Advanced Vision Capabilities

```python
def comprehensive_image_analysis(image_path):
    """Comprehensive image analysis across different aspects"""
    
    image_data = encode_image(image_path)
    
    analysis_tasks = {
        "description": "Provide a detailed description of this image",
        "objects": "List all objects and people visible in this image",
        "text_extraction": "Extract and transcribe any text visible in this image",
        "emotion_analysis": "Analyze the emotions and mood conveyed in this image",
        "technical_details": "Describe the technical aspects: lighting, composition, color scheme"
    }
    
    results = {}
    
    for task_name, task_prompt in analysis_tasks.items():
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
        
        # Different models excel at different vision tasks
        model_choice = {
            "description": "gpt-4.1",                # Best general description
            "objects": "claude-4-opus-20250514",     # Great object detection
            "text_extraction": "gpt-4.1",           # Excellent OCR
            "emotion_analysis": "claude-4-sonnet-20250514", # Good emotional intelligence
            "technical_details": "gemini-2.5-pro"   # Technical analysis
        }.get(task_name, "gpt-4.1")
        
        response = completion(
            model=model_choice,
            messages=messages,
            max_tokens=500,
            temperature=0.3
        )
        
        results[task_name] = {
            "model": model_choice,
            "analysis": response.choices[0].message.content
        }
    
    return results

# Usage
analysis = comprehensive_image_analysis("complex_scene.jpg")
for task, result in analysis.items():
    print(f"\n{task.upper()} ({result['model']}):")
    print(result['analysis'])
```

### Document and Chart Analysis

```python
def analyze_business_document(image_path, doc_type="general"):
    """Specialized analysis for business documents"""
    
    image_data = encode_image(image_path)
    
    prompts = {
        "chart": """
        Analyze this chart/graph and provide:
        1. Chart type and what it displays
        2. Key data points and trends
        3. Main insights and conclusions
        4. Any anomalies or notable patterns
        """,
        "table": """
        Extract and analyze this table:
        1. Column headers and structure
        2. Key data relationships
        3. Notable patterns or outliers
        4. Summary insights
        """,
        "invoice": """
        Extract information from this invoice:
        1. Vendor and customer details
        2. Line items with quantities and prices
        3. Totals and tax information
        4. Payment terms and dates
        """,
        "general": """
        Analyze this business document:
        1. Document type and purpose
        2. Key information and data
        3. Action items or important details
        4. Overall assessment
        """
    }
    
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a business analyst expert at extracting insights from documents.",
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompts.get(doc_type, prompts["general"])},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}",
                        "detail": "high"
                    }
                }
            ]
        }
    ]
    
    # Claude-4 is excellent for business document analysis
    response = completion(
        model="claude-4-opus-20250514",
        messages=messages,
        max_tokens=2000,
        temperature=0.2
    )
    
    return response.choices[0].message.content

# Usage
chart_analysis = analyze_business_document("sales_chart.png", "chart")
invoice_data = analyze_business_document("invoice.pdf", "invoice")
```

### Multimodal Content Generation

```python
def generate_content_from_image(image_path, content_type="description"):
    """Generate various content types from images"""
    
    image_data = encode_image(image_path)
    
    generation_prompts = {
        "description": "Write a vivid, detailed description of this image",
        "story": "Create a short story inspired by this image",
        "marketing": "Write compelling marketing copy based on this image",
        "social": "Create engaging social media posts about this image",
        "technical": "Write technical documentation or instructions based on this image",
        "creative": "Write a creative poem or artistic interpretation of this image"
    }
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": generation_prompts.get(content_type, generation_prompts["description"])
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}",
                        "detail": "high"
                    }
                }
            ]
        }
    ]
    
    # Different models for different creative tasks
    model_choice = {
        "story": "claude-4-opus-20250514",      # Best creative writing
        "marketing": "gpt-4.1",                # Strong persuasive writing
        "technical": "claude-4-sonnet-20250514", # Clear technical writing
        "creative": "gemini-2.5-pro"           # Creative interpretation
    }.get(content_type, "gpt-4.1")
    
    response = completion(
        model=model_choice,
        messages=messages,
        max_tokens=1000,
        temperature=0.8 if content_type in ["story", "creative"] else 0.5
    )
    
    return response.choices[0].message.content

# Generate different content from same image
image_path = "product_photo.jpg"
marketing_copy = generate_content_from_image(image_path, "marketing")
product_story = generate_content_from_image(image_path, "story")
```

### Audio and Video Processing (Gemini)

```python
def process_audio_content(audio_path, task="transcribe"):
    """Process audio with Gemini models"""
    
    # Encode audio file
    with open(audio_path, "rb") as audio_file:
        audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
    
    task_prompts = {
        "transcribe": "Transcribe this audio accurately",
        "summarize": "Listen to this audio and provide a summary",
        "analyze": "Analyze the content, tone, and key points in this audio",
        "translate": "Transcribe and translate this audio to English"
    }
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": task_prompts.get(task, task_prompts["transcribe"])},
                {
                    "type": "audio",
                    "audio": {
                        "data": audio_data,
                        "mime_type": "audio/mp3"  # or "audio/wav", "audio/m4a"
                    }
                }
            ]
        }
    ]
    
    response = completion(
        model="gemini-2.5-pro",  # Only Gemini supports audio
        messages=messages,
        max_tokens=2000
    )
    
    return response.choices[0].message.content

def process_video_content(video_path, task="describe"):
    """Process video with Gemini models"""
    
    # Encode video file
    with open(video_path, "rb") as video_file:
        video_data = base64.b64encode(video_file.read()).decode('utf-8')
    
    task_prompts = {
        "describe": "Describe what happens in this video",
        "analyze": "Analyze the key events and content in this video",
        "extract": "Extract key information and insights from this video",
        "summarize": "Provide a concise summary of this video content"
    }
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": task_prompts.get(task, task_prompts["describe"])},
                {
                    "type": "video",
                    "video": {
                        "data": video_data,
                        "mime_type": "video/mp4"  # or "video/avi", "video/mov"
                    }
                }
            ]
        }
    ]
    
    response = completion(
        model="gemini-2.5-pro",  # Only Gemini supports video
        messages=messages,
        max_tokens=2000
    )
    
    return response.choices[0].message.content

# Usage
audio_transcript = process_audio_content("meeting.mp3", "summarize")
video_analysis = process_video_content("presentation.mp4", "analyze")
```

### Vision Model Comparison for Specific Tasks

| Task | Best Model | Alternative | Notes |
|------|------------|-------------|-------|
| **OCR/Text Extraction** | GPT-4.1 | Claude-4-Opus | Excellent accuracy |
| **Object Detection** | Claude-4-Opus | GPT-4.1 | Detailed object analysis |
| **Creative Description** | Claude-4-Opus | Gemini-2.5-Pro | Rich, vivid descriptions |
| **Technical Analysis** | Gemini-2.5-Pro | GPT-4.1 | Engineering/scientific images |
| **Chart/Graph Analysis** | Claude-4-Opus | GPT-4.1 | Data extraction |
| **Medical Images** | GPT-4.1 | Claude-4-Opus | Healthcare applications |
| **Art Analysis** | Claude-4-Opus | Gemini-2.5-Pro | Artistic interpretation |
| **UI/UX Analysis** | GPT-4.1 | Claude-4-Sonnet | Interface design |

---

## Common Use Cases

### Simple Chat Implementation

```python
class ChatBot:
    """Simple chatbot with 2025 models"""
    
    def __init__(self, model="claude-4-sonnet-20250514", system_prompt=None):
        self.model = model
        self.messages = []
        
        if system_prompt:
            self.messages.append({
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"}  # Cache system prompt
                    }
                ]
            })
    
    def chat(self, user_message, **kwargs):
        """Send message and get response"""
        
        # Add user message
        self.messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Get response
        response = completion(
            model=self.model,
            messages=self.messages,
            max_tokens=kwargs.get('max_tokens', 2000),
            temperature=kwargs.get('temperature', 0.7),
            **kwargs
        )
        
        # Add assistant response
        assistant_message = response.choices[0].message.content
        self.messages.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    def get_conversation_cost(self):
        """Estimate conversation cost using built-in functions"""
        total_tokens = 0
        for msg in self.messages:
            content = msg["content"]
            if isinstance(content, str):
                total_tokens += len(content) // 4  # Rough estimate
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        total_tokens += len(item["text"]) // 4
        
        # Use LiteLLM's built-in cost calculation
        try:
            model_info = litellm.get_model_info(self.model)
            cost_per_token = model_info.get("input_cost_per_token", 0.000002)
            return total_tokens * cost_per_token
        except:
            # Fallback estimation
            return total_tokens * 0.000002

# Usage
bot = ChatBot("claude-4-sonnet-20250514", "You are a helpful AI assistant.")
response1 = bot.chat("What is machine learning?")
response2 = bot.chat("Can you give me practical examples?")
print(f"Estimated cost: ${bot.get_conversation_cost():.6f}")
```

### Advanced Reasoning Tasks

```python
def solve_complex_problem(problem, model="o3-2025-04-16"):
    """Use reasoning models for complex problem solving"""
    
    reasoning_prompt = f"""
    Please solve this complex problem step by step:
    
    {problem}
    
    Requirements:
    1. Break down the problem into components
    2. Show your reasoning process clearly
    3. Consider multiple approaches
    4. Verify your solution
    5. Explain any assumptions made
    """
    
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an expert problem solver. Think carefully and show your work.",
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        },
        {
            "role": "user",
            "content": reasoning_prompt
        }
    ]
    
    # Configure for reasoning models
    if model.startswith(("o3", "o4")):
        # OpenAI reasoning models
        response = completion(
            model=model,
            messages=messages,
            reasoning_effort="high",
            max_reasoning_tokens=15000,
            max_tokens=3000,
            temperature=1.0  # ⚠️ ONLY temperature=1.0 is supported for O3/O4!
        )
    elif "claude-4" in model:
        # Claude reasoning
        response = completion(
            model=model,
            messages=messages,
            reasoning_content=True,
            max_tokens=3000,
            temperature=0.3
        )
    else:
        # Standard models
        response = completion(
            model=model,
            messages=messages,
            max_tokens=3000,
            temperature=0.3
        )
    
    return response

# Example usage
math_problem = """
A factory produces widgets at varying rates throughout the day. 
- Morning shift (8am-12pm): 120 widgets/hour
- Afternoon shift (12pm-6pm): 150 widgets/hour  
- Evening shift (6pm-10pm): 90 widgets/hour

If the factory operates 6 days a week and needs to produce 50,000 widgets per month (30 days), 
will they meet their target? If not, what changes are needed?
"""

solution = solve_complex_problem(math_problem, "o3-2025-04-16")
print(solution.choices[0].message.content)
```

### JSON and Structured Output

```python
def generate_structured_data(prompt, schema, model="gpt-4.1"):
    """Generate structured JSON responses"""
    
    system_prompt = "You are a helpful assistant designed to output JSON that matches the provided schema exactly."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{prompt}\n\nSchema: {schema}"}
    ]
    
    # OpenAI models support structured output
    if model.startswith(("gpt", "o3", "o4")):
        response = completion(
            model=model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "schema": schema
                }
            },
            temperature=0.3
        )
    else:
        # Other models - use JSON mode
        messages[0]["content"] += " Always respond with valid JSON."
        response = completion(
            model=model,
            messages=messages,
            response_format={"type": "json_object"} if "claude" not in model else None,
            temperature=0.3
        )
    
    return response

# Example schema
user_profile_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0},
        "skills": {
            "type": "array",
            "items": {"type": "string"}
        },
        "experience": {
            "type": "object",
            "properties": {
                "years": {"type": "integer"},
                "level": {"type": "string", "enum": ["beginner", "intermediate", "advanced", "expert"]}
            }
        }
    },
    "required": ["name", "age", "skills"]
}

# Generate structured data
profile_data = generate_structured_data(
    "Create a user profile for a software developer named Sarah",
    user_profile_schema,
    "gpt-4.1"
)

import json
parsed_data = json.loads(profile_data.choices[0].message.content)
print(json.dumps(parsed_data, indent=2))
```


### Cost-Optimized Multi-Turn Conversations

```python
class CostOptimizedChat:
    """Chat implementation optimized for cost and performance"""
    
    def __init__(self, primary_model="gemini-2.5-flash", reasoning_model="o4-mini-2025-04-16"):
        self.primary_model = primary_model      # Fast, cheap for general chat
        self.reasoning_model = reasoning_model  # For complex problems
        self.messages = []
        self.total_cost = 0.0
    
    def _estimate_cost(self, model, tokens):
        """Estimate cost for token usage"""
        costs_per_million = {
            "gemini-2.5-flash": 0.30,
            "gemini-2.5-pro": 1.25,
            "claude-4-sonnet-20250514": 3.0,
            "gpt-4.1": 2.0,
            "o4-mini-2025-04-16": 1.1,
            "xai/grok-4": 3.0
        }
        return (tokens / 1000000) * costs_per_million.get(model, 2.0)
    
    def _needs_reasoning(self, message):
        """Determine if message needs reasoning model"""
        reasoning_keywords = [
            "solve", "calculate", "analyze", "compare", "evaluate", 
            "explain why", "step by step", "reasoning", "logic"
        ]
        return any(keyword in message.lower() for keyword in reasoning_keywords)
    
    def chat(self, user_message):
        """Smart model selection based on query complexity"""
        
        # Add user message
        self.messages.append({"role": "user", "content": user_message})
        
        # Choose model based on complexity
        if self._needs_reasoning(user_message):
            model = self.reasoning_model
            print(f"Using reasoning model: {model}")
        else:
            model = self.primary_model
            print(f"Using primary model: {model}")
        
        # Get response with caching
        cached_messages = add_universal_caching(self.messages)
        
        response = completion(
            model=model,
            messages=cached_messages,
            max_tokens=1500,
            temperature=0.7
        )
        
        # Track costs
        if response.usage:
            cost = self._estimate_cost(model, response.usage.total_tokens)
            self.total_cost += cost
            print(f"Tokens: {response.usage.total_tokens}, Cost: ${cost:.6f}")
        
        # Add response
        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message
    
    def get_total_cost(self):
        """Get total conversation cost"""
        return self.total_cost

# Usage
chat = CostOptimizedChat()

# Simple questions use cheap model
response1 = chat.chat("Hello, how are you?")

# Complex questions automatically use reasoning model  
response2 = chat.chat("Solve this step by step: If a train travels 120 km in 1.5 hours, what's its average speed?")

print(f"\nTotal cost: ${chat.get_total_cost():.6f}")
```

---

## Compatibility Matrix

### Complete Feature Support Matrix

| Feature | GPT-4.1 | O3 | O4-Mini | Claude-4-Opus | Claude-4-Sonnet | Gemini-2.5-Pro | Gemini-2.5-Flash | Grok-4 |
|---------|---------|----|---------|--------------|-----------------|-----------------|--------------------|---------|
| **Context Window** | 1.05M | 200K | 200K | 200K | 200K | 1M | 1M | 256K |
| **Vision** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **JSON Mode** | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ❌ |
| **Structured Output** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Streaming** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Reasoning** | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Computer Use** | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Audio Processing** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| **Video Processing** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| **Cache Control** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Reasoning Tokens** | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |

**Legend:**
- ✅ Full Support
- ⚠️ Limited/Partial Support  
- ❌ Not Supported

### Message Role Support

| Role | OpenAI | Anthropic | Google | xAI | Notes |
|------|--------|-----------|--------|-----|-------|
| **system** | ✅ | ✅ | ✅ | ✅ | Universal support |
| **user** | ✅ | ✅ | ✅ | ✅ | Universal support |
| **assistant** | ✅ | ✅ | ✅ | ✅ | Universal support |

### Parameter Compatibility

#### Universal Parameters (Work Everywhere)
```python
universal_params = {
    "model": "any-2025-model",
    "messages": [...],
    "max_tokens": 2000,
    "temperature": 0.7,
    "top_p": 0.9,
    "stream": True,
    "stop": ["<stop>"],
    "n": 1  # Number of responses
}
```

#### OpenAI-Specific Parameters
```python
openai_params = {
    "frequency_penalty": 0.3,    # Reduce repetition
    "presence_penalty": 0.2,     # Encourage new topics  
    "logit_bias": {...},         # Token probability adjustments
    "seed": 12345,               # Reproducible outputs
    "reasoning_effort": "high",  # O3/O4 models only
    "max_reasoning_tokens": 10000, # O3/O4 models only
    "response_format": {"type": "json_object"}, # JSON mode
    "tools": [...],              # Function definitions
    "tool_choice": "auto"        # Tool selection
}
```

#### Anthropic-Specific Parameters
```python
anthropic_params = {
    "system": "System message",     # Alternative system format
    "reasoning_content": True,      # Show reasoning process
    "tools": [...],                 # Function definitions  
    "tool_choice": {"type": "auto"} # Tool selection format
}
```

#### Google-Specific Parameters
```python
google_params = {
    "vertex_ai_project": "project-id",
    "vertex_ai_location": "us-central1", 
    "vertex_ai_credentials": "path/to/creds.json",
    "candidate_count": 1,           # Alternative to 'n'
    "safety_settings": [...]        # Content filtering
}
```

### Error Code Compatibility

| Error Type | OpenAI Code | Anthropic Code | Google Code | Description |
|------------|-------------|----------------|-------------|-------------|
| **Rate Limit** | 429 | 429 | 429 | Too many requests |
| **Invalid API Key** | 401 | 401 | 401 | Authentication failed |
| **Model Not Found** | 404 | 404 | 404 | Invalid model name |
| **Context Too Long** | 400 | 400 | 400 | Exceeds context limit |
| **Invalid Request** | 400 | 400 | 400 | Malformed request |
| **Server Error** | 500 | 500 | 500 | Provider server error |

### Context Window Utilization

```python
def get_context_info(model):
    """Get context window information for any model"""
    
    context_windows = {
        # OpenAI 2025
        "gpt-4.1": {"max_tokens": 1047576, "optimal_usage": 943819},  # 90% of max
        "o3-2025-04-16": {"max_tokens": 200000, "optimal_usage": 180000},
        "o4-mini-2025-04-16": {"max_tokens": 200000, "optimal_usage": 180000},
        
        # Anthropic 2025  
        "claude-4-opus-20250514": {"max_tokens": 200000, "optimal_usage": 180000},
        "claude-4-sonnet-20250514": {"max_tokens": 200000, "optimal_usage": 180000},
        
        # Google 2025
        "gemini-2.5-pro": {"max_tokens": 1048576, "optimal_usage": 943718},
        "gemini-2.5-flash": {"max_tokens": 1048576, "optimal_usage": 943718},
        
        # xAI 2025
        "xai/grok-4": {"max_tokens": 256000, "optimal_usage": 230400}
    }
    
    return context_windows.get(model, {"max_tokens": 200000, "optimal_usage": 180000})

# Check if conversation fits
def validate_context_usage(model, messages):
    """Validate message length against context window"""
    
    info = get_context_info(model)
    estimated_tokens = sum(len(str(msg)) // 4 for msg in messages)
    
    return {
        "estimated_tokens": estimated_tokens,
        "max_tokens": info["max_tokens"],
        "optimal_tokens": info["optimal_usage"], 
        "fits_in_context": estimated_tokens <= info["optimal_usage"],
        "utilization_percent": (estimated_tokens / info["max_tokens"]) * 100
    }

# Usage
validation = validate_context_usage("gpt-4.1", long_conversation)
print(f"Context utilization: {validation['utilization_percent']:.1f}%")
```

---

## Production Essentials

### Robust Error Handling

```python
import time
import random
from typing import Optional

class LiteLLMError(Exception):
    """Base exception for LiteLLM errors"""
    pass

class RateLimitError(LiteLLMError):
    """Rate limit exceeded"""
    pass

class ContextLengthError(LiteLLMError):
    """Context length exceeded"""
    pass

def robust_completion(
    model: str,
    messages: list,
    max_retries: int = 3,
    backoff_base: float = 2.0,
    fallback_models: Optional[list] = None,
    **kwargs
):
    """Completion with comprehensive error handling"""
    
    models_to_try = [model] + (fallback_models or [])
    
    for model_attempt in models_to_try:
        for attempt in range(max_retries):
            try:
                response = completion(
                    model=model_attempt,
                    messages=messages,
                    **kwargs
                )
                
                # Log successful completion
                print(f"✅ Success with {model_attempt} on attempt {attempt + 1}")
                return response
                
            except Exception as e:
                error_message = str(e).lower()
                
                # Handle specific error types
                if "rate limit" in error_message or "429" in error_message:
                    if attempt < max_retries - 1:
                        wait_time = (backoff_base ** attempt) + random.uniform(0, 1)
                        print(f"⏳ Rate limited. Waiting {wait_time:.1f}s before retry {attempt + 2}")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"❌ Rate limit exceeded for {model_attempt}")
                        break
                
                elif "context" in error_message or "token" in error_message:
                    print(f"❌ Context length exceeded for {model_attempt}")
                    # Try next model if available
                    break
                
                elif "authentication" in error_message or "401" in error_message:
                    print(f"❌ Authentication failed for {model_attempt}")
                    # Try next model if available  
                    break
                
                else:
                    # Generic error - retry with backoff
                    if attempt < max_retries - 1:
                        wait_time = (backoff_base ** attempt) + random.uniform(0, 1)
                        print(f"⚠️ Error: {e}. Retrying in {wait_time:.1f}s")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"❌ Max retries exceeded for {model_attempt}")
                        break
    
    # All models and retries failed
    raise LiteLLMError(f"All models failed: {models_to_try}")

# Usage with fallback strategy
fallback_models = [
    "claude-4-sonnet-20250514",  # Primary alternative
    "gemini-2.5-flash",         # Fast backup
    "gpt-4.1"                   # Final fallback
]

response = robust_completion(
    model="claude-4-opus-20250514",
    messages=messages,
    fallback_models=fallback_models,
    max_tokens=2000,
    temperature=0.7
)
```

### Cost Monitoring and Budget Controls

```python
class CostMonitor:
    """Track and control API costs across providers"""
    
    def __init__(self, daily_budget=10.0):
        self.daily_budget = daily_budget
        self.daily_spend = 0.0
        self.total_spend = 0.0
        self.usage_stats = {}
        
        # Cost per 1M tokens (input/output)
        self.model_costs = {
            "gpt-4.1": (2.0, 8.0),
            "o3-2025-04-16": (2.0, 8.0),
            "o4-mini-2025-04-16": (1.1, 4.4),
            "claude-4-opus-20250514": (15.0, 75.0),
            "claude-4-sonnet-20250514": (3.0, 15.0),
            "gemini-2.5-pro": (1.25, 10.0),
            "gemini-2.5-flash": (0.30, 2.50),
            "xai/grok-4": (3.0, 15.0)
        }
    
    def calculate_cost(self, model, input_tokens, output_tokens):
        """Calculate cost for token usage"""
        input_cost, output_cost = self.model_costs.get(model, (2.0, 8.0))
        
        cost = (
            (input_tokens / 1000000) * input_cost +
            (output_tokens / 1000000) * output_cost
        )
        
        return cost
    
    def check_budget(self, estimated_cost):
        """Check if request fits within budget"""
        if self.daily_spend + estimated_cost > self.daily_budget:
            return False, f"Would exceed daily budget: ${self.daily_spend + estimated_cost:.4f} > ${self.daily_budget}"
        return True, "Within budget"
    
    def monitored_completion(self, model, messages, **kwargs):
        """Completion with cost monitoring"""
        
        # Estimate input tokens
        input_tokens = sum(len(str(msg)) // 4 for msg in messages)
        max_output_tokens = kwargs.get('max_tokens', 2000)
        
        # Estimate cost
        estimated_cost = self.calculate_cost(model, input_tokens, max_output_tokens)
        
        # Budget check
        within_budget, message = self.check_budget(estimated_cost)
        if not within_budget:
            raise LiteLLMError(f"Budget exceeded: {message}")
        
        # Make request
        response = completion(model=model, messages=messages, **kwargs)
        
        # Track actual usage
        if response.usage:
            actual_cost = self.calculate_cost(
                model,
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )
            
            self.daily_spend += actual_cost
            self.total_spend += actual_cost
            
            # Update stats
            if model not in self.usage_stats:
                self.usage_stats[model] = {"requests": 0, "tokens": 0, "cost": 0.0}
            
            self.usage_stats[model]["requests"] += 1
            self.usage_stats[model]["tokens"] += response.usage.total_tokens
            self.usage_stats[model]["cost"] += actual_cost
            
            print(f"💰 Cost: ${actual_cost:.6f} | Daily: ${self.daily_spend:.4f}/{self.daily_budget}")
        
        return response
    
    def get_report(self):
        """Generate usage report"""
        report = f"""
📊 COST MONITORING REPORT
========================
Daily Spend: ${self.daily_spend:.4f} / ${self.daily_budget:.2f}
Total Spend: ${self.total_spend:.4f}
Budget Used: {(self.daily_spend/self.daily_budget)*100:.1f}%

Model Usage:
"""
        for model, stats in self.usage_stats.items():
            avg_cost = stats["cost"] / stats["requests"] if stats["requests"] > 0 else 0
            report += f"  {model}: {stats['requests']} requests, ${stats['cost']:.4f} (avg: ${avg_cost:.6f})\n"
        
        return report

# Usage
monitor = CostMonitor(daily_budget=5.0)

try:
    response = monitor.monitored_completion(
        model="claude-4-opus-20250514",
        messages=messages,
        max_tokens=1000
    )
    print(monitor.get_report())
except LiteLLMError as e:
    print(f"❌ {e}")
```

### Model Fallback Strategies

```python
class SmartModelSelector:
    """Intelligent model selection with fallbacks"""
    
    def __init__(self):
        # Model tiers by capability and cost
        self.model_tiers = {
            "premium": [
                "claude-4-opus-20250514",    # Most capable
                "gpt-4.1",                   # Large context
                "o3-2025-04-16"              # Advanced reasoning
            ],
            "balanced": [
                "claude-4-sonnet-20250514",  # Good balance
                "gemini-2.5-pro",           # Strong multimodal
                "xai/grok-4"                # Real-time reasoning
            ],
            "efficient": [
                "gemini-2.5-flash",         # Fast and cheap
                "o4-mini-2025-04-16",       # Efficient reasoning
            ]
        }
        
        # Task-specific model preferences
        self.task_models = {
            "reasoning": ["o3-2025-04-16", "claude-4-opus-20250514", "xai/grok-4"],
            "vision": ["gpt-4.1", "claude-4-opus-20250514", "gemini-2.5-pro"],
            "long_context": ["gpt-4.1", "gemini-2.5-pro", "gemini-2.5-flash"],
            "creative": ["claude-4-opus-20250514", "gemini-2.5-pro", "gpt-4.1"],
            "fast": ["gemini-2.5-flash", "o4-mini-2025-04-16", "claude-4-sonnet-20250514"],
            "computer_use": ["claude-4-opus-20250514", "claude-4-sonnet-20250514"]
        }
    
    def select_model(self, task_type="general", priority="balanced", context_length=0):
        """Select best model for task"""
        
        if task_type in self.task_models:
            candidates = self.task_models[task_type]
        else:
            candidates = self.model_tiers.get(priority, self.model_tiers["balanced"])
        
        # Filter by context length requirements
        if context_length > 500000:  # Large context needed
            large_context_models = ["gpt-4.1", "gemini-2.5-pro", "gemini-2.5-flash"]
            candidates = [m for m in candidates if m in large_context_models]
        
        return candidates
    
    def smart_completion(self, messages, task_type="general", priority="balanced", **kwargs):
        """Completion with smart model selection and fallbacks"""
        
        # Estimate context requirements
        context_length = sum(len(str(msg)) // 4 for msg in messages)
        
        # Get model candidates
        model_candidates = self.select_model(task_type, priority, context_length)
        
        print(f"🎯 Task: {task_type}, Priority: {priority}")
        print(f"📝 Context length: ~{context_length} tokens")
        print(f"🤖 Model candidates: {model_candidates}")
        
        # Try models in order
        for model in model_candidates:
            try:
                print(f"Trying {model}...")
                response = robust_completion(
                    model=model,
                    messages=messages,
                    max_retries=2,
                    **kwargs
                )
                print(f"✅ Success with {model}")
                return response, model
                
            except Exception as e:
                print(f"❌ {model} failed: {str(e)}")
                continue
        
        raise LiteLLMError("All model candidates failed")

# Usage examples
selector = SmartModelSelector()

# Reasoning task - automatically selects O3 or Claude-4-Opus
response, used_model = selector.smart_completion(
    messages=[{"role": "user", "content": "Solve this complex math problem step by step..."}],
    task_type="reasoning",
    priority="premium"
)

# Vision task - automatically selects GPT-4.1 or Claude-4
response, used_model = selector.smart_completion(
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "Analyze this image"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]}],
    task_type="vision"
)

# Large document - automatically selects Gemini-2.5 or GPT-4.1
response, used_model = selector.smart_completion(
    messages=[{"role": "user", "content": very_long_document}],
    task_type="long_context",  
    priority="efficient"
)
```

### Health Monitoring and Alerts

```python
import logging
from datetime import datetime, timedelta
from collections import defaultdict

class HealthMonitor:
    """Monitor API health and performance"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.alerts = []
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def track_request(self, model, start_time, end_time, success, error=None):
        """Track request metrics"""
        
        duration = (end_time - start_time).total_seconds()
        
        self.metrics[model].append({
            "timestamp": start_time,
            "duration": duration,
            "success": success,
            "error": error
        })
        
        # Check for performance issues
        self._check_performance_alerts(model, duration)
        
        # Log request
        if success:
            self.logger.info(f"✅ {model}: {duration:.2f}s")
        else:
            self.logger.error(f"❌ {model}: {error} ({duration:.2f}s)")
    
    def _check_performance_alerts(self, model, duration):
        """Check for performance degradation"""
        
        # Alert on slow responses
        if duration > 30:
            self.alerts.append({
                "type": "slow_response",
                "model": model,
                "duration": duration,
                "timestamp": datetime.now()
            })
        
        # Alert on high error rates
        recent_requests = [r for r in self.metrics[model] 
                          if r["timestamp"] > datetime.now() - timedelta(minutes=10)]
        
        if len(recent_requests) >= 5:
            error_rate = sum(1 for r in recent_requests if not r["success"]) / len(recent_requests)
            
            if error_rate > 0.3:  # 30% error rate
                self.alerts.append({
                    "type": "high_error_rate",
                    "model": model,
                    "error_rate": error_rate,
                    "timestamp": datetime.now()
                })
    
    def monitored_completion(self, model, messages, **kwargs):
        """Completion with health monitoring"""
        
        start_time = datetime.now()
        
        try:
            response = completion(model=model, messages=messages, **kwargs)
            end_time = datetime.now()
            
            self.track_request(model, start_time, end_time, True)
            return response
            
        except Exception as e:
            end_time = datetime.now()
            self.track_request(model, start_time, end_time, False, str(e))
            raise
    
    def get_health_report(self, hours=24):
        """Generate health report"""
        
        cutoff = datetime.now() - timedelta(hours=hours)
        report = f"🏥 HEALTH REPORT (Last {hours}h)\n" + "="*40 + "\n"
        
        for model, requests in self.metrics.items():
            recent = [r for r in requests if r["timestamp"] > cutoff]
            
            if not recent:
                continue
            
            total_requests = len(recent)
            successful = sum(1 for r in recent if r["success"])
            success_rate = (successful / total_requests) * 100
            avg_duration = sum(r["duration"] for r in recent) / total_requests
            
            report += f"\n{model}:\n"
            report += f"  Requests: {total_requests}\n"
            report += f"  Success Rate: {success_rate:.1f}%\n"
            report += f"  Avg Duration: {avg_duration:.2f}s\n"
            
            # Recent errors
            errors = [r["error"] for r in recent if not r["success"]]
            if errors:
                error_counts = defaultdict(int)
                for error in errors:
                    error_counts[error] += 1
                
                report += "  Recent Errors:\n"
                for error, count in error_counts.items():
                    report += f"    {error}: {count}x\n"
        
        # Active alerts
        recent_alerts = [a for a in self.alerts 
                        if a["timestamp"] > datetime.now() - timedelta(hours=1)]
        
        if recent_alerts:
            report += f"\n🚨 ACTIVE ALERTS:\n"
            for alert in recent_alerts:
                report += f"  {alert['type']}: {alert['model']} at {alert['timestamp']}\n"
        
        return report

# Usage
health = HealthMonitor()

# Monitor all requests
response = health.monitored_completion(
    model="claude-4-opus-20250514",
    messages=messages
)

# Get health report
print(health.get_health_report())
```

---

## Model Information & Cost Management

LiteLLM provides comprehensive built-in utilities to access model metadata, pricing, and capabilities - **no need to write custom functions!**

### Built-in Model Information Functions

```python
import litellm

# Get comprehensive model information
info = litellm.get_model_info("gpt-4.1")
print(f"Max tokens: {info['max_tokens']}")                    # 1047576
print(f"Input cost: ${info['input_cost_per_token'] * 1000000}")  # $2.0 per 1M tokens
print(f"Output cost: ${info['output_cost_per_token'] * 1000000}") # $8.0 per 1M tokens
print(f"Supports vision: {info['supports_vision']}")         # True
print(f"Provider: {info['litellm_provider']}")              # openai

# Check model capabilities
print(f"Vision: {litellm.supports_vision('claude-4-opus-20250514')}")
print(f"Reasoning: {litellm.supports_reasoning('o3-2025-04-16')}")
print(f"Computer use: {litellm.supports_computer_use('claude-4-sonnet-20250514')}")

# Get context window limits
max_tokens = litellm.get_max_tokens("gemini-2.5-pro")  # 1048576
```

### Smart Model Selection

```python
def find_best_model(requirements):
    """Find optimal model based on requirements"""
    
    models = [
        "gpt-4.1", "o3-2025-04-16", "o4-mini-2025-04-16",
        "claude-4-opus-20250514", "claude-4-sonnet-20250514",
        "gemini-2.5-pro", "gemini-2.5-flash", "xai/grok-4"
    ]
    
    suitable_models = []
    
    for model in models:
        try:
            info = litellm.get_model_info(model)
            
            # Check requirements
            if requirements.get("vision") and not info.get("supports_vision"):
                continue
            if requirements.get("reasoning") and not info.get("supports_reasoning"):
                continue
            if requirements.get("min_context") and info.get("max_tokens", 0) < requirements["min_context"]:
                continue
            if requirements.get("max_cost_per_1m") and info.get("input_cost_per_token", 0) * 1000000 > requirements["max_cost_per_1m"]:
                continue
            
            suitable_models.append({
                "model": model,
                "cost_per_1m": info.get("input_cost_per_token", 0) * 1000000,
                "context_window": info.get("max_tokens", 0)
            })
        except:
            continue
    
    # Sort by cost (cheapest first)
    return sorted(suitable_models, key=lambda x: x["cost_per_1m"])

# Example: Find cheapest model with vision and reasoning
requirements = {
    "vision": True,
    "reasoning": True,
    "min_context": 100000,
    "max_cost_per_1m": 5.0
}

best_models = find_best_model(requirements)
for model_info in best_models[:3]:  # Top 3
    print(f"{model_info['model']}: ${model_info['cost_per_1m']:.2f}/1M")
```

### Cost Calculation & Monitoring

```python
# Estimate costs before API calls
def estimate_cost(model, input_text, estimated_output_tokens=500):
    input_tokens = len(input_text) // 4  # Rough estimation
    
    input_cost, output_cost = litellm.cost_per_token(
        model=model,
        prompt_tokens=input_tokens,
        completion_tokens=estimated_output_tokens
    )
    
    return {
        "total_estimated_cost": input_cost + output_cost,
        "input_cost": input_cost,
        "output_cost": output_cost
    }

# Calculate actual costs after completion
response = litellm.completion(
    model="claude-4-sonnet-20250514",
    messages=[{"role": "user", "content": "Write a summary"}]
)

actual_cost = litellm.completion_cost(completion_response=response)
print(f"Actual cost: ${actual_cost:.6f}")
```

### Context Window Validation

```python
def validate_context_window(model, messages):
    """Check if messages fit within model's context window"""
    
    max_tokens = litellm.get_max_tokens(model)
    if not max_tokens:
        return {"valid": False, "error": "Unknown context limit"}
    
    # Estimate token usage (4 chars ≈ 1 token)
    total_tokens = sum(len(str(msg)) // 4 for msg in messages)
    safe_limit = int(max_tokens * 0.9)  # 90% safety margin
    
    return {
        "valid": total_tokens <= safe_limit,
        "estimated_tokens": total_tokens,
        "max_tokens": max_tokens,
        "utilization_percent": (total_tokens / max_tokens) * 100
    }

# Usage
validation = validate_context_window("gpt-4.1", long_messages)
if not validation["valid"]:
    print(f"Messages too long: {validation['utilization_percent']:.1f}% of context")
```

### Model Comparison Utility

```python
def compare_models():
    """Generate model comparison table"""
    
    models = ["gpt-4.1", "claude-4-opus-20250514", "gemini-2.5-pro", "xai/grok-4"]
    
    print(f"{'Model':<25} {'Context':<8} {'Cost/1M':<12} {'Vision':<7} {'Reasoning'}")
    print("-" * 65)
    
    for model in models:
        try:
            info = litellm.get_model_info(model)
            context = f"{info.get('max_tokens', 0)//1000}K"
            cost = f"${info.get('input_cost_per_token', 0)*1000000:.1f}"
            vision = "✓" if info.get('supports_vision') else "✗"
            reasoning = "✓" if info.get('supports_reasoning') else "✗"
            
            print(f"{model:<25} {context:<8} {cost:<12} {vision:<7} {reasoning}")
        except:
            print(f"{model:<25} {'Error':<8}")

compare_models()
```

### Custom Model Registration

```python
# Add custom models or override pricing
custom_models = {
    "my-fine-tuned-gpt": {
        "max_tokens": 8192,
        "input_cost_per_token": 0.000001,  # $1 per 1M tokens
        "output_cost_per_token": 0.000002, # $2 per 1M tokens
        "litellm_provider": "openai",
        "supports_vision": False
    }
}

litellm.register_model(custom_models)

# Now use like any other model
info = litellm.get_model_info("my-fine-tuned-gpt")
response = litellm.completion(
    model="my-fine-tuned-gpt",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Key Built-in Functions Summary

| Function | Purpose | Example |
|----------|---------|---------|
| `get_model_info(model)` | Complete model metadata | `info = litellm.get_model_info("gpt-4.1")` |
| `get_max_tokens(model)` | Context window limit | `max_tokens = litellm.get_max_tokens("claude-4-opus")` |
| `supports_vision(model)` | Check vision capability | `has_vision = litellm.supports_vision("gpt-4.1")` |
| `supports_reasoning(model)` | Check reasoning capability | `has_reasoning = litellm.supports_reasoning("o3-2025")` |
| `cost_per_token(model, ...)` | Calculate token costs | `input_cost, output_cost = litellm.cost_per_token(...)` |
| `completion_cost(response)` | Calculate actual cost | `cost = litellm.completion_cost(response)` |
| `register_model(model_info)` | Add custom models | `litellm.register_model(custom_models)` |

**All model data from `model_prices_and_context_window.json` is automatically accessible through these built-in utilities!**

---

## OpenAI Response API with Universal Session Management

### **What is the Response API?**

OpenAI's Response API is a newer conversational API that provides server-side session management, complementing the traditional Chat Completions API. LiteLLM provides full support for the Response API across all providers.

### **Response API vs Chat Completions API**

| Aspect | Response API | Chat Completions API |
|--------|--------------|---------------------|
| **State Management** | Server-side (automatic) | Client-side (manual) |
| **Input Format** | Single `input` parameter | `messages` array |
| **Session Continuity** | `previous_response_id` | Send full history |
| **Context Handling** | Automatic truncation | Manual management |
| **Caching** | Provider-dependent | Universal support |

### **LiteLLM Response API Examples**

```python
import litellm

# Basic Response API usage
response1 = litellm.responses(
    model="openai/gpt-4",
    input="I want to discuss quantum computing",
    store=True  # Enable server-side session storage
)

# Continue conversation with session continuity
response2 = litellm.responses(
    model="openai/gpt-4",
    input="What are the key principles?",
    previous_response_id=response1.id  # Automatic context preservation
)

# Universal multimodal support - Images work with ALL providers
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

image_data = encode_image("chart.png")

response3 = litellm.responses(
    model="anthropic/claude-4-sonnet-20250514",  # Any provider works!
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Analyze this chart"},
                {
                    "type": "input_image",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data}",
                        "detail": "high"
                    }
                }
            ]
        }
    ]
)

# Universal file support - Files work with ALL providers
def encode_file(file_path):
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode('utf-8')

file_data = encode_file("document.pdf")

response4 = litellm.responses(
    model="vertex_ai/gemini-2.5-pro",  # Any provider works!
    input=[
        {
            "role": "user", 
            "content": [
                {"type": "input_text", "text": "Summarize this document"},
                {
                    "type": "input_file",
                    "file_data": file_data,
                    "mime_type": "application/pdf"
                }
            ]
        }
    ],
    vertex_ai_project="your-project-id",
    vertex_ai_location="us-central1"
)
```

### **Universal Session Management Approach**

For maximum compatibility and caching benefits, use a unified approach that works across all providers:

```python
class UniversalSessionManager:
    """Unified session management with optimal caching for all providers"""
    
    def __init__(self):
        self.conversations = {}
        self.session_metadata = {}
    
    def chat(self, 
             session_id: str, 
             user_input: str, 
             model: str,
             use_caching: bool = True,
             **kwargs):
        """Universal chat method with automatic caching optimization"""
        
        # Initialize session if needed
        if session_id not in self.conversations:
            self.conversations[session_id] = []
            self.session_metadata[session_id] = {
                "created_at": datetime.now().isoformat(),
                "models_used": [],
                "total_tokens": 0,
                "total_cost": 0.0
            }
        
        # Add user message to conversation
        self.conversations[session_id].append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Prepare messages with optimal caching
        messages = self.conversations[session_id].copy()
        if use_caching:
            messages = self._optimize_caching(messages, model)
        
        # Use completion API for universal support + optimal caching
        response = litellm.completion(
            model=model,
            messages=messages,
            **kwargs
        )
        
        # Add assistant response to conversation
        assistant_content = response.choices[0].message.content
        self.conversations[session_id].append({
            "role": "assistant",
            "content": assistant_content,
            "model": model,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update metadata
        self._update_session_metadata(session_id, model, response)
        
        return {
            "response": assistant_content,
            "model_used": model,
            "session_id": session_id,
            "caching_enabled": use_caching,
            "metadata": self.session_metadata[session_id]
        }
    
    def _optimize_caching(self, messages, model):
        """Optimize caching based on provider capabilities"""
        
        if model.startswith("openai/"):
            # OpenAI: Automatic caching works best with consistent prefixes
            return self._optimize_for_openai_prefix_cache(messages)
        else:
            # Other providers: Use explicit cache_control
            return self._add_cache_control(messages)
    
    def _optimize_for_openai_prefix_cache(self, messages):
        """Structure for OpenAI's automatic prefix caching"""
        # OpenAI automatically caches prefixes (1024+ tokens)
        # Keep consistent context at the beginning
        return messages
    
    def _add_cache_control(self, messages):
        """Add cache_control for Anthropic/Google models"""
        cached_messages = []
        
        for i, msg in enumerate(messages):
            if isinstance(msg.get("content"), str):
                # Cache system messages and every 5th message
                if msg["role"] == "system" or i % 5 == 0:
                    cached_msg = {
                        "role": msg["role"],
                        "content": [
                            {
                                "type": "text",
                                "text": msg["content"],
                                "cache_control": {"type": "ephemeral"}
                            }
                        ]
                    }
                else:
                    cached_msg = msg
            else:
                cached_msg = msg
            
            cached_messages.append(cached_msg)
        
        return cached_messages
    
    def _update_session_metadata(self, session_id, model, response):
        """Update session statistics"""
        metadata = self.session_metadata[session_id]
        
        if model not in metadata["models_used"]:
            metadata["models_used"].append(model)
        
        if hasattr(response, 'usage') and response.usage:
            metadata["total_tokens"] += response.usage.total_tokens
            try:
                cost = litellm.completion_cost(completion_response=response)
                metadata["total_cost"] += cost
            except:
                pass

# Usage example with optimal caching
manager = UniversalSessionManager()

# Start conversation with OpenAI (benefits from automatic caching)
result1 = manager.chat(
    "project_session",
    "I'm building a chatbot. What architecture should I consider?",
    "openai/gpt-4"
)

# Continue with Anthropic (uses explicit cache_control)
result2 = manager.chat(
    "project_session",
    "What about handling conversation memory?",
    "anthropic/claude-4-sonnet-20250514"
)

# Switch to Google (uses explicit cache_control)
result3 = manager.chat(
    "project_session",
    "How would you add multimodal capabilities?",
    "vertex_ai/gemini-2.5-pro"
)

# Check session statistics
print(f"Models used: {result3['metadata']['models_used']}")
print(f"Total cost: ${result3['metadata']['total_cost']:.6f}")
```

### **Caching Behavior by Provider**

#### **OpenAI Models (Automatic Caching)**
- **Fully automatic** - no configuration needed
- **50% cost reduction** on cached input tokens (1024+ tokens)
- **Up to 80% latency reduction** for repetitive prompts
- **Works with both** completion API and Response API
- **Cache duration**: 5-10 minutes of inactivity

```python
# OpenAI automatic caching - no parameters needed
response = litellm.completion(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "Long prompt that gets cached automatically..."}]
    # Automatic caching provides 50% cost reduction + 80% latency improvement
)
```

#### **Anthropic/Google Models (Explicit Caching)**
- **Requires `cache_control`** parameter for optimization
- **75% cost reduction** on cached tokens
- **Developer controls** what content gets cached
- **Works with completion API** (Response API support varies)

```python
# Anthropic/Google explicit caching
response = litellm.completion(
    model="anthropic/claude-4-sonnet-20250514",
    messages=[
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "System prompt to cache",
                    "cache_control": {"type": "ephemeral"}  # Explicit caching
                }
            ]
        },
        {"role": "user", "content": "User question"}
    ]
)
```

## **Universal Multimodal Support in Response API**

### **🚀 Revolutionary Discovery: Files and Images Work with ALL Providers!**

LiteLLM's Response API supports multimodal inputs universally across **ALL providers** through an intelligent transformation bridge:

```python
# ✅ Works with ANY provider - same code!
def universal_multimodal_analysis(file_path: str, question: str, model: str):
    """Analyze files with any LiteLLM provider using Response API"""
    
    import base64
    import mimetypes
    
    # Encode file
    with open(file_path, "rb") as f:
        file_data = base64.b64encode(f.read()).decode('utf-8')
    
    mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    
    response = litellm.responses(
        model=model,  # ANY provider: OpenAI, Anthropic, Google, xAI, etc.
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": question},
                    {
                        "type": "input_file",
                        "file_data": file_data,
                        "mime_type": mime_type
                    }
                ]
            }
        ]
    )
    
    return response.choices[0].message.content

# Same function works with ALL providers!
anthropic_analysis = universal_multimodal_analysis(
    "financial_report.pdf", 
    "What are the key financial metrics?",
    "anthropic/claude-4-sonnet-20250514"
)

google_analysis = universal_multimodal_analysis(
    "financial_report.pdf",
    "What are the biggest risks?", 
    "vertex_ai/gemini-2.5-pro"
)

openai_analysis = universal_multimodal_analysis(
    "financial_report.pdf",
    "What's your investment recommendation?",
    "openai/gpt-4"
)

xai_analysis = universal_multimodal_analysis(
    "financial_report.pdf",
    "What are the market trends?",
    "xai/grok-4"
)
```

### **Supported Content Types Across ALL Providers**

| Content Type | Format | OpenAI | Anthropic | Google | xAI | Others |
|-------------|---------|---------|-----------|---------|-----|---------|
| **`input_text`** | Plain text | ✅ | ✅ | ✅ | ✅ | ✅ |
| **`input_image`** | Base64/URL | ✅ | ✅ | ✅ | ✅ | ✅ |
| **`input_file`** | Files/PDFs | ✅ | ✅ | ✅ | ✅ | ✅ |
| **`input_audio`** | Audio files | ✅ | ✅ | ✅ | ✅ | ✅ |

### **Universal File Processing Session Manager**

```python
class UniversalMultimodalSessionManager:
    """Process any file type with any provider using Response API"""
    
    def __init__(self):
        self.sessions = {}
    
    def analyze_document(self, session_id: str, file_path: str, question: str, model: str):
        """Analyze any document with any provider"""
        
        file_data = self._encode_file(file_path)
        mime_type = self._get_mime_type(file_path)
        
        response = litellm.responses(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": question},
                        {
                            "type": "input_file",
                            "file_data": file_data,
                            "mime_type": mime_type
                        }
                    ]
                }
            ],
            store=True,
            previous_response_id=self.sessions.get(session_id, {}).get("last_id")
        )
        
        # Update session
        self.sessions[session_id] = {"last_id": response.id, "model": model}
        
        return response.choices[0].message.content
    
    def analyze_image(self, session_id: str, image_path: str, question: str, model: str):
        """Analyze any image with any provider"""
        
        image_data = self._encode_file(image_path)
        
        response = litellm.responses(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": question},
                        {
                            "type": "input_image",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            store=True,
            previous_response_id=self.sessions.get(session_id, {}).get("last_id")
        )
        
        self.sessions[session_id] = {"last_id": response.id, "model": model}
        
        return response.choices[0].message.content
    
    def continue_discussion(self, session_id: str, follow_up: str, model: str = None):
        """Continue discussing files/images with context preservation"""
        
        # Use same model or switch providers mid-conversation
        current_model = model or self.sessions[session_id]["model"]
        
        response = litellm.responses(
            model=current_model,
            input=follow_up,
            previous_response_id=self.sessions[session_id]["last_id"],
            store=True
        )
        
        self.sessions[session_id].update({
            "last_id": response.id,
            "model": current_model
        })
        
        return response.choices[0].message.content
    
    def _encode_file(self, file_path):
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def _get_mime_type(self, file_path):
        import mimetypes
        return mimetypes.guess_type(file_path)[0] or "application/octet-stream"

# Usage Example - Cross-Provider File Analysis
manager = UniversalMultimodalSessionManager()

# Start with Anthropic analyzing a PDF
result1 = manager.analyze_document(
    "analysis_session",
    "business_plan.pdf",
    "What's the market opportunity?",
    "anthropic/claude-4-sonnet-20250514"
)

# Continue with Google for different perspective
result2 = manager.continue_discussion(
    "analysis_session",
    "What are the potential risks?",
    "vertex_ai/gemini-2.5-pro"  # Switch providers mid-session!
)

# Add image analysis with xAI
result3 = manager.analyze_image(
    "analysis_session", 
    "market_chart.png",
    "How does this chart relate to our discussion?",
    "xai/grok-4"  # Another provider switch!
)

# Final insights with OpenAI
result4 = manager.continue_discussion(
    "analysis_session",
    "What's your final investment recommendation?",
    "openai/gpt-4"  # Fourth provider in same session!
)

print("🎉 Single file discussion across 4 different providers with full context!")
```

### **Cross-Provider File Comparison**

```python
def compare_file_analysis_across_providers(file_path: str, question: str):
    """Get multiple AI perspectives on the same file"""
    
    providers = {
        "anthropic/claude-4-sonnet-20250514": "Analytical & Detailed",
        "vertex_ai/gemini-2.5-pro": "Multimodal & Technical", 
        "openai/gpt-4": "Balanced & Comprehensive",
        "xai/grok-4": "Real-time & Insights"
    }
    
    file_data = encode_file(file_path)
    mime_type = get_mime_type(file_path)
    
    results = {}
    
    for provider, description in providers.items():
        try:
            response = litellm.responses(
                model=provider,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": question},
                            {
                                "type": "input_file",
                                "file_data": file_data,
                                "mime_type": mime_type
                            }
                        ]
                    }
                ]
            )
            
            results[provider] = {
                "description": description,
                "analysis": response.choices[0].message.content,
                "status": "✅ Success"
            }
            
        except Exception as e:
            results[provider] = {
                "description": description,
                "analysis": f"Error: {str(e)}",
                "status": "❌ Failed"
            }
    
    return results

# Compare insights from multiple providers
insights = compare_file_analysis_across_providers(
    "quarterly_report.pdf",
    "What are the most important insights from this report?"
)

for provider, result in insights.items():
    print(f"\n--- {result['description']} ({provider}) {result['status']} ---")
    print(result['analysis'])
```

### **How Universal Multimodal Works**

LiteLLM uses an intelligent **transformation bridge** for non-OpenAI providers:

1. **Input**: Response API format (`input_file`, `input_image`)
2. **Transform**: Convert to Chat Completion format automatically
3. **Process**: Forward to provider's Chat Completion API
4. **Transform**: Convert response back to Response API format
5. **Output**: Consistent Response API response

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Response API  │───▶│  Transformation │───▶│ Chat Completion │
│   input_file    │    │     Bridge      │    │    file format  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Provider API  │◀───│  Transformation │◀───│   Provider      │
│   (Anthropic,   │    │     Bridge      │    │   Processing    │
│   Google, etc.) │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **Response API vs Completion API Recommendations**

#### **Use Response API When:**
- **Processing files/images** with any provider (universal support!)
- Want **server-side session management** (OpenAI models)
- Need **simplified state handling** with multimodal content
- Building **provider-agnostic multimodal applications**
- Want **consistent API** across different providers

#### **Use Completion API When:**
- Need **provider-specific optimizations**
- Want **maximum control** over conversation state
- Building **single-provider applications**
- Need **explicit caching control** for all providers

### **Production-Ready Session Management**

```python
class ProductionSessionManager:
    """Production-ready session management with failover and monitoring"""
    
    def __init__(self):
        self.sessions = {}
        self.fallback_providers = [
            "openai/gpt-4",
            "anthropic/claude-4-sonnet-20250514",
            "vertex_ai/gemini-2.5-pro"
        ]
    
    def resilient_chat(self, session_id: str, user_input: str, preferred_model: str):
        """Chat with automatic failover and session preservation"""
        
        providers_to_try = [preferred_model] + [
            p for p in self.fallback_providers if p != preferred_model
        ]
        
        for model in providers_to_try:
            try:
                return self.chat(session_id, user_input, model)
            except Exception as e:
                print(f"❌ {model} failed: {str(e)}")
                continue
        
        raise Exception("All providers failed")
    
    def get_session_stats(self, session_id: str):
        """Get comprehensive session statistics"""
        if session_id not in self.sessions:
            return {}
        
        session = self.sessions[session_id]
        return {
            "message_count": len(session.get("conversation", [])),
            "models_used": session.get("models_used", []),
            "total_cost": session.get("total_cost", 0.0),
            "caching_effectiveness": self._calculate_cache_effectiveness(session_id)
        }
    
    def _calculate_cache_effectiveness(self, session_id):
        """Calculate caching benefits for the session"""
        # Implementation for tracking cache hits and cost savings
        return {"cache_hit_rate": 0.75, "cost_savings": 0.40}

# Usage
manager = ProductionSessionManager()

# Resilient conversation with automatic failover
try:
    result = manager.resilient_chat(
        "production_session",
        "Analyze this complex data",
        "openai/gpt-4"  # Preferred, with automatic fallback
    )
    print(f"✅ Success: {result['response']}")
except Exception as e:
    print(f"❌ All providers failed: {e}")
```

### **Key Benefits of Universal Approach**

✅ **Provider Agnostic** - Same code works with any provider  
✅ **Optimal Caching** - Automatic optimization for each provider  
✅ **Resilient Fallbacks** - Seamless provider switching  
✅ **Cost Effective** - Leverages best caching for each provider  
✅ **Production Ready** - Error handling and monitoring built-in  
✅ **Simple Integration** - Single API for all session management  

This universal approach provides the best of both worlds: the simplicity of Response API session management with the reliability and caching benefits of the completion API across all providers.

---

## Quick Reference

### Top 2025 Models Cheat Sheet

| Model | Context | Input/Output Cost | Best For | Code Example |
|-------|---------|-------------------|----------|--------------|
| **GPT-4.1** | 1.05M | $2/$8 | Large context, vision | `completion(model="gpt-4.1", messages=messages)` |
| **O3** | 200K | $2/$8 | Advanced reasoning | `completion(model="o3-2025-04-16", reasoning_effort="high")` |
| **O4-Mini** | 200K | $1.1/$4.4 | Efficient reasoning | `completion(model="o4-mini-2025-04-16", reasoning_effort="medium")` |
| **Claude-4-Opus** | 200K | $15/$75 | Most capable, computer use | `completion(model="claude-4-opus-20250514", messages=cached_messages)` |
| **Claude-4-Sonnet** | 200K | $3/$15 | Balanced performance | `completion(model="claude-4-sonnet-20250514", messages=cached_messages)` |
| **Gemini-2.5-Pro** | 1M | $1.25/$10 | Multimodal, audio/video | `completion(model="gemini-2.5-pro", messages=multimodal_messages)` |
| **Gemini-2.5-Flash** | 1M | $0.30/$2.50 | Fastest, cheapest | `completion(model="gemini-2.5-flash", messages=messages)` |
| **Grok-4** | 256K | $3/$15 | Real-time reasoning | `completion(model="xai/grok-4", messages=messages)` |

### Universal Parameters

```python
# Works with ALL 2025 models
response = completion(
    model="any-model-above",
    messages=[...],
    max_tokens=2000,        # Output limit
    temperature=0.7,        # Creativity (0-2)
    top_p=0.9,             # Alternative sampling
    stream=True,           # Real-time responses
    stop=["<END>"],        # Stop sequences
    seed=12345             # Reproducible (OpenAI only)
)
```

### Universal Caching (Works Everywhere)

```python
# Add to ANY message for optimal caching
cached_message = {
    "role": "system",
    "content": [
        {
            "type": "text",
            "text": "Your system prompt here",
            "cache_control": {"type": "ephemeral"}  # Works with all providers
        }
    ]
}
```

### Quick Model Selection Guide

```python
def choose_model(task):
    """Dynamic model selection using built-in capability checking"""
    
    # All available 2025 models
    all_models = [
        "gpt-4.1", "o3-2025-04-16", "o4-mini-2025-04-16",
        "claude-4-opus-20250514", "claude-4-sonnet-20250514",
        "gemini-2.5-pro", "gemini-2.5-flash", "xai/grok-4"
    ]
    
    suitable_models = []
    
    for model in all_models:
        try:
            # Check capabilities dynamically
            if task == "reasoning" and not litellm.supports_reasoning(model):
                continue
            elif task == "vision" and not litellm.supports_vision(model):
                continue
            elif task == "computer" and not litellm.supports_computer_use(model):
                continue
            elif task == "multimodal" and not litellm.supports_vision(model):
                continue
            elif task == "long_docs":
                max_tokens = litellm.get_max_tokens(model)
                if not max_tokens or max_tokens < 500000:
                    continue
            
            # Get cost for sorting
            info = litellm.get_model_info(model)
            cost = info.get("input_cost_per_token", 0) * 1000000
            
            suitable_models.append((model, cost))
        except:
            continue
    
    # Sort by cost (cheapest first) and return top 3
    suitable_models.sort(key=lambda x: x[1])
    return [model for model, cost in suitable_models[:3]]

# Usage - now dynamically determines capabilities
models = choose_model("reasoning")
print(f"Best reasoning models: {models}")
```

### Common Error Fixes

| Error | Solution | Code |
|-------|----------|------|
| **Rate Limited** | Add retry with backoff | `time.sleep(2**attempt)` |
| **Context Too Long** | Summarize or use larger context model | `if len(text) > limit: use_model("gpt-4.1")` |
| **Invalid API Key** | Check environment variables | `os.environ["OPENAI_API_KEY"]` |
| **Model Not Found** | Verify model name | Use exact names from table above |
| **JSON Parse Error** | Add JSON mode | `response_format={"type": "json_object"}` |

### Cost Optimization Quick Tips

```python
# 1. Find cheapest model dynamically
def get_cheapest_model(min_capabilities=None):
    models = ["gemini-2.5-flash", "o4-mini-2025-04-16", "claude-4-sonnet-20250514"]
    costs = []
    for model in models:
        try:
            info = litellm.get_model_info(model)
            cost = info.get("input_cost_per_token", 0) * 1000000
            costs.append((model, cost))
        except:
            continue
    costs.sort(key=lambda x: x[1])
    return costs[0][0] if costs else "gemini-2.5-flash"

cheapest = get_cheapest_model()  # Returns actual cheapest model

# 2. Add caching everywhere
cache_control = {"type": "ephemeral"}  # 75% cost reduction

# 3. Calculate optimal token limits
def get_optimal_tokens(model, budget_per_call):
    info = litellm.get_model_info(model)
    output_cost = info.get("output_cost_per_token", 0.000008)
    return int(budget_per_call / output_cost) if output_cost > 0 else 1000

max_tokens = get_optimal_tokens("gpt-4.1", 0.01)  # $0.01 budget

# 4. Use actual cost calculation
actual_cost = litellm.completion_cost(response)
print(f"This call cost: ${actual_cost:.6f}")
```

### Essential Code Snippets

**Basic Chat:**
```python
response = completion(
    model="claude-4-sonnet-20250514",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

**With Caching:**
```python
messages = [{
    "role": "system",
    "content": [{"type": "text", "text": "System prompt", "cache_control": {"type": "ephemeral"}}]
}]
```

**Vision Analysis:**
```python
messages = [{
    "role": "user", 
    "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    ]
}]
```

**Streaming:**
```python
for chunk in completion(model="gpt-4.1", messages=messages, stream=True):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

**Error Handling:**
```python
try:
    response = completion(model="claude-4-opus-20250514", messages=messages)
except Exception as e:
    response = completion(model="gpt-4.1", messages=messages)  # Fallback
```

**JSON Output:**
```python
response = completion(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Generate user data as JSON"}],
    response_format={"type": "json_object"}
)
```

### Provider Selection Flowchart

```
Need computer interaction? → Claude-4-Opus/Sonnet
Need audio/video processing? → Gemini-2.5-Pro/Flash  
Need 1M+ context? → GPT-4.1 or Gemini-2.5-Pro/Flash
Need advanced reasoning? → O3, Claude-4-Opus, or Grok-4
Need fastest/cheapest? → Gemini-2.5-Flash
Need real-time data? → Grok-4
Need best vision? → GPT-4.1 or Claude-4-Opus
Need balanced performance? → Claude-4-Sonnet
```

---

**🎉 You're now ready to use LiteLLM with 2025's most advanced AI models!**

This essential guide covers everything you need for production-ready applications with optimal cost and performance. For additional features and advanced configurations, refer to the comprehensive LiteLLM documentation.

**Happy coding! 🚀**