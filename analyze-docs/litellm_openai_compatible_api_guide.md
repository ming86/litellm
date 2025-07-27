# LiteLLM: OpenAI-Compatible API Calls to Multiple Providers

## Overview

LiteLLM provides a unified OpenAI-compatible interface to 100+ LLM providers. Use the same API calls you're familiar with from OpenAI, but access models from Anthropic, Google, Cohere, Azure, and many more providers.

**Key Benefits:**
- ✅ **Same API Format**: Use OpenAI's familiar `completion()` function
- ✅ **Easy Provider Switching**: Change just the model name
- ✅ **Unified Response Format**: Consistent responses across all providers
- ✅ **Drop-in Replacement**: Minimal changes to existing OpenAI code

## Quick Start

### 1. Installation

```bash
pip install litellm
```

### 2. Basic Usage

```python
from litellm import completion
import os

# Set your API keys
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

# Same function, different providers
messages = [{"role": "user", "content": "Hello, how are you?"}]

# OpenAI
response = completion(model="gpt-4", messages=messages)
print(response.choices[0].message.content)

# Anthropic
response = completion(model="anthropic/claude-3-sonnet-20240229", messages=messages)
print(response.choices[0].message.content)
```

## Provider Examples

### OpenAI
```python
# Standard OpenAI models
response = completion(
    model="gpt-4",  # or "gpt-3.5-turbo", "gpt-4-turbo"
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Anthropic
```python
# Latest Anthropic Claude models (2025)
response = completion(
    model="anthropic/claude-4-sonnet-20250514",  # Latest 2025: claude-4-opus, claude-4-sonnet, claude-3-7-sonnet
    messages=[{"role": "user", "content": "Hello!"}]
)

# 2024 models (still current)
response = completion(
    model="anthropic/claude-3-5-sonnet-20241022",  # or claude-3-5-haiku-20241022
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Google (Vertex AI)
```python
# Set up Google Cloud credentials first
os.environ["VERTEXAI_PROJECT"] = "your-project-id"
os.environ["VERTEXAI_LOCATION"] = "us-central1"

# Latest Google Gemini models (2025)
response = completion(
    model="vertex_ai/gemini-2.5-pro",  # Latest 2025: 1M context, multimodal, reasoning
    messages=[{"role": "user", "content": "Hello!"}]
)

# 2024/2025 models
response = completion(
    model="vertex_ai/gemini-2.0-flash",  # Fast, cost-effective, 1M context
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Cohere
```python
# Latest Cohere models (2025)
response = completion(
    model="cohere/command-a-03-2025",  # Latest 2025: 256K context, function calling
    messages=[{"role": "user", "content": "Hello!"}]
)

# 2024 models (still current)
response = completion(
    model="cohere/command-r-plus",  # or command-r
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Azure OpenAI
```python
os.environ["AZURE_API_KEY"] = "your-azure-key"
os.environ["AZURE_API_BASE"] = "https://your-resource.openai.azure.com"
os.environ["AZURE_API_VERSION"] = "2023-12-01-preview"

response = completion(
    model="azure/your-deployment-name",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Groq
```python
response = completion(
    model="groq/llama3-8b-8192",  # or mixtral-8x7b-32768
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### xAI (Grok)
```python
# Latest Grok models (2025)
response = completion(
    model="xai/grok-3",  # Latest 2025: Grok-3, reasoning, multimodal
    messages=[{"role": "user", "content": "Hello!"}]
)

# Grok models with vision
response = completion(
    model="xai/grok-2-vision-1212",  # Vision capabilities
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Fireworks AI
```python
response = completion(
    model="fireworks_ai/llama-v3p1-70b-instruct",  # Fast inference
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Together AI
```python
response = completion(
    model="together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Cerebras
```python
response = completion(
    model="cerebras/llama3.1-70b",  # Ultra-fast inference
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Complete Feature Reference

### Core Parameters

LiteLLM's `completion()` function supports all OpenAI parameters plus additional features:

```python
from litellm import completion

response = completion(
    # Required
    model="gpt-4",                    # Model to use
    messages=[...],                   # List of message objects
    
    # Text Generation Control
    temperature=0.7,                  # Randomness (0-2, default: 1)
    max_tokens=1000,                  # Max response tokens
    max_completion_tokens=1000,       # Alternative to max_tokens
    top_p=0.9,                       # Nucleus sampling (0-1)
    n=1,                             # Number of completions
    stop=["END", "\n\n"],            # Stop sequences
    presence_penalty=0.0,            # Penalize new tokens (-2 to 2)
    frequency_penalty=0.0,           # Penalize frequent tokens (-2 to 2)
    logit_bias={"50256": -100},      # Modify token probabilities
    seed=42,                         # Reproducible outputs
    
    # Response Format
    response_format={"type": "json_object"},  # JSON mode
    stream=False,                    # Enable streaming
    stream_options={"include_usage": True},  # Streaming options
    
    # Advanced Features
    tools=[...],                     # Function calling tools
    tool_choice="auto",              # Tool selection: "auto", "none", or specific tool
    parallel_tool_calls=True,        # Enable parallel tool execution
    functions=[...],                 # Legacy function calling (deprecated)
    function_call="auto",            # Legacy function control (deprecated)
    
    # Logging & Debugging
    logprobs=True,                   # Return token probabilities
    top_logprobs=3,                  # Number of top logprobs
    user="user123",                  # User identifier
    
    # API Configuration
    timeout=30,                      # Request timeout (seconds)
    api_key="custom-key",            # Override API key
    base_url="https://custom.api",   # Custom API base
    api_version="2023-12-01",        # API version
    
    # LiteLLM Specific
    force_timeout=60,                # Force timeout
    verbose=True,                    # Enable debug logging
    logger_fn=custom_logger,         # Custom logging function
    extra_body={"metadata": {...}},  # Additional request data
    
    # Specialized Features
    web_search_options={"search_context_size": "medium"},  # Web search
    prompt_id="my-prompt-123",       # Prompt management
    prompt_variables={"name": "John"},  # Prompt variables
    prompt_version=1,                # Prompt version
)
```

### Message Types and Roles

LiteLLM supports **5 different message roles** for rich conversations:

```python
messages = [
    # 1. SYSTEM ROLE - Sets context/behavior for the AI
    {
        "role": "system",
        "content": "You are a helpful AI assistant specialized in Python programming."
    },
    
    # 2. USER ROLE - Human input/questions
    {
        "role": "user", 
        "content": "How do I create a list in Python?"
    },
    
    # 3. ASSISTANT ROLE - AI responses
    {
        "role": "assistant",
        "content": "You can create a list in Python using square brackets: my_list = [1, 2, 3]"
    },
    
    # 4. TOOL ROLE - Function/tool call results
    {
        "role": "tool",
        "tool_call_id": "call_123",
        "name": "get_weather",
        "content": '{"temperature": 72, "condition": "sunny"}'
    },
    
    # 5. FUNCTION ROLE - Legacy function call results (deprecated)
    {
        "role": "function",
        "name": "get_current_weather",
        "content": '{"location": "Boston", "temperature": "72", "unit": "fahrenheit"}'
    }
]
```

**Role Details:**

| Role | Purpose | Required Fields | Notes |
|------|---------|----------------|-------|
| `system` | Define AI behavior/context | `content` | First message, optional |
| `user` | Human input | `content` | Required for conversations |
| `assistant` | AI responses | `content` | Can include `tool_calls` |
| `tool` | Tool/function results | `content`, `tool_call_id`, `name` | Modern function calling |
| `function` | Function results | `content`, `name` | Legacy, use `tool` instead |

## Provider Compatibility & Limitations

### ❌ **NOT Universal** - Provider Support Varies

**Message roles are NOT universally supported!** Different providers have different capabilities:

| Role | OpenAI | Anthropic | Google | Cohere | Azure | Groq | Bedrock |
|------|--------|-----------|--------|--------|-------|------|---------|
| `system` | ✅ | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| `user` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `assistant` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `tool` | ✅ | ✅ | ✅ | ⚠️ | ✅ | ⚠️ | ✅ |
| `function` | ✅ | ❌ | ❌ | ❌ | ✅ | ⚠️ | ❌ |

**Legend:** ✅ Full Support | ⚠️ Limited/Conditional | ❌ Not Supported

### Provider-Specific Properties

#### **1. Anthropic-Specific Properties**

```python
# Assistant prefill (guide response start)
{
    "role": "assistant", 
    "content": "{",  # Prefill JSON start
    "prefix": True  # LiteLLM-specific flag for Anthropic
}

# Cache control for cost optimization
{
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "Long document content...",
            "cache_control": {"type": "ephemeral"}  # Anthropic prompt caching
        }
    ]
}

# Reasoning/thinking blocks (Claude 3.7+)
{
    "role": "assistant",
    "content": "The answer is Paris.",
    "reasoning_content": "Let me think about this...",  # Reasoning process
    "thinking_blocks": [                                      # Structured thinking
        {
            "type": "thinking",
            "thinking": "The user is asking about France's capital...",
            "signature": "base64_encoded_signature"
        }
    ]
}
```

#### **2. System Message Limitations**

Some providers don't support system messages and need configuration:

```python
# Models without system message support
unsupported_models = [
    "google/gemma",
    "replicate/llama-2-7b-chat", 
    # many open-source models
]

# LiteLLM handles this automatically, but you can configure:
response = completion(
    model="custom/my-model",
    messages=[
        {"role": "system", "content": "You are helpful"},  # Gets converted to user message
        {"role": "user", "content": "Hello"}
    ],
    supports_system_message=False  # Force conversion
)
```

#### **3. Function Calling Compatibility**

```python
# Modern approach (widely supported)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather", 
            "description": "Get weather",
            "parameters": {...}
        }
    }
]

# Tool response message
{
    "role": "tool",
    "tool_call_id": "call_123",
    "name": "get_weather", 
    "content": '{"temp": 72}'
}

# Legacy approach (limited support)
functions = [{"name": "get_weather", "parameters": {...}}]  # Deprecated

# Function response (legacy)
{
    "role": "function",
    "name": "get_weather",
    "content": '{"temp": 72}'
}
```

#### **4. Provider-Specific Features**

```python
# Google Vertex AI - Built-in search tools
tools = [
    {"googleSearch": {}},           # Google Search
    {"enterpriseWebSearch": {}}     # Enterprise search
]

# Anthropic - File content support
{
    "role": "user",
    "content": [
        {"type": "text", "text": "Analyze this document"},
        {"type": "file", "file": file_object}  # PDF, images, etc.
    ]
}

# Azure OpenAI - Deployment-specific models
model = "azure/my-deployment-name"  # Must match your Azure deployment

# Bedrock - AWS region-specific routing
os.environ["AWS_REGION_NAME"] = "us-east-1"  # Required for Bedrock
```

#### **5. Hidden/Advanced Properties**

```python
# Message-level properties (not in standard docs)
message = {
    "role": "user",
    "content": "Hello",
    "name": "John",                    # Optional: speaker name
    "reasoning_effort": "high",        # Anthropic: thinking intensity
    "cache_control": {"type": "ephemeral"},  # Anthropic: caching
}

# Model-specific capabilities check
from litellm import supports_reasoning, get_model_info

# Check if model supports reasoning
if supports_reasoning("anthropic/claude-3-7-sonnet-20250219"):
    # Use reasoning features
    response = completion(
        model="anthropic/claude-3-7-sonnet-20250219",
        messages=[...],
        reasoning_effort="medium"
    )

# Get detailed model info
model_info = get_model_info("gpt-4")
print(model_info["supports_assistant_prefill"])  # Check capabilities
```

### **Compatibility Testing Function**

```python
def test_provider_compatibility(model, test_cases):
    """Test what message types a provider supports"""
    results = {}
    
    test_messages = {
        "system": [{"role": "system", "content": "You are helpful"}],
        "basic": [{"role": "user", "content": "Hello"}],
        "prefill": [
            {"role": "user", "content": "Say 'JSON:'"}, 
            {"role": "assistant", "content": "{", "prefix": True}
        ],
        "tools": [{"role": "user", "content": "What's the weather?"}]
    }
    
    for test_name, messages in test_messages.items():
        try:
            response = completion(
                model=model, 
                messages=messages,
                max_tokens=10,
                tools=[{
                    "type": "function",
                    "function": {"name": "test", "parameters": {"type": "object"}}
                }] if test_name == "tools" else None
            )
            results[test_name] = "✅ Supported"
        except Exception as e:
            results[test_name] = f"❌ Failed: {str(e)[:50]}"
    
    return results

# Test different providers
providers = ["gpt-4", "anthropic/claude-3-sonnet", "vertex_ai/gemini-pro"]
for provider in providers:
    print(f"\n{provider}:")
    compatibility = test_provider_compatibility(provider, test_messages)
    for test, result in compatibility.items():
        print(f"  {test}: {result}")
```

### **Best Practices for Provider Compatibility**

```python
def universal_completion(model, user_message, system_prompt=None):
    """Create provider-compatible completion calls"""
    
    # Build messages based on provider capabilities
    messages = []
    
    # Handle system message
    if system_prompt:
        # Check if provider supports system messages
        try:
            model_info = get_model_info(model)
            if model_info.get("supports_system_message", True):
                messages.append({"role": "system", "content": system_prompt})
            else:
                # Prepend to user message for unsupported providers
                user_message = f"{system_prompt}\n\nUser: {user_message}"
        except:
            # Default to including system message
            messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": user_message})
    
    return completion(model=model, messages=messages)

# Usage
response = universal_completion(
    model="any-provider/model",
    user_message="Hello!",
    system_prompt="You are a helpful assistant"
)
```

**Special Assistant Features:**

```python
# Assistant with tool calls
{
    "role": "assistant",
    "content": "Let me check the weather for you.",
    "tool_calls": [
        {
            "id": "call_123",
            "type": "function", 
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Boston"}'
            }
        }
    ]
}

# Assistant prefill (Anthropic only)
{
    "role": "assistant", 
    "content": "{",  # Start JSON response
    "prefix": True  # LiteLLM-specific flag
}
```

### Response Structure

All providers return a consistent response format:

```python
{
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-4",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?",
                "tool_calls": None,          # Function calls if any
                "function_call": None        # Legacy function call
            },
            "finish_reason": "stop"          # "stop", "length", "tool_calls"
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    }
}
```

## Multi-Turn Conversations

### 1. Basic Multi-Turn Chat

```python
from litellm import completion

def chat_conversation():
    """Interactive multi-turn conversation"""
    conversation_history = []
    
    print("AI Assistant: Hello! How can I help you today?")
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("AI Assistant: Goodbye!")
            break
        
        # Add user message to history
        conversation_history.append({
            "role": "user", 
            "content": user_input
        })
        
        # Get AI response
        response = completion(
            model="gpt-4",
            messages=conversation_history,
            temperature=0.7,
            max_tokens=150
        )
        
        ai_response = response.choices[0].message.content
        
        # Add AI response to history
        conversation_history.append({
            "role": "assistant",
            "content": ai_response
        })
        
        print(f"AI Assistant: {ai_response}")

# Run the conversation
chat_conversation()
```

### 2. Conversation with System Context

```python
def chat_with_context(system_prompt, model="gpt-4"):
    """Chat with persistent system context"""
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    print(f"Starting conversation with {model}")
    print("System context:", system_prompt)
    print("-" * 50)
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == 'quit':
            break
        
        # Add user message
        messages.append({"role": "user", "content": user_input})
        
        # Get response
        response = completion(
            model=model,
            messages=messages,
            temperature=0.8
        )
        
        ai_response = response.choices[0].message.content
        messages.append({"role": "assistant", "content": ai_response})
        
        print(f"Assistant: {ai_response}")
        
        # Optional: Show token usage
        if response.usage:
            print(f"[Tokens used: {response.usage.total_tokens}]")

# Example usage
chat_with_context(
    "You are a Python programming tutor. Always provide practical examples and explain concepts clearly.",
    model="gpt-4"
)
```

### 3. Cross-Provider Conversation

```python
def multi_provider_conversation():
    """Conversation that switches between providers"""
    providers = [
        "gpt-4",
        "anthropic/claude-3-sonnet-20240229", 
        "vertex_ai/gemini-1.5-pro"
    ]
    current_provider = 0
    
    messages = []
    
    print("Multi-provider conversation started!")
    print("Commands: 'switch' to change provider, 'quit' to exit")
    
    while True:
        current_model = providers[current_provider]
        print(f"\nCurrent provider: {current_model}")
        
        user_input = input("You: ")
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'switch':
            current_provider = (current_provider + 1) % len(providers)
            print(f"Switched to: {providers[current_provider]}")
            continue
        
        messages.append({"role": "user", "content": user_input})
        
        try:
            response = completion(
                model=current_model,
                messages=messages,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            messages.append({"role": "assistant", "content": ai_response})
            
            print(f"Assistant ({current_model}): {ai_response}")
            
        except Exception as e:
            print(f"Error with {current_model}: {e}")
            print("Trying next provider...")
            current_provider = (current_provider + 1) % len(providers)

multi_provider_conversation()
```

### 4. Conversation with Function Calling

```python
import json
from datetime import datetime

def get_current_time():
    """Get the current time"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_weather(location):
    """Mock weather function"""
    return f"The weather in {location} is sunny, 72°F"

def conversation_with_tools():
    """Multi-turn conversation with function calling"""
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current date and time",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function", 
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    available_functions = {
        "get_current_time": get_current_time,
        "get_weather": get_weather,
    }
    
    messages = [
        {
            "role": "system", 
            "content": "You are a helpful assistant with access to current time and weather information."
        }
    ]
    
    print("Assistant with tools ready! Ask about time or weather.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
        
        messages.append({"role": "user", "content": user_input})
        
        # Get initial response
        response = completion(
            model="gpt-4",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        messages.append(response_message)
        
        # Check if model wants to call functions
        if response_message.tool_calls:
            print("Assistant is using tools...")
            
            # Execute each tool call
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Call the function
                function_response = available_functions[function_name](**function_args)
                
                # Add function response to messages
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": str(function_response)
                })
            
            # Get final response with function results
            final_response = completion(
                model="gpt-4",
                messages=messages
            )
            
            final_message = final_response.choices[0].message.content
            messages.append({"role": "assistant", "content": final_message})
            print(f"Assistant: {final_message}")
            
        else:
            # Regular response without function calls
            print(f"Assistant: {response_message.content}")

conversation_with_tools()
```

### 5. Conversation State Management

```python
class ConversationManager:
    """Manage conversation state across multiple sessions"""
    
    def __init__(self, model="gpt-4", max_history=20):
        self.model = model
        self.max_history = max_history
        self.conversations = {}  # session_id -> messages
    
    def start_session(self, session_id, system_prompt=None):
        """Start a new conversation session"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        self.conversations[session_id] = {
            "messages": messages,
            "created_at": datetime.now(),
            "total_tokens": 0
        }
        return session_id
    
    def send_message(self, session_id, user_message, **kwargs):
        """Send a message in a session"""
        if session_id not in self.conversations:
            raise ValueError(f"Session {session_id} not found")
        
        conversation = self.conversations[session_id]
        messages = conversation["messages"]
        
        # Add user message
        messages.append({"role": "user", "content": user_message})
        
        # Trim conversation if too long
        if len(messages) > self.max_history:
            # Keep system message if present
            system_msgs = [msg for msg in messages if msg["role"] == "system"]
            other_msgs = [msg for msg in messages if msg["role"] != "system"]
            messages = system_msgs + other_msgs[-self.max_history:]
            conversation["messages"] = messages
        
        # Get AI response
        response = completion(
            model=self.model,
            messages=messages,
            **kwargs
        )
        
        ai_message = response.choices[0].message.content
        messages.append({"role": "assistant", "content": ai_message})
        
        # Update token count
        if response.usage:
            conversation["total_tokens"] += response.usage.total_tokens
        
        return ai_message
    
    def get_conversation_summary(self, session_id):
        """Get conversation statistics"""
        if session_id not in self.conversations:
            return None
        
        conversation = self.conversations[session_id]
        return {
            "message_count": len(conversation["messages"]),
            "total_tokens": conversation["total_tokens"],
            "created_at": conversation["created_at"],
            "duration": datetime.now() - conversation["created_at"]
        }

# Example usage
manager = ConversationManager(model="gpt-4")

# Start multiple sessions
session1 = manager.start_session("user123", "You are a helpful coding assistant.")
session2 = manager.start_session("user456", "You are a creative writing helper.")

# Have conversations
response1 = manager.send_message(session1, "How do I sort a list in Python?")
print(f"Session 1: {response1}")

response2 = manager.send_message(session2, "Write a short story about a robot.")
print(f"Session 2: {response2}")

# Continue conversations
response1b = manager.send_message(session1, "What about sorting in reverse order?")
print(f"Session 1 (continued): {response1b}")

# Check stats
print("Session 1 stats:", manager.get_conversation_summary(session1))
```

## Practical Examples

### 1. Easy Model Switching

```python
def chat_with_model(model_name, user_message):
    """Chat with any LLM provider using the same function"""
    response = completion(
        model=model_name,
        messages=[{"role": "user", "content": user_message}]
    )
    return response.choices[0].message.content

# Use the same function with different providers
user_msg = "Explain quantum computing in simple terms"

openai_answer = chat_with_model("gpt-4", user_msg)
anthropic_answer = chat_with_model("anthropic/claude-3-sonnet-20240229", user_msg)
google_answer = chat_with_model("vertex_ai/gemini-1.5-pro", user_msg)

print("OpenAI:", openai_answer)
print("Anthropic:", anthropic_answer)
print("Google:", google_answer)
```

### 2. Provider Comparison

```python
def compare_providers(prompt, providers):
    """Compare responses from multiple providers"""
    results = {}
    
    for provider in providers:
        try:
            response = completion(
                model=provider,
                messages=[{"role": "user", "content": prompt}]
            )
            results[provider] = {
                "response": response.choices[0].message.content,
                "tokens": response.usage.total_tokens if response.usage else "N/A"
            }
        except Exception as e:
            results[provider] = {"error": str(e)}
    
    return results

# Compare different providers
providers = [
    "gpt-4",
    "anthropic/claude-3-sonnet-20240229",
    "vertex_ai/gemini-1.5-pro",
    "cohere/command-r-plus"
]

results = compare_providers("Write a haiku about AI", providers)
for provider, result in results.items():
    print(f"\n{provider}:")
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Response: {result['response']}")
        print(f"Tokens: {result['tokens']}")
```

### 3. Automatic Fallback

```python
def resilient_completion(message, providers=None):
    """Try multiple providers until one succeeds"""
    if providers is None:
        providers = [
            "gpt-4",
            "anthropic/claude-3-sonnet-20240229",
            "vertex_ai/gemini-1.5-pro",
            "cohere/command-r-plus"
        ]
    
    for provider in providers:
        try:
            response = completion(
                model=provider,
                messages=[{"role": "user", "content": message}]
            )
            return {
                "provider": provider,
                "response": response.choices[0].message.content,
                "success": True
            }
        except Exception as e:
            print(f"Failed with {provider}: {e}")
            continue
    
    return {"success": False, "error": "All providers failed"}

# This will try providers in order until one succeeds
result = resilient_completion("What is the capital of France?")
print(f"Success: {result['success']}")
if result['success']:
    print(f"Provider: {result['provider']}")
    print(f"Response: {result['response']}")
```

### 4. Streaming Responses

```python
def stream_response(model, message):
    """Stream responses from any provider"""
    response = completion(
        model=model,
        messages=[{"role": "user", "content": message}],
        stream=True
    )
    
    print(f"Streaming from {model}:")
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")

# Stream from different providers
stream_response("gpt-4", "Tell me about the history of AI")
stream_response("anthropic/claude-3-sonnet-20240229", "Tell me about the history of AI")
```

## Configuration & Parameters

### Environment Variables

Set up API keys for the providers you want to use:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Cohere
export COHERE_API_KEY="..."

# Google Cloud
export VERTEXAI_PROJECT="your-project-id"
export VERTEXAI_LOCATION="us-central1"

# Azure
export AZURE_API_KEY="..."
export AZURE_API_BASE="https://your-resource.openai.azure.com"
export AZURE_API_VERSION="2023-12-01-preview"

# Groq
export GROQ_API_KEY="gsk_..."
```

### Common Parameters

These parameters work across all providers:

```python
response = completion(
    model="any-provider/model",
    messages=[{"role": "user", "content": "Hello!"}],
    
    # Standard parameters that work everywhere
    temperature=0.7,        # Controls randomness (0-1)
    max_tokens=1000,        # Maximum response length
    top_p=0.9,             # Nucleus sampling
    stream=False,          # Enable streaming
    stop=["END"],          # Stop sequences
    
    # Provider-specific parameters are also supported
    frequency_penalty=0.1,  # OpenAI specific
    presence_penalty=0.1,   # OpenAI specific
)
```

### Programmatic Configuration

```python
import litellm

# Set API keys programmatically
litellm.openai_key = "sk-..."
litellm.anthropic_key = "sk-ant-..."
litellm.cohere_key = "..."

# Or use the general api_key (tries all providers)
litellm.api_key = "your-key"
```

## Error Handling

```python
from litellm.exceptions import RateLimitError, APIError, AuthenticationError

def safe_completion(model, messages):
    try:
        response = completion(model=model, messages=messages)
        return response.choices[0].message.content
    
    except AuthenticationError:
        return "Error: Invalid API key"
    
    except RateLimitError:
        return "Error: Rate limit exceeded, try again later"
    
    except APIError as e:
        return f"API Error: {str(e)}"
    
    except Exception as e:
        return f"Unexpected error: {str(e)}"

# Safe usage
result = safe_completion("gpt-4", [{"role": "user", "content": "Hello!"}])
print(result)
```

## Cost Tracking

```python
from litellm import completion_cost

response = completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Calculate cost for the response
cost = completion_cost(completion_response=response)
print(f"Cost: ${cost:.6f}")

# Track usage
if response.usage:
    print(f"Prompt tokens: {response.usage.prompt_tokens}")
    print(f"Completion tokens: {response.usage.completion_tokens}")
    print(f"Total tokens: {response.usage.total_tokens}")
```

## Quick Reference: Latest Model Names (2025)

### 🆕 Latest 2025 Models

| Provider | Model Name | Context | Cost/1M | Key Features |
|----------|------------|---------|---------|--------------|
| **OpenAI 2025** | `gpt-4.5-preview-2025-02-27` | 128K | $75/$150 | Vision, function calling, premium |
| | `o3-2025-04-16` | 200K | $2/$8 | Vision, reasoning, function calling |
| | `o3-mini-2025-01-31` | 200K | $1.1/$4.4 | Reasoning, function calling |
| | `o1-pro-2025-03-19` | 200K | $150/$600 | Vision, reasoning, premium |
| **Anthropic 2025** | `anthropic/claude-4-opus-20250514` | 200K | $15/$75 | Vision, reasoning, computer use |
| | `anthropic/claude-4-sonnet-20250514` | 200K | $3/$15 | Vision, reasoning, computer use |
| | `anthropic/claude-3-7-sonnet-20250219` | 200K | $3/$15 | Vision, reasoning, web search |
| **Google 2025** | `vertex_ai/gemini-2.5-pro` | 1M | $1.25/$2.5 | Vision, audio, video, reasoning |
| | `vertex_ai/gemini-2.5-flash` | 1M | $0.30/$2.5 | Vision, audio, multimodal |
| **Cohere 2025** | `cohere/command-a-03-2025` | 256K | $2.5/$10 | Function calling, tool choice |
| **xAI 2025** | `xai/grok-3` | 128K | $5/$15 | Latest Grok, reasoning |
| | `xai/grok-2-vision-1212` | 128K | $2/$10 | Vision capabilities |
| **Fireworks AI** | `fireworks_ai/llama-v3p1-405b-instruct` | 32K | $3/$3 | Ultra-fast inference |
| **Cerebras** | `cerebras/llama3.1-70b` | 8K | $0.6/$0.6 | Fastest inference speeds |

### 📊 Best Current 2024 Models (Still Recommended)

| Provider | Model Name | Context | Cost/1M | Key Features |
|----------|------------|---------|---------|--------------|
| **OpenAI 2024** | `gpt-4o-2024-11-20` | 128K | $2.5/$10 | Latest GPT-4o, vision, functions |
| | `gpt-4o-audio-preview-2024-12-17` | 128K | $2.5/$10 | Audio input/output |
| | `o1-2024-12-17` | 200K | $15/$60 | Latest reasoning model |
| **Anthropic 2024** | `anthropic/claude-3-5-sonnet-20241022` | 200K | $3/$15 | Computer use, web search |
| | `anthropic/claude-3-5-haiku-20241022` | 200K | $0.8/$4 | Fast, vision, cost-effective |
| **Google 2024** | `vertex_ai/gemini-2.0-flash` | 1M | $0.10/$0.40 | Most cost-effective vision |

### 💰 Cost Comparison (Input/Output per 1M tokens)

**Most Affordable:**
1. `vertex_ai/gemini-2.0-flash`: $0.10/$0.40
2. `vertex_ai/gemini-2.5-flash`: $0.30/$2.50  
3. `anthropic/claude-3-5-haiku-20241022`: $0.8/$4.00
4. `o3-mini-2025-01-31`: $1.1/$4.4

**Best Value:**
1. `vertex_ai/gemini-2.5-pro`: $1.25/$2.5 (1M context!)
2. `o3-2025-04-16`: $2/$8 (reasoning)
3. `gpt-4o-2024-11-20`: $2.5/$10
4. `anthropic/claude-4-sonnet-20250514`: $3/$15

**Premium/Specialized:**
1. `o1-pro-2025-03-19`: $150/$600 (highest reasoning)
2. `gpt-4.5-preview-2025-02-27`: $75/$150
3. `anthropic/claude-4-opus-20250514`: $15/$75
4. `xai/grok-3`: $5/$15 (latest Grok reasoning)
5. `fireworks_ai/llama-v3p1-405b-instruct`: $3/$3 (ultra-fast)

## Embedding Models

LiteLLM supports **62+ embedding models** across multiple providers for semantic search, document similarity, and RAG applications.

### Text Embeddings

```python
from litellm import embedding

# OpenAI embeddings
response = embedding(
    model="text-embedding-3-large",  # Latest OpenAI: 3072 dimensions
    input=["Hello world", "How are you?"],
    dimensions=1024  # Optional: reduce dimensions
)

# Access embeddings
embeddings = response.data[0]['embedding']  # First text embedding
print(f"Embedding dimensions: {len(embeddings)}")

# Azure OpenAI embeddings
response = embedding(
    model="azure/text-embedding-ada-002",
    input="Your text here"
)

# Cohere embeddings
response = embedding(
    model="cohere/embed-english-v3.0",
    input="Your text here",
    input_type="search_query"  # or "search_document", "classification"
)

# Voyage AI embeddings
response = embedding(
    model="voyage/voyage-large-2",
    input="Your text here"
)
```

### Provider-Specific Embedding Features

```python
# OpenAI - Dimension reduction
response = embedding(
    model="text-embedding-3-large",
    input="Your text",
    dimensions=512  # Reduce from 3072 to 512 dimensions
)

# Cohere - Input type specification
response = embedding(
    model="cohere/embed-multilingual-v3.0",
    input=["Query text", "Document text"],
    input_type="search_query",  # Optimize for search queries
    encoding_format="float"     # or "base64"
)

# Vertex AI - Multimodal embeddings
response = embedding(
    model="vertex_ai/textembedding-gecko@003",
    input="Your text here",
    vertex_ai_project="your-project-id",
    vertex_ai_location="us-central1"
)
```

### Batch Embedding Processing

```python
def batch_embed_documents(texts, model="text-embedding-3-small", batch_size=100):
    """Process large document collections efficiently"""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        response = embedding(
            model=model,
            input=batch
        )
        
        batch_embeddings = [item['embedding'] for item in response.data]
        all_embeddings.extend(batch_embeddings)
        
        print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} documents")
    
    return all_embeddings

# Process large document collection
documents = ["Document 1 text...", "Document 2 text..."]  # Your documents
embeddings = batch_embed_documents(documents)
```

### Embedding Model Comparison

| Provider | Model | Dimensions | Cost/1M tokens | Best For |
|----------|-------|------------|----------------|----------|
| **OpenAI** | `text-embedding-3-large` | 3072 | $0.13 | Highest accuracy |
| **OpenAI** | `text-embedding-3-small` | 1536 | $0.02 | Cost-effective |
| **Cohere** | `embed-english-v3.0` | 1024 | $0.10 | English text |
| **Cohere** | `embed-multilingual-v3.0` | 1024 | $0.10 | Multilingual |
| **Voyage** | `voyage-large-2` | 1536 | $0.12 | General purpose |
| **Azure** | `text-embedding-ada-002` | 1536 | $0.10 | Enterprise |

## Image Generation

LiteLLM supports **60+ image generation models** including DALL-E, Stable Diffusion, and more.

### DALL-E Models

```python
from litellm import image_generation

# DALL-E 3 (latest)
response = image_generation(
    model="dall-e-3",
    prompt="A futuristic city with flying cars",
    size="1024x1024",  # "1792x1024", "1024x1792"
    quality="hd",      # "standard" or "hd"
    style="vivid",     # "vivid" or "natural"
    n=1                # Number of images
)

# Access generated image
image_url = response.data[0].url
print(f"Generated image: {image_url}")

# DALL-E 2 (cost-effective)
response = image_generation(
    model="dall-e-2",
    prompt="A sunset over mountains",
    size="512x512",    # "256x256", "512x512", "1024x1024"
    n=2                # Generate multiple variations
)
```

### Stable Diffusion Models

```python
# Stability AI models via Bedrock
response = image_generation(
    model="bedrock/stability.stable-diffusion-xl-v1",
    prompt="A serene lake surrounded by mountains",
    width=1024,
    height=1024,
    steps=50,              # Inference steps
    cfg_scale=7.0,         # Prompt adherence
    aws_access_key_id="your-key",
    aws_secret_access_key="your-secret",
    aws_region_name="us-east-1"
)
```

### Azure Image Generation

```python
# Azure DALL-E
response = image_generation(
    model="azure/dall-e-3",
    prompt="Abstract art with vibrant colors",
    size="1024x1024",
    api_key="your-azure-key",
    api_base="https://your-resource.openai.azure.com",
    api_version="2024-02-01"
)
```

### Image Generation with Custom Parameters

```python
def generate_image_variations(prompt, styles=["vivid", "natural"]):
    """Generate multiple style variations of the same prompt"""
    results = []
    
    for style in styles:
        response = image_generation(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="hd",
            style=style,
            n=1
        )
        
        results.append({
            "style": style,
            "url": response.data[0].url,
            "revised_prompt": response.data[0].revised_prompt  # DALL-E's interpretation
        })
    
    return results

# Generate variations
variations = generate_image_variations("A cyberpunk cityscape at night")
for var in variations:
    print(f"Style: {var['style']} - URL: {var['url']}")
```

### Image Generation Pricing

| Model | Size | Cost per Image | Quality |
|-------|------|----------------|---------|
| **DALL-E 3 HD** | 1024×1024 | $0.040 | Highest |
| **DALL-E 3 Standard** | 1024×1024 | $0.020 | High |
| **DALL-E 2** | 1024×1024 | $0.020 | Good |
| **DALL-E 2** | 512×512 | $0.018 | Good |
| **DALL-E 2** | 256×256 | $0.016 | Basic |

## Computer Use

LiteLLM supports **36 models** with computer use capabilities, primarily Claude models that can interact with computer interfaces.

### Claude Computer Use

```python
from litellm import completion

# Latest Claude models with computer use
response = completion(
    model="anthropic/claude-3-5-sonnet-20241022",  # Computer use enabled
    messages=[
        {
            "role": "user",
            "content": "Take a screenshot and describe what you see on the screen"
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

# Check if Claude wants to use computer
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        print(f"Computer action: {tool_call.function.name}")
        print(f"Parameters: {tool_call.function.arguments}")
```

### Computer Use Actions

Claude can perform various computer actions:

```python
# Available computer actions
computer_tools = [
    {
        "type": "computer_20241022",
        "name": "computer",
        "display_width_px": 1920,
        "display_height_px": 1080,
        "display_number": 1
    }
]

# Claude can:
# - Take screenshots
# - Click on elements
# - Type text
# - Scroll pages
# - Navigate applications
# - Read screen content

response = completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[
        {
            "role": "user",
            "content": "Please open a web browser and search for 'LiteLLM documentation'"
        }
    ],
    tools=computer_tools,
    tool_choice="auto"
)
```

### Computer Use Models

| Model | Provider | Computer Use | Context | Notes |
|-------|----------|--------------|---------|-------|
| **Claude 3.5 Sonnet** | Anthropic | ✅ | 200K | Best computer use |
| **Claude 4 Sonnet** | Anthropic | ✅ | 200K | Latest 2025 |
| **Claude 4 Opus** | Anthropic | ✅ | 200K | Most capable |
| **Claude 3.7 Sonnet** | Anthropic | ✅ | 200K | With reasoning |

### Computer Use Safety

```python
# Always implement safety measures
def safe_computer_use(prompt, model="anthropic/claude-3-5-sonnet-20241022"):
    """Computer use with safety constraints"""
    
    safety_prompt = f"""
    {prompt}
    
    SAFETY CONSTRAINTS:
    - Only interact with approved applications
    - Do not access sensitive files or data
    - Confirm destructive actions before proceeding
    - Respect privacy and security boundaries
    """
    
    response = completion(
        model=model,
        messages=[{"role": "user", "content": safety_prompt}],
        tools=[{
            "type": "computer_20241022",
            "name": "computer",
            "display_width_px": 1920,
            "display_height_px": 1080
        }],
        tool_choice="auto"
    )
    
    return response

# Safe computer use
result = safe_computer_use("Help me organize my desktop files")
```

## Reranking Models

LiteLLM supports reranking models for improving search result relevance:

```python
from litellm import rerank

# Cohere reranking
response = rerank(
    model="cohere/rerank-english-v3.0",
    query="What is machine learning?",
    documents=[
        "Machine learning is a subset of AI",
        "Python is a programming language", 
        "Deep learning uses neural networks",
        "Statistics is used in data analysis"
    ],
    top_n=2  # Return top 2 most relevant
)

# Access reranked results
for result in response.results:
    print(f"Rank: {result.index}, Score: {result.relevance_score}")
    print(f"Document: {result.document.text}\n")

# Amazon Titan reranking (via Bedrock)
response = rerank(
    model="bedrock/amazon.rerank-v1:0",
    query="Python programming tutorials",
    documents=[...],
    aws_access_key_id="your-key",
    aws_secret_access_key="your-secret",
    aws_region_name="us-east-1"
)
```

## Advanced Features

### 1. Streaming Responses

```python
from litellm import completion

def stream_chat(model, message):
    """Stream responses in real-time"""
    response = completion(
        model=model,
        messages=[{"role": "user", "content": message}],
        stream=True,
        stream_options={"include_usage": True}  # Include token usage in stream
    )
    
    print(f"Streaming from {model}:")
    full_response = ""
    
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content
        
        # Check for usage information in the final chunk
        if hasattr(chunk, 'usage') and chunk.usage:
            print(f"\n[Tokens used: {chunk.usage.total_tokens}]")
    
    print("\n")
    return full_response

# Try streaming with different providers
stream_chat("gpt-4", "Explain quantum computing in simple terms")
stream_chat("anthropic/claude-3-sonnet-20240229", "Write a haiku about AI")
```

### 2. JSON Mode

```python
def get_structured_response(model, prompt):
    """Get structured JSON responses"""
    response = completion(
        model=model,
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful assistant that responds with valid JSON."
            },
            {
                "role": "user", 
                "content": f"{prompt}. Respond with JSON only."
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.1
    )
    
    import json
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response", "content": response.choices[0].message.content}

# Example usage
result = get_structured_response(
    "gpt-4", 
    "List 3 benefits of renewable energy with explanations"
)
print(json.dumps(result, indent=2))
```

### 3. Batch Processing

```python
from litellm import batch_completion

def process_multiple_prompts(model, prompts):
    """Process multiple prompts in batch"""
    messages_list = [
        [{"role": "user", "content": prompt}] 
        for prompt in prompts
    ]
    
    responses = batch_completion(
        model=model,
        messages=messages_list,
        temperature=0.7,
        max_tokens=100
    )
    
    results = []
    for i, response in enumerate(responses):
        results.append({
            "prompt": prompts[i],
            "response": response.choices[0].message.content,
            "tokens": response.usage.total_tokens if response.usage else "N/A"
        })
    
    return results

# Example usage
prompts = [
    "What is the capital of France?",
    "Explain photosynthesis briefly",
    "Write a short joke"
]

results = process_multiple_prompts("gpt-3.5-turbo", prompts)
for result in results:
    print(f"Q: {result['prompt']}")
    print(f"A: {result['response']}")
    print(f"Tokens: {result['tokens']}\n")
```

### 4. Web Search Integration

```python
def web_enhanced_chat(query):
    """Chat with web search capabilities"""
    response = completion(
        model="openai/gpt-4o-search-preview",  # Web search enabled model
        messages=[
            {
                "role": "user",
                "content": query
            }
        ],
        web_search_options={
            "search_context_size": "medium"  # "low", "medium", "high"
        }
    )
    
    return response.choices[0].message.content

# Example usage
answer = web_enhanced_chat("What are the latest developments in AI in 2024?")
print(f"Web-enhanced answer: {answer}")
```

### 5. Vision and Multimodal Support

## Vision/Image Support Across Providers

LiteLLM provides **comprehensive vision capabilities** across multiple providers. Based on the latest model database (1,245+ models with 292 vision-enabled), here's the **2025 vision support status**:

### ✅ Latest Vision Models by Provider (2025/2024)

| Provider | Latest Models | Context | Cost/1K Input | Key Features |
|----------|---------------|---------|---------------|--------------|
| **OpenAI** | gpt-4.1-turbo, gpt-4.1-nano, o3-pro (2025) | 1M-1.05M | $0.1-$20 | Vision, reasoning, function calling |
| **Anthropic** | claude-4-sonnet, claude-4-opus, claude-3.7-sonnet (2025) | 200K | $3-$15 | Computer use, vision, reasoning |
| **Google** | gemini-2.5-pro, gemini-2.0-flash (2025) | 1M-2M | $0.075-$1.25 | Multimodal, audio/video, function calling |
| **Meta/Vertex** | llama-4-scout-17b (2025) | **10M** | $1.5 | **Largest context window**, vision |
| **Amazon** | nova-pro (2025) | 300K | $0.8 | Enterprise multimodal, PDF support |
| **Azure OpenAI** | gpt-4.1 variants, o3-pro (2025) | 1M | $0.1-$20 | Same as OpenAI + regional pricing |
| **Groq** | llama-3.2-11b-vision-preview | 8K | $0.18 | **Fastest inference**, function calling |
| **Lambda** | llama3.2-11b-vision | 131K | $0.015 | **Most cost-effective vision** |
| **NSScale** | qwen2.5-coder-3b-instruct | 32K | $0.01 | Coding specialist with vision |

### 🆕 2025 Model Highlights

**🚀 OpenAI 2025 Models:**
- **GPT-4.1-Turbo**: 1.05M context, $2.5/1K tokens, vision + reasoning
- **GPT-4.1-Nano**: 1M context, $0.1/1K tokens, most cost-effective vision
- **O3-Pro**: Advanced reasoning, $20/1K tokens, problem-solving specialist

**🧠 Anthropic 2025 Models:**
- **Claude-4-Opus**: $15/1K tokens, computer use, enhanced vision 
- **Claude-4-Sonnet**: $3/1K tokens, computer use, vision capabilities
- **Claude-3.7-Sonnet**: $3/1K tokens, web search, thinking capabilities

**💎 Google 2025 Models:**
- **Gemini-2.5-Pro**: Up to 2M context, audio/video processing
- **Gemini-2.0-Flash**: Cost-effective at $0.075/1K tokens, 1M context

**🔥 New Standout Models:**
- **Llama-4-Scout-17B**: **10M context window** (largest available), $1.5/1K
- **Amazon Nova-Pro**: Enterprise multimodal, 300K context, $0.8/1K

### 📊 Cost Comparison (Input Cost per 1K Tokens)

**Most Affordable Vision Models:**
1. **NSScale Qwen2.5-Coder-3B**: $0.01
2. **Lambda Llama3.2-11B-Vision**: $0.015
3. **Gemini 2.0 Flash** (Google): $0.075
4. **GPT-4.1-Nano** (OpenAI 2025): $0.1  
5. **Groq Llama-3.2-11B-Vision**: $0.18

**Premium Vision Models:**
1. **O3-Pro** (OpenAI): $20/1K tokens
2. **Claude-4-Opus**: $15/1K tokens
3. **GPT-4.1-Turbo**: $2.5/1K tokens
4. **Claude-4-Sonnet**: $3/1K tokens

### 🎯 Provider-Specific Strengths

**OpenAI**: GPT-4.1 series, O3-Pro reasoning, vision + function calling
**Anthropic**: Claude-4 with computer use, enhanced reasoning, safety
**Google**: Gemini-2.5 with audio/video, 2M context, cost-effective  
**Meta/Vertex**: Llama-4-Scout with 10M context (largest available)
**Amazon**: Nova-Pro enterprise features, PDF support
**Lambda**: Most cost-effective vision models ($0.015/1K)
**NSScale**: Ultra-low-cost coding models with vision ($0.01/1K)
**Groq**: Fastest inference speeds for vision models

## Vision Examples

### 1. Basic Image Analysis

```python
import base64
from litellm import completion

def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image(model, image_path, question):
    """Analyze images with vision models"""
    base64_image = encode_image(image_path)
    
    response = completion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=300
    )
    
    return response.choices[0].message.content

# Try with different latest providers (2025 models)
providers = [
    "gpt-4.1-turbo",                             # OpenAI 2025 - 1.05M context
    "gpt-4.1-nano",                              # OpenAI 2025 - Most cost-effective vision  
    "anthropic/claude-4-sonnet-20250514",       # Anthropic 2025 - Computer use
    "vertex_ai/gemini-2.5-pro",                 # Google 2025 - Audio/video support
    "vertex_ai/llama-4-scout-17b",              # Meta 2025 - 10M context
    "amazon-nova/nova-pro",                      # Amazon 2025 - Enterprise
    "lambda/llama3.2-11b-vision",               # Lambda - Most affordable ($0.015/1K)
    "nscale/qwen2.5-coder-3b-instruct"          # NSScale - Ultra-low-cost ($0.01/1K)
]

for provider in providers:
    try:
        result = analyze_image(provider, "image.jpg", "What's in this image?")
        print(f"✅ {provider}: {result}\n")
    except Exception as e:
        print(f"❌ {provider} failed: {e}\n")
```

### 2. Image from URL

```python
def analyze_image_url(model, image_url, question):
    """Analyze image from URL"""
    response = completion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ]
            }
        ]
    )
    return response.choices[0].message.content

# Example with public image URL
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
result = analyze_image_url("gpt-4o", image_url, "Describe this landscape")
print(result)
```

### 3. Multi-Image Analysis

```python
def analyze_multiple_images(model, image_paths, question):
    """Analyze multiple images at once"""
    content = [{"type": "text", "text": question}]
    
    # Add each image to content
    for image_path in image_paths:
        base64_image = encode_image(image_path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    
    response = completion(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=500
    )
    
    return response.choices[0].message.content

# Compare multiple images
images = ["image1.jpg", "image2.jpg", "image3.jpg"]
result = analyze_multiple_images(
    "gpt-4o", 
    images, 
    "Compare these images and describe the differences"
)
print(result)
```

### 4. Vision in Conversations

```python
def vision_conversation():
    """Multi-turn conversation with images"""
    messages = []
    
    while True:
        user_input = input("\nEnter text (or 'image:path' to add image, 'quit' to exit): ")
        
        if user_input.lower() == 'quit':
            break
        
        if user_input.startswith('image:'):
            # Add image to conversation
            image_path = user_input[6:]  # Remove 'image:' prefix
            base64_image = encode_image(image_path)
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please analyze this image"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            })
        else:
            # Add text message
            messages.append({
                "role": "user",
                "content": user_input
            })
        
        # Get AI response
        response = completion(
            model="gpt-4o",
            messages=messages,
            max_tokens=300
        )
        
        ai_response = response.choices[0].message.content
        messages.append({
            "role": "assistant",
            "content": ai_response
        })
        
        print(f"AI: {ai_response}")

# Run vision conversation
# vision_conversation()
```

### 5. Provider-Specific Vision Features

```python
def provider_specific_vision():
    """Demonstrate provider-specific vision capabilities"""
    
    # OpenAI - Detailed image analysis
    openai_response = completion(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image in detail, including colors, objects, composition, and mood"},
                {"type": "image_url", "image_url": {"url": "path/to/image.jpg"}}
            ]
        }],
        max_tokens=500
    )
    
    # Anthropic - Image with specific format request
    anthropic_response = completion(
        model="anthropic/claude-3-sonnet-20240229",
        messages=[{
            "role": "user", 
            "content": [
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,...",
                        "format": "image/jpeg"  # Anthropic supports format specification
                    }
                }
            ]
        }]
    )
    
    # Google Vertex AI - Gemini with image
    vertex_response = completion(
        model="vertex_ai/gemini-1.5-pro",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "gs://bucket/image.jpg"}}  # GCS URL support
            ]
        }]
    )
    
    return {
        "openai": openai_response.choices[0].message.content,
        "anthropic": anthropic_response.choices[0].message.content, 
        "vertex": vertex_response.choices[0].message.content
    }
```

### 6. Image Format Support

```python
def test_image_formats():
    """Test different image formats"""
    
    # Supported formats
    formats = {
        "JPEG": "data:image/jpeg;base64,",
        "PNG": "data:image/png;base64,", 
        "GIF": "data:image/gif;base64,",
        "WebP": "data:image/webp;base64,",
        "SVG": "data:image/svg+xml;base64,"
    }
    
    for format_name, prefix in formats.items():
        try:
            # Test with encoded image
            response = completion(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"What format is this {format_name} image?"},
                        {"type": "image_url", "image_url": {"url": prefix + "..."}}
                    ]
                }]
            )
            print(f"{format_name}: Supported ✅")
        except Exception as e:
            print(f"{format_name}: Error - {e}")
```

### 7. Vision with Tools/Functions

```python
import json

def image_analysis_with_tools():
    """Combine vision with function calling"""
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "save_image_analysis",
                "description": "Save image analysis results",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "objects": {"type": "array", "items": {"type": "string"}},
                        "colors": {"type": "array", "items": {"type": "string"}},
                        "mood": {"type": "string"},
                        "description": {"type": "string"}
                    },
                    "required": ["objects", "description"]
                }
            }
        }
    ]
    
    response = completion(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "Analyze this image and save the results using the save_image_analysis function"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,..."}
                }
            ]
        }],
        tools=tools,
        tool_choice="auto"
    )
    
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        analysis_data = json.loads(tool_call.function.arguments)
        print("Image Analysis Results:", analysis_data)
        
        # Continue conversation with tool result
        # ... (tool execution logic)
```

## Multimodal Embeddings

Some providers also support multimodal embeddings:

```python
from litellm import embedding

# Vertex AI multimodal embeddings
response = embedding(
    model="vertex_ai/multimodalembedding@001",
    input=[
        "A description of the image",
        "data:image/jpeg;base64,..."  # Base64 image
    ]
)

print("Text embedding:", response.data[0]['embedding'])
print("Image embedding:", response.data[1]['embedding'])
```

## Best Practices for Vision

### 1. **Image Size Optimization**
```python
def optimize_image_for_vision(image_path, max_size=(1024, 1024)):
    """Resize image for optimal vision processing"""
    from PIL import Image
    
    with Image.open(image_path) as img:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save optimized version
        optimized_path = f"optimized_{image_path}"
        img.save(optimized_path, "JPEG", quality=85)
        return optimized_path
```

### 2. **Provider Selection for Vision**
```python
def select_vision_provider(task_type):
    """Select best provider for vision task"""
    providers = {
        "detailed_analysis": "gpt-4o",  # Best for detailed descriptions
        "creative": "anthropic/claude-3-sonnet-20240229",  # Good for creative analysis
        "technical": "vertex_ai/gemini-1.5-pro",  # Good for technical details
        "fast": "gpt-4o",  # Fastest response
        "cost_effective": "vertex_ai/gemini-1.5-flash"  # Most cost-effective
    }
    return providers.get(task_type, "gpt-4o")
```

### 3. **Error Handling for Vision**
```python
def robust_vision_analysis(image_path, question, fallback_providers=None):
    """Vision analysis with fallback providers"""
    if fallback_providers is None:
        fallback_providers = [
            "gpt-4o",
            "anthropic/claude-3-sonnet-20240229",
            "vertex_ai/gemini-1.5-pro"
        ]
    
    base64_image = encode_image(image_path)
    
    for provider in fallback_providers:
        try:
            response = completion(
                model=provider,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }],
                max_tokens=300
            )
            
            return {
                "provider": provider,
                "result": response.choices[0].message.content,
                "success": True
            }
            
        except Exception as e:
            print(f"Provider {provider} failed: {e}")
            continue
    
    return {"success": False, "error": "All vision providers failed"}
```

## Summary

**LiteLLM provides excellent vision support** across major providers:
- ✅ **Universal API**: Same code works across OpenAI, Anthropic, Google, Azure
- ✅ **Multiple Input Methods**: Base64, URLs, file paths
- ✅ **Rich Conversations**: Mix text and images in multi-turn chats
- ✅ **Advanced Features**: Multi-image analysis, tool integration
- ✅ **Provider Flexibility**: Easy fallback between providers

**Not all providers support vision**, but LiteLLM makes it easy to work with those that do using the same OpenAI-compatible interface!

### 6. Reproducible Outputs

```python
def reproducible_generation(prompt, model="gpt-4"):
    """Generate reproducible responses using seed"""
    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        seed=42,  # Fixed seed for reproducibility
        temperature=0.7,
        max_tokens=100
    )
    
    return {
        "response": response.choices[0].message.content,
        "seed": 42,
        "model": model,
        "fingerprint": getattr(response, 'system_fingerprint', 'N/A')
    }

# Generate the same response multiple times
for i in range(3):
    result = reproducible_generation("Write a creative opening line for a story")
    print(f"Attempt {i+1}: {result['response']}")
```

### 7. Token Usage Monitoring

```python
from litellm import completion_cost

def monitored_completion(model, messages, **kwargs):
    """Completion with detailed monitoring"""
    response = completion(
        model=model,
        messages=messages,
        **kwargs
    )
    
    # Calculate costs
    cost = completion_cost(completion_response=response)
    
    # Compile monitoring data
    monitoring_data = {
        "response": response.choices[0].message.content,
        "model": response.model,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0
        },
        "cost": cost,
        "finish_reason": response.choices[0].finish_reason
    }
    
    return monitoring_data

# Example usage
result = monitored_completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Explain machine learning"}],
    max_tokens=200
)

print(f"Response: {result['response']}")
print(f"Cost: ${result['cost']:.6f}")
print(f"Tokens: {result['usage']['total_tokens']}")
```

## Prompt Caching

LiteLLM supports provider-specific prompt caching to reduce costs and improve response times. **Caching behavior varies by provider** - some have implicit caching, while others require explicit configuration.

### Implicit vs Explicit Caching

**✅ Implicit Caching (Automatic)**
- **OpenAI**: Automatic caching of recent requests (no configuration needed)
- **Deepseek**: Built-in conversation context caching
- Most providers have some form of internal caching

**⚙️ Explicit Caching (Requires Configuration)**
- **Anthropic**: Supports ephemeral prompt caching with `cache_control`
- **Google Gemini/Vertex AI**: Context caching with `cache_control`
- **Custom providers**: Varies by implementation

### Does LiteLLM Use Provider Caching by Default?

**Yes, LiteLLM automatically utilizes provider caching when available:**

```python
from litellm import completion

# This automatically benefits from provider-specific caching
# No special configuration needed for implicit caching
response = completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)

# For providers with implicit caching, repeated identical requests
# will be served from cache automatically
response2 = completion(
    model="gpt-4", 
    messages=[{"role": "user", "content": "Hello!"}]  # Same request = cached
)
```

### Explicit Prompt Caching (Anthropic)

For providers that support explicit caching, use the `cache_control` parameter:

```python
def anthropic_caching_example():
    """Anthropic explicit prompt caching"""
    
    # Cache system context (long instructions, documents, etc.)
    response = completion(
        model="anthropic/claude-3-5-sonnet-20240620",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an AI assistant tasked with analyzing legal documents."
                    },
                    {
                        "type": "text",
                        "text": "Here is the full text of a complex legal agreement: " + 
                               "Very long document content here..." * 400,
                        "cache_control": {"type": "ephemeral"}  # Cache this expensive content
                    }
                ]
            },
            {
                "role": "user",
                "content": "What are the key terms and conditions?"
            }
        ]
    )
    
    # Check cache usage in response
    if response.usage:
        print(f"Total tokens: {response.usage.total_tokens}")
        if hasattr(response.usage, 'prompt_tokens_details'):
            cached_tokens = response.usage.prompt_tokens_details.get('cached_tokens', 0)
            print(f"Cached tokens: {cached_tokens}")
    
    return response

# Run example
result = anthropic_caching_example()
```

### Multi-Turn Conversation Caching

Cache conversation context across multiple turns:

```python
def multi_turn_caching():
    """Cache conversation context across turns"""
    
    # Initial conversation with cached context
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant with access to this large knowledge base: " +
                           "Large knowledge base content..." * 300,
                    "cache_control": {"type": "ephemeral"}  # Cache knowledge base
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is the main topic?",
                    "cache_control": {"type": "ephemeral"}  # Cache this turn for follow-ups
                }
            ]
        }
    ]
    
    response1 = completion(
        model="anthropic/claude-3-5-sonnet-20240620",
        messages=messages
    )
    
    # Add response to conversation
    messages.append({
        "role": "assistant", 
        "content": response1.choices[0].message.content
    })
    
    # Follow-up question (benefits from cached context)
    messages.append({
        "role": "user",
        "content": "Can you elaborate on that?"
    })
    
    response2 = completion(
        model="anthropic/claude-3-5-sonnet-20240620",
        messages=messages  # Previous context is cached
    )
    
    return response1, response2

# Subsequent calls benefit from caching
response1, response2 = multi_turn_caching()
```

### Tool Definition Caching

Cache expensive tool definitions:

```python
def cached_tool_definitions():
    """Cache tool definitions for repeated use"""
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "analyze_document",
                "description": "Analyze legal documents with extensive context and examples...",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "document_type": {"type": "string"},
                        # ... complex schema
                    }
                },
                "cache_control": {"type": "ephemeral"}  # Cache tool definition
            }
        }
    ]
    
    response = completion(
        model="anthropic/claude-3-5-sonnet-20240620",
        messages=[{"role": "user", "content": "Analyze this contract"}],
        tools=tools,
        tool_choice="auto"
    )
    
    return response

result = cached_tool_definitions()
```

### Gemini/Vertex AI Context Caching

Google models also support caching:

```python
def gemini_caching_example():
    """Gemini context caching"""
    
    response = completion(
        model="vertex_ai/gemini-1.5-pro",
        messages=[
            {
                "role": "system",
                "content": "Large system context...",
                "cache_control": {"type": "ephemeral"}  # Cache for Gemini
            },
            {
                "role": "user",
                "content": "What's the summary?"
            }
        ],
        vertex_ai_project="your-project-id",
        vertex_ai_location="us-central1"
    )
    
    return response

result = gemini_caching_example()
```

### Monitoring Cache Usage

Track cache hits and costs:

```python
def monitor_cache_usage(model, messages):
    """Monitor caching effectiveness"""
    
    response = completion(model=model, messages=messages)
    
    cache_info = {
        "total_tokens": 0,
        "cached_tokens": 0,
        "cache_hit_rate": 0,
        "cost_savings": 0
    }
    
    if response.usage:
        cache_info["total_tokens"] = response.usage.total_tokens
        
        # Check for cached tokens (Anthropic format)
        if hasattr(response.usage, 'prompt_tokens_details'):
            details = response.usage.prompt_tokens_details
            if isinstance(details, dict) and 'cached_tokens' in details:
                cache_info["cached_tokens"] = details['cached_tokens']
                cache_info["cache_hit_rate"] = (
                    cache_info["cached_tokens"] / response.usage.prompt_tokens * 100
                    if response.usage.prompt_tokens > 0 else 0
                )
        
        # Estimate cost savings (cached tokens typically cost 75% less)
        if cache_info["cached_tokens"] > 0:
            cache_info["cost_savings"] = cache_info["cached_tokens"] * 0.75
    
    return response, cache_info

# Example usage
response, cache_stats = monitor_cache_usage(
    "anthropic/claude-3-5-sonnet-20240620",
    [{"role": "user", "content": "Hello!"}]
)

print(f"Cache hit rate: {cache_stats['cache_hit_rate']:.1f}%")
print(f"Tokens saved: {cache_stats['cached_tokens']}")
```

### Best Practices for Caching

**1. Cache Long, Stable Content**
```python
# ✅ Good for caching
system_prompt = {
    "role": "system",
    "content": [
        {
            "type": "text",
            "text": "Long, stable instructions or context documents...",
            "cache_control": {"type": "ephemeral"}
        }
    ]
}
```

**2. Cache Tool Definitions**
```python
# ✅ Cache expensive tool schemas
complex_tool = {
    "type": "function",
    "function": {
        "name": "complex_analysis",
        "description": "Extensive tool description...",
        "cache_control": {"type": "ephemeral"}  # Cache the tool definition
    }
}
```

**3. Strategic Cache Points in Conversations**
```python
# ✅ Cache at strategic conversation points
messages = [
    # Cache initial context
    {"role": "system", "content": [...], "cache_control": {"type": "ephemeral"}},
    
    # Regular conversation
    {"role": "user", "content": "Question 1"},
    {"role": "assistant", "content": "Answer 1"},
    
    # Cache before expensive operations
    {
        "role": "user", 
        "content": [
            {
                "type": "text",
                "text": "Now analyze this complex data...",
                "cache_control": {"type": "ephemeral"}  # Cache before analysis
            }
        ]
    }
]
```

**4. Don't Cache Dynamic Content**
```python
# ❌ Don't cache frequently changing content
dynamic_content = {
    "role": "user",
    "content": f"Current time: {datetime.now()}",  # Changes every call
    # Don't add cache_control here
}
```

### Provider Caching Summary

| Provider | Caching Type | Configuration Required | Cache Parameter |
|----------|--------------|------------------------|-----------------|
| **OpenAI** | Implicit | None | Automatic |
| **Anthropic** | Explicit | Yes | `cache_control: {"type": "ephemeral"}` |
| **Google/Gemini** | Explicit | Yes | `cache_control: {"type": "ephemeral"}` |
| **Deepseek** | Implicit | None | Automatic |
| **Azure OpenAI** | Implicit | None | Automatic |
| **Cohere** | Implicit | None | Automatic |
| **Other Providers** | Varies | Check docs | Provider-specific |

### Key Takeaways

1. **Automatic Benefits**: LiteLLM automatically uses provider caching when available
2. **No Configuration Needed**: Most providers have implicit caching that works out-of-the-box
3. **Explicit Control**: Use `cache_control` for providers like Anthropic and Gemini
4. **Cost Savings**: Cached tokens typically cost 75% less than non-cached tokens
5. **Performance**: Cached responses are served much faster
6. **Cross-Provider**: Same caching patterns work across different providers through LiteLLM

### Global Caching Configuration

**Q: Can I enable prompt caching for ALL providers at once?**

**A: There's no single global setting, but here are your options:**

#### 1. Automatic for Most Providers

Most providers (OpenAI, Deepseek, Azure, Cohere) have implicit caching that's **always enabled** - no configuration needed:

```python
# These automatically use provider caching:
completion(model="gpt-4", messages=[...])           # ✅ Auto-cached
completion(model="azure/gpt-4", messages=[...])     # ✅ Auto-cached  
completion(model="deepseek/deepseek-chat", messages=[...])  # ✅ Auto-cached
completion(model="cohere/command-r-plus", messages=[...])   # ✅ Auto-cached
```

#### 2. Helper Function for Explicit Caching

Create a wrapper to automatically add cache controls for providers that support it:

```python
def smart_completion(model, messages, **kwargs):
    """Completion with automatic caching for supported providers"""
    
    # Providers that support explicit caching
    explicit_cache_providers = [
        "anthropic/", "claude-", 
        "vertex_ai/", "gemini/", "gemini-"
    ]
    
    # Check if provider supports explicit caching
    supports_explicit_cache = any(
        model.startswith(provider) or provider.replace("/", "") in model 
        for provider in explicit_cache_providers
    )
    
    if supports_explicit_cache:
        # Auto-add cache_control to system messages and long content
        enhanced_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                # Cache system messages automatically
                if isinstance(msg["content"], str):
                    enhanced_msg = {
                        "role": "system",
                        "content": [
                            {
                                "type": "text", 
                                "text": msg["content"],
                                "cache_control": {"type": "ephemeral"}
                            }
                        ]
                    }
                else:
                    # Already structured content - add cache_control to long text
                    enhanced_msg = msg.copy()
                    if isinstance(msg["content"], list):
                        for item in enhanced_msg["content"]:
                            if (item.get("type") == "text" and 
                                len(item.get("text", "")) > 1000):  # Cache long content
                                item["cache_control"] = {"type": "ephemeral"}
                
                enhanced_messages.append(enhanced_msg)
            else:
                enhanced_messages.append(msg)
        
        return completion(model=model, messages=enhanced_messages, **kwargs)
    else:
        # For implicit caching providers, use as-is
        return completion(model=model, messages=messages, **kwargs)

# Usage - automatically optimized for each provider
response1 = smart_completion("gpt-4", messages)                    # Implicit caching
response2 = smart_completion("anthropic/claude-3-sonnet", messages)  # Auto cache_control added
response3 = smart_completion("vertex_ai/gemini-pro", messages)     # Auto cache_control added
```

#### 3. Router-Level Caching Strategy

Use LiteLLM Router with caching-optimized model configurations:

```python
from litellm import Router

# Router with caching-optimized configurations
router = Router(
    model_list=[
        # OpenAI - implicit caching (always on)
        {
            "model_name": "gpt-4",
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"caching": "implicit"}
        },
        
        # Anthropic - explicit caching configuration
        {
            "model_name": "claude-sonnet",
            "litellm_params": {"model": "anthropic/claude-3-sonnet-20240229"},
            "model_info": {"caching": "explicit"}
        },
        
        # Gemini - explicit caching configuration  
        {
            "model_name": "gemini-pro",
            "litellm_params": {
                "model": "vertex_ai/gemini-1.5-pro",
                "vertex_ai_project": "your-project",
                "vertex_ai_location": "us-central1"
            },
            "model_info": {"caching": "explicit"}
        }
    ],
    # Router handles caching optimization automatically
    routing_strategy="lowest-latency"  # Benefits from cached responses
)

# Router automatically uses optimal caching for each provider
response = router.completion(
    model="claude-sonnet", 
    messages=[{"role": "user", "content": "Hello!"}]
)
```

#### 4. Conversation Manager with Universal Caching

Create a conversation manager that handles caching across all providers:

```python
class CachingConversationManager:
    """Manages conversations with optimal caching for all providers"""
    
    def __init__(self):
        self.conversations = {}
        
        # Provider caching capabilities
        self.explicit_cache_providers = {
            "anthropic/", "claude-", "vertex_ai/", "gemini/"
        }
    
    def _optimize_messages_for_caching(self, messages, model):
        """Add cache controls based on provider capabilities"""
        
        supports_explicit = any(
            provider in model for provider in self.explicit_cache_providers
        )
        
        if not supports_explicit:
            return messages  # Implicit caching providers
        
        optimized = []
        for i, msg in enumerate(messages):
            if msg["role"] == "system":
                # Always cache system messages
                optimized.append(self._add_cache_control(msg))
            elif (msg["role"] == "user" and 
                  len(msg.get("content", "")) > 500):  # Cache long user messages
                optimized.append(self._add_cache_control(msg))
            elif (i > 0 and i % 10 == 0):  # Cache every 10th message as checkpoint
                optimized.append(self._add_cache_control(msg))
            else:
                optimized.append(msg)
        
        return optimized
    
    def _add_cache_control(self, message):
        """Add cache_control to a message"""
        if isinstance(message["content"], str):
            return {
                "role": message["role"],
                "content": [
                    {
                        "type": "text",
                        "text": message["content"],
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            }
        return message  # Already structured
    
    def send_message(self, conversation_id, model, messages):
        """Send message with optimal caching"""
        
        # Optimize messages for the specific provider
        optimized_messages = self._optimize_messages_for_caching(messages, model)
        
        # Make the completion call
        response = completion(model=model, messages=optimized_messages)
        
        # Store conversation state
        self.conversations[conversation_id] = {
            "messages": optimized_messages,
            "model": model,
            "last_response": response
        }
        
        return response

# Usage across all providers with optimal caching
manager = CachingConversationManager()

# All these will use optimal caching for their respective providers
openai_response = manager.send_message("conv1", "gpt-4", messages)
anthropic_response = manager.send_message("conv2", "anthropic/claude-3-sonnet", messages)  
gemini_response = manager.send_message("conv3", "vertex_ai/gemini-pro", messages)
```

#### 5. Environment-Based Cache Configuration

Set provider-specific cache defaults via configuration:

```python
import os
from litellm import completion

# Configure default caching behavior
CACHING_CONFIG = {
    "anthropic": {"default_cache": True, "cache_system_messages": True},
    "vertex_ai": {"default_cache": True, "cache_long_messages": True},
    "gemini": {"default_cache": True, "cache_tools": True},
    # OpenAI, Azure, Cohere, etc. use implicit caching (always on)
}

def configured_completion(model, messages, **kwargs):
    """Completion with configured caching defaults"""
    
    provider = model.split('/')[0] if '/' in model else model.split('-')[0]
    cache_config = CACHING_CONFIG.get(provider, {})
    
    if cache_config.get("default_cache", False):
        # Apply caching based on configuration
        if cache_config.get("cache_system_messages", False):
            messages = auto_cache_system_messages(messages)
        
        if cache_config.get("cache_long_messages", False):
            messages = auto_cache_long_messages(messages)
    
    return completion(model=model, messages=messages, **kwargs)

# Use configured completion everywhere
response = configured_completion("anthropic/claude-3-sonnet", messages)
```

### Summary: Caching Options

| Approach | Scope | Configuration | Best For |
|----------|-------|---------------|----------|
| **Automatic (Built-in)** | Most providers | None | Simple applications |
| **Smart Completion Wrapper** | All providers | Helper function | Single model calls |
| **Router Strategy** | Multiple models | Router config | Load balancing + caching |
| **Conversation Manager** | Full conversations | Class-based | Complex chat apps |
| **Configuration-Based** | Environment-driven | Config file | Enterprise applications |

**Recommendation**: Use the **Smart Completion Wrapper** for most cases - it automatically optimizes caching for each provider without requiring global configuration changes.

### Universal Cache Control: Adding cache_control to ALL Providers

**Q: Can I just add `cache_control: {"type": "ephemeral"}` to all provider calls?**

**A: YES! LiteLLM handles this gracefully:**

```python
def universal_caching_approach():
    """Add cache_control to ALL providers - LiteLLM handles compatibility"""
    
    # Universal message format with cache_control
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant with extensive knowledge.",
                    "cache_control": {"type": "ephemeral"}  # ✅ Add to ALL providers
                }
            ]
        },
        {
            "role": "user", 
            "content": [
                {
                    "type": "text",
                    "text": "What is machine learning?",
                    "cache_control": {"type": "ephemeral"}  # ✅ Add everywhere
                }
            ]
        }
    ]
    
    # This works with ALL providers:
    
    # ✅ Anthropic - Uses cache_control (explicit caching)
    anthropic_response = completion(
        model="anthropic/claude-3-sonnet-20240229",
        messages=messages
    )
    
    # ✅ OpenAI - Ignores cache_control, uses implicit caching  
    openai_response = completion(
        model="gpt-4",
        messages=messages
    )
    
    # ✅ Gemini - Uses cache_control (explicit caching)
    gemini_response = completion(
        model="vertex_ai/gemini-1.5-pro",
        messages=messages,
        vertex_ai_project="your-project",
        vertex_ai_location="us-central1"
    )
    
    # ✅ Azure - Ignores cache_control, uses implicit caching
    azure_response = completion(
        model="azure/gpt-4",
        messages=messages
    )
    
    # ✅ Cohere - Ignores cache_control, uses implicit caching  
    cohere_response = completion(
        model="cohere/command-r-plus",
        messages=messages
    )
    
    return {
        "anthropic": anthropic_response,
        "openai": openai_response, 
        "gemini": gemini_response,
        "azure": azure_response,
        "cohere": cohere_response
    }

# Run with all providers
results = universal_caching_approach()
```

### How Each Provider Handles cache_control

| Provider | Behavior | Result |
|----------|----------|---------|
| **Anthropic** | ✅ **Uses cache_control** | Explicit caching with cost savings |
| **Gemini/Vertex AI** | ✅ **Uses cache_control** | Context caching enabled |  
| **OpenAI** | ⚠️ **Ignores cache_control** | Falls back to implicit caching |
| **Azure OpenAI** | ⚠️ **Ignores cache_control** | Falls back to implicit caching |
| **Cohere** | ⚠️ **Ignores cache_control** | Falls back to implicit caching |
| **Deepseek** | ⚠️ **Ignores cache_control** | Falls back to implicit caching |
| **Other Providers** | ⚠️ **Ignores cache_control** | Provider-specific behavior |

### Simple Universal Wrapper

Create a wrapper that adds cache_control to everything:

```python
def add_cache_control_to_all(messages):
    """Add cache_control to all messages for universal caching"""
    
    cached_messages = []
    
    for message in messages:
        if isinstance(message.get("content"), str):
            # Convert string content to structured format with cache_control
            cached_message = {
                "role": message["role"],
                "content": [
                    {
                        "type": "text",
                        "text": message["content"],
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            }
        elif isinstance(message.get("content"), list):
            # Add cache_control to existing structured content
            cached_message = message.copy()
            for item in cached_message["content"]:
                if item.get("type") == "text":
                    item["cache_control"] = {"type": "ephemeral"}
        else:
            cached_message = message
        
        cached_messages.append(cached_message)
    
    return cached_messages

# Usage with any provider
def universal_cached_completion(model, messages, **kwargs):
    """Completion with cache_control added to all messages"""
    cached_messages = add_cache_control_to_all(messages)
    return completion(model=model, messages=cached_messages, **kwargs)

# Works with ALL providers - each handles cache_control appropriately
response1 = universal_cached_completion("gpt-4", messages)                    # Ignored, uses implicit
response2 = universal_cached_completion("anthropic/claude-3-sonnet", messages) # Used for explicit caching  
response3 = universal_cached_completion("vertex_ai/gemini-pro", messages)     # Used for context caching
response4 = universal_cached_completion("cohere/command-r-plus", messages)   # Ignored, uses implicit
```

### Conversation Class with Universal Caching

```python
class UniversalCachingChat:
    """Chat class that adds cache_control to everything"""
    
    def __init__(self, model, system_prompt=None):
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
    
    def send_message(self, user_message):
        """Send message with universal cache_control"""
        
        # Add user message with cache_control
        self.messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": user_message,
                    "cache_control": {"type": "ephemeral"}  # Cache all user messages
                }
            ]
        })
        
        # Get response (cache_control is handled per provider)
        response = completion(model=self.model, messages=self.messages)
        
        # Add assistant response to conversation
        assistant_message = response.choices[0].message.content
        self.messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": assistant_message,
                    "cache_control": {"type": "ephemeral"}  # Cache responses too
                }
            ]
        })
        
        return assistant_message

# Usage - same code works with any provider
openai_chat = UniversalCachingChat("gpt-4", "You are a helpful assistant.")
anthropic_chat = UniversalCachingChat("anthropic/claude-3-sonnet", "You are a helpful assistant.")
gemini_chat = UniversalCachingChat("vertex_ai/gemini-1.5-pro", "You are a helpful assistant.")

# All conversations use optimal caching for their respective providers
openai_response = openai_chat.send_message("Hello!")      # Implicit caching
anthropic_response = anthropic_chat.send_message("Hello!") # Explicit caching  
gemini_response = gemini_chat.send_message("Hello!")      # Context caching
```

### Benefits of Universal cache_control

**✅ Advantages:**
- **Single codebase** works across all providers
- **Optimal caching** for providers that support it
- **Graceful fallback** for providers that don't
- **Future-proof** - new providers with caching support work automatically
- **No provider detection** logic needed

**⚠️ Considerations:**
- Slightly more verbose message format
- Small overhead for providers that ignore it
- All text must be in structured format

### Recommendation

**YES, use `cache_control: {"type": "ephemeral"}` universally!**

```python
# ✅ RECOMMENDED: Universal approach
messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "Your system prompt here",
                "cache_control": {"type": "ephemeral"}  # Add everywhere
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text", 
                "text": "Your question here",
                "cache_control": {"type": "ephemeral"}  # Add everywhere
            }
        ]
    }
]

# This format works optimally with ALL providers
response = completion(model="any-provider/model", messages=messages)
```

**Result**: You get the best of both worlds - explicit caching where supported, implicit caching where not, with zero provider-specific code!

## Tips & Best Practices

### 1. **Start Simple**
Begin with basic completion calls before adding complexity.

### 2. **Test Provider Availability**
Not all providers may be available in your region or with your API keys.

```python
def test_provider(model):
    try:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10
        )
        return True
    except:
        return False

providers_to_test = ["gpt-4", "anthropic/claude-3-haiku-20240307", "vertex_ai/gemini-1.5-flash"]
available_providers = [p for p in providers_to_test if test_provider(p)]
print("Available providers:", available_providers)
```

### 3. **Provider-Specific Features**
Some parameters only work with specific providers:

```python
# OpenAI-specific parameters
response = completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    functions=[...],           # Function calling
    function_call="auto",      # Function calling mode
    response_format={"type": "json_object"}  # JSON mode
)

# Anthropic-specific parameters
response = completion(
    model="anthropic/claude-3-sonnet-20240229",
    messages=[{"role": "user", "content": "Hello"}],
    system="You are a helpful assistant"  # System message for Claude
)
```

### 4. **Load Balancing with Router**
For production use, consider LiteLLM's Router for automatic load balancing:

```python
from litellm import Router

router = Router(model_list=[
    {"model_name": "gpt-4", "litellm_params": {"model": "gpt-4"}},
    {"model_name": "gpt-4", "litellm_params": {"model": "anthropic/claude-3-sonnet-20240229"}},
])

# Router automatically distributes requests
response = router.completion(model="gpt-4", messages=[...])
```

## Troubleshooting

### Common Issues:

1. **Authentication Errors**: Check your API keys are set correctly
2. **Model Not Found**: Verify the model name format for your provider
3. **Rate Limits**: Implement retry logic or use multiple providers
4. **Region Restrictions**: Some providers have geographic limitations

### Debug Mode:
```python
import litellm
litellm.set_verbose = True  # Enable debug logging

response = completion(model="gpt-4", messages=[...])
```

## Conclusion

LiteLLM's OpenAI-compatible API makes it incredibly easy to:
- **Switch between providers** without changing your code structure
- **Compare responses** from different models
- **Build resilient applications** with automatic fallbacks
- **Optimize costs** by choosing the right provider for each task

The unified interface means you can focus on building your application instead of learning different APIs for each provider.

**Next Steps:**
- Try the examples above with your API keys
- Experiment with different providers and models
- Build fallback logic for production applications
- Explore LiteLLM's Router for advanced load balancing

Happy coding! 🚀
