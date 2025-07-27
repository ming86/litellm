# Universal AI Assistant Platform: A Comprehensive LiteLLM Implementation Guide

## Overview

This guide provides a complete implementation strategy for building a production-ready Universal AI Assistant Platform using the LiteLLM Python SDK. The platform demonstrates LiteLLM's unified interface to 100+ LLM providers through a multi-modal, scalable application architecture.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Implementation Guide](#implementation-guide)
4. [Code Examples](#code-examples)
5. [Configuration Management](#configuration-management)
6. [Deployment Strategy](#deployment-strategy)
7. [Best Practices](#best-practices)

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────┬───────────────────┬───────────────────┤
│    Web UI           │    REST API       │   WebSocket API   │
│  (Streamlit)        │   (FastAPI)       │   (Real-time)     │
└─────────────────────┴───────────────────┴───────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                     Service Layer                           │
├──────────────┬──────────────┬──────────────┬───────────────┤
│ ChatService  │DocumentSrvc  │ AudioService │AnalyticsSrvc  │
└──────────────┴──────────────┴──────────────┴───────────────┘
┌─────────────────────────────────────────────────────────────┐
│                LiteLLM Integration Layer                    │
├──────────────┬──────────────┬──────────────┬───────────────┤
│RouterManager │ProviderConfig│ResponseProc. │ ErrorHandler  │
└──────────────┴──────────────┴──────────────┴───────────────┘
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                             │
├──────────────┬──────────────┬──────────────┬───────────────┤
│   Config     │   Sessions   │  Analytics   │   Documents   │
│   Storage    │   Storage    │   Storage    │   Storage     │
└──────────────┴──────────────┴──────────────┴───────────────┘
```

### Key Features

- **Multi-Provider Support**: Seamless integration with OpenAI, Anthropic, Cohere, Google, Azure, and 100+ providers
- **Intelligent Routing**: Load balancing, cost optimization, and automatic failover
- **Multi-Modal Capabilities**: Chat, document Q&A, embeddings, reranking, audio processing
- **Real-Time Analytics**: Cost tracking, performance monitoring, provider health checks
- **Production Ready**: Error handling, logging, monitoring, scalability, security

## Core Components

### 1. RouterManager - Multi-Provider Orchestration

The RouterManager is the heart of the platform, managing LiteLLM's Router for intelligent request distribution.

```python
from litellm import Router
from typing import Dict, List, Optional
import yaml
import os

class RouterManager:
    """
    Manages LiteLLM Router for multi-provider orchestration
    """
    
    def __init__(self, config_path: str = "config/providers.yaml"):
        self.router: Optional[Router] = None
        self.config_path = config_path
        self._model_configs = {}
        self._initialize_router()
    
    def _initialize_router(self):
        """Initialize LiteLLM Router with provider configurations"""
        model_list = self._load_model_configurations()
        
        self.router = Router(
            model_list=model_list,
            routing_strategy="simple-shuffle",  # Options: "least-busy", "lowest-cost", "lowest-latency"
            fallbacks=[
                {"gpt-4": ["claude-3-sonnet-20240229", "gemini-1.5-pro"]},
                {"gpt-3.5-turbo": ["claude-3-haiku-20240307", "gemini-1.5-flash"]},
            ],
            retry_strategy={
                "num_retries": 3,
                "retry_delay": 1,
                "exponential_backoff": True
            },
            timeout=120,
            set_verbose=True
        )
    
    def _load_model_configurations(self) -> List[Dict]:
        """Load model configurations from YAML file"""
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        model_list = []
        for provider_name, provider_config in config['providers'].items():
            for model_config in provider_config['models']:
                # Expand environment variables in API keys
                litellm_params = model_config['litellm_params'].copy()
                if 'api_key' in litellm_params:
                    litellm_params['api_key'] = os.path.expandvars(litellm_params['api_key'])
                
                model_entry = {
                    "model_name": model_config['model_name'],
                    "litellm_params": litellm_params
                }
                
                # Add optional parameters
                if 'tpm' in model_config:
                    model_entry['tpm'] = model_config['tpm']
                if 'rpm' in model_config:
                    model_entry['rpm'] = model_config['rpm']
                
                model_list.append(model_entry)
                self._model_configs[model_config['model_name']] = model_config
        
        return model_list
    
    async def complete(self, **kwargs) -> 'ModelResponse':
        """Execute completion with automatic provider selection and fallback"""
        try:
            return await self.router.acompletion(**kwargs)
        except Exception as e:
            # Enhanced error handling and fallback logic
            return await self._handle_completion_error(e, kwargs)
    
    async def _handle_completion_error(self, error, kwargs):
        """Handle completion errors with intelligent fallback"""
        from litellm.exceptions import RateLimitError, APIError, ContextWindowExceededError
        
        if isinstance(error, RateLimitError):
            # Switch to alternative provider with higher rate limits
            kwargs['model'] = self._get_fallback_model(kwargs['model'], criteria='rate_limit')
        elif isinstance(error, ContextWindowExceededError):
            # Switch to model with larger context window
            kwargs['model'] = self._get_fallback_model(kwargs['model'], criteria='context_window')
        
        return await self.router.acompletion(**kwargs)
    
    def _get_fallback_model(self, original_model: str, criteria: str = 'general') -> str:
        """Get fallback model based on specific criteria"""
        fallback_map = {
            'gpt-4': 'claude-3-sonnet-20240229',
            'gpt-3.5-turbo': 'claude-3-haiku-20240307',
            'claude-3-sonnet-20240229': 'gemini-1.5-pro',
            'claude-3-haiku-20240307': 'gemini-1.5-flash'
        }
        return fallback_map.get(original_model, 'gpt-3.5-turbo')
```

### 2. ChatService - Conversational AI

Manages multi-turn conversations with context retention and streaming support.

```python
from typing import List, Dict, AsyncGenerator
import json
import uuid
from datetime import datetime

class ChatService:
    """
    Manages conversational AI with multi-provider support
    """
    
    def __init__(self, router_manager: RouterManager):
        self.router_manager = router_manager
        self.conversations = {}  # In production, use persistent storage
        self.analytics_service = None  # Injected dependency
    
    async def start_conversation(self, user_id: str, model: str = "gpt-4") -> str:
        """Start a new conversation session"""
        conversation_id = str(uuid.uuid4())
        self.conversations[conversation_id] = {
            "id": conversation_id,
            "user_id": user_id,
            "model": model,
            "messages": [],
            "created_at": datetime.utcnow(),
            "metadata": {
                "total_tokens": 0,
                "total_cost": 0.0,
                "provider_usage": {}
            }
        }
        return conversation_id
    
    async def send_message(
        self, 
        conversation_id: str, 
        message: str, 
        stream: bool = False
    ) -> AsyncGenerator[str, None] if stream else str:
        """Send message and get AI response"""
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            raise ValueError("Conversation not found")
        
        # Add user message to conversation
        conversation['messages'].append({
            "role": "user",
            "content": message,
            "timestamp": datetime.utcnow()
        })
        
        # Prepare messages for LiteLLM
        litellm_messages = [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in conversation['messages']
        ]
        
        if stream:
            return self._stream_response(conversation, litellm_messages)
        else:
            return await self._get_response(conversation, litellm_messages)
    
    async def _get_response(self, conversation: Dict, messages: List[Dict]) -> str:
        """Get non-streaming response"""
        response = await self.router_manager.complete(
            model=conversation['model'],
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        
        # Extract response content
        assistant_message = response.choices[0].message.content
        
        # Update conversation
        conversation['messages'].append({
            "role": "assistant",
            "content": assistant_message,
            "timestamp": datetime.utcnow(),
            "model_used": response.model,
            "usage": response.usage.dict() if response.usage else {}
        })
        
        # Update metadata
        if response.usage:
            conversation['metadata']['total_tokens'] += response.usage.total_tokens
            
        # Track analytics
        if self.analytics_service:
            await self.analytics_service.track_completion(
                conversation_id=conversation['id'],
                response=response,
                provider=response.model.split('/')[0] if '/' in response.model else 'openai'
            )
        
        return assistant_message
    
    async def _stream_response(self, conversation: Dict, messages: List[Dict]) -> AsyncGenerator[str, None]:
        """Get streaming response"""
        response_chunks = []
        
        async for chunk in await self.router_manager.router.acompletion(
            model=conversation['model'],
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=2000
        ):
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                response_chunks.append(content)
                yield content
        
        # Store complete response
        complete_response = ''.join(response_chunks)
        conversation['messages'].append({
            "role": "assistant",
            "content": complete_response,
            "timestamp": datetime.utcnow(),
            "model_used": conversation['model']
        })
```

### 3. DocumentService - RAG Pipeline

Implements document intelligence with embeddings, reranking, and retrieval-augmented generation.

```python
from litellm import embedding, rerank
import numpy as np
from typing import List, Dict, Any
import asyncio

class DocumentService:
    """
    Document intelligence service with RAG capabilities
    """
    
    def __init__(self, router_manager: RouterManager):
        self.router_manager = router_manager
        self.vector_store = None  # Use Pinecone, Weaviate, or similar
        self.document_chunks = {}  # Document storage
    
    async def upload_document(self, file_path: str, collection: str = "default") -> str:
        """Upload and process document for RAG"""
        document_id = str(uuid.uuid4())
        
        # 1. Extract text from document
        text_content = await self._extract_text(file_path)
        
        # 2. Chunk document
        chunks = self._chunk_document(text_content)
        
        # 3. Generate embeddings for chunks
        embeddings_response = await embedding(
            model="text-embedding-3-small",  # or "cohere/embed-english-v3.0"
            input=[chunk['content'] for chunk in chunks]
        )
        
        # 4. Store in vector database
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings_response.data[i]['embedding']
            chunk['document_id'] = document_id
            chunk['collection'] = collection
            await self._store_chunk(chunk)
        
        self.document_chunks[document_id] = {
            "file_path": file_path,
            "chunks": chunks,
            "collection": collection,
            "created_at": datetime.utcnow()
        }
        
        return document_id
    
    async def query_documents(
        self, 
        query: str, 
        collection: str = "default",
        top_k: int = 10,
        rerank_top_n: int = 5
    ) -> Dict[str, Any]:
        """Query documents using RAG pipeline"""
        
        # 1. Generate query embedding
        query_embedding_response = await embedding(
            model="text-embedding-3-small",
            input=[query]
        )
        query_embedding = query_embedding_response.data[0]['embedding']
        
        # 2. Retrieve similar chunks
        similar_chunks = await self._similarity_search(
            query_embedding, 
            collection, 
            top_k
        )
        
        # 3. Rerank for relevance
        if len(similar_chunks) > rerank_top_n:
            rerank_response = await rerank(
                model="cohere/rerank-english-v3.0",
                query=query,
                documents=[chunk['content'] for chunk in similar_chunks],
                top_n=rerank_top_n
            )
            
            # Reorder chunks based on reranking scores
            reranked_chunks = []
            for result in rerank_response.results:
                reranked_chunks.append(similar_chunks[result.index])
            similar_chunks = reranked_chunks
        
        # 4. Create context from top chunks
        context = "\n\n".join([chunk['content'] for chunk in similar_chunks[:rerank_top_n]])
        
        # 5. Generate response with context
        response = await self.router_manager.complete(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant. Answer questions based on the provided context. If the context doesn't contain relevant information, say so."
                },
                {
                    "role": "user", 
                    "content": f"Context:\n{context}\n\nQuestion: {query}"
                }
            ],
            temperature=0.1
        )
        
        return {
            "answer": response.choices[0].message.content,
            "sources": similar_chunks[:rerank_top_n],
            "query": query,
            "model_used": response.model,
            "context_chunks": len(similar_chunks)
        }
    
    def _chunk_document(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """Split document into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                "content": chunk_text,
                "start_index": i,
                "end_index": min(i + chunk_size, len(words)),
                "word_count": len(chunk_words)
            })
        
        return chunks
    
    async def _similarity_search(self, query_embedding: List[float], collection: str, top_k: int) -> List[Dict]:
        """Perform similarity search in vector database"""
        # Implement with your chosen vector database
        # This is a simplified example
        all_chunks = []
        for doc_chunks in self.document_chunks.values():
            if doc_chunks['collection'] == collection:
                all_chunks.extend(doc_chunks['chunks'])
        
        # Calculate cosine similarity
        similarities = []
        for chunk in all_chunks:
            similarity = np.dot(query_embedding, chunk['embedding']) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk['embedding'])
            )
            similarities.append((similarity, chunk))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in similarities[:top_k]]
```

### 4. AudioService - Speech Processing

Handles speech-to-text and text-to-speech with multiple providers.

```python
from litellm import transcription, speech
from pathlib import Path
import tempfile
import aiofiles

class AudioService:
    """
    Audio processing service for speech-to-text and text-to-speech
    """
    
    def __init__(self, router_manager: RouterManager):
        self.router_manager = router_manager
        self.supported_formats = ['.mp3', '.wav', '.m4a', '.ogg', '.flac']
    
    async def transcribe_audio(self, audio_file_path: str, language: str = None) -> Dict[str, Any]:
        """Transcribe audio to text using multiple providers"""
        file_path = Path(audio_file_path)
        
        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported audio format. Supported: {self.supported_formats}")
        
        try:
            # Try OpenAI Whisper first
            with open(audio_file_path, "rb") as audio_file:
                response = await transcription(
                    model="whisper-1",
                    file=audio_file,
                    language=language,
                    response_format="verbose_json"
                )
            
            return {
                "text": response.text,
                "language": getattr(response, 'language', 'unknown'),
                "duration": getattr(response, 'duration', 0),
                "segments": getattr(response, 'segments', []),
                "provider": "openai",
                "model": "whisper-1"
            }
            
        except Exception as e:
            # Fallback to ElevenLabs or other providers
            try:
                with open(audio_file_path, "rb") as audio_file:
                    response = await transcription(
                        model="elevenlabs/scribe_v1",
                        file=audio_file,
                        api_key=os.getenv("ELEVENLABS_API_KEY")
                    )
                
                return {
                    "text": response.text,
                    "provider": "elevenlabs",
                    "model": "scribe_v1"
                }
            except Exception as fallback_error:
                raise Exception(f"All transcription providers failed: {str(e)}, {str(fallback_error)}")
    
    async def text_to_speech(
        self, 
        text: str, 
        voice: str = "alloy",
        output_format: str = "mp3",
        provider: str = "openai"
    ) -> str:
        """Convert text to speech and return file path"""
        
        # Create temporary file for audio output
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=f".{output_format}"
        )
        temp_file.close()
        
        try:
            if provider == "openai":
                response = await speech(
                    model="tts-1",
                    voice=voice,
                    input=text,
                    response_format=output_format
                )
            elif provider == "elevenlabs":
                response = await speech(
                    model="elevenlabs/eleven_monolingual_v1",
                    voice=voice,
                    input=text,
                    api_key=os.getenv("ELEVENLABS_API_KEY")
                )
            else:
                raise ValueError(f"Unsupported TTS provider: {provider}")
            
            # Stream response to file
            response.stream_to_file(temp_file.name)
            
            return temp_file.name
            
        except Exception as e:
            # Clean up temp file on error
            Path(temp_file.name).unlink(missing_ok=True)
            raise Exception(f"Text-to-speech failed: {str(e)}")
    
    async def process_audio_conversation(
        self, 
        audio_file_path: str, 
        conversation_id: str,
        chat_service: 'ChatService'
    ) -> str:
        """Process audio input and return audio response"""
        
        # 1. Transcribe audio to text
        transcription_result = await self.transcribe_audio(audio_file_path)
        user_message = transcription_result['text']
        
        # 2. Get text response from chat service
        response_text = await chat_service.send_message(conversation_id, user_message)
        
        # 3. Convert response to speech
        audio_response_path = await self.text_to_speech(
            text=response_text,
            voice="alloy",
            provider="openai"
        )
        
        return audio_response_path
```

### 5. AnalyticsService - Monitoring & Cost Tracking

Comprehensive analytics for cost, performance, and usage monitoring.

```python
from litellm import completion_cost
from typing import Dict, List, Any
from datetime import datetime, timedelta
import asyncio

class AnalyticsService:
    """
    Analytics service for cost tracking and performance monitoring
    """
    
    def __init__(self):
        self.metrics_storage = {}  # Use proper database in production
        self.provider_health = {}
    
    async def track_completion(
        self, 
        conversation_id: str,
        response: 'ModelResponse',
        provider: str,
        latency_ms: float = None
    ):
        """Track completion metrics"""
        
        cost = completion_cost(completion_response=response)
        
        metric = {
            "id": str(uuid.uuid4()),
            "conversation_id": conversation_id,
            "provider": provider,
            "model": response.model,
            "timestamp": datetime.utcnow(),
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0
            },
            "cost": cost,
            "latency_ms": latency_ms,
            "success": True
        }
        
        await self._store_metric(metric)
        await self._update_provider_health(provider, True, latency_ms)
    
    async def track_error(
        self, 
        provider: str, 
        model: str, 
        error: Exception,
        latency_ms: float = None
    ):
        """Track error metrics"""
        
        metric = {
            "id": str(uuid.uuid4()),
            "provider": provider,
            "model": model,
            "timestamp": datetime.utcnow(),
            "error": str(error),
            "error_type": type(error).__name__,
            "latency_ms": latency_ms,
            "success": False
        }
        
        await self._store_metric(metric)
        await self._update_provider_health(provider, False, latency_ms)
    
    async def get_cost_analytics(
        self, 
        time_range: str = "24h",
        group_by: str = "provider"
    ) -> Dict[str, Any]:
        """Get cost analytics for specified time range"""
        
        end_time = datetime.utcnow()
        if time_range == "24h":
            start_time = end_time - timedelta(hours=24)
        elif time_range == "7d":
            start_time = end_time - timedelta(days=7)
        elif time_range == "30d":
            start_time = end_time - timedelta(days=30)
        else:
            start_time = end_time - timedelta(hours=24)
        
        # Filter metrics by time range
        filtered_metrics = [
            metric for metric in self.metrics_storage.values()
            if start_time <= metric['timestamp'] <= end_time and metric['success']
        ]
        
        if group_by == "provider":
            return self._group_by_provider(filtered_metrics)
        elif group_by == "model":
            return self._group_by_model(filtered_metrics)
        elif group_by == "day":
            return self._group_by_day(filtered_metrics)
        else:
            return self._group_by_provider(filtered_metrics)
    
    async def get_performance_analytics(self, time_range: str = "24h") -> Dict[str, Any]:
        """Get performance analytics"""
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24 if time_range == "24h" else 168)
        
        filtered_metrics = [
            metric for metric in self.metrics_storage.values()
            if start_time <= metric['timestamp'] <= end_time
        ]
        
        # Calculate success rates
        total_requests = len(filtered_metrics)
        successful_requests = len([m for m in filtered_metrics if m['success']])
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
        
        # Calculate average latency by provider
        provider_latencies = {}
        for metric in filtered_metrics:
            if metric.get('latency_ms') and metric['success']:
                provider = metric['provider']
                if provider not in provider_latencies:
                    provider_latencies[provider] = []
                provider_latencies[provider].append(metric['latency_ms'])
        
        avg_latencies = {
            provider: sum(latencies) / len(latencies)
            for provider, latencies in provider_latencies.items()
        }
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "success_rate": success_rate,
            "average_latencies": avg_latencies,
            "time_range": time_range,
            "generated_at": datetime.utcnow()
        }
    
    async def get_provider_health(self) -> Dict[str, Any]:
        """Get current provider health status"""
        return {
            provider: {
                "status": health.get("status", "unknown"),
                "success_rate": health.get("success_rate", 0),
                "average_latency": health.get("average_latency", 0),
                "last_success": health.get("last_success"),
                "last_error": health.get("last_error"),
                "total_requests": health.get("total_requests", 0)
            }
            for provider, health in self.provider_health.items()
        }
    
    def _group_by_provider(self, metrics: List[Dict]) -> Dict[str, Any]:
        """Group metrics by provider"""
        grouped = {}
        for metric in metrics:
            provider = metric['provider']
            if provider not in grouped:
                grouped[provider] = {
                    "total_cost": 0,
                    "total_tokens": 0,
                    "request_count": 0
                }
            
            grouped[provider]["total_cost"] += metric.get('cost', 0)
            grouped[provider]["total_tokens"] += metric['usage']['total_tokens']
            grouped[provider]["request_count"] += 1
        
        return grouped
    
    async def _store_metric(self, metric: Dict):
        """Store metric (implement with your database)"""
        self.metrics_storage[metric['id']] = metric
    
    async def _update_provider_health(self, provider: str, success: bool, latency_ms: float):
        """Update provider health metrics"""
        if provider not in self.provider_health:
            self.provider_health[provider] = {
                "total_requests": 0,
                "successful_requests": 0,
                "latencies": [],
                "last_success": None,
                "last_error": None
            }
        
        health = self.provider_health[provider]
        health["total_requests"] += 1
        
        if success:
            health["successful_requests"] += 1
            health["last_success"] = datetime.utcnow()
            if latency_ms:
                health["latencies"].append(latency_ms)
                # Keep only last 100 latencies for memory efficiency
                health["latencies"] = health["latencies"][-100:]
        else:
            health["last_error"] = datetime.utcnow()
        
        # Update calculated fields
        health["success_rate"] = (health["successful_requests"] / health["total_requests"]) * 100
        health["average_latency"] = sum(health["latencies"]) / len(health["latencies"]) if health["latencies"] else 0
        health["status"] = "healthy" if health["success_rate"] > 95 else "degraded" if health["success_rate"] > 80 else "unhealthy"
```

## Configuration Management

### Provider Configuration (config/providers.yaml)

```yaml
providers:
  openai:
    models:
      - model_name: "gpt-4"
        litellm_params:
          model: "openai/gpt-4"
          api_key: "${OPENAI_API_KEY}"
        tpm: 300000  # tokens per minute
        rpm: 5000    # requests per minute
        
      - model_name: "gpt-3.5-turbo"
        litellm_params:
          model: "openai/gpt-3.5-turbo"
          api_key: "${OPENAI_API_KEY}"
        tpm: 1000000
        rpm: 10000

  anthropic:
    models:
      - model_name: "claude-3-sonnet-20240229"
        litellm_params:
          model: "anthropic/claude-3-sonnet-20240229"
          api_key: "${ANTHROPIC_API_KEY}"
        tpm: 200000
        rpm: 4000
        
      - model_name: "claude-3-haiku-20240307"
        litellm_params:
          model: "anthropic/claude-3-haiku-20240307"
          api_key: "${ANTHROPIC_API_KEY}"
        tpm: 300000
        rpm: 5000

  google:
    models:
      - model_name: "gemini-1.5-pro"
        litellm_params:
          model: "vertex_ai/gemini-1.5-pro"
          vertex_project: "${GOOGLE_CLOUD_PROJECT}"
          vertex_location: "${GOOGLE_CLOUD_LOCATION}"
        tpm: 300000
        rpm: 1000
        
      - model_name: "gemini-1.5-flash"
        litellm_params:
          model: "vertex_ai/gemini-1.5-flash"
          vertex_project: "${GOOGLE_CLOUD_PROJECT}"
          vertex_location: "${GOOGLE_CLOUD_LOCATION}"
        tpm: 1000000
        rpm: 2000

  cohere:
    models:
      - model_name: "command-r-plus"
        litellm_params:
          model: "cohere/command-r-plus"
          api_key: "${COHERE_API_KEY}"
        tpm: 10000
        rpm: 100

  # Embedding models
  embeddings:
    - model_name: "text-embedding-3-small"
      litellm_params:
        model: "openai/text-embedding-3-small"
        api_key: "${OPENAI_API_KEY}"
    
    - model_name: "embed-english-v3.0"
      litellm_params:
        model: "cohere/embed-english-v3.0"
        api_key: "${COHERE_API_KEY}"

  # Reranking models
  rerankers:
    - model_name: "rerank-english-v3.0"
      litellm_params:
        model: "cohere/rerank-english-v3.0"
        api_key: "${COHERE_API_KEY}"
```

### Application Settings (config/settings.yaml)

```yaml
app:
  name: "Universal AI Assistant Platform"
  version: "1.0.0"
  debug: false
  
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  
database:
  url: "${DATABASE_URL}"
  pool_size: 10
  
redis:
  url: "${REDIS_URL}"
  
storage:
  type: "s3"  # or "local", "gcs"
  bucket: "${S3_BUCKET}"
  
logging:
  level: "INFO"
  format: "json"
  
monitoring:
  prometheus:
    enabled: true
    port: 9090
  
  sentry:
    dsn: "${SENTRY_DSN}"
    environment: "${ENVIRONMENT}"
```

## Implementation Guide

### Project Structure

```
universal_ai_assistant/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── router_manager.py
│   │   ├── provider_config.py
│   │   ├── response_processor.py
│   │   └── error_handler.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── chat_service.py
│   │   ├── document_service.py
│   │   ├── audio_service.py
│   │   ├── analytics_service.py
│   │   └── provider_service.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── chat_endpoints.py
│   │   ├── document_endpoints.py
│   │   ├── audio_endpoints.py
│   │   └── analytics_endpoints.py
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── streamlit_app.py
│   │   ├── components/
│   │   │   ├── chat_interface.py
│   │   │   ├── document_upload.py
│   │   │   ├── analytics_dashboard.py
│   │   │   └── provider_status.py
│   │   └── pages/
│   │       ├── chat.py
│   │       ├── documents.py
│   │       ├── audio.py
│   │       └── analytics.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── requests.py
│   │   ├── responses.py
│   │   └── config.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       ├── validation.py
│       └── monitoring.py
├── config/
│   ├── providers.yaml
│   ├── models.yaml
│   └── settings.yaml
├── tests/
│   ├── __init__.py
│   ├── test_chat_service.py
│   ├── test_document_service.py
│   ├── test_audio_service.py
│   └── test_analytics_service.py
├── docs/
│   ├── README.md
│   ├── API.md
│   └── deployment.md
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── docker-compose.prod.yml
├── scripts/
│   ├── setup.sh
│   ├── run_dev.sh
│   └── deploy.sh
├── requirements.txt
├── pyproject.toml
└── README.md
```

### FastAPI Application (src/api/main.py)

```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn

from src.core.router_manager import RouterManager
from src.services import ChatService, DocumentService, AudioService, AnalyticsService
from src.api import chat_endpoints, document_endpoints, audio_endpoints, analytics_endpoints

# Initialize services
router_manager = RouterManager()
chat_service = ChatService(router_manager)
document_service = DocumentService(router_manager)
audio_service = AudioService(router_manager)
analytics_service = AnalyticsService()

# Inject analytics into chat service
chat_service.analytics_service = analytics_service

app = FastAPI(
    title="Universal AI Assistant Platform",
    description="Multi-provider AI platform powered by LiteLLM",
    version="1.0.0"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include routers
app.include_router(
    chat_endpoints.router, 
    prefix="/api/v1/chat", 
    tags=["chat"]
)
app.include_router(
    document_endpoints.router, 
    prefix="/api/v1/documents", 
    tags=["documents"]
)
app.include_router(
    audio_endpoints.router, 
    prefix="/api/v1/audio", 
    tags=["audio"]
)
app.include_router(
    analytics_endpoints.router, 
    prefix="/api/v1/analytics", 
    tags=["analytics"]
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    provider_health = await analytics_service.get_provider_health()
    return {
        "status": "healthy",
        "version": "1.0.0",
        "providers": provider_health
    }

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )
```

### Streamlit UI (src/ui/streamlit_app.py)

```python
import streamlit as st
import asyncio
from datetime import datetime
import requests
import json

# Configure page
st.set_page_config(
    page_title="Universal AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL
API_BASE = "http://localhost:8000/api/v1"

def main():
    st.title("🤖 Universal AI Assistant Platform")
    st.markdown("*Powered by LiteLLM - Supporting 100+ AI Providers*")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Choose a feature:",
            ["💬 Chat", "📄 Documents", "🎤 Audio", "📊 Analytics", "⚙️ Settings"]
        )
        
        # Provider status
        st.header("Provider Status")
        try:
            response = requests.get(f"{API_BASE}/analytics/provider-health")
            if response.status_code == 200:
                provider_health = response.json()
                for provider, health in provider_health.items():
                    status_color = {
                        "healthy": "🟢",
                        "degraded": "🟡", 
                        "unhealthy": "🔴"
                    }.get(health["status"], "⚪")
                    st.write(f"{status_color} {provider}: {health['success_rate']:.1f}%")
        except:
            st.write("Unable to fetch provider status")
    
    # Main content based on selected page
    if page == "💬 Chat":
        render_chat_page()
    elif page == "📄 Documents":
        render_documents_page()
    elif page == "🎤 Audio":
        render_audio_page()
    elif page == "📊 Analytics":
        render_analytics_page()
    elif page == "⚙️ Settings":
        render_settings_page()

def render_chat_page():
    st.header("💬 AI Chat Assistant")
    
    # Model selection
    col1, col2 = st.columns([3, 1])
    with col1:
        model = st.selectbox(
            "Select Model:",
            ["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "gemini-1.5-pro"]
        )
    with col2:
        stream = st.checkbox("Stream responses", value=True)
    
    # Initialize conversation
    if "conversation_id" not in st.session_state:
        response = requests.post(f"{API_BASE}/chat/conversations", json={"model": model})
        if response.status_code == 200:
            st.session_state.conversation_id = response.json()["conversation_id"]
        else:
            st.error("Failed to initialize conversation")
            return
    
    # Chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and "metadata" in message:
                with st.expander("Message Details"):
                    st.json(message["metadata"])
    
    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            if stream:
                # Streaming response
                response = requests.post(
                    f"{API_BASE}/chat/conversations/{st.session_state.conversation_id}/messages/stream",
                    json={"message": prompt},
                    stream=True
                )
                
                full_response = ""
                for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
                    if chunk:
                        full_response += chunk
                        message_placeholder.write(full_response + "▌")
                
                message_placeholder.write(full_response)
            else:
                # Non-streaming response
                response = requests.post(
                    f"{API_BASE}/chat/conversations/{st.session_state.conversation_id}/messages",
                    json={"message": prompt}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    full_response = result["response"]
                    message_placeholder.write(full_response)
                else:
                    full_response = "Sorry, I encountered an error processing your request."
                    message_placeholder.write(full_response)
        
        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

def render_documents_page():
    st.header("📄 Document Intelligence")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Upload Documents")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "txt", "md", "docx"],
            help="Upload documents to query with AI"
        )
        
        collection = st.text_input("Collection Name", value="default")
        
        if uploaded_file and st.button("Upload & Process"):
            # Save uploaded file temporarily
            with open(f"temp_{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Upload to API
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            data = {"collection": collection}
            
            response = requests.post(
                f"{API_BASE}/documents/upload",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                st.success("Document uploaded and processed!")
                st.json(response.json())
            else:
                st.error("Failed to upload document")
    
    with col2:
        st.subheader("Query Documents")
        
        query = st.text_input("Ask a question about your documents:")
        query_collection = st.text_input("Collection to search", value="default")
        
        col2a, col2b = st.columns(2)
        with col2a:
            top_k = st.slider("Retrieve chunks", 5, 20, 10)
        with col2b:
            rerank_n = st.slider("Rerank top", 3, 10, 5)
        
        if query and st.button("Search"):
            response = requests.post(
                f"{API_BASE}/documents/query",
                json={
                    "query": query,
                    "collection": query_collection,
                    "top_k": top_k,
                    "rerank_top_n": rerank_n
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                st.subheader("Answer")
                st.write(result["answer"])
                
                st.subheader("Sources")
                for i, source in enumerate(result["sources"]):
                    with st.expander(f"Source {i+1}"):
                        st.write(source["content"])
            else:
                st.error("Failed to query documents")

def render_analytics_page():
    st.header("📊 Analytics Dashboard")
    
    # Time range selection
    time_range = st.selectbox("Time Range", ["24h", "7d", "30d"])
    
    # Get analytics data
    try:
        cost_response = requests.get(f"{API_BASE}/analytics/costs?time_range={time_range}")
        perf_response = requests.get(f"{API_BASE}/analytics/performance?time_range={time_range}")
        
        if cost_response.status_code == 200 and perf_response.status_code == 200:
            cost_data = cost_response.json()
            perf_data = perf_response.json()
            
            # Cost metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💰 Cost Breakdown")
                total_cost = sum(provider["total_cost"] for provider in cost_data.values())
                st.metric("Total Cost", f"${total_cost:.4f}")
                
                # Provider cost chart
                provider_costs = {provider: data["total_cost"] for provider, data in cost_data.items()}
                st.bar_chart(provider_costs)
            
            with col2:
                st.subheader("⚡ Performance Metrics")
                st.metric("Success Rate", f"{perf_data['success_rate']:.1f}%")
                st.metric("Total Requests", perf_data['total_requests'])
                
                # Latency chart
                if perf_data['average_latencies']:
                    st.bar_chart(perf_data['average_latencies'])
            
            # Detailed breakdown
            st.subheader("Detailed Breakdown")
            st.json(cost_data)
            
    except Exception as e:
        st.error(f"Failed to load analytics: {str(e)}")

if __name__ == "__main__":
    main()
```

## Deployment Strategy

### Docker Configuration

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/aiassistant
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - COHERE_API_KEY=${COHERE_API_KEY}
      - GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT}
      - GOOGLE_CLOUD_LOCATION=${GOOGLE_CLOUD_LOCATION}
    depends_on:
      - postgres
      - redis
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs

  ui:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    environment:
      - API_BASE_URL=http://api:8000/api/v1
    depends_on:
      - api

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=aiassistant
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

### Requirements.txt

```txt
# Core dependencies
litellm>=1.65.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
streamlit>=1.28.0
pydantic>=2.5.0
python-multipart>=0.0.6

# Database & Storage
sqlalchemy>=2.0.0
asyncpg>=0.29.0
redis>=5.0.0
psycopg2-binary>=2.9.0

# Document processing
pypdf2>=3.0.0
python-docx>=1.1.0
tiktoken>=0.5.0

# Vector storage
pinecone-client>=2.2.0
# weaviate-client>=3.25.0
# chromadb>=0.4.0

# Audio processing
pydub>=0.25.0

# Monitoring & Logging
prometheus-client>=0.19.0
structlog>=23.2.0
sentry-sdk[fastapi]>=1.40.0

# Utilities
python-dotenv>=1.0.0
pyyaml>=6.0.0
aiofiles>=23.2.0
httpx>=0.25.0
tenacity>=8.2.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0

# Development
black>=23.10.0
isort>=5.12.0
flake8>=6.1.0
mypy>=1.7.0
```

## Best Practices

### 1. Error Handling & Resilience

```python
from tenacity import retry, stop_after_attempt, wait_exponential
from litellm.exceptions import RateLimitError, APIError, ContextWindowExceededError

class ResilientLiteLLMService:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def resilient_completion(self, **kwargs):
        """Completion with automatic retries and error handling"""
        try:
            return await self.router_manager.complete(**kwargs)
        except RateLimitError:
            # Switch to different provider
            kwargs['model'] = self._get_alternative_model(kwargs['model'])
            return await self.router_manager.complete(**kwargs)
        except ContextWindowExceededError:
            # Truncate messages or switch to larger context model
            kwargs = self._handle_context_overflow(kwargs)
            return await self.router_manager.complete(**kwargs)
        except APIError as e:
            # Log error and potentially switch provider
            logger.error(f"API Error: {e}")
            raise
```

### 2. Cost Optimization

```python
class CostOptimizer:
    def __init__(self):
        self.model_costs = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "claude-3-sonnet": {"input": 0.015, "output": 0.075},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125}
        }
    
    def select_optimal_model(self, requirements: Dict) -> str:
        """Select the most cost-effective model for requirements"""
        if requirements.get("complexity") == "high":
            return min(["gpt-4", "claude-3-sonnet"], 
                      key=lambda m: self.model_costs[m]["input"])
        else:
            return min(["gpt-3.5-turbo", "claude-3-haiku"], 
                      key=lambda m: self.model_costs[m]["input"])
```

### 3. Security & Compliance

```python
from cryptography.fernet import Fernet
import hashlib

class SecurityManager:
    def __init__(self):
        self.cipher_suite = Fernet(self._get_encryption_key())
    
    def encrypt_api_key(self, api_key: str) -> str:
        """Encrypt API keys for secure storage"""
        return self.cipher_suite.encrypt(api_key.encode()).decode()
    
    def decrypt_api_key(self, encrypted_key: str) -> str:
        """Decrypt API keys for use"""
        return self.cipher_suite.decrypt(encrypted_key.encode()).decode()
    
    def sanitize_input(self, user_input: str) -> str:
        """Sanitize user input to prevent injection attacks"""
        # Implement input sanitization logic
        return user_input.strip()
    
    def audit_log(self, user_id: str, action: str, details: Dict):
        """Log user actions for compliance"""
        log_entry = {
            "timestamp": datetime.utcnow(),
            "user_id": hashlib.sha256(user_id.encode()).hexdigest(),
            "action": action,
            "details": details
        }
        # Store in secure audit log
```

### 4. Performance Monitoring

```python
import time
from contextlib import asynccontextmanager

class PerformanceMonitor:
    @asynccontextmanager
    async def track_request(self, operation: str, provider: str):
        """Context manager for tracking request performance"""
        start_time = time.time()
        try:
            yield
        except Exception as e:
            # Log error metrics
            await self._log_error(operation, provider, str(e))
            raise
        finally:
            duration = time.time() - start_time
            await self._log_performance(operation, provider, duration)
    
    async def _log_performance(self, operation: str, provider: str, duration: float):
        """Log performance metrics"""
        # Send to monitoring service (Prometheus, DataDog, etc.)
        pass
```

## Conclusion

This Universal AI Assistant Platform demonstrates the full power of LiteLLM's unified interface to multiple AI providers. The architecture is designed to be:

- **Production-Ready**: Comprehensive error handling, monitoring, and security
- **Scalable**: Modular design supporting horizontal scaling
- **Cost-Effective**: Intelligent provider routing and cost optimization
- **Extensible**: Easy to add new providers, models, and features
- **Educational**: Clear examples of LiteLLM best practices

The platform showcases real-world use cases including conversational AI, document intelligence, audio processing, and comprehensive analytics - all while abstracting away the complexity of working with multiple AI providers.

## Next Steps

1. **Implementation**: Start with the core RouterManager and ChatService
2. **Database Integration**: Add persistent storage for conversations and documents
3. **Vector Database**: Integrate Pinecone, Weaviate, or ChromaDB for document embeddings
4. **Authentication**: Implement user management and API key security
5. **Monitoring**: Set up Prometheus, Grafana, and alerting
6. **CI/CD**: Create deployment pipelines for production
7. **Testing**: Comprehensive test suite with provider mocking
8. **Documentation**: API documentation and user guides

This implementation guide provides everything needed to build a comprehensive, production-ready AI platform using LiteLLM's powerful capabilities.