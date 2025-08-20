# Integration Guide

## Overview

This guide provides step-by-step instructions for integrating the Coherence-Aware AI Framework (CAAF) with your AI application. CAAF works with any LLM or AI system to provide coherence-aware interactions.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Integration Patterns](#integration-patterns)
3. [Platform-Specific Guides](#platform-specific-guides)
4. [Best Practices](#best-practices)
5. [Troubleshooting](#troubleshooting)

## Quick Start

### 1. Install CAAF

```bash
# Clone the repository
git clone https://github.com/GreatPyreneseDad/CAAF.git
cd CAAF

# Install dependencies
pip install -r requirements.txt

# Start the API server
python -m src.api
```

### 2. Basic Integration

```python
import requests

class CoherenceAwareAI:
    def __init__(self, caaf_url="http://localhost:8000"):
        self.caaf_url = caaf_url
    
    def get_response(self, user_message, user_id):
        # Step 1: Assess coherence
        coherence = requests.post(
            f"{self.caaf_url}/assess-coherence",
            json={"message": user_message, "user_id": user_id}
        ).json()
        
        # Step 2: Generate AI response (your existing code)
        ai_response = your_ai_model.generate(user_message)
        
        # Step 3: Optimize response based on coherence
        optimized = requests.post(
            f"{self.caaf_url}/optimize-response",
            json={
                "original_response": ai_response,
                "user_id": user_id
            }
        ).json()
        
        return optimized["optimized_response"]
```

## Integration Patterns

### Pattern 1: Synchronous Integration

Best for: Real-time chat applications, customer support bots

```python
def handle_message(user_message, user_id):
    # Assess and optimize in sequence
    coherence = assess_coherence(user_message, user_id)
    ai_response = generate_response(user_message)
    optimized = optimize_response(ai_response, user_id, coherence)
    return optimized
```

### Pattern 2: Asynchronous Integration

Best for: High-volume applications, streaming responses

```python
import asyncio
import aiohttp

async def handle_message_async(user_message, user_id):
    async with aiohttp.ClientSession() as session:
        # Assess coherence and generate response in parallel
        coherence_task = assess_coherence_async(session, user_message, user_id)
        response_task = generate_response_async(user_message)
        
        coherence, ai_response = await asyncio.gather(
            coherence_task, response_task
        )
        
        # Optimize based on coherence
        optimized = await optimize_response_async(
            session, ai_response, user_id, coherence
        )
        return optimized
```

### Pattern 3: Middleware Integration

Best for: Existing API services, microservices architecture

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

class CAFMiddleware:
    def __init__(self, app, caaf_url="http://localhost:8000"):
        self.app = app
        self.caaf_url = caaf_url
        self.init_app(app)
    
    def init_app(self, app):
        app.before_request(self.assess_coherence)
        app.after_request(self.optimize_response)
    
    def assess_coherence(self):
        if request.method == "POST" and request.json.get("message"):
            # Store coherence in request context
            request.coherence = requests.post(
                f"{self.caaf_url}/assess-coherence",
                json={
                    "message": request.json["message"],
                    "user_id": request.json.get("user_id", "anonymous")
                }
            ).json()
    
    def optimize_response(self, response):
        if hasattr(request, "coherence") and response.json:
            # Optimize the response
            optimized = requests.post(
                f"{self.caaf_url}/optimize-response",
                json={
                    "original_response": response.json.get("response"),
                    "user_id": request.json.get("user_id", "anonymous")
                }
            ).json()
            
            response.json["response"] = optimized["optimized_response"]
            response.json["coherence_info"] = request.coherence
        
        return response
```

## Platform-Specific Guides

### OpenAI / ChatGPT

```python
import openai
from caaf_client import CAFClient  # Hypothetical client library

client = CAFClient()

def coherence_aware_chatgpt(user_message, user_id, model="gpt-4"):
    # Assess coherence
    coherence = client.assess_coherence(user_message, user_id)
    
    # Add coherence context to system message
    system_message = f"""
    You are a helpful assistant. The user's current coherence state is {coherence['state']}.
    Coherence score: {coherence['coherence_score']:.2f}
    Please adapt your response style accordingly:
    - Crisis/Low: Be supportive, use simple language, focus on immediate help
    - Medium: Balance support with gentle growth opportunities  
    - High/Optimal: Engage with complex ideas, challenge thinking constructively
    """
    
    # Generate response
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    )
    
    # Optimize response
    optimized = client.optimize_response(
        response.choices[0].message.content,
        user_id
    )
    
    return optimized
```

### Anthropic / Claude

```python
import anthropic
from caaf_client import CAFClient

client = CAFClient()
claude = anthropic.Anthropic()

def coherence_aware_claude(user_message, user_id):
    # Assess coherence
    coherence = client.assess_coherence(user_message, user_id)
    
    # Create coherence-aware system prompt
    system_prompt = generate_coherence_prompt(coherence)
    
    # Generate response
    response = claude.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=500,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )
    
    # Optimize response
    return client.optimize_response(response.content[0].text, user_id)
```

### LangChain Integration

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks import BaseCallbackHandler

class CoherenceAwareCallback(BaseCallbackHandler):
    def __init__(self, caaf_client, user_id):
        self.caaf_client = caaf_client
        self.user_id = user_id
        self.coherence_state = None
    
    def on_chat_model_start(self, serialized, messages, **kwargs):
        # Assess coherence from user message
        user_message = next((m.content for m in messages if isinstance(m, HumanMessage)), "")
        self.coherence_state = self.caaf_client.assess_coherence(user_message, self.user_id)
    
    def on_llm_end(self, response, **kwargs):
        # Optimize the response
        original = response.generations[0][0].text
        optimized = self.caaf_client.optimize_response(original, self.user_id)
        response.generations[0][0].text = optimized["optimized_response"]

# Usage
chat = ChatOpenAI(
    callbacks=[CoherenceAwareCallback(caaf_client, "user123")]
)
response = chat([HumanMessage(content="I'm feeling overwhelmed")])
```

### Hugging Face Transformers

```python
from transformers import pipeline
from caaf_client import CAFClient

class CoherenceAwareGenerator:
    def __init__(self, model_name="gpt2"):
        self.generator = pipeline("text-generation", model=model_name)
        self.caaf = CAFClient()
    
    def generate(self, prompt, user_id, max_length=100):
        # Assess coherence
        coherence = self.caaf.assess_coherence(prompt, user_id)
        
        # Adjust generation parameters based on coherence
        if coherence["state"] in ["crisis", "low"]:
            # Simpler, shorter responses
            temperature = 0.7
            max_length = 50
        else:
            temperature = 0.9
        
        # Generate
        response = self.generator(
            prompt,
            max_length=max_length,
            temperature=temperature
        )[0]["generated_text"]
        
        # Optimize
        optimized = self.caaf.optimize_response(response, user_id)
        return optimized["optimized_response"]
```

## Best Practices

### 1. User ID Management

```python
import uuid
import hashlib

class UserIDManager:
    @staticmethod
    def generate_anonymous_id(session_id):
        """Generate consistent anonymous user ID from session"""
        return hashlib.sha256(session_id.encode()).hexdigest()[:16]
    
    @staticmethod
    def get_or_create_user_id(request):
        """Get user ID from auth or create anonymous one"""
        if request.user.is_authenticated:
            return f"user_{request.user.id}"
        else:
            session_id = request.session.session_key or str(uuid.uuid4())
            return UserIDManager.generate_anonymous_id(session_id)
```

### 2. Error Handling

```python
def safe_coherence_assessment(message, user_id, fallback_state="medium"):
    try:
        response = requests.post(
            "http://localhost:8000/assess-coherence",
            json={"message": message, "user_id": user_id},
            timeout=2.0  # 2 second timeout
        )
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, KeyError) as e:
        # Log error and return safe fallback
        logger.error(f"Coherence assessment failed: {e}")
        return {
            "state": fallback_state,
            "coherence_score": 0.5,
            "variables": {"psi": 0.5, "rho": 0.5, "q": 0.5, "f": 0.5}
        }
```

### 3. Caching

```python
from functools import lru_cache
from datetime import datetime, timedelta

class CoherenceCache:
    def __init__(self, ttl_seconds=300):  # 5 minute cache
        self.cache = {}
        self.ttl = timedelta(seconds=ttl_seconds)
    
    def get_or_assess(self, message, user_id, assess_func):
        cache_key = f"{user_id}:{hash(message)}"
        
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.ttl:
                return cached_data
        
        # Assess and cache
        result = assess_func(message, user_id)
        self.cache[cache_key] = (result, datetime.now())
        return result
```

### 4. Batch Processing

```python
async def process_messages_batch(messages, user_id):
    """Process multiple messages efficiently"""
    # Assess all messages
    assessments = await asyncio.gather(*[
        assess_coherence_async(msg, user_id) for msg in messages
    ])
    
    # Generate all responses
    responses = await generate_responses_batch(messages)
    
    # Optimize all responses
    optimized = await asyncio.gather(*[
        optimize_response_async(resp, user_id, assess)
        for resp, assess in zip(responses, assessments)
    ])
    
    return optimized
```

## Troubleshooting

### Common Issues

#### 1. API Connection Errors

**Problem:** Cannot connect to CAAF API

**Solution:**
```bash
# Check if API is running
curl http://localhost:8000/health

# Check logs
tail -f caaf.log

# Restart API
python -m src.api
```

#### 2. Slow Response Times

**Problem:** Integration adds significant latency

**Solutions:**
- Implement caching (see Best Practices)
- Use asynchronous patterns
- Deploy CAAF closer to your application
- Increase API worker count

#### 3. Inconsistent Coherence Assessments

**Problem:** Same message gets different coherence scores

**Solutions:**
- Ensure consistent user IDs
- Check for timing issues (assessments too close together)
- Review context data being passed

### Performance Optimization

```python
# Connection pooling
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

session = requests.Session()
retry = Retry(total=3, backoff_factor=0.3)
adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100)
session.mount("http://", adapter)

# Use session for all requests
response = session.post("http://localhost:8000/assess-coherence", json=data)
```

### Monitoring Integration

```python
import logging
from datetime import datetime

class CAFMonitor:
    def __init__(self):
        self.metrics = {
            "assessments": 0,
            "optimizations": 0,
            "errors": 0,
            "avg_latency": 0
        }
    
    def log_assessment(self, start_time, success=True):
        latency = (datetime.now() - start_time).total_seconds()
        self.metrics["assessments"] += 1
        if not success:
            self.metrics["errors"] += 1
        
        # Update rolling average
        self.metrics["avg_latency"] = (
            (self.metrics["avg_latency"] * (self.metrics["assessments"] - 1) + latency) 
            / self.metrics["assessments"]
        )
        
        logging.info(f"CAAF Metrics: {self.metrics}")
```

## Next Steps

1. **Test Integration**: Use the `/integration-test` endpoint to verify your setup
2. **Monitor Metrics**: Set up the Streamlit dashboard for real-time monitoring
3. **Customize**: Adjust optimization strategies based on your use case
4. **Scale**: Consider deploying CAAF as a microservice for production use

For more information, see:
- [API Reference](api_reference.md)
- [Coherence Theory](coherence_theory.md)
- [Examples](../examples/)