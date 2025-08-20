# Coherence-Aware AI Framework (CAAF)

A drop-in coherence measurement and response optimization system that transforms generic AI responses into personalized, coherence-enhancing interactions based on Grounded Coherence Theory.

## Overview

CAAF enables any LLM to:
- Assess user coherence state in real-time
- Optimize responses based on coherence levels
- Track coherence patterns over time
- Prevent harmful interactions during low coherence states

## Quick Start

```python
from caaf import CoherenceEngine, ResponseOptimizer

# Initialize the framework
engine = CoherenceEngine()
optimizer = ResponseOptimizer()

# Assess coherence from user message
coherence = engine.process_message(
    message="I'm feeling overwhelmed with everything lately",
    user_id="user123"
)

# Optimize AI response based on coherence state
optimized_response = optimizer.optimize_response(
    original_response="Have you tried making a to-do list?",
    coherence_state=coherence.state,
    user_profile=coherence.profile
)
```

## Installation

```bash
pip install coherence-aware-ai
```

## Core Components

### 1. GCT Core Engine
Mathematical implementation of Grounded Coherence Theory with four key variables:
- **Ψ (Psi)**: Internal Consistency (0.0-1.0)
- **ρ (Rho)**: Accumulated Wisdom (0.0-1.0)
- **q**: Moral Activation Energy (0.0-1.0)
- **f**: Social Belonging Architecture (0.0-1.0)

### 2. Coherence Engine
Real-time assessment of user coherence from conversation data.

### 3. Response Optimizer
Modifies AI responses based on user's coherence state:
- High coherence: Complex ideas, challenging discussions
- Medium coherence: Balanced support and growth
- Low coherence: Grounding, simplification, emotional support
- Crisis mode: Immediate stabilization protocols

### 4. Metrics Tracker
Stores and analyzes coherence patterns over time while maintaining privacy.

### 5. REST API
Simple integration endpoints for any platform.

## Integration Examples

### ChatGPT Integration
```python
import openai
from caaf import CoherenceEngine, ResponseOptimizer

engine = CoherenceEngine()
optimizer = ResponseOptimizer()

def coherence_aware_chat(user_message, user_id):
    # Assess coherence
    coherence = engine.process_message(user_message, user_id)
    
    # Get ChatGPT response
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": user_message}]
    )
    
    # Optimize for coherence
    return optimizer.optimize_response(
        response.choices[0].message.content,
        coherence.state,
        coherence.profile
    )
```

### API Integration
```bash
# Assess coherence
curl -X POST http://localhost:8000/assess-coherence \
  -H "Content-Type: application/json" \
  -d '{"message": "I'm struggling today", "user_id": "user123"}'

# Optimize response
curl -X POST http://localhost:8000/optimize-response \
  -H "Content-Type: application/json" \
  -d '{"original_response": "Try harder", "user_id": "user123"}'
```

## Dashboard

Access the monitoring dashboard at `http://localhost:8501` after running:
```bash
streamlit run dashboard/app.py
```

## Key Features

- **Real-time Assessment**: <500ms response time
- **Privacy-First**: No raw message storage
- **Platform Agnostic**: Works with any LLM
- **Research-Backed**: Based on Grounded Coherence Theory
- **Measurable Impact**: Demonstrable improvement in user outcomes

## Documentation

- [API Reference](docs/api_reference.md)
- [Integration Guide](docs/integration_guide.md)  
- [Coherence Theory](docs/coherence_theory.md)

## License

MIT License - See LICENSE for details

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Research

Based on Grounded Coherence Theory. For academic citations and research papers, see [docs/research](docs/research).