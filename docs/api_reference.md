# API Reference

## Overview

The Coherence-Aware AI Framework (CAAF) provides a REST API for integrating coherence awareness into any AI system. The API allows you to assess user coherence states and optimize AI responses accordingly.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. For production deployments, implement API key authentication by setting the `CAAF_API_KEY` environment variable.

## Endpoints

### Health Check

#### `GET /health`

Check if the API is running and all components are initialized.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "coherence_engine": true,
    "response_optimizer": true,
    "metrics_tracker": true
  }
}
```

### Assess Coherence

#### `POST /assess-coherence`

Assess the coherence state from a user message.

**Request Body:**
```json
{
  "message": "I'm feeling overwhelmed with everything lately",
  "user_id": "user123",
  "context": {
    "user_age": 30,
    "session_start": "2024-01-15T10:00:00Z"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Parameters:**
- `message` (required): The user's message to analyze
- `user_id` (required): Unique identifier for the user
- `context` (optional): Additional context information
- `timestamp` (optional): Message timestamp (defaults to current time)

**Response:**
```json
{
  "user_id": "user123",
  "timestamp": "2024-01-15T10:30:00Z",
  "coherence_score": 0.35,
  "state": "low",
  "confidence": 0.85,
  "variables": {
    "psi": 0.3,
    "rho": 0.4,
    "q": 0.35,
    "f": 0.25
  },
  "intervention_priorities": {
    "psi": 0.35,
    "f": 0.4,
    "q": 0.15,
    "rho": 0.1
  },
  "risk_factors": ["poor_internal_consistency", "social_isolation"],
  "protective_factors": ["accumulated_wisdom"]
}
```

**Response Fields:**
- `coherence_score`: Overall coherence score (0.0-1.0)
- `state`: Coherence state (crisis, low, medium, high, optimal)
- `confidence`: Confidence in the assessment (0.0-1.0)
- `variables`: Individual coherence variables
  - `psi`: Internal consistency (0.0-1.0)
  - `rho`: Accumulated wisdom (0.0-1.0)
  - `q`: Moral activation energy (0.0-1.0)
  - `f`: Social belonging architecture (0.0-1.0)
- `intervention_priorities`: Suggested focus areas for improvement
- `risk_factors`: Identified risk factors
- `protective_factors`: Identified protective factors

### Optimize Response

#### `POST /optimize-response`

Optimize an AI response based on the user's coherence state.

**Request Body:**
```json
{
  "original_response": "Have you tried making a to-do list to organize your tasks?",
  "user_id": "user123",
  "conversation_history": [
    {
      "role": "user",
      "content": "I'm feeling overwhelmed"
    },
    {
      "role": "assistant",
      "content": "I understand you're feeling overwhelmed"
    }
  ],
  "optimization_level": "automatic"
}
```

**Parameters:**
- `original_response` (required): The AI's original response to optimize
- `user_id` (required): User identifier for coherence lookup
- `conversation_history` (optional): Recent conversation context
- `optimization_level` (optional): Level of optimization (automatic, minimal, moderate, aggressive)

**Response:**
```json
{
  "optimized_response": "I hear that you're going through something difficult. Let's take this one step at a time. What feels most pressing right now?",
  "optimization_strategy": "grounding",
  "modifications_applied": ["simplified", "structure_enhanced", "social_support_added"],
  "coherence_state": {
    "state": "low",
    "score": 0.35,
    "variables": {
      "psi": 0.3,
      "rho": 0.4,
      "q": 0.35,
      "f": 0.25
    }
  },
  "safety_checks_passed": true
}
```

### Get Coherence Profile

#### `GET /coherence-profile/{user_id}`

Retrieve a user's coherence profile based on their history.

**Response:**
```json
{
  "user_id": "user123",
  "baseline_coherence": 0.55,
  "baseline_variables": {
    "psi": 0.5,
    "rho": 0.6,
    "q": 0.55,
    "f": 0.5
  },
  "volatility": 0.15,
  "trend": "improving",
  "risk_factors": ["high_volatility"],
  "protective_factors": ["accumulated_wisdom", "strong_moral_framework"],
  "assessment_count": 25,
  "last_updated": "2024-01-15T10:30:00Z"
}
```

### Get Coherence Metrics

#### `GET /coherence-metrics/{user_id}`

Retrieve detailed coherence metrics for analysis.

**Query Parameters:**
- `timeframe`: Time period to analyze (1h, 24h, 7d, 30d, all)
- `metric_type`: Type of metrics (summary, detailed, patterns)

**Example Request:**
```
GET /coherence-metrics/user123?timeframe=7d&metric_type=summary
```

**Response (Summary):**
```json
{
  "user_id": "user123",
  "timeframe": "7d",
  "status": "success",
  "metrics": {
    "avg_coherence": 0.58,
    "min_coherence": 0.25,
    "max_coherence": 0.85,
    "std_coherence": 0.18,
    "total_assessments": 156,
    "crisis_count": 5,
    "optimal_count": 12,
    "improvement_metrics": {
      "improvement_rate": 0.23,
      "coherence_change": 0.15,
      "stability_change": 0.08,
      "crisis_reduction": 0.6
    }
  }
}
```

### Integration Test

#### `POST /integration-test`

Test the CAAF integration with sample data.

**Request Body:**
```json
{
  "test_messages": [
    "I'm feeling lost",
    "Things are getting better",
    "I'm ready to tackle new challenges"
  ],
  "test_responses": [
    "Just stay positive!",
    "That's great to hear",
    "What specific challenges interest you?"
  ],
  "user_id": "test_user"
}
```

**Response:**
```json
{
  "user_id": "test_user",
  "assessments": [
    {
      "message_index": 0,
      "coherence_score": 0.3,
      "state": "low",
      "success": true
    }
  ],
  "optimizations": [
    {
      "response_index": 0,
      "original_length": 18,
      "optimized_length": 95,
      "changed": true,
      "success": true
    }
  ],
  "summary": {
    "total_assessments": 3,
    "successful_assessments": 3,
    "total_optimizations": 3,
    "successful_optimizations": 3,
    "responses_changed": 2
  },
  "overall_status": "success"
}
```

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200 OK`: Successful request
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

Error responses include a detail message:

```json
{
  "detail": "User ID cannot be empty"
}
```

## Rate Limiting

The API currently does not implement rate limiting. For production deployments, consider implementing rate limiting based on API keys or IP addresses.

## WebSocket Support

Future versions will include WebSocket support for real-time coherence monitoring at:

```
ws://localhost:8000/ws/coherence/{user_id}
```

## Examples

### Python Example

```python
import requests

# Assess coherence
response = requests.post(
    "http://localhost:8000/assess-coherence",
    json={
        "message": "I'm struggling today",
        "user_id": "user123"
    }
)
coherence_data = response.json()

# Optimize response
response = requests.post(
    "http://localhost:8000/optimize-response",
    json={
        "original_response": "Try harder!",
        "user_id": "user123"
    }
)
optimized_data = response.json()
print(optimized_data["optimized_response"])
```

### JavaScript Example

```javascript
// Assess coherence
const assessResponse = await fetch('http://localhost:8000/assess-coherence', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: "I'm struggling today",
    user_id: "user123"
  })
});
const coherenceData = await assessResponse.json();

// Optimize response
const optimizeResponse = await fetch('http://localhost:8000/optimize-response', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    original_response: "Try harder!",
    user_id: "user123"
  })
});
const optimizedData = await optimizeResponse.json();
console.log(optimizedData.optimized_response);
```

### cURL Example

```bash
# Assess coherence
curl -X POST http://localhost:8000/assess-coherence \
  -H "Content-Type: application/json" \
  -d '{"message": "I am struggling today", "user_id": "user123"}'

# Optimize response
curl -X POST http://localhost:8000/optimize-response \
  -H "Content-Type: application/json" \
  -d '{"original_response": "Try harder!", "user_id": "user123"}'
```