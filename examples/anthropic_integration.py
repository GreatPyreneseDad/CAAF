#!/usr/bin/env python3
"""
Anthropic (Claude) Integration Example for Coherence-Aware AI Framework

This example shows how to integrate CAAF with Anthropic's Claude API to create
coherence-aware conversations.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Optional
from datetime import datetime
import anthropic
from dotenv import load_dotenv

from src.coherence_engine import CoherenceEngine
from src.response_optimizer import ResponseOptimizer, CoherenceState
from src.metrics_tracker import MetricsTracker, CoherenceRecord

# Load environment variables
load_dotenv()


class CoherenceAwareClaude:
    """
    A wrapper around Claude that adds coherence awareness.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet-20240229"):
        """
        Initialize the coherence-aware Claude wrapper.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Claude model to use
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key parameter.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        
        # Initialize CAAF components
        self.coherence_engine = CoherenceEngine()
        self.response_optimizer = ResponseOptimizer()
        self.metrics_tracker = MetricsTracker(database_url="sqlite:///claude_coherence.db")
        
        # Conversation history
        self.conversations: Dict[str, List[Dict]] = {}
    
    def chat(self, 
             message: str, 
             user_id: str, 
             system_prompt: Optional[str] = None,
             temperature: float = 0.7,
             max_tokens: int = 500) -> Dict[str, any]:
        """
        Have a coherence-aware conversation with Claude.
        
        Args:
            message: User's message
            user_id: Unique user identifier
            system_prompt: Optional system prompt
            temperature: Claude temperature parameter
            max_tokens: Maximum response tokens
            
        Returns:
            Dict containing response and coherence information
        """
        # Step 1: Assess user's coherence
        print(f"[CAAF] Assessing coherence for user {user_id}...")
        assessment = self.coherence_engine.process_message(
            message=message,
            user_id=user_id
        )
        
        print(f"[CAAF] Coherence state: {assessment.state.value} (score: {assessment.coherence_score:.2f})")
        
        # Step 2: Get conversation history
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        history = self.conversations[user_id]
        
        # Step 3: Prepare prompt for Claude
        full_prompt = self._prepare_prompt(
            message, 
            history, 
            system_prompt, 
            assessment
        )
        
        # Step 4: Get Claude response
        print("[Claude] Generating response...")
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=self._get_system_message(assessment, system_prompt),
                messages=full_prompt
            )
            
            original_response = response.content[0].text
            
        except Exception as e:
            print(f"[Claude] Error: {e}")
            original_response = "I'm having trouble generating a response right now. Please try again."
        
        # Step 5: Optimize response based on coherence
        print("[CAAF] Optimizing response for coherence state...")
        
        coherence_state = CoherenceState(
            state=assessment.state,
            score=assessment.coherence_score,
            psi=assessment.variables.psi,
            rho=assessment.variables.rho,
            q=assessment.variables.q,
            f=assessment.variables.f,
            velocity=self.coherence_engine.get_coherence_velocity(user_id)
        )
        
        profile = self.coherence_engine.get_user_coherence_profile(user_id)
        
        optimized_response = self.response_optimizer.optimize_response(
            original_response=original_response,
            coherence_state=coherence_state,
            user_profile=profile
        )
        
        # Step 6: Record metrics
        self._record_interaction(
            user_id=user_id,
            assessment=assessment,
            original_response=original_response,
            optimized_response=optimized_response
        )
        
        # Step 7: Update conversation history
        self.conversations[user_id].append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat(),
            "coherence_score": assessment.coherence_score
        })
        
        self.conversations[user_id].append({
            "role": "assistant",
            "content": optimized_response,
            "timestamp": datetime.now().isoformat(),
            "optimized": optimized_response != original_response
        })
        
        # Keep only last 10 messages
        if len(self.conversations[user_id]) > 20:
            self.conversations[user_id] = self.conversations[user_id][-20:]
        
        return {
            "response": optimized_response,
            "original_response": original_response,
            "was_optimized": optimized_response != original_response,
            "coherence_assessment": {
                "score": assessment.coherence_score,
                "state": assessment.state.value,
                "variables": {
                    "psi": assessment.variables.psi,
                    "rho": assessment.variables.rho,
                    "q": assessment.variables.q,
                    "f": assessment.variables.f
                },
                "intervention_priorities": assessment.intervention_priorities
            },
            "user_profile": {
                "trend": profile.trend if profile else "unknown",
                "risk_factors": profile.risk_factors if profile else [],
                "protective_factors": profile.protective_factors if profile else []
            }
        }
    
    def _prepare_prompt(self, 
                       message: str, 
                       history: List[Dict],
                       system_prompt: Optional[str],
                       assessment) -> List[Dict]:
        """Prepare messages for Claude with coherence context."""
        messages = []
        
        # Add recent history (last 6 messages) for Claude format
        for msg in history[-6:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        return messages
    
    def _get_system_message(self, assessment, custom_prompt: Optional[str] = None) -> str:
        """Generate system message with coherence context."""
        # Base system prompt
        base_prompt = custom_prompt or (
            "You are a helpful, empathetic AI assistant. "
            "You adapt your communication style to best support the user's current state."
        )
        
        # Add coherence context
        coherence_context = self._get_coherence_context(assessment)
        
        return f"{base_prompt}\n\n{coherence_context}"
    
    def _get_coherence_context(self, assessment) -> str:
        """Generate coherence context for system prompt."""
        state = assessment.state.value
        
        contexts = {
            "crisis": (
                "CRITICAL: The user appears to be in crisis. "
                "Provide immediate emotional support and grounding. "
                "Use simple, compassionate language. "
                "Avoid complex explanations or challenging ideas. "
                "Focus on safety and stabilization."
            ),
            "low": (
                "The user is struggling with low coherence. "
                "Be extra supportive and validating. "
                "Use clear, simple language and break down complex ideas. "
                "Offer encouragement and practical, manageable suggestions. "
                "Help them feel heard and understood."
            ),
            "medium": (
                "The user is in a moderate coherence state. "
                "Balance emotional support with gentle opportunities for growth. "
                "Use clear language and check for understanding. "
                "You can introduce new ideas but keep them accessible."
            ),
            "high": (
                "The user is in a high coherence state. "
                "They're ready for deeper engagement and thoughtful challenges. "
                "You can explore complex ideas and nuanced perspectives. "
                "Encourage reflection and intellectual exploration."
            ),
            "optimal": (
                "The user is in an optimal coherence state. "
                "Engage with sophisticated concepts and philosophical depth. "
                "Challenge their thinking constructively. "
                "Explore multiple perspectives and embrace complexity. "
                "Foster wisdom and insight."
            )
        }
        
        base_context = contexts.get(state, contexts["medium"])
        
        # Add specific variable guidance
        variable_guidance = []
        
        if assessment.variables.psi < 0.4:
            variable_guidance.append(
                "Note: Internal consistency is low. "
                "Help structure their thoughts and provide clear frameworks."
            )
        
        if assessment.variables.f < 0.4:
            variable_guidance.append(
                "Note: Social connection appears low. "
                "Acknowledge their experiences and remind them they're not alone."
            )
        
        if assessment.variables.q < 0.4:
            variable_guidance.append(
                "Note: Moral engagement is low. "
                "Gently connect to their values and what matters to them."
            )
        
        if assessment.variables.rho > 0.7:
            variable_guidance.append(
                "Note: The user shows high wisdom. "
                "Acknowledge their insights and build on their understanding."
            )
        
        if variable_guidance:
            return f"{base_context}\n\n" + "\n".join(variable_guidance)
        
        return base_context
    
    def _record_interaction(self, 
                          user_id: str,
                          assessment,
                          original_response: str,
                          optimized_response: str):
        """Record the interaction metrics."""
        record = CoherenceRecord(
            user_id=user_id,
            timestamp=assessment.timestamp,
            psi=assessment.variables.psi,
            rho=assessment.variables.rho,
            q=assessment.variables.q,
            f=assessment.variables.f,
            coherence_score=assessment.coherence_score,
            coherence_velocity=self.coherence_engine.get_coherence_velocity(user_id),
            coherence_state=assessment.state.value,
            message_hash=assessment.message_hash,
            response_optimized=original_response != optimized_response
        )
        
        self.metrics_tracker.record_assessment(record)
    
    def get_conversation_insights(self, user_id: str) -> Dict:
        """Get detailed insights about the conversation."""
        if user_id not in self.conversations:
            return {"error": "No conversation found for user"}
        
        # Get comprehensive metrics
        metrics = self.metrics_tracker.get_user_metrics(user_id)
        improvement = self.metrics_tracker.calculate_improvement_metrics(user_id)
        patterns = self.metrics_tracker.get_pattern_analysis(user_id)
        profile = self.coherence_engine.get_user_coherence_profile(user_id)
        
        # Calculate optimization effectiveness
        if not metrics.empty and 'response_optimized' in metrics:
            optimization_rate = metrics['response_optimized'].mean()
        else:
            optimization_rate = 0
        
        return {
            "conversation": {
                "length": len(self.conversations[user_id]),
                "duration_minutes": self._calculate_conversation_duration(user_id)
            },
            "coherence": {
                "current_state": profile.trend if profile else "unknown",
                "baseline_score": profile.baseline_coherence if profile else 0.5,
                "volatility": profile.volatility if profile else 0.1,
                "improvement_rate": improvement['improvement_rate']
            },
            "optimization": {
                "rate": optimization_rate,
                "total_optimized": int(metrics['response_optimized'].sum()) if not metrics.empty and 'response_optimized' in metrics else 0
            },
            "risk_profile": {
                "risk_factors": profile.risk_factors if profile else [],
                "protective_factors": profile.protective_factors if profile else []
            },
            "recommendations": self._generate_recommendations(profile, patterns)
        }
    
    def _calculate_conversation_duration(self, user_id: str) -> float:
        """Calculate conversation duration in minutes."""
        if user_id not in self.conversations or len(self.conversations[user_id]) < 2:
            return 0
        
        messages = self.conversations[user_id]
        first_time = datetime.fromisoformat(messages[0]['timestamp'])
        last_time = datetime.fromisoformat(messages[-1]['timestamp'])
        
        return (last_time - first_time).total_seconds() / 60
    
    def _generate_recommendations(self, profile, patterns) -> List[str]:
        """Generate recommendations based on profile and patterns."""
        recommendations = []
        
        if not profile:
            return ["Continue conversations to build coherence profile"]
        
        if profile.trend == "declining":
            recommendations.append("Consider more frequent check-ins")
            recommendations.append("Focus on grounding and stabilization techniques")
        
        if "chronic_low_coherence" in profile.risk_factors:
            recommendations.append("Professional support may be beneficial")
            recommendations.append("Implement daily coherence-building practices")
        
        if "strong_social_support" in profile.protective_factors:
            recommendations.append("Leverage social connections for continued growth")
        
        if profile.volatility > 0.3:
            recommendations.append("Work on consistency and stability practices")
        
        return recommendations


def demo_conversation():
    """Run a demo conversation showing coherence awareness with Claude."""
    print("=== Coherence-Aware Claude Demo ===\n")
    
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: Please set ANTHROPIC_API_KEY environment variable")
        print("You can get an API key from https://console.anthropic.com/")
        print("\nFor demo purposes, we'll simulate responses instead.")
        return demo_conversation_simulated()
    
    # Initialize coherence-aware Claude
    ca_claude = CoherenceAwareClaude()
    
    # Simulate a conversation with evolving coherence
    user_id = "claude_demo_user"
    
    conversation_flow = [
        {
            "message": "I feel lost and disconnected from everything. Nothing makes sense anymore.",
            "context": "User expressing crisis-level distress"
        },
        {
            "message": "Sometimes I wonder if anyone would even notice if I wasn't here.",
            "context": "Continuing crisis state - needs immediate support"
        },
        {
            "message": "Thank you for listening. It helps to know someone understands.",
            "context": "Beginning to stabilize with support"
        },
        {
            "message": "I used to enjoy painting, but I haven't touched my brushes in months.",
            "context": "Reconnecting with past interests"
        },
        {
            "message": "Maybe I could try painting something small today, just to see how it feels.",
            "context": "Moving toward positive action"
        }
    ]
    
    print(f"Starting coherence-aware conversation with Claude\n")
    print(f"User ID: {user_id}\n")
    
    for i, turn in enumerate(conversation_flow, 1):
        print(f"\n{'='*60}")
        print(f"Turn {i}: {turn['context']}")
        print(f"{'='*60}\n")
        
        print(f"User: {turn['message']}\n")
        
        # Get coherence-aware response
        result = ca_claude.chat(turn['message'], user_id)
        
        # Display coherence assessment
        print(f"Coherence Assessment:")
        print(f"  State: {result['coherence_assessment']['state']}")
        print(f"  Score: {result['coherence_assessment']['score']:.2f}")
        print(f"  Variables: Ψ={result['coherence_assessment']['variables']['psi']:.2f}, "
              f"ρ={result['coherence_assessment']['variables']['rho']:.2f}, "
              f"q={result['coherence_assessment']['variables']['q']:.2f}, "
              f"f={result['coherence_assessment']['variables']['f']:.2f}")
        
        print(f"\nClaude's Response: {result['response']}")
        
        if result['was_optimized']:
            print(f"\n[Response was optimized for coherence state]")
            if len(result['original_response']) > 150:
                print(f"Original (truncated): {result['original_response'][:150]}...")
            else:
                print(f"Original: {result['original_response']}")
        
        # Show intervention priorities if in low state
        if result['coherence_assessment']['score'] < 0.4:
            print(f"\nIntervention Priorities:")
            for var, priority in result['coherence_assessment']['intervention_priorities'].items():
                if priority > 0.2:
                    print(f"  - {var}: {priority:.2f}")
        
        input("\nPress Enter to continue...")
    
    # Show conversation insights
    print(f"\n{'='*60}")
    print("Conversation Insights")
    print(f"{'='*60}\n")
    
    insights = ca_claude.get_conversation_insights(user_id)
    
    print(f"Conversation Summary:")
    print(f"  - Duration: {insights['conversation']['duration_minutes']:.1f} minutes")
    print(f"  - Messages: {insights['conversation']['length']}")
    print(f"  - Optimization rate: {insights['optimization']['rate']:.0%}")
    
    print(f"\nCoherence Journey:")
    print(f"  - Baseline: {insights['coherence']['baseline_score']:.2f}")
    print(f"  - Improvement: {insights['coherence']['improvement_rate']:.0%}")
    print(f"  - Current trend: {insights['coherence']['current_state']}")
    
    if insights['risk_profile']['risk_factors']:
        print(f"\nRisk Factors: {', '.join(insights['risk_profile']['risk_factors'])}")
    
    if insights['risk_profile']['protective_factors']:
        print(f"Protective Factors: {', '.join(insights['risk_profile']['protective_factors'])}")
    
    if insights['recommendations']:
        print(f"\nRecommendations:")
        for rec in insights['recommendations']:
            print(f"  • {rec}")


def demo_conversation_simulated():
    """Simulated demo when Anthropic API key is not available."""
    print("\n[Running simulated demo without Anthropic API]\n")
    
    # Initialize CAAF components only
    coherence_engine = CoherenceEngine()
    response_optimizer = ResponseOptimizer()
    
    user_id = "claude_demo_simulated"
    
    # Simulated conversation showing Claude's adaptive responses
    exchanges = [
        {
            "user": "I feel lost and disconnected from everything. Nothing makes sense anymore.",
            "claude_original": "I understand you're going through a difficult time. Let's explore what specific areas feel most confusing and work on strategies to address each one systematically.",
            "expected_state": "crisis",
            "context": "Crisis state - Claude's original response is too complex"
        },
        {
            "user": "Thank you for listening. It helps to know someone understands.",
            "claude_original": "You're welcome. Building on this foundation of understanding, what aspects of your situation would you like to explore further?",
            "expected_state": "low",
            "context": "Beginning to stabilize - needs continued support"
        },
        {
            "user": "I've been thinking about what gives my life meaning. It's a complex question.",
            "claude_original": "That's a profound question. Meaning often emerges from our relationships, values, and contributions. What areas resonate most with you?",
            "expected_state": "medium",
            "context": "Improving coherence - ready for deeper exploration"
        }
    ]
    
    for i, exchange in enumerate(exchanges, 1):
        print(f"\n{'='*60}")
        print(f"Turn {i}: {exchange['context']}")
        print(f"{'='*60}\n")
        
        print(f"User: {exchange['user']}\n")
        
        # Assess coherence
        assessment = coherence_engine.process_message(
            exchange['user'],
            user_id
        )
        
        # Create coherence state
        coherence_state = CoherenceState(
            state=assessment.state,
            score=assessment.coherence_score,
            psi=assessment.variables.psi,
            rho=assessment.variables.rho,
            q=assessment.variables.q,
            f=assessment.variables.f
        )
        
        # Optimize response
        profile = coherence_engine.get_user_coherence_profile(user_id)
        optimized = response_optimizer.optimize_response(
            original_response=exchange['claude_original'],
            coherence_state=coherence_state,
            user_profile=profile
        )
        
        print(f"Coherence Assessment:")
        print(f"  State: {assessment.state.value} (expected: {exchange['expected_state']})")
        print(f"  Score: {assessment.coherence_score:.2f}")
        print(f"  Variables: Ψ={assessment.variables.psi:.2f}, "
              f"ρ={assessment.variables.rho:.2f}, "
              f"q={assessment.variables.q:.2f}, "
              f"f={assessment.variables.f:.2f}")
        
        print(f"\nClaude (Original): {exchange['claude_original']}")
        print(f"\nClaude (Optimized): {optimized}")
        
        if optimized != exchange['claude_original']:
            print(f"\n[✓ Response optimized for {assessment.state.value} coherence state]")


def api_integration_example():
    """Show how to integrate with Claude using the CAAF API."""
    print("\n=== Claude API Integration Example ===\n")
    
    print("To integrate CAAF with Claude using the REST API:\n")
    
    example_code = '''```python
import requests
import anthropic

# Initialize Claude client
client = anthropic.Anthropic(api_key="your-api-key")

# Step 1: Assess user coherence
coherence_response = requests.post(
    'http://localhost:8000/assess-coherence',
    json={
        'message': user_message,
        'user_id': user_id
    }
)
coherence_data = coherence_response.json()

# Step 2: Create coherence-aware system prompt
coherence_state = coherence_data['state']
system_prompt = f"""You are a helpful AI assistant. 
Current user coherence state: {coherence_state}
Please adapt your response style accordingly."""

# Step 3: Get Claude response
response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=500,
    system=system_prompt,
    messages=[{"role": "user", "content": user_message}]
)
claude_response = response.content[0].text

# Step 4: Optimize response based on coherence
optimized_response = requests.post(
    'http://localhost:8000/optimize-response',
    json={
        'original_response': claude_response,
        'user_id': user_id
    }
)

final_response = optimized_response.json()['optimized_response']
```'''

    print(example_code)
    
    print("\n\nAdvanced Integration with Streaming:\n")
    
    streaming_example = '''```python
# For streaming responses with coherence awareness
async def stream_coherent_response(message, user_id):
    # Assess coherence first
    coherence = await assess_coherence(message, user_id)
    
    # Stream from Claude with coherence context
    stream = await client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=500,
        stream=True,
        system=get_coherence_system_prompt(coherence),
        messages=[{"role": "user", "content": message}]
    )
    
    # Collect full response for optimization
    full_response = ""
    async for chunk in stream:
        full_response += chunk.content
        # Could yield chunks here for real-time display
    
    # Optimize complete response
    optimized = await optimize_response(full_response, user_id)
    return optimized
```'''

    print(streaming_example)


def compare_models_example():
    """Show how different models adapt to coherence states."""
    print("\n=== Model Comparison: Coherence Adaptations ===\n")
    
    print("CAAF works with any LLM. Here's how different models might adapt:\n")
    
    comparison = """
    User in CRISIS state: "Everything is falling apart"
    
    Claude (Optimized):
    "I hear that you're going through something really difficult right now. 
    Let's take this one step at a time. Right now, focusing on this moment 
    might help. You don't have to go through this alone."
    
    GPT-4 (Optimized):
    "It sounds like things are really tough for you at the moment. I'm here 
    to listen and support you. Sometimes when things feel overwhelming, 
    starting with something small can help. What feels most pressing right now?"
    
    ---
    
    User in OPTIMAL state: "I've been reflecting on consciousness and emergence"
    
    Claude (Optimized):
    "Your reflection on consciousness and emergence touches on fascinating 
    territory. The relationship between simple rules and complex phenomena 
    raises profound questions about reductionism versus holism. Have you 
    considered how this might connect with your broader understanding? 
    What patterns do you notice emerging here?"
    
    GPT-4 (Optimized):
    "That's a sophisticated area of inquiry. The emergence of consciousness 
    from neural substrates exemplifies how complex systems can exhibit 
    properties not present in their components. How do you see this 
    connecting with other emergent phenomena you've observed?"
    """
    
    print(comparison)


def main():
    """Run the Claude integration examples."""
    print("COHERENCE-AWARE CLAUDE INTEGRATION\n")
    print("=" * 50)
    
    while True:
        print("\nSelect an option:")
        print("1. Run demo conversation")
        print("2. Show API integration example")
        print("3. Compare model adaptations")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            demo_conversation()
        elif choice == "2":
            api_integration_example()
        elif choice == "3":
            compare_models_example()
        elif choice == "4":
            print("\nExiting...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()