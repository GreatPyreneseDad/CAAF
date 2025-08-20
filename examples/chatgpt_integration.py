#!/usr/bin/env python3
"""
ChatGPT Integration Example for Coherence-Aware AI Framework

This example shows how to integrate CAAF with OpenAI's ChatGPT to create
coherence-aware conversations.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Optional
from datetime import datetime
import openai
from dotenv import load_dotenv

from src.coherence_engine import CoherenceEngine
from src.response_optimizer import ResponseOptimizer, CoherenceState
from src.metrics_tracker import MetricsTracker, CoherenceRecord

# Load environment variables
load_dotenv()


class CoherenceAwareChatGPT:
    """
    A wrapper around ChatGPT that adds coherence awareness.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize the coherence-aware ChatGPT wrapper.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: OpenAI model to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter.")
        
        openai.api_key = self.api_key
        self.model = model
        
        # Initialize CAAF components
        self.coherence_engine = CoherenceEngine()
        self.response_optimizer = ResponseOptimizer()
        self.metrics_tracker = MetricsTracker(database_url="sqlite:///chatgpt_coherence.db")
        
        # Conversation history
        self.conversations: Dict[str, List[Dict]] = {}
    
    def chat(self, 
             message: str, 
             user_id: str, 
             system_prompt: Optional[str] = None,
             temperature: float = 0.7,
             max_tokens: int = 500) -> Dict[str, any]:
        """
        Have a coherence-aware conversation with ChatGPT.
        
        Args:
            message: User's message
            user_id: Unique user identifier
            system_prompt: Optional system prompt
            temperature: ChatGPT temperature parameter
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
        
        # Step 3: Prepare messages for ChatGPT
        messages = self._prepare_messages(
            message, 
            history, 
            system_prompt, 
            assessment
        )
        
        # Step 4: Get ChatGPT response
        print("[ChatGPT] Generating response...")
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            original_response = response.choices[0].message.content
            
        except Exception as e:
            print(f"[ChatGPT] Error: {e}")
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
    
    def _prepare_messages(self, 
                         message: str, 
                         history: List[Dict],
                         system_prompt: Optional[str],
                         assessment) -> List[Dict]:
        """Prepare messages for ChatGPT with coherence context."""
        # Base system prompt
        if not system_prompt:
            system_prompt = (
                "You are a helpful, empathetic AI assistant. "
                "You adapt your communication style to best support the user's current state."
            )
        
        # Add coherence context to system prompt
        coherence_context = self._get_coherence_context(assessment)
        enhanced_prompt = f"{system_prompt}\n\n{coherence_context}"
        
        messages = [{"role": "system", "content": enhanced_prompt}]
        
        # Add recent history (last 6 messages)
        for msg in history[-6:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        return messages
    
    def _get_coherence_context(self, assessment) -> str:
        """Generate coherence context for system prompt."""
        state = assessment.state.value
        
        if state == "crisis":
            return (
                "IMPORTANT: The user appears to be in crisis. "
                "Be extremely gentle, supportive, and avoid complex explanations. "
                "Focus on immediate emotional support and grounding. "
                "Keep responses simple and compassionate."
            )
        elif state == "low":
            return (
                "The user seems to be struggling. "
                "Use simple, clear language and be extra supportive. "
                "Break down complex ideas into manageable pieces. "
                "Offer encouragement and validation."
            )
        elif state == "medium":
            return (
                "The user is in a moderate state. "
                "Balance support with gentle challenges for growth. "
                "Use clear explanations and check for understanding."
            )
        elif state == "high":
            return (
                "The user is in a good state and ready for deeper engagement. "
                "You can explore complex ideas and offer thoughtful challenges. "
                "Encourage reflection and growth."
            )
        else:  # optimal
            return (
                "The user is in an optimal state. "
                "Engage with sophisticated ideas and nuanced perspectives. "
                "Challenge thinking and explore philosophical depths."
            )
    
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
    
    def get_conversation_summary(self, user_id: str) -> Dict:
        """Get a summary of the conversation with coherence insights."""
        if user_id not in self.conversations:
            return {"error": "No conversation found for user"}
        
        # Get metrics
        metrics = self.metrics_tracker.get_user_metrics(user_id)
        improvement = self.metrics_tracker.calculate_improvement_metrics(user_id)
        patterns = self.metrics_tracker.get_pattern_analysis(user_id)
        
        return {
            "conversation_length": len(self.conversations[user_id]),
            "coherence_metrics": {
                "current_score": metrics['coherence_score'].iloc[-1] if not metrics.empty else None,
                "average_score": metrics['coherence_score'].mean() if not metrics.empty else None,
                "improvement_rate": improvement['improvement_rate'],
                "trend": patterns['patterns'].get('trend', 'unknown') if patterns['status'] != 'insufficient_data' else 'unknown'
            },
            "optimization_stats": {
                "responses_optimized": int(metrics['response_optimized'].sum()) if not metrics.empty and 'response_optimized' in metrics else 0,
                "total_responses": len([m for m in self.conversations[user_id] if m['role'] == 'assistant'])
            }
        }


def demo_conversation():
    """Run a demo conversation showing coherence awareness."""
    print("=== Coherence-Aware ChatGPT Demo ===\n")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        print("You can get an API key from https://platform.openai.com/api-keys")
        print("\nFor demo purposes, we'll simulate responses instead.")
        return demo_conversation_simulated()
    
    # Initialize coherence-aware ChatGPT
    ca_chatgpt = CoherenceAwareChatGPT(model="gpt-3.5-turbo")  # Using 3.5 for cost efficiency
    
    # Simulate a conversation with changing coherence states
    user_id = "demo_user_001"
    
    conversation_turns = [
        "I'm feeling really overwhelmed with everything going on in my life right now.",
        "Work has been stressful, and I haven't been sleeping well. I feel like I'm falling behind on everything.",
        "You're right, I should probably try to focus on one thing at a time. Maybe I'll start with organizing my work tasks.",
        "I made a list of my priorities and already feel a bit better. Thanks for the suggestion.",
        "Actually, I've been thinking about the bigger picture. What strategies do you recommend for long-term stress management?"
    ]
    
    print(f"Starting conversation with user: {user_id}\n")
    
    for i, user_message in enumerate(conversation_turns, 1):
        print(f"\n--- Turn {i} ---")
        print(f"User: {user_message}\n")
        
        # Get coherence-aware response
        result = ca_chatgpt.chat(user_message, user_id)
        
        print(f"Coherence State: {result['coherence_assessment']['state']} "
              f"(score: {result['coherence_assessment']['score']:.2f})")
        
        print(f"Response: {result['response']}")
        
        if result['was_optimized']:
            print(f"\n[Note: Response was optimized from original]")
            print(f"Original: {result['original_response'][:100]}...")
        
        print("\nCoherence Variables:")
        for var, value in result['coherence_assessment']['variables'].items():
            print(f"  {var}: {value:.2f}")
        
        input("\nPress Enter to continue...")
    
    # Show conversation summary
    print("\n=== Conversation Summary ===")
    summary = ca_chatgpt.get_conversation_summary(user_id)
    print(f"Total turns: {summary['conversation_length']}")
    print(f"Average coherence: {summary['coherence_metrics']['average_score']:.2f}")
    print(f"Improvement rate: {summary['coherence_metrics']['improvement_rate']:.2%}")
    print(f"Responses optimized: {summary['optimization_stats']['responses_optimized']}/{summary['optimization_stats']['total_responses']}")


def demo_conversation_simulated():
    """Simulated demo when OpenAI API key is not available."""
    print("\n[Running simulated demo without OpenAI API]\n")
    
    # Initialize CAAF components only
    coherence_engine = CoherenceEngine()
    response_optimizer = ResponseOptimizer()
    
    user_id = "demo_user_simulated"
    
    # Simulated conversation
    exchanges = [
        {
            "user": "I'm feeling really overwhelmed with everything going on in my life right now.",
            "ai_original": "You should try to organize your tasks better and maintain a positive attitude.",
            "expected_state": "low"
        },
        {
            "user": "Work has been stressful, and I haven't been sleeping well. I feel like I'm falling behind.",
            "ai_original": "Have you considered implementing a comprehensive time management system? You could use techniques like the Pomodoro method, Eisenhower matrix, and GTD methodology.",
            "expected_state": "low"
        },
        {
            "user": "You're right, I should probably try to focus on one thing at a time. Maybe I'll start with organizing my work tasks.",
            "ai_original": "That's a great approach! Starting with one area can help build momentum.",
            "expected_state": "medium"
        }
    ]
    
    for i, exchange in enumerate(exchanges, 1):
        print(f"\n--- Turn {i} ---")
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
            original_response=exchange['ai_original'],
            coherence_state=coherence_state,
            user_profile=profile
        )
        
        print(f"Coherence State: {assessment.state.value} "
              f"(score: {assessment.coherence_score:.2f})")
        print(f"Expected State: {exchange['expected_state']}")
        
        print(f"\nOriginal AI: {exchange['ai_original']}")
        print(f"\nOptimized AI: {optimized}")
        
        print("\nCoherence Variables:")
        print(f"  Ψ (Internal Consistency): {assessment.variables.psi:.2f}")
        print(f"  ρ (Wisdom): {assessment.variables.rho:.2f}")
        print(f"  q (Moral): {assessment.variables.q:.2f}")
        print(f"  f (Social): {assessment.variables.f:.2f}")


def api_integration_example():
    """Show how to integrate with ChatGPT using the CAAF API."""
    print("\n=== API Integration Example ===\n")
    
    print("To integrate CAAF with ChatGPT using the REST API:\n")
    
    print("1. Start the CAAF API server:")
    print("   python -m src.api\n")
    
    print("2. Use these endpoints in your ChatGPT integration:\n")
    
    print("```python")
    print("import requests")
    print("import openai")
    print()
    print("# Assess user coherence")
    print("coherence_response = requests.post(")
    print("    'http://localhost:8000/assess-coherence',")
    print("    json={")
    print("        'message': user_message,")
    print("        'user_id': user_id")
    print("    }")
    print(")")
    print()
    print("# Get ChatGPT response")
    print("chatgpt_response = openai.ChatCompletion.create(...)")
    print()
    print("# Optimize response based on coherence")
    print("optimized_response = requests.post(")
    print("    'http://localhost:8000/optimize-response',")
    print("    json={")
    print("        'original_response': chatgpt_response,")
    print("        'user_id': user_id")
    print("    }")
    print(")")
    print("```")


def main():
    """Run the ChatGPT integration examples."""
    print("COHERENCE-AWARE CHATGPT INTEGRATION\n")
    print("=" * 50)
    
    while True:
        print("\nSelect an option:")
        print("1. Run demo conversation")
        print("2. Show API integration example")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            demo_conversation()
        elif choice == "2":
            api_integration_example()
        elif choice == "3":
            print("\nExiting...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()