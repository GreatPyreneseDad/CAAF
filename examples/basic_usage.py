#!/usr/bin/env python3
"""
Basic usage example for the Coherence-Aware AI Framework (CAAF)

This example demonstrates how to use CAAF to assess coherence and optimize
AI responses based on user state.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from src.coherence_engine import CoherenceEngine
from src.response_optimizer import ResponseOptimizer, CoherenceState
from src.metrics_tracker import MetricsTracker, CoherenceRecord
from src.gct_core import GCTEngine


def basic_coherence_assessment():
    """Demonstrate basic coherence assessment from text."""
    print("=== Basic Coherence Assessment ===\n")
    
    # Initialize engine
    engine = CoherenceEngine()
    
    # Example messages representing different coherence states
    messages = [
        {
            "text": "I'm feeling really overwhelmed lately. Everything seems to be falling apart and I can't focus on anything. I don't know what to do anymore.",
            "user_id": "user_001",
            "expected_state": "low"
        },
        {
            "text": "I've been reflecting on my experiences this past year, and I realize how much I've grown. The challenges taught me valuable lessons about resilience and perspective.",
            "user_id": "user_002",
            "expected_state": "high"
        },
        {
            "text": "Today was productive. I completed my tasks, had a good meeting with the team, and I'm looking forward to the weekend plans with friends.",
            "user_id": "user_003",
            "expected_state": "medium-high"
        }
    ]
    
    for msg_data in messages:
        print(f"User: {msg_data['user_id']}")
        print(f"Message: \"{msg_data['text']}\"\n")
        
        # Assess coherence
        assessment = engine.process_message(
            message=msg_data['text'],
            user_id=msg_data['user_id']
        )
        
        print(f"Coherence Score: {assessment.coherence_score:.2f}")
        print(f"State: {assessment.state.value}")
        print(f"Variables:")
        print(f"  - Ψ (Internal Consistency): {assessment.variables.psi:.2f}")
        print(f"  - ρ (Accumulated Wisdom): {assessment.variables.rho:.2f}")
        print(f"  - q (Moral Activation): {assessment.variables.q:.2f}")
        print(f"  - f (Social Belonging): {assessment.variables.f:.2f}")
        print(f"Confidence: {assessment.confidence:.2f}")
        print(f"Expected State: {msg_data['expected_state']}")
        print("-" * 50 + "\n")


def response_optimization_demo():
    """Demonstrate response optimization based on coherence state."""
    print("\n=== Response Optimization Demo ===\n")
    
    # Initialize components
    engine = CoherenceEngine()
    optimizer = ResponseOptimizer()
    
    # Simulate a conversation
    user_id = "demo_user"
    
    # User in crisis state
    crisis_message = "I can't do this anymore. Everything is too much and I feel like giving up."
    
    # Assess coherence
    assessment = engine.process_message(crisis_message, user_id)
    
    # Create coherence state object
    coherence_state = CoherenceState(
        state=assessment.state,
        score=assessment.coherence_score,
        psi=assessment.variables.psi,
        rho=assessment.variables.rho,
        q=assessment.variables.q,
        f=assessment.variables.f
    )
    
    # Original AI responses (generic)
    generic_responses = [
        "Have you tried making a to-do list to organize your tasks?",
        "Things will get better if you just stay positive and work hard.",
        "You should consider the following strategies for improving your situation: First, analyze the root causes of your problems. Second, develop a comprehensive action plan. Third, implement systematic changes to your daily routine.",
    ]
    
    print(f"User state: {coherence_state.state.value} (score: {coherence_state.score:.2f})\n")
    
    for i, original in enumerate(generic_responses, 1):
        print(f"Example {i}:")
        print(f"Original: {original}")
        
        # Optimize for user's state
        optimized = optimizer.optimize_response(
            original_response=original,
            coherence_state=coherence_state
        )
        
        print(f"Optimized: {optimized}")
        print("-" * 80 + "\n")


def tracking_metrics_example():
    """Demonstrate metrics tracking and analysis."""
    print("\n=== Metrics Tracking Example ===\n")
    
    # Initialize tracker (uses in-memory SQLite by default)
    tracker = MetricsTracker(database_url="sqlite:///demo_metrics.db")
    
    # Simulate coherence journey over time
    user_id = "tracking_demo_user"
    
    # Create sample coherence records
    from datetime import timedelta
    base_time = datetime.now() - timedelta(hours=5)
    
    # Simulate improving coherence over time
    coherence_journey = [
        (0.25, "crisis"),    # Hour 0: Crisis
        (0.30, "low"),       # Hour 1: Slight improvement
        (0.35, "low"),       # Hour 2: Continued improvement
        (0.45, "medium"),    # Hour 3: Breaking through
        (0.55, "medium"),    # Hour 4: Stabilizing
        (0.65, "high"),      # Hour 5: Good progress
    ]
    
    for hour, (score, state) in enumerate(coherence_journey):
        record = CoherenceRecord(
            user_id=user_id,
            timestamp=base_time + timedelta(hours=hour),
            psi=score + 0.05,  # Slightly higher internal consistency
            rho=score,
            q=score - 0.05,    # Slightly lower moral activation
            f=score + 0.1,     # Higher social connection
            coherence_score=score,
            coherence_velocity=0.05 if hour > 0 else 0,
            coherence_state=state,
            response_optimized=hour > 2  # Started optimizing after hour 2
        )
        tracker.record_assessment(record)
    
    # Analyze improvement
    improvement = tracker.calculate_improvement_metrics(user_id, window_days=1)
    
    print("Coherence Journey Analysis:")
    print(f"Starting coherence: {coherence_journey[0][0]:.2f} ({coherence_journey[0][1]})")
    print(f"Current coherence: {coherence_journey[-1][0]:.2f} ({coherence_journey[-1][1]})")
    print(f"Improvement rate: {improvement['improvement_rate']:.2%}")
    print(f"Coherence change: {improvement['coherence_change']:.2%}")
    print(f"Crisis reduction: {improvement['crisis_reduction']:.2%}")
    
    # Get pattern analysis
    patterns = tracker.get_pattern_analysis(user_id, window_days=1)
    if patterns['status'] != 'insufficient_data':
        print(f"\nIntervention effectiveness: {patterns['patterns']['intervention_effectiveness']:.2%}")


def integrated_workflow_example():
    """Demonstrate complete integrated workflow."""
    print("\n=== Integrated Workflow Example ===\n")
    
    # Initialize all components
    gct_engine = GCTEngine()
    coherence_engine = CoherenceEngine()
    response_optimizer = ResponseOptimizer()
    metrics_tracker = MetricsTracker(database_url="sqlite:///integrated_demo.db")
    
    # Simulate a user conversation
    user_id = "integrated_user"
    
    conversation = [
        {
            "user": "I've been struggling with anxiety about my job. The pressure is getting to me.",
            "ai": "I understand work pressure can be challenging. What specific aspects are causing the most anxiety?"
        },
        {
            "user": "Everything feels uncertain. I don't know if I'm good enough, and my manager seems disappointed.",
            "ai": "It sounds like you're dealing with self-doubt and concerns about how others perceive you."
        },
        {
            "user": "Yes, exactly. I used to be confident, but lately I question every decision I make.",
            "ai": "This shift from confidence to self-doubt must be difficult. Have you noticed when this change began?"
        }
    ]
    
    print("Processing conversation...\n")
    
    for turn in conversation:
        # Assess user's coherence
        assessment = coherence_engine.process_message(
            turn["user"],
            user_id
        )
        
        print(f"User: \"{turn['user']}\"")
        print(f"Coherence: {assessment.coherence_score:.2f} ({assessment.state.value})")
        
        # Create coherence state
        coherence_state = CoherenceState(
            state=assessment.state,
            score=assessment.coherence_score,
            psi=assessment.variables.psi,
            rho=assessment.variables.rho,
            q=assessment.variables.q,
            f=assessment.variables.f
        )
        
        # Optimize AI response
        profile = coherence_engine.get_user_coherence_profile(user_id)
        optimized_response = response_optimizer.optimize_response(
            original_response=turn["ai"],
            coherence_state=coherence_state,
            user_profile=profile
        )
        
        print(f"Original AI: \"{turn['ai']}\"")
        print(f"Optimized AI: \"{optimized_response}\"")
        
        # Record metrics
        record = CoherenceRecord(
            user_id=user_id,
            timestamp=datetime.now(),
            psi=assessment.variables.psi,
            rho=assessment.variables.rho,
            q=assessment.variables.q,
            f=assessment.variables.f,
            coherence_score=assessment.coherence_score,
            coherence_velocity=0.0,
            coherence_state=assessment.state.value,
            response_optimized=True
        )
        metrics_tracker.record_assessment(record)
        
        print("-" * 80 + "\n")
    
    # Show final profile
    final_profile = coherence_engine.get_user_coherence_profile(user_id)
    print("Final User Profile:")
    print(f"Baseline coherence: {final_profile.baseline_coherence:.2f}")
    print(f"Trend: {final_profile.trend}")
    print(f"Risk factors: {final_profile.risk_factors}")
    print(f"Protective factors: {final_profile.protective_factors}")


def mathematical_foundations_demo():
    """Demonstrate the mathematical foundations of GCT."""
    print("\n=== Mathematical Foundations Demo ===\n")
    
    engine = GCTEngine()
    
    # Example coherence variable sets
    scenarios = [
        {
            "name": "Optimal State",
            "psi": 0.9,  # High internal consistency
            "rho": 0.8,  # Strong wisdom
            "q": 0.85,   # Active moral framework
            "f": 0.9     # Strong social connections
        },
        {
            "name": "Crisis State",
            "psi": 0.2,  # Low internal consistency
            "rho": 0.3,  # Limited wisdom application
            "q": 0.1,    # Moral disengagement
            "f": 0.15    # Social isolation
        },
        {
            "name": "Growth State",
            "psi": 0.6,  # Moderate consistency
            "rho": 0.7,  # Good wisdom
            "q": 0.5,    # Developing moral framework
            "f": 0.6     # Building connections
        }
    ]
    
    print("Coherence Equation: C = Ψ + (ρ × Ψ) + q + (f × Ψ)\n")
    
    for scenario in scenarios:
        variables = engine.CoherenceVariables(
            psi=scenario["psi"],
            rho=scenario["rho"],
            q=scenario["q"],
            f=scenario["f"]
        )
        
        coherence = engine.calculate_coherence(variables)
        
        print(f"{scenario['name']}:")
        print(f"  Ψ={scenario['psi']:.1f}, ρ={scenario['rho']:.1f}, q={scenario['q']:.1f}, f={scenario['f']:.1f}")
        print(f"  Coherence Score: {coherence:.3f}")
        print(f"  State: {engine.get_coherence_state_label(coherence)}")
        
        # Show component contributions
        psi_contrib = scenario['psi']
        rho_contrib = scenario['rho'] * scenario['psi']
        q_contrib = scenario['q']
        f_contrib = scenario['f'] * scenario['psi']
        
        print(f"  Component contributions:")
        print(f"    - Ψ contribution: {psi_contrib:.3f}")
        print(f"    - ρ×Ψ contribution: {rho_contrib:.3f}")
        print(f"    - q contribution: {q_contrib:.3f}")
        print(f"    - f×Ψ contribution: {f_contrib:.3f}")
        print()


def main():
    """Run all demonstrations."""
    print("COHERENCE-AWARE AI FRAMEWORK - DEMONSTRATION\n")
    print("=" * 80)
    
    try:
        # Run demonstrations
        basic_coherence_assessment()
        response_optimization_demo()
        tracking_metrics_example()
        integrated_workflow_example()
        mathematical_foundations_demo()
        
        print("\n" + "=" * 80)
        print("Demonstration complete!")
        print("\nNext steps:")
        print("1. Run the API server: python -m src.api")
        print("2. Try the integration examples: python examples/chatgpt_integration.py")
        print("3. Launch the dashboard: streamlit run dashboard/app.py")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()