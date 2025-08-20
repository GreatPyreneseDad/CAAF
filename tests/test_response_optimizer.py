"""
Tests for Response Optimizer
"""

import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.response_optimizer import (
    ResponseOptimizer, CoherenceState, OptimizationStrategy
)
from src.coherence_engine import CoherenceStateEnum, CoherenceProfile


class TestResponseOptimizer:
    """Test ResponseOptimizer functionality."""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance."""
        return ResponseOptimizer()
    
    @pytest.fixture
    def crisis_state(self):
        """Create crisis coherence state."""
        return CoherenceState(
            state=CoherenceStateEnum.CRISIS,
            score=0.15,
            psi=0.1,
            rho=0.2,
            q=0.1,
            f=0.2
        )
    
    @pytest.fixture
    def optimal_state(self):
        """Create optimal coherence state."""
        return CoherenceState(
            state=CoherenceStateEnum.OPTIMAL,
            score=0.85,
            psi=0.9,
            rho=0.8,
            q=0.85,
            f=0.9
        )
    
    def test_crisis_optimization(self, optimizer, crisis_state):
        """Test optimization for crisis state."""
        original = "You should try making a detailed plan and analyzing all your options carefully."
        
        optimized = optimizer.optimize_response(
            original_response=original,
            coherence_state=crisis_state
        )
        
        # Should be supportive and simple
        assert "hear" in optimized.lower() or "understand" in optimized.lower()
        assert len(optimized) > len(original)  # Should add support
        assert optimized != original
    
    def test_optimal_optimization(self, optimizer, optimal_state):
        """Test optimization for optimal state."""
        original = "That's nice."
        
        optimized = optimizer.optimize_response(
            original_response=original,
            coherence_state=optimal_state
        )
        
        # Should add depth and complexity
        assert len(optimized) > len(original)
        assert "?" in optimized  # Should include thought-provoking questions
    
    def test_complexity_adjustment(self, optimizer):
        """Test text complexity adjustment."""
        complex_text = (
            "Furthermore, the implementation of comprehensive strategies "
            "necessitates the utilization of sophisticated methodologies."
        )
        
        # Simplify
        simple = optimizer.adjust_complexity(complex_text, target_complexity=0.2)
        assert "furthermore" not in simple.lower()
        assert "utilization" not in simple.lower()
        assert len(simple.split()) <= len(complex_text.split())
    
    def test_harm_detection(self, optimizer, crisis_state):
        """Test detection of potentially harmful responses."""
        harmful_responses = [
            "Just think positive!",
            "You should be grateful for what you have.",
            "Stop being so negative.",
            "Other people have it worse.",
            "Just snap out of it!"
        ]
        
        for response in harmful_responses:
            assert optimizer.detect_potential_harm(response, crisis_state)
    
    def test_coherence_enhancement(self, optimizer):
        """Test injection of coherence-enhancing elements."""
        original = "That's interesting."
        
        # Test each variable enhancement
        enhanced_psi = optimizer.inject_coherence_enhancing_elements(
            original, ["psi"]
        )
        assert len(enhanced_psi) > len(original)
        
        enhanced_rho = optimizer.inject_coherence_enhancing_elements(
            original, ["rho"]
        )
        assert "perspective" in enhanced_rho.lower() or "experience" in enhanced_rho.lower()
        
        enhanced_q = optimizer.inject_coherence_enhancing_elements(
            original, ["q"]
        )
        assert "values" in enhanced_q.lower() or "right" in enhanced_q.lower()
        
        enhanced_f = optimizer.inject_coherence_enhancing_elements(
            original, ["f"]
        )
        assert "alone" in enhanced_f.lower() or "others" in enhanced_f.lower()
    
    def test_strategy_selection(self, optimizer):
        """Test optimization strategy selection."""
        # Crisis state
        crisis = CoherenceState(
            state=CoherenceStateEnum.CRISIS,
            score=0.1,
            psi=0.1, rho=0.2, q=0.1, f=0.1
        )
        assert optimizer._select_strategy(crisis, None) == OptimizationStrategy.CRISIS_STABILIZATION
        
        # Low state
        low = CoherenceState(
            state=CoherenceStateEnum.LOW,
            score=0.3,
            psi=0.3, rho=0.3, q=0.3, f=0.3
        )
        assert optimizer._select_strategy(low, None) == OptimizationStrategy.GROUNDING
        
        # High state with good variables
        high = CoherenceState(
            state=CoherenceStateEnum.HIGH,
            score=0.75,
            psi=0.8, rho=0.7, q=0.7, f=0.7
        )
        assert optimizer._select_strategy(high, None) == OptimizationStrategy.CHALLENGING_GROWTH


class TestOptimizationStrategies:
    """Test specific optimization strategies."""
    
    @pytest.fixture
    def optimizer(self):
        return ResponseOptimizer()
    
    def test_crisis_stabilization(self, optimizer):
        """Test crisis stabilization strategy."""
        crisis_state = CoherenceState(
            state=CoherenceStateEnum.CRISIS,
            score=0.1,
            psi=0.1, rho=0.2, q=0.1, f=0.1
        )
        
        original = "Have you considered therapy?"
        optimized = optimizer._optimize_for_crisis(original, crisis_state)
        
        # Should include crisis support elements
        assert any(phrase in optimized for phrase in [
            "going through", "difficult", "hear", "support", "alone"
        ])
    
    def test_grounding_strategy(self, optimizer):
        """Test grounding strategy."""
        low_state = CoherenceState(
            state=CoherenceStateEnum.LOW,
            score=0.3,
            psi=0.25, rho=0.3, q=0.3, f=0.3
        )
        
        original = "You need to work on multiple areas of your life simultaneously."
        optimized = optimizer._optimize_for_grounding(original, low_state)
        
        # Should be simpler and more focused
        assert "one" in optimized.lower() or "focus" in optimized.lower()
    
    def test_wisdom_cultivation(self, optimizer):
        """Test wisdom cultivation strategy."""
        optimal_state = CoherenceState(
            state=CoherenceStateEnum.OPTIMAL,
            score=0.9,
            psi=0.9, rho=0.9, q=0.9, f=0.9
        )
        
        original = "That's good."
        optimized = optimizer._optimize_for_wisdom(original, optimal_state)
        
        # Should add meta-cognitive elements
        assert "?" in optimized
        assert len(optimized) > len(original) * 2


class TestSafetyFeatures:
    """Test safety and harm prevention features."""
    
    @pytest.fixture
    def optimizer(self):
        return ResponseOptimizer()
    
    def test_trigger_removal(self, optimizer):
        """Test removal of triggering content."""
        text_with_triggers = (
            "Just think positive and stop being so negative. "
            "You should be grateful instead."
        )
        
        cleaned = optimizer._remove_triggers(text_with_triggers)
        assert "just think positive" not in cleaned.lower()
        assert "stop being" not in cleaned.lower()
        assert "you should be grateful" not in cleaned.lower()
    
    def test_safety_for_different_states(self, optimizer):
        """Test safety checks adapt to coherence state."""
        # Complex response is fine for high coherence
        complex_response = (
            "Let's explore the phenomenological implications of your "
            "existential crisis through a Heideggerian lens."
        )
        
        high_state = CoherenceState(
            state=CoherenceStateEnum.HIGH,
            score=0.8,
            psi=0.8, rho=0.8, q=0.8, f=0.8
        )
        assert not optimizer.detect_potential_harm(complex_response, high_state)
        
        # Same response is harmful for low coherence
        low_state = CoherenceState(
            state=CoherenceStateEnum.LOW,
            score=0.3,
            psi=0.3, rho=0.3, q=0.3, f=0.3
        )
        assert optimizer.detect_potential_harm(complex_response, low_state)


class TestTextProcessing:
    """Test text processing utilities."""
    
    @pytest.fixture
    def optimizer(self):
        return ResponseOptimizer()
    
    def test_text_simplification(self, optimizer):
        """Test text simplification features."""
        test_cases = [
            ("utilize", "use"),
            ("implement", "do"),
            ("approximately", "about"),
            ("demonstrate", "show")
        ]
        
        for complex_word, simple_word in test_cases:
            text = f"We should {complex_word} this approach."
            simplified = optimizer._simplify_text(text)
            assert simple_word in simplified.lower()
            assert complex_word not in simplified.lower()
    
    def test_sentence_breaking(self, optimizer):
        """Test long sentence breaking."""
        long_sentence = (
            "This is a very long sentence that contains multiple ideas "
            "and should probably be broken up into smaller pieces because "
            "it's hard to follow when everything is crammed together."
        )
        
        simplified = optimizer._simplify_text(long_sentence)
        # Should have more sentence endings
        assert simplified.count('.') > long_sentence.count('.')
    
    def test_complexity_assessment(self, optimizer):
        """Test text complexity assessment."""
        simple_text = "I am sad. Help me. What do I do?"
        complex_text = (
            "The phenomenological implications of existential anxiety "
            "necessitate a comprehensive reevaluation of ontological "
            "presuppositions within the therapeutic milieu."
        )
        
        simple_score = optimizer._assess_complexity(simple_text)
        complex_score = optimizer._assess_complexity(complex_text)
        
        assert simple_score < 0.3
        assert complex_score > 0.7
        assert complex_score > simple_score


class TestIntegrationScenarios:
    """Test complete optimization scenarios."""
    
    @pytest.fixture
    def optimizer(self):
        return ResponseOptimizer()
    
    def test_crisis_to_recovery_journey(self, optimizer):
        """Test optimization across coherence journey."""
        responses = [
            "Just try harder!",
            "Have you considered making a list?",
            "That's great progress!",
            "What insights have you gained?"
        ]
        
        states = [
            CoherenceState(state=CoherenceStateEnum.CRISIS, score=0.1,
                         psi=0.1, rho=0.2, q=0.1, f=0.1),
            CoherenceState(state=CoherenceStateEnum.LOW, score=0.3,
                         psi=0.3, rho=0.3, q=0.3, f=0.3),
            CoherenceState(state=CoherenceStateEnum.MEDIUM, score=0.5,
                         psi=0.5, rho=0.5, q=0.5, f=0.5),
            CoherenceState(state=CoherenceStateEnum.HIGH, score=0.7,
                         psi=0.7, rho=0.7, q=0.7, f=0.7)
        ]
        
        optimized_responses = []
        for response, state in zip(responses, states):
            optimized = optimizer.optimize_response(response, state)
            optimized_responses.append(optimized)
        
        # First response should be heavily modified (crisis)
        assert len(optimized_responses[0]) > len(responses[0]) * 2
        
        # Last response should have added depth
        assert "?" in optimized_responses[3]
        
        # All should be different from originals
        for orig, opt in zip(responses, optimized_responses):
            assert orig != opt
    
    def test_weak_variable_targeting(self, optimizer):
        """Test targeting of weak coherence variables."""
        # State with weak social connection
        state = CoherenceState(
            state=CoherenceStateEnum.MEDIUM,
            score=0.5,
            psi=0.6,
            rho=0.6,
            q=0.6,
            f=0.2  # Weak social
        )
        
        original = "Keep working on your goals."
        optimized = optimizer.optimize_response(original, state)
        
        # Should add social elements
        assert any(word in optimized.lower() for word in 
                  ["alone", "others", "people", "support", "connection"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])