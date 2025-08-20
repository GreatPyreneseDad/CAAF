"""
Tests for GCT Core Engine
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gct_core import GCTEngine, CoherenceVariables


class TestCoherenceVariables:
    """Test CoherenceVariables dataclass."""
    
    def test_valid_initialization(self):
        """Test creating variables with valid values."""
        vars = CoherenceVariables(psi=0.5, rho=0.6, q=0.7, f=0.8)
        assert vars.psi == 0.5
        assert vars.rho == 0.6
        assert vars.q == 0.7
        assert vars.f == 0.8
        assert isinstance(vars.timestamp, datetime)
    
    def test_invalid_values(self):
        """Test that invalid values raise errors."""
        with pytest.raises(ValueError):
            CoherenceVariables(psi=1.5, rho=0.5, q=0.5, f=0.5)
        
        with pytest.raises(ValueError):
            CoherenceVariables(psi=-0.1, rho=0.5, q=0.5, f=0.5)
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        vars = CoherenceVariables(psi=0.5, rho=0.6, q=0.7, f=0.8)
        d = vars.to_dict()
        
        assert d['psi'] == 0.5
        assert d['rho'] == 0.6
        assert d['q'] == 0.7
        assert d['f'] == 0.8
        assert 'timestamp' in d


class TestGCTEngine:
    """Test GCT Engine functionality."""
    
    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return GCTEngine()
    
    def test_calculate_coherence(self, engine):
        """Test coherence calculation."""
        # Test with all zeros
        vars = CoherenceVariables(psi=0, rho=0, q=0, f=0)
        assert engine.calculate_coherence(vars) == 0
        
        # Test with all ones
        vars = CoherenceVariables(psi=1, rho=1, q=1, f=1)
        expected = (1 + 1 + 1 + 1) / 4.0  # Normalized
        assert engine.calculate_coherence(vars) == expected
        
        # Test intermediate values
        vars = CoherenceVariables(psi=0.5, rho=0.5, q=0.5, f=0.5)
        coherence = engine.calculate_coherence(vars)
        assert 0 <= coherence <= 1
        
        # Test specific calculation
        vars = CoherenceVariables(psi=0.6, rho=0.7, q=0.5, f=0.8)
        expected = (0.6 + (0.7 * 0.6) + 0.5 + (0.8 * 0.6)) / 4.0
        assert abs(engine.calculate_coherence(vars) - expected) < 0.001
    
    def test_calculate_coherence_velocity(self, engine):
        """Test coherence velocity calculation."""
        # Test with single point (should return 0)
        history = [CoherenceVariables(psi=0.5, rho=0.5, q=0.5, f=0.5)]
        assert engine.calculate_coherence_velocity(history) == 0
        
        # Test with two points
        t1 = datetime.now()
        t2 = t1 + timedelta(hours=1)
        
        history = [
            CoherenceVariables(psi=0.4, rho=0.4, q=0.4, f=0.4, timestamp=t1),
            CoherenceVariables(psi=0.6, rho=0.5, q=0.5, f=0.5, timestamp=t2)
        ]
        
        velocity = engine.calculate_coherence_velocity(history)
        assert velocity > 0  # Should be positive (improving)
    
    def test_optimize_q_parameter(self, engine):
        """Test q parameter optimization."""
        # Test with default data
        individual_data = {}
        q_opt = engine.optimize_q_parameter(individual_data)
        assert 0 <= q_opt <= 1
        
        # Test with specific data
        individual_data = {
            'age': 30,
            'moral_development_stage': 5,
            'ethical_sensitivity': 0.8,
            'context_factors': {
                'stress_level': 0.3,
                'social_support': 0.7
            }
        }
        q_opt = engine.optimize_q_parameter(individual_data)
        assert 0 <= q_opt <= 1
        assert q_opt > 0.5  # Should be relatively high given good factors
    
    def test_assess_coherence_from_text(self, engine):
        """Test text-based coherence assessment."""
        # Test empty text
        vars = engine.assess_coherence_from_text("")
        assert vars.psi == 0.5  # Should return defaults
        
        # Test coherent text
        coherent_text = """
        I've been reflecting on my experiences and realize how much I've learned. 
        My friends and family have been incredibly supportive. 
        I feel a strong sense of purpose and direction in my life.
        """
        vars = engine.assess_coherence_from_text(coherent_text)
        assert vars.psi > 0.5  # Should show good consistency
        assert vars.f > 0.5   # Should show social connection
        
        # Test low coherence text
        crisis_text = """
        Everything is falling apart. I can't think straight. 
        Nothing makes sense anymore. I'm all alone.
        """
        vars = engine.assess_coherence_from_text(crisis_text)
        assert vars.psi < 0.5  # Should show low consistency
        assert vars.f < 0.5    # Should show low social connection
    
    def test_get_coherence_state_label(self, engine):
        """Test state labeling."""
        assert engine.get_coherence_state_label(0.1) == "Crisis"
        assert engine.get_coherence_state_label(0.3) == "Low"
        assert engine.get_coherence_state_label(0.5) == "Medium"
        assert engine.get_coherence_state_label(0.7) == "High"
        assert engine.get_coherence_state_label(0.9) == "Optimal"
    
    def test_calculate_intervention_priority(self, engine):
        """Test intervention priority calculation."""
        # Test with low variables
        current = CoherenceVariables(psi=0.2, rho=0.8, q=0.5, f=0.3)
        priorities = engine.calculate_intervention_priority(current)
        
        assert priorities['psi'] > priorities['rho']  # Psi should be highest priority
        assert priorities['f'] > priorities['q']      # f should be higher than q
        assert sum(priorities.values()) == pytest.approx(1.0)  # Should sum to 1
        
        # Test with history showing decline
        history = [
            CoherenceVariables(psi=0.6, rho=0.6, q=0.6, f=0.6),
            CoherenceVariables(psi=0.4, rho=0.5, q=0.5, f=0.4)
        ]
        current = CoherenceVariables(psi=0.2, rho=0.4, q=0.4, f=0.2)
        priorities = engine.calculate_intervention_priority(current, history)
        
        # All priorities should be elevated due to decline
        assert all(p > 0.2 for p in priorities.values())


class TestMathematicalProperties:
    """Test mathematical properties of GCT."""
    
    @pytest.fixture
    def engine(self):
        return GCTEngine()
    
    def test_coherence_bounds(self, engine):
        """Test that coherence is always bounded [0, 1]."""
        # Test 1000 random combinations
        for _ in range(1000):
            vars = CoherenceVariables(
                psi=np.random.random(),
                rho=np.random.random(),
                q=np.random.random(),
                f=np.random.random()
            )
            coherence = engine.calculate_coherence(vars)
            assert 0 <= coherence <= 1
    
    def test_psi_criticality(self, engine):
        """Test that Psi=0 severely limits coherence."""
        # When Psi=0, only q contributes
        vars = CoherenceVariables(psi=0, rho=1, q=0.5, f=1)
        coherence = engine.calculate_coherence(vars)
        assert coherence == 0.5 / 4.0  # Only q/4
    
    def test_multiplicative_effects(self, engine):
        """Test multiplicative interactions."""
        # High Psi amplifies rho and f effects
        vars1 = CoherenceVariables(psi=0.2, rho=0.8, q=0.5, f=0.8)
        vars2 = CoherenceVariables(psi=0.8, rho=0.8, q=0.5, f=0.8)
        
        c1 = engine.calculate_coherence(vars1)
        c2 = engine.calculate_coherence(vars2)
        
        assert c2 > c1  # Higher Psi should yield higher coherence
        assert (c2 - c1) > 0.1  # Difference should be substantial


class TestTextAnalysis:
    """Test natural language processing capabilities."""
    
    @pytest.fixture
    def engine(self):
        return GCTEngine()
    
    def test_sentiment_consistency(self, engine):
        """Test that mixed sentiments lower Psi."""
        consistent_text = "I'm happy and excited. Everything is going well. Life is good."
        inconsistent_text = "I'm happy but also sad. Things are good but terrible. I love and hate this."
        
        consistent_vars = engine.assess_coherence_from_text(consistent_text)
        inconsistent_vars = engine.assess_coherence_from_text(inconsistent_text)
        
        assert consistent_vars.psi > inconsistent_vars.psi
    
    def test_wisdom_detection(self, engine):
        """Test wisdom indicator detection."""
        wise_text = """
        Looking back on my experiences, I've learned that there are multiple perspectives 
        to consider. On the other hand, what seemed like failures were actually opportunities. 
        It depends on how you frame things.
        """
        
        unwise_text = "This is bad. That's it. No other way to see it."
        
        wise_vars = engine.assess_coherence_from_text(wise_text)
        unwise_vars = engine.assess_coherence_from_text(unwise_text)
        
        assert wise_vars.rho > unwise_vars.rho
    
    def test_moral_language(self, engine):
        """Test moral activation detection."""
        moral_text = """
        I should help others. It's the right thing to do. 
        We have a responsibility to our community. 
        I care about doing what's ethical and fair.
        """
        
        amoral_text = "I do whatever I want. Who cares about others?"
        
        moral_vars = engine.assess_coherence_from_text(moral_text)
        amoral_vars = engine.assess_coherence_from_text(amoral_text)
        
        assert moral_vars.q > amoral_vars.q
    
    def test_social_connection(self, engine):
        """Test social belonging detection."""
        connected_text = """
        My friends and I had a great time together. 
        Our team really supports each other. 
        We're all in this together with our families.
        """
        
        isolated_text = "I'm alone. Nobody understands. It's just me against the world."
        
        connected_vars = engine.assess_coherence_from_text(connected_text)
        isolated_vars = engine.assess_coherence_from_text(isolated_text)
        
        assert connected_vars.f > isolated_vars.f


if __name__ == "__main__":
    pytest.main([__file__, "-v"])