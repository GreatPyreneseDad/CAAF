"""
Tests for Coherence Engine
"""

import pytest
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.coherence_engine import (
    CoherenceEngine, CoherenceAssessment, CoherenceProfile,
    CoherenceStateEnum
)
from src.gct_core import CoherenceVariables


class TestCoherenceEngine:
    """Test CoherenceEngine functionality."""
    
    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return CoherenceEngine()
    
    def test_process_message(self, engine):
        """Test message processing."""
        # Process a message
        assessment = engine.process_message(
            message="I'm feeling good today and making progress",
            user_id="test_user"
        )
        
        assert isinstance(assessment, CoherenceAssessment)
        assert assessment.user_id == "test_user"
        assert 0 <= assessment.coherence_score <= 1
        assert assessment.state in CoherenceStateEnum
        assert assessment.confidence > 0
        assert isinstance(assessment.variables, CoherenceVariables)
    
    def test_user_history_tracking(self, engine):
        """Test that user history is properly tracked."""
        user_id = "history_test_user"
        
        # Process multiple messages
        messages = [
            "I'm struggling today",
            "Things are getting a bit better",
            "I feel more confident now"
        ]
        
        for msg in messages:
            engine.process_message(msg, user_id)
        
        # Check history
        history = engine._get_user_history(user_id)
        assert len(history) == 3
        
        # Verify chronological order
        for i in range(1, len(history)):
            assert history[i].timestamp >= history[i-1].timestamp
    
    def test_profile_creation_and_update(self, engine):
        """Test user profile management."""
        user_id = "profile_test_user"
        
        # First assessment should create profile
        assessment1 = engine.process_message(
            "Initial message for testing",
            user_id
        )
        
        profile = engine.get_user_coherence_profile(user_id)
        assert profile is not None
        assert profile.user_id == user_id
        assert profile.assessment_count == 1
        
        # Additional assessments should update profile
        for i in range(5):
            engine.process_message(f"Message {i}", user_id)
        
        updated_profile = engine.get_user_coherence_profile(user_id)
        assert updated_profile.assessment_count == 6
    
    def test_crisis_detection(self, engine):
        """Test crisis detection functionality."""
        user_id = "crisis_test_user"
        
        # Non-crisis message
        assessment1 = engine.process_message(
            "I'm doing okay, just a normal day",
            user_id
        )
        assert not engine.detect_coherence_crisis(user_id)
        
        # Crisis message
        assessment2 = engine.process_message(
            "I want to die. I can't go on anymore.",
            user_id
        )
        assert assessment2.state == CoherenceStateEnum.CRISIS
        assert assessment2.coherence_score < 0.3
        assert engine.detect_coherence_crisis(user_id)
    
    def test_coherence_velocity(self, engine):
        """Test coherence velocity calculation."""
        user_id = "velocity_test_user"
        
        # Create assessments with known timestamps
        base_time = datetime.now()
        
        # Declining coherence
        engine.process_message("I'm doing great!", user_id, timestamp=base_time)
        engine.process_message("Things are getting tough", user_id, 
                             timestamp=base_time + timedelta(hours=1))
        engine.process_message("I'm really struggling", user_id,
                             timestamp=base_time + timedelta(hours=2))
        
        velocity = engine.get_coherence_velocity(user_id)
        assert velocity < 0  # Should be negative (declining)
    
    def test_optimal_response_timing(self, engine):
        """Test response timing predictions."""
        user_id = "timing_test_user"
        
        # Insufficient data case
        timing = engine.predict_optimal_response_timing(user_id)
        assert timing['recommendation'] == 'insufficient_data'
        
        # Create history
        for i in range(10):
            engine.process_message(f"Message {i}", user_id)
        
        # Normal state timing
        timing = engine.predict_optimal_response_timing(user_id)
        assert 'wait_time_minutes' in timing
        assert timing['priority'] in ['critical', 'high', 'medium', 'normal']
        
        # Crisis state timing
        engine.process_message("I want to end it all", user_id)
        timing = engine.predict_optimal_response_timing(user_id)
        assert timing['recommendation'] == 'immediate'
        assert timing['wait_time_minutes'] == 0
    
    def test_trend_detection(self, engine):
        """Test coherence trend detection."""
        user_id = "trend_test_user"
        
        # Create improving trend
        messages = [
            "Everything is terrible",
            "Still struggling a lot",
            "Maybe things will improve",
            "I'm starting to feel better",
            "Today was actually good",
            "I'm doing much better now",
            "Life is getting back on track",
            "I feel strong and capable",
            "Everything is going well",
            "I'm thriving!"
        ]
        
        for i, msg in enumerate(messages):
            engine.process_message(msg, user_id,
                                 timestamp=datetime.now() + timedelta(minutes=i))
        
        profile = engine.get_user_coherence_profile(user_id)
        assert profile.trend == "improving"
    
    def test_risk_and_protective_factors(self, engine):
        """Test identification of risk and protective factors."""
        user_id = "factors_test_user"
        
        # Create user with risk factors
        risk_messages = [
            "I'm always alone",
            "Nothing ever makes sense",
            "I can't focus on anything",
            "Everything keeps changing",
            "I feel so isolated"
        ]
        
        for msg in risk_messages:
            engine.process_message(msg, user_id)
        
        profile = engine.get_user_coherence_profile(user_id)
        assert len(profile.risk_factors) > 0
        assert any('isolation' in factor for factor in profile.risk_factors)
        
        # Create user with protective factors
        user_id2 = "protective_test_user"
        protective_messages = [
            "My friends are amazing and always there for me",
            "I've learned so much from my experiences",
            "I have strong values that guide me",
            "My community supports me through everything",
            "I feel deeply connected to others"
        ]
        
        for msg in protective_messages:
            engine.process_message(msg, user_id2)
        
        profile2 = engine.get_user_coherence_profile(user_id2)
        assert len(profile2.protective_factors) > 0


class TestCoherenceStates:
    """Test coherence state transitions and properties."""
    
    @pytest.fixture
    def engine(self):
        return CoherenceEngine()
    
    def test_state_thresholds(self, engine):
        """Test that states are assigned correctly based on scores."""
        test_cases = [
            (0.1, CoherenceStateEnum.CRISIS),
            (0.25, CoherenceStateEnum.LOW),
            (0.5, CoherenceStateEnum.MEDIUM),
            (0.7, CoherenceStateEnum.HIGH),
            (0.85, CoherenceStateEnum.OPTIMAL)
        ]
        
        for score, expected_state in test_cases:
            state = engine._determine_state(score)
            assert state == expected_state
    
    def test_state_transitions(self, engine):
        """Test tracking state transitions."""
        user_id = "transition_test_user"
        
        # Create specific state sequence
        state_messages = {
            CoherenceStateEnum.CRISIS: "I want to die",
            CoherenceStateEnum.LOW: "I'm really struggling",
            CoherenceStateEnum.MEDIUM: "Things are okay I guess",
            CoherenceStateEnum.HIGH: "I'm doing well and feeling good",
            CoherenceStateEnum.OPTIMAL: "Life is amazing, I'm helping others grow"
        }
        
        assessments = []
        for state, message in state_messages.items():
            assessment = engine.process_message(message, user_id)
            assessments.append(assessment)
        
        # Verify we hit different states
        states_seen = {a.state for a in assessments}
        assert len(states_seen) >= 3  # Should see at least 3 different states


class TestPrivacyFeatures:
    """Test privacy-preserving features."""
    
    @pytest.fixture
    def engine(self):
        return CoherenceEngine()
    
    def test_message_hashing(self, engine):
        """Test that messages are hashed, not stored."""
        user_id = "privacy_test_user"
        message = "This is a private message"
        
        assessment = engine.process_message(message, user_id)
        
        # Check that message_hash exists and is not the original message
        assert assessment.message_hash
        assert assessment.message_hash != message
        assert len(assessment.message_hash) == 16  # Truncated hash
    
    def test_no_raw_message_storage(self, engine):
        """Verify no raw messages are stored in history."""
        user_id = "storage_test_user"
        sensitive_message = "My SSN is 123-45-6789"  # Obviously fake
        
        engine.process_message(sensitive_message, user_id)
        
        # Check that raw message is not in history
        history = engine._get_user_history(user_id)
        
        # Verify no assessment contains the raw message
        for assessment in history:
            assert sensitive_message not in str(assessment)
            assert "123-45-6789" not in str(assessment)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def engine(self):
        return CoherenceEngine()
    
    def test_empty_message(self, engine):
        """Test handling of empty messages."""
        assessment = engine.process_message("", "test_user")
        assert assessment.coherence_score == 0.5  # Should return default
    
    def test_very_long_message(self, engine):
        """Test handling of very long messages."""
        long_message = "This is a test. " * 1000
        assessment = engine.process_message(long_message, "test_user")
        assert assessment is not None
        assert 0 <= assessment.coherence_score <= 1
    
    def test_special_characters(self, engine):
        """Test handling of special characters."""
        special_message = "Test with émojis 😀 and spëcial çharacters!!! @#$%"
        assessment = engine.process_message(special_message, "test_user")
        assert assessment is not None
    
    def test_concurrent_users(self, engine):
        """Test that different users don't interfere."""
        user1 = "user1"
        user2 = "user2"
        
        # Process messages for both users
        engine.process_message("User 1 message", user1)
        engine.process_message("User 2 message", user2)
        
        # Verify histories are separate
        history1 = engine._get_user_history(user1)
        history2 = engine._get_user_history(user2)
        
        assert len(history1) == 1
        assert len(history2) == 1
        assert history1[0].user_id == user1
        assert history2[0].user_id == user2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])