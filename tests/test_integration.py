"""
Integration tests for the complete CAAF system
"""

import pytest
import requests
import asyncio
from datetime import datetime, timedelta
import sys
import os
import time
from multiprocessing import Process

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import app
from src.coherence_engine import CoherenceEngine
from src.response_optimizer import ResponseOptimizer
from src.metrics_tracker import MetricsTracker
import uvicorn


class TestAPIIntegration:
    """Test REST API integration."""
    
    @classmethod
    def setup_class(cls):
        """Start API server for testing."""
        def run_server():
            uvicorn.run(app, host="127.0.0.1", port=8001)
        
        cls.server_process = Process(target=run_server)
        cls.server_process.start()
        time.sleep(2)  # Give server time to start
        cls.base_url = "http://127.0.0.1:8001"
    
    @classmethod
    def teardown_class(cls):
        """Stop API server."""
        cls.server_process.terminate()
        cls.server_process.join()
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = requests.get(f"{self.base_url}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert all(data["components"].values())
    
    def test_assess_coherence_endpoint(self):
        """Test coherence assessment via API."""
        payload = {
            "message": "I'm feeling overwhelmed with work",
            "user_id": "api_test_user"
        }
        
        response = requests.post(
            f"{self.base_url}/assess-coherence",
            json=payload
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "coherence_score" in data
        assert 0 <= data["coherence_score"] <= 1
        assert "state" in data
        assert "variables" in data
        assert all(var in data["variables"] for var in ["psi", "rho", "q", "f"])
    
    def test_optimize_response_endpoint(self):
        """Test response optimization via API."""
        # First assess coherence
        assess_payload = {
            "message": "Everything is falling apart",
            "user_id": "optimize_test_user"
        }
        requests.post(f"{self.base_url}/assess-coherence", json=assess_payload)
        
        # Then optimize response
        optimize_payload = {
            "original_response": "Just stay positive!",
            "user_id": "optimize_test_user"
        }
        
        response = requests.post(
            f"{self.base_url}/optimize-response",
            json=optimize_payload
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "optimized_response" in data
        assert data["optimized_response"] != optimize_payload["original_response"]
        assert "optimization_strategy" in data
        assert data["safety_checks_passed"]
    
    def test_coherence_profile_endpoint(self):
        """Test profile retrieval via API."""
        user_id = "profile_api_test"
        
        # Create some history
        for i in range(5):
            payload = {
                "message": f"Test message {i}",
                "user_id": user_id
            }
            requests.post(f"{self.base_url}/assess-coherence", json=payload)
        
        # Get profile
        response = requests.get(f"{self.base_url}/coherence-profile/{user_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["user_id"] == user_id
        assert "baseline_coherence" in data
        assert "trend" in data
        assert data["assessment_count"] == 5
    
    def test_integration_test_endpoint(self):
        """Test the integration test endpoint."""
        payload = {
            "test_messages": [
                "I'm in crisis",
                "Feeling a bit better",
                "Making good progress"
            ],
            "test_responses": [
                "Cheer up!",
                "Keep going",
                "That's great"
            ],
            "user_id": "integration_test"
        }
        
        response = requests.post(
            f"{self.base_url}/integration-test",
            json=payload
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["overall_status"] == "success"
        assert len(data["assessments"]) == 3
        assert len(data["optimizations"]) == 3
        assert data["summary"]["responses_changed"] > 0


class TestEndToEndWorkflow:
    """Test complete workflows."""
    
    def test_conversation_flow(self):
        """Test a complete conversation workflow."""
        engine = CoherenceEngine()
        optimizer = ResponseOptimizer()
        tracker = MetricsTracker(database_url="sqlite:///test_e2e.db")
        
        user_id = "e2e_test_user"
        conversation = [
            {
                "user": "I've been feeling really anxious lately",
                "ai": "I understand anxiety can be challenging"
            },
            {
                "user": "Work pressure is getting to me and I can't sleep",
                "ai": "Let's explore some strategies to help"
            },
            {
                "user": "I tried meditation and it's helping a bit",
                "ai": "That's wonderful progress"
            }
        ]
        
        for turn in conversation:
            # Assess coherence
            assessment = engine.process_message(turn["user"], user_id)
            
            # Create coherence state
            from src.response_optimizer import CoherenceState
            coherence_state = CoherenceState(
                state=assessment.state,
                score=assessment.coherence_score,
                psi=assessment.variables.psi,
                rho=assessment.variables.rho,
                q=assessment.variables.q,
                f=assessment.variables.f
            )
            
            # Optimize response
            profile = engine.get_user_coherence_profile(user_id)
            optimized = optimizer.optimize_response(
                turn["ai"],
                coherence_state,
                profile
            )
            
            # Track metrics
            from src.metrics_tracker import CoherenceRecord
            record = CoherenceRecord(
                user_id=user_id,
                timestamp=assessment.timestamp,
                psi=assessment.variables.psi,
                rho=assessment.variables.rho,
                q=assessment.variables.q,
                f=assessment.variables.f,
                coherence_score=assessment.coherence_score,
                coherence_velocity=0,
                coherence_state=assessment.state.value
            )
            tracker.record_assessment(record)
            
            # Verify optimization happened
            assert optimized != turn["ai"] or assessment.state.value in ["high", "optimal"]
        
        # Check final metrics
        improvement = tracker.calculate_improvement_metrics(user_id, window_days=1)
        assert improvement["total_assessments"] == 3
        
        # Clean up
        os.remove("test_e2e.db")
    
    def test_crisis_intervention_flow(self):
        """Test crisis detection and intervention."""
        engine = CoherenceEngine()
        optimizer = ResponseOptimizer()
        
        user_id = "crisis_flow_test"
        
        # Crisis message
        crisis_message = "I don't want to live anymore"
        assessment = engine.process_message(crisis_message, user_id)
        
        assert assessment.state.value == "crisis"
        assert engine.detect_coherence_crisis(user_id)
        
        # AI response that needs optimization
        ai_response = "Things will get better if you just try harder"
        
        from src.response_optimizer import CoherenceState
        coherence_state = CoherenceState(
            state=assessment.state,
            score=assessment.coherence_score,
            psi=assessment.variables.psi,
            rho=assessment.variables.rho,
            q=assessment.variables.q,
            f=assessment.variables.f
        )
        
        # Verify harmful response is detected and optimized
        assert optimizer.detect_potential_harm(ai_response, coherence_state)
        
        optimized = optimizer.optimize_response(
            ai_response,
            coherence_state
        )
        
        # Optimized response should be supportive
        assert "support" in optimized.lower() or "help" in optimized.lower()
        assert "try harder" not in optimized.lower()
    
    def test_coherence_improvement_tracking(self):
        """Test tracking coherence improvement over time."""
        engine = CoherenceEngine()
        tracker = MetricsTracker(database_url="sqlite:///test_improvement.db")
        
        user_id = "improvement_test_user"
        
        # Simulate improving coherence over time
        messages = [
            ("I'm completely lost", 0.2),  # Low
            ("Starting to see some hope", 0.35),  # Low-Medium
            ("Things are getting clearer", 0.5),  # Medium
            ("I'm feeling more confident", 0.65),  # Medium-High
            ("Life is good and I'm helping others", 0.8)  # High
        ]
        
        base_time = datetime.now()
        for i, (message, expected_range) in enumerate(messages):
            assessment = engine.process_message(
                message,
                user_id,
                timestamp=base_time + timedelta(hours=i)
            )
            
            # Record assessment
            from src.metrics_tracker import CoherenceRecord
            record = CoherenceRecord(
                user_id=user_id,
                timestamp=assessment.timestamp,
                psi=assessment.variables.psi,
                rho=assessment.variables.rho,
                q=assessment.variables.q,
                f=assessment.variables.f,
                coherence_score=assessment.coherence_score,
                coherence_velocity=engine.get_coherence_velocity(user_id),
                coherence_state=assessment.state.value
            )
            tracker.record_assessment(record)
        
        # Check improvement metrics
        improvement = tracker.calculate_improvement_metrics(user_id, window_days=1)
        
        assert improvement["improvement_rate"] > 0
        assert improvement["coherence_change"] > 0
        assert improvement["second_period_avg"] > improvement["first_period_avg"]
        
        # Check pattern analysis
        patterns = tracker.get_pattern_analysis(user_id, window_days=1)
        assert patterns["status"] != "insufficient_data"
        
        # Clean up
        os.remove("test_improvement.db")


class TestPerformance:
    """Test performance characteristics."""
    
    def test_response_time(self):
        """Test that operations complete within acceptable time."""
        engine = CoherenceEngine()
        optimizer = ResponseOptimizer()
        
        message = "This is a test message to assess performance"
        user_id = "performance_test"
        
        # Test coherence assessment time
        start = time.time()
        assessment = engine.process_message(message, user_id)
        assessment_time = time.time() - start
        
        assert assessment_time < 0.5  # Should complete within 500ms
        
        # Test optimization time
        from src.response_optimizer import CoherenceState
        coherence_state = CoherenceState(
            state=assessment.state,
            score=assessment.coherence_score,
            psi=assessment.variables.psi,
            rho=assessment.variables.rho,
            q=assessment.variables.q,
            f=assessment.variables.f
        )
        
        start = time.time()
        optimized = optimizer.optimize_response(
            "This is a test response",
            coherence_state
        )
        optimization_time = time.time() - start
        
        assert optimization_time < 0.1  # Should complete within 100ms
    
    def test_concurrent_users(self):
        """Test handling multiple concurrent users."""
        engine = CoherenceEngine()
        
        async def process_user(user_id, message_count):
            """Process messages for a single user."""
            for i in range(message_count):
                engine.process_message(f"Message {i} from {user_id}", user_id)
        
        async def run_concurrent_test():
            """Run concurrent processing test."""
            users = [f"concurrent_user_{i}" for i in range(10)]
            tasks = [process_user(user, 5) for user in users]
            
            start = time.time()
            await asyncio.gather(*tasks)
            total_time = time.time() - start
            
            # Should handle 50 messages (10 users × 5 messages) quickly
            assert total_time < 5.0
            
            # Verify all users have correct history
            for user in users:
                history = engine._get_user_history(user)
                assert len(history) == 5
        
        # Run the async test
        asyncio.run(run_concurrent_test())
    
    def test_memory_efficiency(self):
        """Test memory usage with history limits."""
        engine = CoherenceEngine(history_window=10)
        user_id = "memory_test_user"
        
        # Process many messages
        for i in range(100):
            engine.process_message(f"Message {i}", user_id)
        
        # History should be limited to window size
        history = engine._get_user_history(user_id)
        assert len(history) <= 10
        
        # Most recent messages should be preserved
        assert any("Message 99" in str(h) for h in history)
        assert not any("Message 0" in str(h) for h in history)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_malformed_api_requests(self):
        """Test API handling of malformed requests."""
        base_url = "http://127.0.0.1:8001"
        
        # Missing required fields
        response = requests.post(
            f"{base_url}/assess-coherence",
            json={"message": "Test"}  # Missing user_id
        )
        assert response.status_code == 422
        
        # Empty message
        response = requests.post(
            f"{base_url}/assess-coherence",
            json={"message": "", "user_id": "test"}
        )
        assert response.status_code == 422
        
        # Invalid endpoint
        response = requests.get(f"{base_url}/invalid-endpoint")
        assert response.status_code == 404
    
    def test_resilience_to_errors(self):
        """Test system resilience to various errors."""
        engine = CoherenceEngine()
        optimizer = ResponseOptimizer()
        
        # Very long message
        long_message = "word " * 10000
        assessment = engine.process_message(long_message, "test_user")
        assert assessment is not None
        
        # Unicode and special characters
        unicode_message = "Test 测试 🎉 émojis ñ special"
        assessment = engine.process_message(unicode_message, "test_user")
        assert assessment is not None
        
        # Empty optimization
        from src.response_optimizer import CoherenceState
        state = CoherenceState(
            state=assessment.state,
            score=0.5,
            psi=0.5, rho=0.5, q=0.5, f=0.5
        )
        optimized = optimizer.optimize_response("", state)
        assert optimized == ""  # Should handle empty input gracefully


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])