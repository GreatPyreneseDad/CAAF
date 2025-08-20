"""
Coherence Engine - Real-time coherence assessment from conversation data

This module provides the core functionality for assessing user coherence
states from text messages and conversation history.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import hashlib
import json
import logging
from collections import deque

from .gct_core import GCTEngine, CoherenceVariables

logger = logging.getLogger(__name__)


class CoherenceStateEnum(Enum):
    """Enumeration of coherence states."""
    CRISIS = "crisis"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    OPTIMAL = "optimal"


@dataclass
class CoherenceAssessment:
    """Results of a coherence assessment."""
    user_id: str
    timestamp: datetime
    message_hash: str  # Privacy: store hash instead of content
    variables: CoherenceVariables
    coherence_score: float
    state: CoherenceStateEnum
    confidence: float = 0.8  # Confidence in assessment
    intervention_priorities: Dict[str, float] = field(default_factory=dict)
    profile: Optional['CoherenceProfile'] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat(),
            'message_hash': self.message_hash,
            'variables': self.variables.to_dict(),
            'coherence_score': self.coherence_score,
            'state': self.state.value,
            'confidence': self.confidence,
            'intervention_priorities': self.intervention_priorities,
            'profile': self.profile.to_dict() if self.profile else None
        }


@dataclass
class CoherenceProfile:
    """User's coherence profile based on historical data."""
    user_id: str
    baseline_coherence: float = 0.5
    baseline_psi: float = 0.5
    baseline_rho: float = 0.5
    baseline_q: float = 0.5
    baseline_f: float = 0.5
    volatility: float = 0.1  # How much coherence typically varies
    trend: str = "stable"  # trending up, down, or stable
    risk_factors: List[str] = field(default_factory=list)
    protective_factors: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    assessment_count: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'user_id': self.user_id,
            'baseline_coherence': self.baseline_coherence,
            'baseline_psi': self.baseline_psi,
            'baseline_rho': self.baseline_rho,
            'baseline_q': self.baseline_q,
            'baseline_f': self.baseline_f,
            'volatility': self.volatility,
            'trend': self.trend,
            'risk_factors': self.risk_factors,
            'protective_factors': self.protective_factors,
            'last_updated': self.last_updated.isoformat(),
            'assessment_count': self.assessment_count
        }


class CoherenceEngine:
    """
    Main engine for real-time coherence assessment from conversation data.
    """
    
    def __init__(self, 
                 history_window: int = 100,
                 crisis_threshold: float = 0.2,
                 optimal_threshold: float = 0.8):
        """
        Initialize the Coherence Engine.
        
        Args:
            history_window: Number of historical assessments to keep per user
            crisis_threshold: Coherence score below this triggers crisis mode
            optimal_threshold: Coherence score above this indicates optimal state
        """
        self.gct_engine = GCTEngine()
        self.history_window = history_window
        self.crisis_threshold = crisis_threshold
        self.optimal_threshold = optimal_threshold
        
        # User data storage (in production, use proper database)
        self.user_histories: Dict[str, deque] = {}
        self.user_profiles: Dict[str, CoherenceProfile] = {}
        
        # Pattern detection
        self.crisis_patterns = [
            'want to die', 'kill myself', 'end it all', 'no point',
            'hopeless', 'can\'t go on', 'give up', 'worthless'
        ]
        
        self.growth_patterns = [
            'learning', 'growing', 'improving', 'progress', 'better',
            'understanding', 'insight', 'realized', 'discovered'
        ]
    
    def process_message(self, 
                       message: str, 
                       user_id: str,
                       timestamp: Optional[datetime] = None,
                       context: Optional[Dict] = None) -> CoherenceAssessment:
        """
        Process a message and assess coherence state.
        
        Args:
            message: User's message text
            user_id: Unique user identifier
            timestamp: Message timestamp (defaults to now)
            context: Optional context information
            
        Returns:
            CoherenceAssessment: Complete assessment results
        """
        if not timestamp:
            timestamp = datetime.now()
        
        # Hash message for privacy
        message_hash = hashlib.sha256(message.encode()).hexdigest()[:16]
        
        # Get user history and profile
        history = self._get_user_history(user_id)
        profile = self._get_or_create_profile(user_id)
        
        # Prepare context with user data
        enhanced_context = context or {}
        enhanced_context['coherence_history'] = history
        enhanced_context['user_profile'] = {
            'baseline_wisdom': profile.baseline_rho,
            'volatility': profile.volatility
        }
        
        # Assess coherence variables from text
        variables = self.gct_engine.assess_coherence_from_text(message, enhanced_context)
        
        # Calculate overall coherence score
        coherence_score = self.gct_engine.calculate_coherence(variables)
        
        # Apply crisis detection
        if self._detect_crisis_indicators(message):
            coherence_score *= 0.5  # Reduce score if crisis indicators present
            variables.psi *= 0.7  # Internal consistency likely compromised
        
        # Determine state
        state = self._determine_state(coherence_score)
        
        # Calculate intervention priorities
        intervention_priorities = self.gct_engine.calculate_intervention_priority(
            variables, 
            [h.variables for h in history]
        )
        
        # Create assessment
        assessment = CoherenceAssessment(
            user_id=user_id,
            timestamp=timestamp,
            message_hash=message_hash,
            variables=variables,
            coherence_score=coherence_score,
            state=state,
            confidence=self._calculate_confidence(message, history),
            intervention_priorities=intervention_priorities,
            profile=profile
        )
        
        # Update history and profile
        self._update_user_history(user_id, assessment)
        self._update_user_profile(user_id, assessment)
        
        return assessment
    
    def get_user_coherence_profile(self, user_id: str) -> Optional[CoherenceProfile]:
        """
        Get user's coherence profile.
        
        Args:
            user_id: User identifier
            
        Returns:
            CoherenceProfile or None if user not found
        """
        return self.user_profiles.get(user_id)
    
    def predict_optimal_response_timing(self, user_id: str) -> Dict:
        """
        Predict optimal timing for interventions based on coherence patterns.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with timing recommendations
        """
        history = self._get_user_history(user_id)
        profile = self._get_or_create_profile(user_id)
        
        if len(history) < 5:
            return {
                'recommendation': 'insufficient_data',
                'message': 'Need more interaction history',
                'wait_time_minutes': 5
            }
        
        # Analyze coherence patterns over time
        recent_scores = [h.coherence_score for h in history[-10:]]
        recent_times = [h.timestamp for h in history[-10:]]
        
        # Calculate average time between messages
        if len(recent_times) > 1:
            time_deltas = []
            for i in range(1, len(recent_times)):
                delta = (recent_times[i] - recent_times[i-1]).total_seconds() / 60
                time_deltas.append(delta)
            avg_response_time = np.mean(time_deltas)
        else:
            avg_response_time = 10  # Default 10 minutes
        
        # Determine timing based on state and trends
        current_state = history[-1].state if history else CoherenceStateEnum.MEDIUM
        
        if current_state == CoherenceStateEnum.CRISIS:
            return {
                'recommendation': 'immediate',
                'message': 'User in crisis - respond immediately',
                'wait_time_minutes': 0,
                'priority': 'critical'
            }
        
        elif current_state == CoherenceStateEnum.LOW:
            return {
                'recommendation': 'prompt',
                'message': 'Low coherence - respond within 2-3 minutes',
                'wait_time_minutes': min(avg_response_time * 0.5, 3),
                'priority': 'high'
            }
        
        elif profile.trend == 'declining':
            return {
                'recommendation': 'attentive',
                'message': 'Declining trend - maintain close engagement',
                'wait_time_minutes': min(avg_response_time * 0.7, 5),
                'priority': 'medium'
            }
        
        else:
            return {
                'recommendation': 'natural',
                'message': 'Stable coherence - respond at natural pace',
                'wait_time_minutes': avg_response_time,
                'priority': 'normal'
            }
    
    def detect_coherence_crisis(self, user_id: str) -> bool:
        """
        Detect if user is in coherence crisis requiring immediate intervention.
        
        Args:
            user_id: User identifier
            
        Returns:
            bool: True if crisis detected
        """
        history = self._get_user_history(user_id)
        
        if not history:
            return False
        
        # Check current state
        if history[-1].state == CoherenceStateEnum.CRISIS:
            return True
        
        # Check rapid decline
        if len(history) >= 3:
            recent_scores = [h.coherence_score for h in history[-3:]]
            if all(recent_scores[i] < recent_scores[i-1] for i in range(1, len(recent_scores))):
                if recent_scores[-1] < 0.3:  # Rapid decline to low levels
                    return True
        
        # Check sustained low coherence
        if len(history) >= 5:
            recent_states = [h.state for h in history[-5:]]
            crisis_count = sum(1 for s in recent_states if s in [CoherenceStateEnum.CRISIS, CoherenceStateEnum.LOW])
            if crisis_count >= 4:
                return True
        
        return False
    
    def get_coherence_velocity(self, user_id: str) -> float:
        """
        Get current coherence velocity (rate of change).
        
        Args:
            user_id: User identifier
            
        Returns:
            float: Coherence velocity
        """
        history = self._get_user_history(user_id)
        
        if len(history) < 2:
            return 0.0
        
        return self.gct_engine.calculate_coherence_velocity(
            [h.variables for h in history]
        )
    
    def _get_user_history(self, user_id: str) -> List[CoherenceAssessment]:
        """Get user's assessment history."""
        if user_id not in self.user_histories:
            self.user_histories[user_id] = deque(maxlen=self.history_window)
        return list(self.user_histories[user_id])
    
    def _get_or_create_profile(self, user_id: str) -> CoherenceProfile:
        """Get or create user profile."""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = CoherenceProfile(user_id=user_id)
        return self.user_profiles[user_id]
    
    def _update_user_history(self, user_id: str, assessment: CoherenceAssessment):
        """Update user's assessment history."""
        if user_id not in self.user_histories:
            self.user_histories[user_id] = deque(maxlen=self.history_window)
        self.user_histories[user_id].append(assessment)
    
    def _update_user_profile(self, user_id: str, assessment: CoherenceAssessment):
        """Update user's coherence profile based on new assessment."""
        profile = self._get_or_create_profile(user_id)
        history = self._get_user_history(user_id)
        
        profile.assessment_count += 1
        
        # Update baselines with exponential moving average
        alpha = 0.1  # Learning rate
        profile.baseline_coherence = (1 - alpha) * profile.baseline_coherence + alpha * assessment.coherence_score
        profile.baseline_psi = (1 - alpha) * profile.baseline_psi + alpha * assessment.variables.psi
        profile.baseline_rho = (1 - alpha) * profile.baseline_rho + alpha * assessment.variables.rho
        profile.baseline_q = (1 - alpha) * profile.baseline_q + alpha * assessment.variables.q
        profile.baseline_f = (1 - alpha) * profile.baseline_f + alpha * assessment.variables.f
        
        # Update volatility
        if len(history) > 5:
            recent_scores = [h.coherence_score for h in history[-10:]]
            profile.volatility = np.std(recent_scores)
        
        # Determine trend
        if len(history) >= 10:
            older_avg = np.mean([h.coherence_score for h in history[-20:-10]])
            newer_avg = np.mean([h.coherence_score for h in history[-10:]])
            
            if newer_avg > older_avg + 0.1:
                profile.trend = "improving"
            elif newer_avg < older_avg - 0.1:
                profile.trend = "declining"
            else:
                profile.trend = "stable"
        
        # Update risk and protective factors
        profile.risk_factors = self._identify_risk_factors(history)
        profile.protective_factors = self._identify_protective_factors(history)
        
        profile.last_updated = datetime.now()
    
    def _determine_state(self, coherence_score: float) -> CoherenceStateEnum:
        """Determine coherence state from score."""
        if coherence_score < 0.2:
            return CoherenceStateEnum.CRISIS
        elif coherence_score < 0.4:
            return CoherenceStateEnum.LOW
        elif coherence_score < 0.6:
            return CoherenceStateEnum.MEDIUM
        elif coherence_score < 0.8:
            return CoherenceStateEnum.HIGH
        else:
            return CoherenceStateEnum.OPTIMAL
    
    def _detect_crisis_indicators(self, message: str) -> bool:
        """Detect crisis indicators in message."""
        message_lower = message.lower()
        return any(pattern in message_lower for pattern in self.crisis_patterns)
    
    def _calculate_confidence(self, message: str, history: List[CoherenceAssessment]) -> float:
        """Calculate confidence in assessment."""
        confidence = 0.8  # Base confidence
        
        # Lower confidence for very short messages
        if len(message.split()) < 5:
            confidence *= 0.8
        
        # Higher confidence with more history
        if len(history) > 20:
            confidence *= 1.1
        
        # Lower confidence if high volatility
        if history and len(history) > 5:
            recent_scores = [h.coherence_score for h in history[-5:]]
            volatility = np.std(recent_scores)
            if volatility > 0.3:
                confidence *= 0.9
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _identify_risk_factors(self, history: List[CoherenceAssessment]) -> List[str]:
        """Identify risk factors from assessment history."""
        risk_factors = []
        
        if not history:
            return risk_factors
        
        # Low baseline coherence
        recent_scores = [h.coherence_score for h in history[-10:]] if len(history) >= 10 else [h.coherence_score for h in history]
        if np.mean(recent_scores) < 0.4:
            risk_factors.append("chronic_low_coherence")
        
        # High volatility
        if len(recent_scores) > 3 and np.std(recent_scores) > 0.3:
            risk_factors.append("high_volatility")
        
        # Weak specific variables
        recent_vars = [h.variables for h in history[-5:]] if len(history) >= 5 else [h.variables for h in history]
        avg_psi = np.mean([v.psi for v in recent_vars])
        avg_f = np.mean([v.f for v in recent_vars])
        
        if avg_psi < 0.3:
            risk_factors.append("poor_internal_consistency")
        if avg_f < 0.3:
            risk_factors.append("social_isolation")
        
        # Declining trend
        if len(history) >= 10:
            older_avg = np.mean([h.coherence_score for h in history[-20:-10]])
            newer_avg = np.mean([h.coherence_score for h in history[-10:]])
            if newer_avg < older_avg - 0.15:
                risk_factors.append("declining_trajectory")
        
        return risk_factors
    
    def _identify_protective_factors(self, history: List[CoherenceAssessment]) -> List[str]:
        """Identify protective factors from assessment history."""
        protective_factors = []
        
        if not history:
            return protective_factors
        
        recent_vars = [h.variables for h in history[-5:]] if len(history) >= 5 else [h.variables for h in history]
        
        # Strong wisdom
        avg_rho = np.mean([v.rho for v in recent_vars])
        if avg_rho > 0.7:
            protective_factors.append("accumulated_wisdom")
        
        # Strong social connections
        avg_f = np.mean([v.f for v in recent_vars])
        if avg_f > 0.7:
            protective_factors.append("strong_social_support")
        
        # Moral grounding
        avg_q = np.mean([v.q for v in recent_vars])
        if avg_q > 0.7:
            protective_factors.append("strong_moral_framework")
        
        # Resilience (bounces back from low points)
        if len(history) > 10:
            scores = [h.coherence_score for h in history]
            # Look for recovery patterns
            for i in range(2, len(scores)-2):
                if scores[i] < 0.3 and scores[i+2] > 0.5:
                    protective_factors.append("demonstrated_resilience")
                    break
        
        return protective_factors