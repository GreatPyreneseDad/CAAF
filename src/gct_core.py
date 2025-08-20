"""
Grounded Coherence Theory (GCT) Core Engine

Implements the mathematical foundation of GCT with four key variables:
- Ψ (Psi): Internal Consistency (0.0-1.0)
- ρ (Rho): Accumulated Wisdom (0.0-1.0)
- q: Moral Activation Energy (0.0-1.0)
- f: Social Belonging Architecture (0.0-1.0)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
from transformers import pipeline
import logging

# Initialize NLP tools
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)


@dataclass
class CoherenceVariables:
    """Container for the four fundamental coherence variables."""
    psi: float = 0.5  # Internal Consistency (0.0-1.0)
    rho: float = 0.5  # Accumulated Wisdom (0.0-1.0)
    q: float = 0.5    # Moral Activation Energy (0.0-1.0)
    f: float = 0.5    # Social Belonging Architecture (0.0-1.0)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate variable ranges."""
        for var_name, var_value in [('psi', self.psi), ('rho', self.rho), 
                                     ('q', self.q), ('f', self.f)]:
            if not 0.0 <= var_value <= 1.0:
                raise ValueError(f"{var_name} must be between 0.0 and 1.0, got {var_value}")
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return {
            'psi': self.psi,
            'rho': self.rho,
            'q': self.q,
            'f': self.f,
            'timestamp': self.timestamp.isoformat()
        }


class GCTEngine:
    """
    Core engine implementing Grounded Coherence Theory calculations.
    """
    
    def __init__(self):
        """Initialize the GCT engine with NLP models."""
        self._init_nlp_models()
        
    def _init_nlp_models(self):
        """Initialize NLP models for text analysis."""
        try:
            # Sentiment analyzer
            self.sia = SentimentIntensityAnalyzer()
            
            # Load spaCy model for linguistic analysis
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
            
            # Load transformer model for deeper analysis
            try:
                self.emotion_classifier = pipeline(
                    "text-classification", 
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True
                )
            except:
                logger.warning("Emotion classifier not available. Install transformers and torch.")
                self.emotion_classifier = None
                
        except Exception as e:
            logger.error(f"Error initializing NLP models: {e}")
    
    def calculate_coherence(self, variables: CoherenceVariables) -> float:
        """
        Calculate overall coherence score using the GCT equation:
        C = Ψ + (ρ × Ψ) + q_optimal + (f × Ψ)
        
        Args:
            variables: CoherenceVariables instance
            
        Returns:
            float: Overall coherence score (0.0-4.0 range, normalized to 0.0-1.0)
        """
        # Core GCT equation
        raw_coherence = (
            variables.psi + 
            (variables.rho * variables.psi) + 
            variables.q + 
            (variables.f * variables.psi)
        )
        
        # Normalize to 0.0-1.0 range (max theoretical value is 4.0)
        normalized_coherence = min(raw_coherence / 4.0, 1.0)
        
        return normalized_coherence
    
    def calculate_coherence_velocity(self, history: List[CoherenceVariables]) -> float:
        """
        Calculate the rate of change of coherence over time.
        dC/dt = Ψ̇(1 + ρ + f) + ρ̇ × Ψ + q̇_optimal + ḟ × Ψ
        
        Args:
            history: List of coherence variables over time (at least 2 points)
            
        Returns:
            float: Coherence velocity (positive = improving, negative = declining)
        """
        if len(history) < 2:
            return 0.0
        
        # Get most recent two measurements
        current = history[-1]
        previous = history[-2]
        
        # Calculate time delta in hours
        time_delta = (current.timestamp - previous.timestamp).total_seconds() / 3600.0
        
        if time_delta <= 0:
            return 0.0
        
        # Calculate derivatives
        dpsi_dt = (current.psi - previous.psi) / time_delta
        drho_dt = (current.rho - previous.rho) / time_delta
        dq_dt = (current.q - previous.q) / time_delta
        df_dt = (current.f - previous.f) / time_delta
        
        # Apply dynamic coherence equation
        velocity = (
            dpsi_dt * (1 + current.rho + current.f) +
            drho_dt * current.psi +
            dq_dt +
            df_dt * current.psi
        )
        
        return velocity
    
    def optimize_q_parameter(self, individual_data: Dict) -> float:
        """
        Calculate optimal q (moral activation energy) for an individual.
        Uses sigmoid curve with personalized parameters.
        
        Args:
            individual_data: Dictionary containing:
                - age: int
                - moral_development_stage: int (1-6, Kohlberg scale)
                - ethical_sensitivity: float (0.0-1.0)
                - context_factors: dict
                
        Returns:
            float: Optimized q value (0.0-1.0)
        """
        # Default values if data missing
        age = individual_data.get('age', 30)
        moral_stage = individual_data.get('moral_development_stage', 4)
        ethical_sensitivity = individual_data.get('ethical_sensitivity', 0.5)
        context = individual_data.get('context_factors', {})
        
        # Age-based baseline (sigmoid curve peaking around 25-35)
        age_factor = 1 / (1 + np.exp(-0.1 * (age - 30))) * 0.8 + 0.2
        
        # Moral development factor (higher stages = higher baseline q)
        moral_factor = (moral_stage - 1) / 5.0  # Normalize to 0.0-1.0
        
        # Context modifiers
        stress_level = context.get('stress_level', 0.5)
        social_support = context.get('social_support', 0.5)
        
        # Calculate optimal q with weighted factors
        q_optimal = (
            0.3 * age_factor +
            0.3 * moral_factor +
            0.2 * ethical_sensitivity +
            0.1 * (1 - stress_level) +
            0.1 * social_support
        )
        
        return np.clip(q_optimal, 0.0, 1.0)
    
    def assess_coherence_from_text(self, text: str, context: Optional[Dict] = None) -> CoherenceVariables:
        """
        Extract coherence variables from text using NLP analysis.
        
        Args:
            text: Input text to analyze
            context: Optional context dictionary with user history, demographics, etc.
            
        Returns:
            CoherenceVariables: Assessed coherence state
        """
        if not text:
            return CoherenceVariables()
        
        # Initialize variables
        psi = 0.5  # Internal consistency
        rho = 0.5  # Accumulated wisdom
        q = 0.5    # Moral activation
        f = 0.5    # Social belonging
        
        # 1. Analyze Psi (Internal Consistency)
        psi = self._analyze_internal_consistency(text)
        
        # 2. Analyze Rho (Accumulated Wisdom)
        rho = self._analyze_wisdom_indicators(text, context)
        
        # 3. Analyze q (Moral Activation)
        q = self._analyze_moral_activation(text)
        
        # 4. Analyze f (Social Belonging)
        f = self._analyze_social_belonging(text)
        
        # Apply context modifiers if available
        if context:
            # Adjust based on user history
            if 'coherence_history' in context:
                history_avg = np.mean([h.psi for h in context['coherence_history'][-10:]])
                psi = 0.7 * psi + 0.3 * history_avg  # Smooth with history
            
            # Adjust based on user profile
            if 'user_profile' in context:
                profile = context['user_profile']
                if 'baseline_wisdom' in profile:
                    rho = 0.8 * rho + 0.2 * profile['baseline_wisdom']
        
        return CoherenceVariables(psi=psi, rho=rho, q=q, f=f)
    
    def _analyze_internal_consistency(self, text: str) -> float:
        """
        Analyze internal consistency (Psi) from text.
        Looks for logical structure, coherent thought patterns, and consistency.
        """
        psi_score = 0.5
        
        # 1. Sentiment consistency
        sentences = nltk.sent_tokenize(text)
        if sentences:
            sentiments = [self.sia.polarity_scores(sent)['compound'] for sent in sentences]
            if len(sentiments) > 1:
                # Low variance in sentiment = higher consistency
                sentiment_variance = np.var(sentiments)
                psi_score += (1 - min(sentiment_variance * 2, 1)) * 0.2
        
        # 2. Linguistic coherence using spaCy
        if self.nlp and len(text) > 10:
            doc = self.nlp(text)
            
            # Check for complete sentences
            complete_sentences = sum(1 for sent in doc.sents if len(sent) > 3)
            sentence_completeness = complete_sentences / max(len(list(doc.sents)), 1)
            psi_score += sentence_completeness * 0.2
            
            # Check for logical connectors
            logical_connectors = ['because', 'therefore', 'however', 'thus', 'since']
            connector_count = sum(1 for token in doc if token.text.lower() in logical_connectors)
            connector_ratio = min(connector_count / (len(doc) / 50), 1)  # Normalize
            psi_score += connector_ratio * 0.1
        
        # 3. Topic consistency (simplified without advanced models)
        # Check for repeated key concepts
        words = text.lower().split()
        if len(words) > 10:
            word_freq = {}
            for word in words:
                if len(word) > 4:  # Focus on meaningful words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Higher frequency of key terms = more focused/consistent
            if word_freq:
                max_freq = max(word_freq.values())
                consistency_ratio = max_freq / len(words)
                psi_score += min(consistency_ratio * 5, 0.3)
        
        return np.clip(psi_score, 0.0, 1.0)
    
    def _analyze_wisdom_indicators(self, text: str, context: Optional[Dict] = None) -> float:
        """
        Analyze accumulated wisdom (Rho) from text.
        Looks for perspective-taking, nuanced thinking, and experience-based insights.
        """
        rho_score = 0.5
        
        # 1. Perspective-taking phrases
        perspective_phrases = [
            'on the other hand', 'from another perspective', 'considering',
            'it depends', 'in my experience', 'i\'ve learned', 'looking back',
            'alternatively', 'multiple factors', 'it\'s complex'
        ]
        
        text_lower = text.lower()
        perspective_count = sum(1 for phrase in perspective_phrases if phrase in text_lower)
        rho_score += min(perspective_count * 0.1, 0.3)
        
        # 2. Temporal references (past experience)
        temporal_markers = ['used to', 'previously', 'in the past', 'learned from',
                          'experience taught', 'over time', 'gradually']
        temporal_count = sum(1 for marker in temporal_markers if marker in text_lower)
        rho_score += min(temporal_count * 0.05, 0.2)
        
        # 3. Balanced emotional tone (wisdom often shows emotional regulation)
        sentiment = self.sia.polarity_scores(text)
        # Moderate sentiment (not extreme) indicates wisdom
        emotional_balance = 1 - abs(sentiment['compound'])
        rho_score += emotional_balance * 0.2
        
        # 4. Context-based adjustments
        if context and 'user_age' in context:
            # Older users might have higher baseline wisdom
            age = context['user_age']
            if age > 40:
                rho_score += 0.1
            elif age > 60:
                rho_score += 0.2
        
        return np.clip(rho_score, 0.0, 1.0)
    
    def _analyze_moral_activation(self, text: str) -> float:
        """
        Analyze moral activation energy (q) from text.
        Looks for ethical considerations, values-based reasoning, and moral language.
        """
        q_score = 0.5
        
        # 1. Moral/ethical vocabulary
        moral_terms = [
            'right', 'wrong', 'should', 'shouldn\'t', 'ought', 'duty',
            'responsibility', 'ethical', 'moral', 'fair', 'unfair',
            'justice', 'honest', 'integrity', 'values', 'principle',
            'conscience', 'harm', 'help', 'care', 'respect'
        ]
        
        text_lower = text.lower()
        words = text_lower.split()
        moral_count = sum(1 for word in words if word in moral_terms)
        moral_density = moral_count / max(len(words), 1)
        q_score += min(moral_density * 10, 0.3)
        
        # 2. Consideration of others
        other_focused = ['they', 'them', 'others', 'people', 'someone', 'everyone',
                        'community', 'society', 'together']
        other_count = sum(1 for word in words if word in other_focused)
        other_ratio = other_count / max(len(words), 1)
        q_score += min(other_ratio * 5, 0.2)
        
        # 3. Emotion classification for moral emotions
        if self.emotion_classifier:
            try:
                emotions = self.emotion_classifier(text[:512])  # Limit length
                # Look for moral emotions: guilt, shame, pride, gratitude
                moral_emotions = ['guilt', 'gratitude', 'pride']
                for emotion_set in emotions:
                    for emotion in emotion_set:
                        if emotion['label'].lower() in moral_emotions:
                            q_score += emotion['score'] * 0.2
            except:
                pass
        
        # 4. Question patterns (moral questioning)
        moral_questions = ['should i', 'is it right', 'would it be wrong',
                          'what\'s the right', 'am i being']
        question_count = sum(1 for pattern in moral_questions if pattern in text_lower)
        q_score += min(question_count * 0.1, 0.2)
        
        return np.clip(q_score, 0.0, 1.0)
    
    def _analyze_social_belonging(self, text: str) -> float:
        """
        Analyze social belonging architecture (f) from text.
        Looks for social connections, relationships, and community orientation.
        """
        f_score = 0.5
        
        # 1. Social pronouns and references
        social_terms = [
            'we', 'us', 'our', 'together', 'friend', 'family', 'partner',
            'team', 'group', 'community', 'everyone', 'colleagues', 'loved ones'
        ]
        
        text_lower = text.lower()
        words = text_lower.split()
        social_count = sum(1 for word in words if word in social_terms)
        social_density = social_count / max(len(words), 1)
        f_score += min(social_density * 8, 0.3)
        
        # 2. Relationship quality indicators
        positive_social = ['support', 'help', 'care', 'love', 'trust', 'share',
                          'connect', 'belong', 'appreciate', 'grateful']
        negative_social = ['alone', 'isolated', 'lonely', 'rejected', 'excluded',
                          'disconnected', 'abandoned']
        
        positive_count = sum(1 for word in words if word in positive_social)
        negative_count = sum(1 for word in words if word in negative_social)
        
        # Net social sentiment
        net_social = (positive_count - negative_count) / max(len(words), 1)
        f_score += np.clip(net_social * 5, -0.2, 0.3)
        
        # 3. Emotion analysis for social emotions
        sentiment = self.sia.polarity_scores(text)
        # Positive sentiment often correlates with social connection
        if sentiment['pos'] > sentiment['neg']:
            f_score += sentiment['pos'] * 0.2
        
        # 4. Inclusive language
        inclusive_terms = ['everyone', 'all of us', 'anybody', 'anybody can',
                          'we all', 'together we', 'our shared']
        inclusive_count = sum(1 for term in inclusive_terms if term in text_lower)
        f_score += min(inclusive_count * 0.1, 0.2)
        
        return np.clip(f_score, 0.0, 1.0)
    
    def get_coherence_state_label(self, coherence_score: float) -> str:
        """
        Convert coherence score to human-readable state label.
        
        Args:
            coherence_score: Normalized coherence score (0.0-1.0)
            
        Returns:
            str: State label (Crisis, Low, Medium, High, Optimal)
        """
        if coherence_score < 0.2:
            return "Crisis"
        elif coherence_score < 0.4:
            return "Low"
        elif coherence_score < 0.6:
            return "Medium"
        elif coherence_score < 0.8:
            return "High"
        else:
            return "Optimal"
    
    def calculate_intervention_priority(self, 
                                      current: CoherenceVariables,
                                      history: Optional[List[CoherenceVariables]] = None) -> Dict[str, float]:
        """
        Calculate which coherence variables need intervention priority.
        
        Args:
            current: Current coherence state
            history: Optional historical data
            
        Returns:
            Dict with intervention priorities for each variable
        """
        priorities = {}
        
        # Base priority on how far from optimal each variable is
        priorities['psi'] = 1.0 - current.psi
        priorities['rho'] = 1.0 - current.rho
        priorities['q'] = 1.0 - current.q
        priorities['f'] = 1.0 - current.f
        
        # Adjust based on velocity if history available
        if history and len(history) >= 2:
            velocity = self.calculate_coherence_velocity(history + [current])
            
            # If declining rapidly, increase priority
            if velocity < -0.1:
                for key in priorities:
                    priorities[key] *= 1.5
        
        # Psi (internal consistency) is foundational - prioritize if very low
        if current.psi < 0.3:
            priorities['psi'] *= 2.0
        
        # Normalize priorities to sum to 1.0
        total = sum(priorities.values())
        if total > 0:
            priorities = {k: v/total for k, v in priorities.items()}
        
        return priorities