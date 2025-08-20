"""
Response Optimizer - Modifies AI responses based on user's coherence state

This module transforms generic AI responses into coherence-aware responses
that are optimized for the user's current psychological state.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import re
import numpy as np
import logging
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from .coherence_engine import CoherenceStateEnum, CoherenceProfile

# Initialize NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    english_stopwords = set(stopwords.words('english'))
except:
    english_stopwords = set()

logger = logging.getLogger(__name__)


@dataclass
class CoherenceState:
    """Current coherence state information."""
    state: CoherenceStateEnum
    score: float
    psi: float  # Internal consistency
    rho: float  # Wisdom
    q: float    # Moral activation
    f: float    # Social belonging
    velocity: float = 0.0  # Rate of change
    risk_factors: List[str] = None
    protective_factors: List[str] = None
    
    def __post_init__(self):
        if self.risk_factors is None:
            self.risk_factors = []
        if self.protective_factors is None:
            self.protective_factors = []


class OptimizationStrategy(Enum):
    """Response optimization strategies based on coherence state."""
    CRISIS_STABILIZATION = "crisis_stabilization"
    GROUNDING = "grounding"
    SUPPORTIVE_GROWTH = "supportive_growth"
    CHALLENGING_GROWTH = "challenging_growth"
    WISDOM_CULTIVATION = "wisdom_cultivation"


class ResponseOptimizer:
    """
    Optimizes AI responses based on user's coherence state.
    """
    
    def __init__(self):
        """Initialize the Response Optimizer."""
        self._init_response_templates()
        self._init_complexity_markers()
    
    def _init_response_templates(self):
        """Initialize response modification templates."""
        self.crisis_phrases = {
            'opening': [
                "I hear that you're going through something really difficult right now.",
                "It sounds like things are really tough for you at the moment.",
                "I can sense you're in a lot of pain right now."
            ],
            'grounding': [
                "Let's take this one step at a time.",
                "Right now, focusing on this moment might help.",
                "Sometimes when things feel overwhelming, starting with something small can help."
            ],
            'support': [
                "You don't have to go through this alone.",
                "It's okay to feel this way.",
                "These feelings are valid and understandable."
            ],
            'resources': [
                "If you're having thoughts of self-harm, please reach out to a crisis helpline.",
                "Speaking with a mental health professional could provide additional support.",
                "There are people who want to help and support you."
            ]
        }
        
        self.growth_phrases = {
            'acknowledgment': [
                "I appreciate your thoughtful perspective on this.",
                "Your insight here shows real depth of understanding.",
                "That's a nuanced way of looking at it."
            ],
            'challenge': [
                "Have you considered how this might look from another angle?",
                "What would happen if we pushed this thinking a bit further?",
                "There's an interesting tension here worth exploring."
            ],
            'wisdom': [
                "Your experience seems to have given you valuable perspective.",
                "This reflects the kind of wisdom that comes from real reflection.",
                "You're integrating multiple viewpoints in a sophisticated way."
            ]
        }
    
    def _init_complexity_markers(self):
        """Initialize markers for text complexity."""
        self.simple_words = {
            'complex': 'hard', 'utilize': 'use', 'implement': 'do',
            'facilitate': 'help', 'demonstrate': 'show', 'indicate': 'show',
            'approximately': 'about', 'subsequent': 'next', 'prior': 'before'
        }
        
        self.complex_connectors = [
            'furthermore', 'nevertheless', 'consequently', 'notwithstanding',
            'henceforth', 'insofar', 'wherein', 'whereby'
        ]
        
        self.simple_connectors = [
            'and', 'but', 'so', 'because', 'also', 'then', 'next'
        ]
    
    def optimize_response(self,
                         original_response: str,
                         coherence_state: CoherenceState,
                         user_profile: Optional[CoherenceProfile] = None,
                         conversation_context: Optional[Dict] = None) -> str:
        """
        Optimize an AI response based on user's coherence state.
        
        Args:
            original_response: The original AI response
            coherence_state: Current coherence state
            user_profile: Optional user profile with history
            conversation_context: Optional conversation context
            
        Returns:
            str: Optimized response
        """
        # Select optimization strategy
        strategy = self._select_strategy(coherence_state, user_profile)
        
        # Apply base optimizations
        optimized = original_response
        
        # Strategy-specific optimizations
        if strategy == OptimizationStrategy.CRISIS_STABILIZATION:
            optimized = self._optimize_for_crisis(optimized, coherence_state)
        elif strategy == OptimizationStrategy.GROUNDING:
            optimized = self._optimize_for_grounding(optimized, coherence_state)
        elif strategy == OptimizationStrategy.SUPPORTIVE_GROWTH:
            optimized = self._optimize_for_supportive_growth(optimized, coherence_state)
        elif strategy == OptimizationStrategy.CHALLENGING_GROWTH:
            optimized = self._optimize_for_challenging_growth(optimized, coherence_state)
        elif strategy == OptimizationStrategy.WISDOM_CULTIVATION:
            optimized = self._optimize_for_wisdom(optimized, coherence_state)
        
        # Apply coherence-enhancing elements based on weak variables
        optimized = self._enhance_weak_variables(optimized, coherence_state)
        
        # Safety check
        if self.detect_potential_harm(optimized, coherence_state):
            optimized = self._make_response_safer(optimized, coherence_state)
        
        # Final adjustments
        optimized = self._apply_final_adjustments(optimized, coherence_state, user_profile)
        
        return optimized
    
    def adjust_complexity(self, text: str, target_complexity: float) -> str:
        """
        Adjust text complexity to match target level.
        
        Args:
            text: Input text
            target_complexity: Target complexity (0.0=simple, 1.0=complex)
            
        Returns:
            str: Adjusted text
        """
        current_complexity = self._assess_complexity(text)
        
        if abs(current_complexity - target_complexity) < 0.1:
            return text  # Already at target
        
        if target_complexity < current_complexity:
            # Simplify
            return self._simplify_text(text)
        else:
            # Complexify (less common, but possible)
            return self._complexify_text(text)
    
    def inject_coherence_enhancing_elements(self, 
                                          response: str, 
                                          weak_variables: List[str]) -> str:
        """
        Inject elements that strengthen weak coherence variables.
        
        Args:
            response: Original response
            weak_variables: List of weak variables ['psi', 'rho', 'q', 'f']
            
        Returns:
            str: Enhanced response
        """
        enhanced = response
        
        for var in weak_variables:
            if var == 'psi':
                enhanced = self._enhance_internal_consistency(enhanced)
            elif var == 'rho':
                enhanced = self._enhance_wisdom_elements(enhanced)
            elif var == 'q':
                enhanced = self._enhance_moral_elements(enhanced)
            elif var == 'f':
                enhanced = self._enhance_social_elements(enhanced)
        
        return enhanced
    
    def detect_potential_harm(self, response: str, user_state: CoherenceState) -> bool:
        """
        Detect if response could be harmful given user's state.
        
        Args:
            response: Response text
            user_state: Current coherence state
            
        Returns:
            bool: True if potential harm detected
        """
        response_lower = response.lower()
        
        # Check for potentially triggering content in crisis states
        if user_state.state in [CoherenceStateEnum.CRISIS, CoherenceStateEnum.LOW]:
            harmful_patterns = [
                'just think positive', 'just get over it', 'stop being',
                'you should be grateful', 'other people have it worse',
                'it\'s not that bad', 'you\'re overreacting', 'calm down',
                'just try harder', 'snap out of it'
            ]
            
            if any(pattern in response_lower for pattern in harmful_patterns):
                return True
        
        # Check for overly complex content when coherence is low
        if user_state.score < 0.4:
            complexity = self._assess_complexity(response)
            if complexity > 0.7:
                return True  # Too complex for current state
        
        # Check for lack of empathy in crisis
        if user_state.state == CoherenceStateEnum.CRISIS:
            empathy_markers = ['understand', 'hear', 'feel', 'support', 'help']
            if not any(marker in response_lower for marker in empathy_markers):
                return True
        
        return False
    
    def _select_strategy(self, 
                        coherence_state: CoherenceState, 
                        user_profile: Optional[CoherenceProfile]) -> OptimizationStrategy:
        """Select optimization strategy based on coherence state."""
        if coherence_state.state == CoherenceStateEnum.CRISIS:
            return OptimizationStrategy.CRISIS_STABILIZATION
        
        elif coherence_state.state == CoherenceStateEnum.LOW:
            return OptimizationStrategy.GROUNDING
        
        elif coherence_state.state == CoherenceStateEnum.MEDIUM:
            # Check trajectory
            if coherence_state.velocity > 0.1:
                return OptimizationStrategy.SUPPORTIVE_GROWTH
            else:
                return OptimizationStrategy.GROUNDING
        
        elif coherence_state.state == CoherenceStateEnum.HIGH:
            # Check if ready for challenge
            if coherence_state.psi > 0.7 and coherence_state.f > 0.6:
                return OptimizationStrategy.CHALLENGING_GROWTH
            else:
                return OptimizationStrategy.SUPPORTIVE_GROWTH
        
        else:  # OPTIMAL
            # Focus on wisdom cultivation
            return OptimizationStrategy.WISDOM_CULTIVATION
    
    def _optimize_for_crisis(self, response: str, state: CoherenceState) -> str:
        """Optimize response for crisis state."""
        # Start with empathetic opening
        opening = np.random.choice(self.crisis_phrases['opening'])
        
        # Simplify the response
        simplified = self._simplify_text(response)
        
        # Add grounding element
        grounding = np.random.choice(self.crisis_phrases['grounding'])
        
        # Add support statement
        support = np.random.choice(self.crisis_phrases['support'])
        
        # Construct crisis-optimized response
        optimized_parts = [opening]
        
        # Keep only most essential part of original response
        sentences = sent_tokenize(simplified)
        if sentences:
            # Take first 1-2 most relevant sentences
            core_message = '. '.join(sentences[:2])
            optimized_parts.append(core_message)
        
        optimized_parts.extend([grounding, support])
        
        # Add resources if very low coherence
        if state.score < 0.2:
            resource = np.random.choice(self.crisis_phrases['resources'])
            optimized_parts.append(resource)
        
        return ' '.join(optimized_parts)
    
    def _optimize_for_grounding(self, response: str, state: CoherenceState) -> str:
        """Optimize response for grounding (low coherence)."""
        # Simplify language
        simplified = self._simplify_text(response)
        
        # Break into smaller chunks
        sentences = sent_tokenize(simplified)
        
        # Add breathing room between ideas
        spaced_sentences = []
        for i, sent in enumerate(sentences):
            spaced_sentences.append(sent)
            if i < len(sentences) - 1 and len(sent.split()) > 10:
                spaced_sentences.append("")  # Add paragraph break
        
        grounded_response = '\n'.join(spaced_sentences)
        
        # Add a grounding statement
        if state.psi < 0.4:  # Low internal consistency
            grounding = "Let's focus on one thing at a time. "
            grounded_response = grounding + grounded_response
        
        return grounded_response
    
    def _optimize_for_supportive_growth(self, response: str, state: CoherenceState) -> str:
        """Optimize response for supportive growth (medium coherence)."""
        # Moderate complexity
        balanced = self.adjust_complexity(response, 0.5)
        
        # Add acknowledgment if showing improvement
        if state.velocity > 0:
            acknowledgment = np.random.choice([
                "I can see you're making progress with this. ",
                "Your thinking on this is developing nicely. ",
                "You're working through this thoughtfully. "
            ])
            balanced = acknowledgment + balanced
        
        # Add gentle encouragement
        if state.f < 0.5:  # Low social connection
            connection = "\n\nRemember, you're not alone in working through this."
            balanced += connection
        
        return balanced
    
    def _optimize_for_challenging_growth(self, response: str, state: CoherenceState) -> str:
        """Optimize response for challenging growth (high coherence)."""
        # Can handle more complexity
        response = self.adjust_complexity(response, 0.7)
        
        # Add thought-provoking elements
        sentences = sent_tokenize(response)
        
        # Insert a challenging question
        if len(sentences) > 2:
            challenge = np.random.choice(self.growth_phrases['challenge'])
            sentences.insert(len(sentences)//2, challenge)
        
        # Add acknowledgment of capability
        acknowledgment = np.random.choice(self.growth_phrases['acknowledgment'])
        
        return acknowledgment + ' ' + ' '.join(sentences)
    
    def _optimize_for_wisdom(self, response: str, state: CoherenceState) -> str:
        """Optimize response for wisdom cultivation (optimal coherence)."""
        # Maintain complexity
        response = self.adjust_complexity(response, 0.8)
        
        # Add wisdom acknowledgment
        wisdom_ack = np.random.choice(self.growth_phrases['wisdom'])
        
        # Add meta-cognitive elements
        meta_additions = [
            "\n\nHow does this connect with your broader understanding?",
            "\n\nWhat patterns do you notice emerging here?",
            "\n\nHow might this insight apply in other contexts?"
        ]
        
        meta_element = np.random.choice(meta_additions)
        
        return wisdom_ack + ' ' + response + meta_element
    
    def _enhance_weak_variables(self, response: str, state: CoherenceState) -> str:
        """Enhance response based on weak coherence variables."""
        weak_vars = []
        
        # Identify weak variables (below 0.4)
        if state.psi < 0.4:
            weak_vars.append('psi')
        if state.rho < 0.4:
            weak_vars.append('rho')
        if state.q < 0.4:
            weak_vars.append('q')
        if state.f < 0.4:
            weak_vars.append('f')
        
        if weak_vars:
            response = self.inject_coherence_enhancing_elements(response, weak_vars)
        
        return response
    
    def _enhance_internal_consistency(self, response: str) -> str:
        """Enhance elements that support internal consistency (Psi)."""
        # Add structure and logical flow
        sentences = sent_tokenize(response)
        
        if len(sentences) > 2:
            # Add transitional phrases
            transitions = ['First, ', 'Next, ', 'Finally, ']
            for i in range(min(3, len(sentences))):
                if not sentences[i].startswith(('First', 'Next', 'Finally', 'However', 'Therefore')):
                    sentences[i] = transitions[i] + sentences[i].lower()
        
        # Add summary if response is long
        if len(sentences) > 4:
            summary = "\n\nIn summary, the key point here is about finding clarity and coherence in your thoughts."
            return ' '.join(sentences) + summary
        
        return ' '.join(sentences)
    
    def _enhance_wisdom_elements(self, response: str) -> str:
        """Enhance elements that support wisdom (Rho)."""
        # Add perspective-taking language
        wisdom_additions = [
            " It might help to consider different perspectives on this.",
            " Experience often teaches us that there are multiple ways to view this.",
            " Time and reflection can bring new insights to situations like these."
        ]
        
        addition = np.random.choice(wisdom_additions)
        return response + addition
    
    def _enhance_moral_elements(self, response: str) -> str:
        """Enhance elements that support moral activation (q)."""
        # Add values-based language
        moral_additions = [
            " What feels right to you in this situation?",
            " Consider what aligns with your core values here.",
            " Think about what would feel most authentic to who you are."
        ]
        
        addition = np.random.choice(moral_additions)
        return response + addition
    
    def _enhance_social_elements(self, response: str) -> str:
        """Enhance elements that support social belonging (f)."""
        # Add connection-oriented language
        social_additions = [
            " You're not alone in facing challenges like this.",
            " Many people have navigated similar situations.",
            " Connecting with others who understand can be valuable."
        ]
        
        addition = np.random.choice(social_additions)
        return response + addition
    
    def _simplify_text(self, text: str) -> str:
        """Simplify text for better comprehension."""
        # Replace complex words
        for complex_word, simple_word in self.simple_words.items():
            text = re.sub(r'\b' + complex_word + r'\b', simple_word, text, flags=re.IGNORECASE)
        
        # Replace complex connectors
        for complex_conn in self.complex_connectors:
            simple_replacement = np.random.choice(self.simple_connectors[:3])
            text = re.sub(r'\b' + complex_conn + r'\b', simple_replacement, text, flags=re.IGNORECASE)
        
        # Shorten sentences
        sentences = sent_tokenize(text)
        simplified_sentences = []
        
        for sent in sentences:
            words = sent.split()
            if len(words) > 20:
                # Break long sentences
                mid_point = len(words) // 2
                # Find good breaking point
                for i in range(mid_point - 5, mid_point + 5):
                    if i < len(words) and words[i] in [',', ';', 'and', 'but']:
                        first_part = ' '.join(words[:i])
                        second_part = ' '.join(words[i+1:])
                        simplified_sentences.extend([first_part + '.', second_part.capitalize()])
                        break
                else:
                    simplified_sentences.append(sent)
            else:
                simplified_sentences.append(sent)
        
        return ' '.join(simplified_sentences)
    
    def _complexify_text(self, text: str) -> str:
        """Add complexity to text (rarely needed)."""
        # This is a placeholder - in practice, complexifying is rarely needed
        # as AI responses tend to be appropriately complex already
        return text
    
    def _assess_complexity(self, text: str) -> float:
        """Assess text complexity (0.0=simple, 1.0=complex)."""
        complexity_score = 0.0
        
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        if not words or not sentences:
            return 0.5
        
        # Average sentence length
        avg_sent_length = len(words) / len(sentences)
        complexity_score += min(avg_sent_length / 30, 0.3)  # Cap at 0.3
        
        # Complex word ratio
        complex_word_count = sum(1 for word in words if len(word) > 8 and word not in english_stopwords)
        complex_ratio = complex_word_count / len(words)
        complexity_score += min(complex_ratio * 3, 0.3)  # Cap at 0.3
        
        # Vocabulary diversity
        unique_words = set(words) - english_stopwords
        diversity = len(unique_words) / len(words) if words else 0
        complexity_score += min(diversity * 2, 0.2)  # Cap at 0.2
        
        # Complex connector usage
        complex_conn_count = sum(1 for conn in self.complex_connectors if conn in text.lower())
        complexity_score += min(complex_conn_count * 0.1, 0.2)  # Cap at 0.2
        
        return min(complexity_score, 1.0)
    
    def _make_response_safer(self, response: str, state: CoherenceState) -> str:
        """Make response safer for user's current state."""
        if state.state == CoherenceStateEnum.CRISIS:
            # Remove any potentially triggering content
            safe_response = self._remove_triggers(response)
            # Add crisis support
            safe_response += "\n\nYour wellbeing is important. Please reach out for support if you need it."
            return safe_response
        else:
            # Just simplify if needed
            return self._simplify_text(response)
    
    def _remove_triggers(self, text: str) -> str:
        """Remove potentially triggering content."""
        triggers = [
            'just think positive', 'just get over it', 'stop being',
            'you should be grateful', 'other people have it worse',
            'it\'s not that bad', 'you\'re overreacting', 'calm down'
        ]
        
        for trigger in triggers:
            text = re.sub(trigger, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _apply_final_adjustments(self, 
                                response: str, 
                                state: CoherenceState,
                                profile: Optional[CoherenceProfile]) -> str:
        """Apply final adjustments based on state and profile."""
        # Ensure appropriate length
        if state.state == CoherenceStateEnum.CRISIS:
            # Keep it concise
            sentences = sent_tokenize(response)
            if len(sentences) > 5:
                response = ' '.join(sentences[:5])
        
        # Add hope in low states
        if state.score < 0.4 and 'hope' not in response.lower():
            hope_additions = [
                " Things can improve with time and support.",
                " This is a difficult moment, but it's not permanent.",
                " Small steps forward are still progress."
            ]
            response += ' ' + np.random.choice(hope_additions)
        
        # Clean up any double spaces or formatting issues
        response = re.sub(r'\s+', ' ', response)
        response = re.sub(r'\n\n+', '\n\n', response)
        
        return response.strip()