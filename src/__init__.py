"""
Coherence-Aware AI Framework (CAAF)

A drop-in coherence measurement and response optimization system
that transforms generic AI responses into personalized, coherence-enhancing interactions.
"""

from .gct_core import GCTEngine, CoherenceVariables
from .coherence_engine import CoherenceEngine, CoherenceAssessment, CoherenceProfile
from .response_optimizer import ResponseOptimizer, CoherenceState
from .metrics_tracker import MetricsTracker, CoherenceRecord

__version__ = "0.1.0"
__author__ = "CAAF Team"

__all__ = [
    "GCTEngine",
    "CoherenceVariables",
    "CoherenceEngine", 
    "CoherenceAssessment",
    "CoherenceProfile",
    "ResponseOptimizer",
    "CoherenceState",
    "MetricsTracker",
    "CoherenceRecord"
]