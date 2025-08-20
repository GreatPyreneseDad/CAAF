"""
REST API for Coherence-Aware AI Framework

Provides simple integration endpoints for any LLM platform.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import os
import logging
from contextlib import asynccontextmanager
import asyncio
from functools import lru_cache

from .coherence_engine import CoherenceEngine, CoherenceProfile, CoherenceStateEnum
from .response_optimizer import ResponseOptimizer, CoherenceState
from .metrics_tracker import MetricsTracker, CoherenceRecord
from .gct_core import CoherenceVariables

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
coherence_engine = None
response_optimizer = None
metrics_tracker = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    global coherence_engine, response_optimizer, metrics_tracker
    
    logger.info("Initializing CAAF components...")
    coherence_engine = CoherenceEngine()
    response_optimizer = ResponseOptimizer()
    
    # Initialize metrics tracker with optional database URL
    db_url = os.getenv("CAAF_DATABASE_URL")
    metrics_tracker = MetricsTracker(database_url=db_url)
    
    logger.info("CAAF API ready")
    
    yield
    
    # Shutdown
    logger.info("Shutting down CAAF API")


# Initialize FastAPI app
app = FastAPI(
    title="Coherence-Aware AI Framework API",
    description="Transform generic AI responses into coherence-aware interactions",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class AssessCoherenceRequest(BaseModel):
    """Request model for coherence assessment."""
    message: str = Field(..., description="User message to assess")
    user_id: str = Field(..., description="Unique user identifier")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Optional context information")
    timestamp: Optional[datetime] = Field(default=None, description="Message timestamp")
    
    @validator('message')
    def message_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        return v
    
    @validator('user_id')
    def user_id_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('User ID cannot be empty')
        return v


class AssessCoherenceResponse(BaseModel):
    """Response model for coherence assessment."""
    user_id: str
    timestamp: str
    coherence_score: float = Field(..., ge=0.0, le=1.0)
    state: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    variables: Dict[str, float]
    intervention_priorities: Dict[str, float]
    risk_factors: List[str]
    protective_factors: List[str]


class OptimizeResponseRequest(BaseModel):
    """Request model for response optimization."""
    original_response: str = Field(..., description="Original AI response to optimize")
    user_id: str = Field(..., description="User ID for coherence lookup")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None, 
        description="Recent conversation history"
    )
    optimization_level: Optional[str] = Field(
        default="automatic",
        description="Optimization level: automatic, minimal, moderate, aggressive"
    )


class OptimizeResponseResponse(BaseModel):
    """Response model for response optimization."""
    optimized_response: str
    optimization_strategy: str
    modifications_applied: List[str]
    coherence_state: Dict[str, Any]
    safety_checks_passed: bool


class CoherenceProfileResponse(BaseModel):
    """Response model for coherence profile."""
    user_id: str
    baseline_coherence: float
    baseline_variables: Dict[str, float]
    volatility: float
    trend: str
    risk_factors: List[str]
    protective_factors: List[str]
    assessment_count: int
    last_updated: str


class CoherenceMetricsRequest(BaseModel):
    """Request model for coherence metrics."""
    user_id: str
    timeframe: str = Field(default="7d", description="Timeframe: 1h, 24h, 7d, 30d, all")
    metric_type: Optional[str] = Field(default="summary", description="Type: summary, detailed, patterns")


class IntegrationTestRequest(BaseModel):
    """Request model for integration testing."""
    test_messages: List[str] = Field(..., description="Test messages to process")
    test_responses: List[str] = Field(..., description="Test AI responses to optimize")
    user_id: Optional[str] = Field(default="test_user", description="Test user ID")


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Coherence-Aware AI Framework",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "POST /assess-coherence": "Assess coherence from user message",
            "POST /optimize-response": "Optimize AI response for coherence",
            "GET /coherence-profile/{user_id}": "Get user's coherence profile",
            "GET /coherence-metrics/{user_id}": "Get coherence metrics",
            "POST /integration-test": "Test the integration"
        }
    }


@app.post("/assess-coherence", response_model=AssessCoherenceResponse)
async def assess_coherence(
    request: AssessCoherenceRequest,
    background_tasks: BackgroundTasks
):
    """
    Assess coherence from a user message.
    
    This endpoint analyzes a user's message to determine their current coherence state
    based on Grounded Coherence Theory.
    """
    try:
        # Process message
        assessment = coherence_engine.process_message(
            message=request.message,
            user_id=request.user_id,
            timestamp=request.timestamp,
            context=request.context
        )
        
        # Get user profile
        profile = coherence_engine.get_user_coherence_profile(request.user_id)
        
        # Record metrics in background
        background_tasks.add_task(
            record_metrics,
            assessment=assessment
        )
        
        # Prepare response
        return AssessCoherenceResponse(
            user_id=request.user_id,
            timestamp=assessment.timestamp.isoformat(),
            coherence_score=assessment.coherence_score,
            state=assessment.state.value,
            confidence=assessment.confidence,
            variables={
                "psi": assessment.variables.psi,
                "rho": assessment.variables.rho,
                "q": assessment.variables.q,
                "f": assessment.variables.f
            },
            intervention_priorities=assessment.intervention_priorities,
            risk_factors=profile.risk_factors if profile else [],
            protective_factors=profile.protective_factors if profile else []
        )
        
    except Exception as e:
        logger.error(f"Error in assess_coherence: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize-response", response_model=OptimizeResponseResponse)
async def optimize_response(
    request: OptimizeResponseRequest,
    background_tasks: BackgroundTasks
):
    """
    Optimize an AI response based on user's coherence state.
    
    This endpoint modifies AI responses to be more appropriate for the user's
    current psychological coherence state.
    """
    try:
        # Get user's latest coherence state
        profile = coherence_engine.get_user_coherence_profile(request.user_id)
        
        if not profile:
            # No history - assess from conversation if available
            if request.conversation_history and len(request.conversation_history) > 0:
                last_message = request.conversation_history[-1].get("user", "")
                assessment = coherence_engine.process_message(
                    message=last_message,
                    user_id=request.user_id
                )
                profile = coherence_engine.get_user_coherence_profile(request.user_id)
            else:
                # Use default medium coherence
                coherence_state = CoherenceState(
                    state=CoherenceStateEnum.MEDIUM,
                    score=0.5,
                    psi=0.5,
                    rho=0.5,
                    q=0.5,
                    f=0.5
                )
        else:
            # Get latest assessment
            history = coherence_engine._get_user_history(request.user_id)
            if history:
                latest = history[-1]
                coherence_state = CoherenceState(
                    state=latest.state,
                    score=latest.coherence_score,
                    psi=latest.variables.psi,
                    rho=latest.variables.rho,
                    q=latest.variables.q,
                    f=latest.variables.f,
                    velocity=coherence_engine.get_coherence_velocity(request.user_id),
                    risk_factors=profile.risk_factors,
                    protective_factors=profile.protective_factors
                )
            else:
                # Use profile baselines
                coherence_state = CoherenceState(
                    state=CoherenceStateEnum.MEDIUM,
                    score=profile.baseline_coherence,
                    psi=profile.baseline_psi,
                    rho=profile.baseline_rho,
                    q=profile.baseline_q,
                    f=profile.baseline_f
                )
        
        # Optimize response
        optimized = response_optimizer.optimize_response(
            original_response=request.original_response,
            coherence_state=coherence_state,
            user_profile=profile,
            conversation_context={"history": request.conversation_history}
        )
        
        # Track optimization metrics
        background_tasks.add_task(
            record_optimization,
            user_id=request.user_id,
            coherence_state=coherence_state,
            optimized=optimized != request.original_response
        )
        
        # Determine modifications applied
        modifications = []
        if len(optimized) < len(request.original_response) * 0.8:
            modifications.append("simplified")
        if coherence_state.state == CoherenceStateEnum.CRISIS:
            modifications.append("crisis_support_added")
        if coherence_state.psi < 0.4:
            modifications.append("structure_enhanced")
        if coherence_state.f < 0.4:
            modifications.append("social_support_added")
        
        return OptimizeResponseResponse(
            optimized_response=optimized,
            optimization_strategy=response_optimizer._select_strategy(
                coherence_state, profile
            ).value,
            modifications_applied=modifications,
            coherence_state={
                "state": coherence_state.state.value,
                "score": coherence_state.score,
                "variables": {
                    "psi": coherence_state.psi,
                    "rho": coherence_state.rho,
                    "q": coherence_state.q,
                    "f": coherence_state.f
                }
            },
            safety_checks_passed=not response_optimizer.detect_potential_harm(
                optimized, coherence_state
            )
        )
        
    except Exception as e:
        logger.error(f"Error in optimize_response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/coherence-profile/{user_id}", response_model=CoherenceProfileResponse)
async def get_coherence_profile(user_id: str):
    """
    Get a user's coherence profile.
    
    Returns comprehensive coherence profile including baselines, trends,
    and risk/protective factors.
    """
    try:
        profile = coherence_engine.get_user_coherence_profile(user_id)
        
        if not profile:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        return CoherenceProfileResponse(
            user_id=user_id,
            baseline_coherence=profile.baseline_coherence,
            baseline_variables={
                "psi": profile.baseline_psi,
                "rho": profile.baseline_rho,
                "q": profile.baseline_q,
                "f": profile.baseline_f
            },
            volatility=profile.volatility,
            trend=profile.trend,
            risk_factors=profile.risk_factors,
            protective_factors=profile.protective_factors,
            assessment_count=profile.assessment_count,
            last_updated=profile.last_updated.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_coherence_profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/coherence-metrics/{user_id}")
async def get_coherence_metrics(
    user_id: str,
    timeframe: str = "7d",
    metric_type: str = "summary"
):
    """
    Get coherence metrics for a user.
    
    Returns various metrics based on the requested type and timeframe.
    """
    try:
        # Parse timeframe
        if timeframe == "1h":
            start_date = datetime.now() - timedelta(hours=1)
        elif timeframe == "24h":
            start_date = datetime.now() - timedelta(days=1)
        elif timeframe == "7d":
            start_date = datetime.now() - timedelta(days=7)
        elif timeframe == "30d":
            start_date = datetime.now() - timedelta(days=30)
        else:
            start_date = None
        
        if metric_type == "summary":
            # Get summary metrics
            df = metrics_tracker.get_user_metrics(user_id, start_date=start_date)
            
            if df.empty:
                return {
                    "user_id": user_id,
                    "timeframe": timeframe,
                    "status": "no_data",
                    "metrics": {}
                }
            
            summary = {
                "avg_coherence": float(df['coherence_score'].mean()),
                "min_coherence": float(df['coherence_score'].min()),
                "max_coherence": float(df['coherence_score'].max()),
                "std_coherence": float(df['coherence_score'].std()),
                "total_assessments": len(df),
                "crisis_count": int((df['coherence_state'] == 'crisis').sum()),
                "optimal_count": int((df['coherence_state'] == 'optimal').sum()),
                "improvement_metrics": metrics_tracker.calculate_improvement_metrics(
                    user_id, 
                    window_days=30 if timeframe == "30d" else 7
                )
            }
            
            return {
                "user_id": user_id,
                "timeframe": timeframe,
                "status": "success",
                "metrics": summary
            }
            
        elif metric_type == "detailed":
            # Get detailed time series
            df = metrics_tracker.get_user_metrics(user_id, start_date=start_date)
            
            if df.empty:
                return {
                    "user_id": user_id,
                    "timeframe": timeframe,
                    "status": "no_data",
                    "data": []
                }
            
            # Convert to list of dicts
            data = df.to_dict('records')
            
            # Convert timestamps to ISO format
            for record in data:
                record['timestamp'] = record['timestamp'].isoformat()
            
            return {
                "user_id": user_id,
                "timeframe": timeframe,
                "status": "success",
                "data": data
            }
            
        elif metric_type == "patterns":
            # Get pattern analysis
            window_days = 30 if timeframe == "30d" else 7
            patterns = metrics_tracker.get_pattern_analysis(user_id, window_days)
            
            return {
                "user_id": user_id,
                "timeframe": timeframe,
                "status": "success",
                "patterns": patterns
            }
            
        else:
            raise HTTPException(status_code=400, detail="Invalid metric_type")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_coherence_metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/integration-test")
async def integration_test(request: IntegrationTestRequest):
    """
    Test the CAAF integration with sample data.
    
    Useful for verifying that the framework is working correctly.
    """
    try:
        results = {
            "user_id": request.user_id,
            "assessments": [],
            "optimizations": [],
            "overall_status": "success"
        }
        
        # Test coherence assessment
        for i, message in enumerate(request.test_messages):
            try:
                assessment = coherence_engine.process_message(
                    message=message,
                    user_id=request.user_id,
                    timestamp=datetime.now() + timedelta(minutes=i)
                )
                
                results["assessments"].append({
                    "message_index": i,
                    "coherence_score": assessment.coherence_score,
                    "state": assessment.state.value,
                    "success": True
                })
            except Exception as e:
                results["assessments"].append({
                    "message_index": i,
                    "error": str(e),
                    "success": False
                })
                results["overall_status"] = "partial_failure"
        
        # Test response optimization
        profile = coherence_engine.get_user_coherence_profile(request.user_id)
        
        for i, response in enumerate(request.test_responses):
            try:
                # Get latest coherence state
                history = coherence_engine._get_user_history(request.user_id)
                if history:
                    latest = history[-1]
                    coherence_state = CoherenceState(
                        state=latest.state,
                        score=latest.coherence_score,
                        psi=latest.variables.psi,
                        rho=latest.variables.rho,
                        q=latest.variables.q,
                        f=latest.variables.f
                    )
                else:
                    coherence_state = CoherenceState(
                        state=CoherenceStateEnum.MEDIUM,
                        score=0.5,
                        psi=0.5,
                        rho=0.5,
                        q=0.5,
                        f=0.5
                    )
                
                optimized = response_optimizer.optimize_response(
                    original_response=response,
                    coherence_state=coherence_state,
                    user_profile=profile
                )
                
                results["optimizations"].append({
                    "response_index": i,
                    "original_length": len(response),
                    "optimized_length": len(optimized),
                    "changed": optimized != response,
                    "success": True
                })
            except Exception as e:
                results["optimizations"].append({
                    "response_index": i,
                    "error": str(e),
                    "success": False
                })
                results["overall_status"] = "partial_failure"
        
        # Add summary
        results["summary"] = {
            "total_assessments": len(request.test_messages),
            "successful_assessments": sum(1 for a in results["assessments"] if a["success"]),
            "total_optimizations": len(request.test_responses),
            "successful_optimizations": sum(1 for o in results["optimizations"] if o["success"]),
            "responses_changed": sum(1 for o in results["optimizations"] if o.get("changed", False))
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error in integration_test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "coherence_engine": coherence_engine is not None,
            "response_optimizer": response_optimizer is not None,
            "metrics_tracker": metrics_tracker is not None
        }
    }


# Background tasks
async def record_metrics(assessment):
    """Record assessment metrics in background."""
    try:
        record = CoherenceRecord(
            user_id=assessment.user_id,
            timestamp=assessment.timestamp,
            psi=assessment.variables.psi,
            rho=assessment.variables.rho,
            q=assessment.variables.q,
            f=assessment.variables.f,
            coherence_score=assessment.coherence_score,
            coherence_velocity=0.0,  # Will be calculated by tracker
            coherence_state=assessment.state.value,
            message_hash=assessment.message_hash
        )
        
        metrics_tracker.record_assessment(record)
        
    except Exception as e:
        logger.error(f"Error recording metrics: {e}")


async def record_optimization(user_id: str, coherence_state: CoherenceState, optimized: bool):
    """Record optimization event in background."""
    try:
        record = CoherenceRecord(
            user_id=user_id,
            timestamp=datetime.now(),
            psi=coherence_state.psi,
            rho=coherence_state.rho,
            q=coherence_state.q,
            f=coherence_state.f,
            coherence_score=coherence_state.score,
            coherence_velocity=coherence_state.velocity,
            coherence_state=coherence_state.state.value,
            response_optimized=optimized
        )
        
        metrics_tracker.record_assessment(record)
        
    except Exception as e:
        logger.error(f"Error recording optimization: {e}")


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred"}
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run with: python -m src.api
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )