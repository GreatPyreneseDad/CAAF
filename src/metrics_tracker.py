"""
Metrics Tracker - Store and analyze coherence data over time

This module provides privacy-preserving storage and analysis of coherence
metrics for research and improvement purposes.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
import sqlite3
import logging
from contextlib import contextmanager
import os

from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func

logger = logging.getLogger(__name__)

Base = declarative_base()


@dataclass
class CoherenceRecord:
    """Record of a single coherence assessment."""
    user_id: str
    timestamp: datetime
    psi: float
    rho: float
    q: float
    f: float
    coherence_score: float
    coherence_velocity: float
    coherence_state: str
    context_tags: List[str] = field(default_factory=list)
    message_hash: str = ""  # Privacy: only store hash
    intervention_applied: bool = False
    intervention_type: Optional[str] = None
    response_optimized: bool = False
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat(),
            'psi': self.psi,
            'rho': self.rho,
            'q': self.q,
            'f': self.f,
            'coherence_score': self.coherence_score,
            'coherence_velocity': self.coherence_velocity,
            'coherence_state': self.coherence_state,
            'context_tags': json.dumps(self.context_tags),
            'message_hash': self.message_hash,
            'intervention_applied': self.intervention_applied,
            'intervention_type': self.intervention_type,
            'response_optimized': self.response_optimized,
            'session_id': self.session_id
        }


class CoherenceMetric(Base):
    """SQLAlchemy model for coherence metrics."""
    __tablename__ = 'coherence_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(128), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    psi = Column(Float, nullable=False)
    rho = Column(Float, nullable=False)
    q = Column(Float, nullable=False)
    f = Column(Float, nullable=False)
    coherence_score = Column(Float, nullable=False)
    coherence_velocity = Column(Float, default=0.0)
    coherence_state = Column(String(50), nullable=False)
    context_tags = Column(Text, default='[]')
    message_hash = Column(String(64))
    intervention_applied = Column(Boolean, default=False)
    intervention_type = Column(String(100))
    response_optimized = Column(Boolean, default=False)
    session_id = Column(String(128))
    created_at = Column(DateTime, default=func.now())


class AggregateMetric(Base):
    """SQLAlchemy model for aggregate metrics."""
    __tablename__ = 'aggregate_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(128), nullable=False, index=True)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    period_type = Column(String(20))  # hourly, daily, weekly, monthly
    avg_coherence = Column(Float)
    min_coherence = Column(Float)
    max_coherence = Column(Float)
    std_coherence = Column(Float)
    avg_psi = Column(Float)
    avg_rho = Column(Float)
    avg_q = Column(Float)
    avg_f = Column(Float)
    crisis_count = Column(Integer, default=0)
    optimal_count = Column(Integer, default=0)
    total_assessments = Column(Integer, default=0)
    improvement_rate = Column(Float)
    created_at = Column(DateTime, default=func.now())


class MetricsTracker:
    """
    Tracks and analyzes coherence metrics over time with privacy protection.
    """
    
    def __init__(self, 
                 database_url: Optional[str] = None,
                 enable_aggregation: bool = True,
                 retention_days: int = 90):
        """
        Initialize the Metrics Tracker.
        
        Args:
            database_url: SQLAlchemy database URL (defaults to SQLite)
            enable_aggregation: Whether to compute aggregate metrics
            retention_days: Days to retain detailed records
        """
        if database_url is None:
            # Default to SQLite database
            db_path = Path.home() / '.caaf' / 'metrics.db'
            db_path.parent.mkdir(parents=True, exist_ok=True)
            database_url = f'sqlite:///{db_path}'
        
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        self.enable_aggregation = enable_aggregation
        self.retention_days = retention_days
        
        # Privacy settings
        self.hash_user_ids = True
        self.anonymize_after_days = 30
    
    @contextmanager
    def get_session(self) -> Session:
        """Get a database session."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def record_assessment(self, record: CoherenceRecord) -> bool:
        """
        Record a coherence assessment.
        
        Args:
            record: CoherenceRecord to store
            
        Returns:
            bool: Success status
        """
        try:
            with self.get_session() as session:
                # Hash user ID for privacy if enabled
                user_id = self._hash_user_id(record.user_id) if self.hash_user_ids else record.user_id
                
                metric = CoherenceMetric(
                    user_id=user_id,
                    timestamp=record.timestamp,
                    psi=record.psi,
                    rho=record.rho,
                    q=record.q,
                    f=record.f,
                    coherence_score=record.coherence_score,
                    coherence_velocity=record.coherence_velocity,
                    coherence_state=record.coherence_state,
                    context_tags=json.dumps(record.context_tags),
                    message_hash=record.message_hash,
                    intervention_applied=record.intervention_applied,
                    intervention_type=record.intervention_type,
                    response_optimized=record.response_optimized,
                    session_id=record.session_id
                )
                
                session.add(metric)
                
            # Trigger aggregation if enabled
            if self.enable_aggregation:
                self._update_aggregates(user_id, record.timestamp)
            
            # Clean old records
            self._clean_old_records()
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording assessment: {e}")
            return False
    
    def get_user_metrics(self, 
                        user_id: str,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get user's coherence metrics.
        
        Args:
            user_id: User identifier
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum records to return
            
        Returns:
            pd.DataFrame: User metrics
        """
        with self.get_session() as session:
            query = session.query(CoherenceMetric)
            
            # Apply user filter
            hashed_id = self._hash_user_id(user_id) if self.hash_user_ids else user_id
            query = query.filter(CoherenceMetric.user_id == hashed_id)
            
            # Apply date filters
            if start_date:
                query = query.filter(CoherenceMetric.timestamp >= start_date)
            if end_date:
                query = query.filter(CoherenceMetric.timestamp <= end_date)
            
            # Order and limit
            query = query.order_by(CoherenceMetric.timestamp.desc())
            if limit:
                query = query.limit(limit)
            
            # Convert to DataFrame
            records = query.all()
            if not records:
                return pd.DataFrame()
            
            data = []
            for record in records:
                data.append({
                    'timestamp': record.timestamp,
                    'psi': record.psi,
                    'rho': record.rho,
                    'q': record.q,
                    'f': record.f,
                    'coherence_score': record.coherence_score,
                    'coherence_velocity': record.coherence_velocity,
                    'coherence_state': record.coherence_state,
                    'intervention_applied': record.intervention_applied,
                    'response_optimized': record.response_optimized
                })
            
            return pd.DataFrame(data)
    
    def get_aggregate_metrics(self,
                            user_id: Optional[str] = None,
                            period_type: str = 'daily',
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get aggregate metrics.
        
        Args:
            user_id: Optional user filter
            period_type: Aggregation period (hourly, daily, weekly, monthly)
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            pd.DataFrame: Aggregate metrics
        """
        with self.get_session() as session:
            query = session.query(AggregateMetric)
            
            # Apply filters
            if user_id:
                hashed_id = self._hash_user_id(user_id) if self.hash_user_ids else user_id
                query = query.filter(AggregateMetric.user_id == hashed_id)
            
            query = query.filter(AggregateMetric.period_type == period_type)
            
            if start_date:
                query = query.filter(AggregateMetric.period_start >= start_date)
            if end_date:
                query = query.filter(AggregateMetric.period_end <= end_date)
            
            # Convert to DataFrame
            records = query.all()
            if not records:
                return pd.DataFrame()
            
            return pd.DataFrame([{
                'period_start': r.period_start,
                'period_end': r.period_end,
                'avg_coherence': r.avg_coherence,
                'min_coherence': r.min_coherence,
                'max_coherence': r.max_coherence,
                'std_coherence': r.std_coherence,
                'avg_psi': r.avg_psi,
                'avg_rho': r.avg_rho,
                'avg_q': r.avg_q,
                'avg_f': r.avg_f,
                'crisis_count': r.crisis_count,
                'optimal_count': r.optimal_count,
                'total_assessments': r.total_assessments,
                'improvement_rate': r.improvement_rate
            } for r in records])
    
    def calculate_improvement_metrics(self, 
                                    user_id: str,
                                    window_days: int = 30) -> Dict[str, float]:
        """
        Calculate improvement metrics for a user.
        
        Args:
            user_id: User identifier
            window_days: Days to analyze
            
        Returns:
            Dict with improvement metrics
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=window_days)
        mid_date = start_date + timedelta(days=window_days // 2)
        
        # Get metrics for two halves of the window
        first_half = self.get_user_metrics(
            user_id, start_date, mid_date
        )
        second_half = self.get_user_metrics(
            user_id, mid_date, end_date
        )
        
        if first_half.empty or second_half.empty:
            return {
                'improvement_rate': 0.0,
                'coherence_change': 0.0,
                'stability_change': 0.0,
                'crisis_reduction': 0.0
            }
        
        # Calculate improvements
        first_avg = first_half['coherence_score'].mean()
        second_avg = second_half['coherence_score'].mean()
        coherence_change = (second_avg - first_avg) / first_avg if first_avg > 0 else 0
        
        # Stability (inverse of standard deviation)
        first_stability = 1 / (1 + first_half['coherence_score'].std())
        second_stability = 1 / (1 + second_half['coherence_score'].std())
        stability_change = (second_stability - first_stability) / first_stability if first_stability > 0 else 0
        
        # Crisis reduction
        first_crisis = (first_half['coherence_state'] == 'crisis').sum()
        second_crisis = (second_half['coherence_state'] == 'crisis').sum()
        crisis_reduction = (first_crisis - second_crisis) / first_crisis if first_crisis > 0 else 0
        
        # Overall improvement rate
        improvement_rate = (coherence_change + stability_change + crisis_reduction) / 3
        
        return {
            'improvement_rate': improvement_rate,
            'coherence_change': coherence_change,
            'stability_change': stability_change,
            'crisis_reduction': crisis_reduction,
            'first_period_avg': first_avg,
            'second_period_avg': second_avg,
            'total_assessments': len(first_half) + len(second_half)
        }
    
    def get_pattern_analysis(self, 
                           user_id: str,
                           window_days: int = 30) -> Dict[str, Any]:
        """
        Analyze patterns in user's coherence data.
        
        Args:
            user_id: User identifier
            window_days: Days to analyze
            
        Returns:
            Dict with pattern analysis
        """
        df = self.get_user_metrics(
            user_id,
            start_date=datetime.now() - timedelta(days=window_days)
        )
        
        if df.empty:
            return {'status': 'insufficient_data'}
        
        # Time-based patterns
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        hourly_avg = df.groupby('hour')['coherence_score'].mean()
        daily_avg = df.groupby('day_of_week')['coherence_score'].mean()
        
        # Variable correlations
        correlations = {
            'psi_rho': df['psi'].corr(df['rho']),
            'psi_q': df['psi'].corr(df['q']),
            'psi_f': df['psi'].corr(df['f']),
            'rho_q': df['rho'].corr(df['q']),
            'rho_f': df['rho'].corr(df['f']),
            'q_f': df['q'].corr(df['f'])
        }
        
        # State transitions
        states = df['coherence_state'].tolist()
        transitions = {}
        for i in range(1, len(states)):
            transition = f"{states[i-1]}_to_{states[i]}"
            transitions[transition] = transitions.get(transition, 0) + 1
        
        # Identify patterns
        patterns = {
            'best_hour': int(hourly_avg.idxmax()) if not hourly_avg.empty else None,
            'worst_hour': int(hourly_avg.idxmin()) if not hourly_avg.empty else None,
            'best_day': int(daily_avg.idxmax()) if not daily_avg.empty else None,
            'worst_day': int(daily_avg.idxmin()) if not daily_avg.empty else None,
            'strongest_correlation': max(correlations.items(), key=lambda x: abs(x[1])),
            'most_common_transition': max(transitions.items(), key=lambda x: x[1]) if transitions else None,
            'intervention_effectiveness': self._calculate_intervention_effectiveness(df)
        }
        
        return {
            'patterns': patterns,
            'hourly_averages': hourly_avg.to_dict(),
            'daily_averages': daily_avg.to_dict(),
            'correlations': correlations,
            'state_transitions': transitions
        }
    
    def export_anonymized_data(self, 
                             output_path: str,
                             format: str = 'csv') -> bool:
        """
        Export anonymized data for research.
        
        Args:
            output_path: Output file path
            format: Export format (csv, json, parquet)
            
        Returns:
            bool: Success status
        """
        try:
            with self.get_session() as session:
                # Get all records older than anonymize threshold
                cutoff_date = datetime.now() - timedelta(days=self.anonymize_after_days)
                
                query = session.query(CoherenceMetric).filter(
                    CoherenceMetric.timestamp <= cutoff_date
                )
                
                records = query.all()
                
                if not records:
                    logger.info("No records to export")
                    return False
                
                # Convert to DataFrame and anonymize
                data = []
                for record in records:
                    data.append({
                        'anonymous_id': hashlib.sha256(record.user_id.encode()).hexdigest()[:8],
                        'timestamp': record.timestamp,
                        'psi': round(record.psi, 3),
                        'rho': round(record.rho, 3),
                        'q': round(record.q, 3),
                        'f': round(record.f, 3),
                        'coherence_score': round(record.coherence_score, 3),
                        'coherence_state': record.coherence_state,
                        'intervention_applied': record.intervention_applied,
                        'intervention_type': record.intervention_type if record.intervention_applied else None
                    })
                
                df = pd.DataFrame(data)
                
                # Export based on format
                if format == 'csv':
                    df.to_csv(output_path, index=False)
                elif format == 'json':
                    df.to_json(output_path, orient='records', date_format='iso')
                elif format == 'parquet':
                    df.to_parquet(output_path, index=False)
                else:
                    raise ValueError(f"Unsupported format: {format}")
                
                logger.info(f"Exported {len(df)} anonymized records to {output_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False
    
    def _hash_user_id(self, user_id: str) -> str:
        """Hash user ID for privacy."""
        return hashlib.sha256(user_id.encode()).hexdigest()
    
    def _update_aggregates(self, user_id: str, timestamp: datetime):
        """Update aggregate metrics after new assessment."""
        try:
            # Update hourly aggregate
            self._update_aggregate_for_period(user_id, timestamp, 'hourly', timedelta(hours=1))
            
            # Update daily aggregate
            self._update_aggregate_for_period(user_id, timestamp, 'daily', timedelta(days=1))
            
            # Update weekly aggregate
            self._update_aggregate_for_period(user_id, timestamp, 'weekly', timedelta(days=7))
            
            # Update monthly aggregate
            self._update_aggregate_for_period(user_id, timestamp, 'monthly', timedelta(days=30))
            
        except Exception as e:
            logger.error(f"Error updating aggregates: {e}")
    
    def _update_aggregate_for_period(self, 
                                   user_id: str,
                                   timestamp: datetime,
                                   period_type: str,
                                   period_delta: timedelta):
        """Update aggregate for specific period."""
        with self.get_session() as session:
            # Determine period boundaries
            if period_type == 'hourly':
                period_start = timestamp.replace(minute=0, second=0, microsecond=0)
            elif period_type == 'daily':
                period_start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            elif period_type == 'weekly':
                days_since_monday = timestamp.weekday()
                period_start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_since_monday)
            else:  # monthly
                period_start = timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            period_end = period_start + period_delta
            
            # Get metrics for this period
            metrics = session.query(CoherenceMetric).filter(
                CoherenceMetric.user_id == user_id,
                CoherenceMetric.timestamp >= period_start,
                CoherenceMetric.timestamp < period_end
            ).all()
            
            if not metrics:
                return
            
            # Calculate aggregates
            coherence_scores = [m.coherence_score for m in metrics]
            
            # Check if aggregate already exists
            existing = session.query(AggregateMetric).filter(
                AggregateMetric.user_id == user_id,
                AggregateMetric.period_start == period_start,
                AggregateMetric.period_type == period_type
            ).first()
            
            if existing:
                # Update existing
                aggregate = existing
            else:
                # Create new
                aggregate = AggregateMetric(
                    user_id=user_id,
                    period_start=period_start,
                    period_end=period_end,
                    period_type=period_type
                )
            
            # Update values
            aggregate.avg_coherence = np.mean(coherence_scores)
            aggregate.min_coherence = np.min(coherence_scores)
            aggregate.max_coherence = np.max(coherence_scores)
            aggregate.std_coherence = np.std(coherence_scores)
            aggregate.avg_psi = np.mean([m.psi for m in metrics])
            aggregate.avg_rho = np.mean([m.rho for m in metrics])
            aggregate.avg_q = np.mean([m.q for m in metrics])
            aggregate.avg_f = np.mean([m.f for m in metrics])
            aggregate.crisis_count = sum(1 for m in metrics if m.coherence_state == 'crisis')
            aggregate.optimal_count = sum(1 for m in metrics if m.coherence_state == 'optimal')
            aggregate.total_assessments = len(metrics)
            
            # Calculate improvement rate (simplified)
            if len(coherence_scores) > 1:
                first_half = coherence_scores[:len(coherence_scores)//2]
                second_half = coherence_scores[len(coherence_scores)//2:]
                if first_half and second_half:
                    improvement = (np.mean(second_half) - np.mean(first_half)) / np.mean(first_half)
                    aggregate.improvement_rate = improvement
            
            if not existing:
                session.add(aggregate)
    
    def _clean_old_records(self):
        """Clean records older than retention period."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            with self.get_session() as session:
                # Delete old detailed records
                deleted = session.query(CoherenceMetric).filter(
                    CoherenceMetric.timestamp < cutoff_date
                ).delete()
                
                if deleted > 0:
                    logger.info(f"Cleaned {deleted} old records")
                    
        except Exception as e:
            logger.error(f"Error cleaning old records: {e}")
    
    def _calculate_intervention_effectiveness(self, df: pd.DataFrame) -> float:
        """Calculate effectiveness of interventions."""
        if df.empty or 'intervention_applied' not in df.columns:
            return 0.0
        
        # Get records where intervention was applied
        intervention_df = df[df['intervention_applied'] == True]
        
        if len(intervention_df) < 5:
            return 0.0  # Not enough data
        
        # For each intervention, check if coherence improved in next assessment
        improvements = []
        
        for idx in intervention_df.index:
            # Find next assessment
            next_idx = idx + 1
            if next_idx < len(df):
                current_score = df.loc[idx, 'coherence_score']
                next_score = df.loc[next_idx, 'coherence_score']
                
                improvement = (next_score - current_score) / current_score if current_score > 0 else 0
                improvements.append(improvement)
        
        if improvements:
            return np.mean(improvements)
        
        return 0.0