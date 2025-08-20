#!/usr/bin/env python3
"""
Streamlit Dashboard for Coherence-Aware AI Framework

Real-time monitoring and visualization of coherence metrics.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
import time
from typing import Dict, List, Optional
import numpy as np

# Configure page
st.set_page_config(
    page_title="CAAF Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .crisis-alert {
        background-color: #ff4b4b;
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-alert {
        background-color: #00cc00;
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = st.sidebar.text_input("API URL", "http://localhost:8000")

# Initialize session state
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 5
if 'test_user_id' not in st.session_state:
    st.session_state.test_user_id = "dashboard_test_user"
if 'api_connected' not in st.session_state:
    st.session_state.api_connected = False


def check_api_connection():
    """Check if API is accessible."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_user_metrics(user_id: str, timeframe: str = "24h") -> Optional[Dict]:
    """Fetch user metrics from API."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/coherence-metrics/{user_id}",
            params={"timeframe": timeframe, "metric_type": "detailed"}
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_user_profile(user_id: str) -> Optional[Dict]:
    """Fetch user profile from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/coherence-profile/{user_id}")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def assess_coherence(message: str, user_id: str) -> Optional[Dict]:
    """Assess coherence for a message."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/assess-coherence",
            json={"message": message, "user_id": user_id}
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def create_coherence_gauge(score: float, title: str = "Coherence Score") -> go.Figure:
    """Create a gauge chart for coherence score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': 0.5},
        gauge={
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.2], 'color': "red"},
                {'range': [0.2, 0.4], 'color': "orange"},
                {'range': [0.4, 0.6], 'color': "yellow"},
                {'range': [0.6, 0.8], 'color': "lightgreen"},
                {'range': [0.8, 1], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 0.9
            }
        }
    ))
    fig.update_layout(height=300)
    return fig


def create_variables_radar(variables: Dict[str, float]) -> go.Figure:
    """Create radar chart for coherence variables."""
    categories = ['Ψ (Consistency)', 'ρ (Wisdom)', 'q (Moral)', 'f (Social)']
    values = [variables.get('psi', 0), variables.get('rho', 0), 
              variables.get('q', 0), variables.get('f', 0)]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Coherence Variables'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Coherence Variables Breakdown"
    )
    return fig


def create_time_series_chart(data: pd.DataFrame) -> go.Figure:
    """Create time series chart of coherence scores."""
    fig = px.line(data, x='timestamp', y='coherence_score', 
                  title='Coherence Score Over Time',
                  labels={'coherence_score': 'Coherence Score', 'timestamp': 'Time'})
    
    # Add state regions
    state_colors = {
        'crisis': 'rgba(255, 0, 0, 0.2)',
        'low': 'rgba(255, 165, 0, 0.2)',
        'medium': 'rgba(255, 255, 0, 0.2)',
        'high': 'rgba(144, 238, 144, 0.2)',
        'optimal': 'rgba(0, 128, 0, 0.2)'
    }
    
    # Add colored regions for different states
    for state, color in state_colors.items():
        state_data = data[data['coherence_state'] == state]
        if not state_data.empty:
            for _, row in state_data.iterrows():
                fig.add_vrect(
                    x0=row['timestamp'] - pd.Timedelta(minutes=1),
                    x1=row['timestamp'] + pd.Timedelta(minutes=1),
                    fillcolor=color,
                    layer="below",
                    line_width=0,
                )
    
    fig.update_layout(height=400)
    return fig


def main():
    st.title("🧠 Coherence-Aware AI Framework Dashboard")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Check API connection
    if st.sidebar.button("Test API Connection"):
        if check_api_connection():
            st.sidebar.success("✅ API Connected")
            st.session_state.api_connected = True
        else:
            st.sidebar.error("❌ API Connection Failed")
            st.session_state.api_connected = False
    
    # Auto-refresh
    st.sidebar.subheader("Auto-Refresh")
    auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh")
    if auto_refresh:
        st.session_state.refresh_interval = st.sidebar.slider(
            "Refresh Interval (seconds)", 1, 60, 5
        )
    
    # User Selection
    st.sidebar.subheader("User Selection")
    user_id = st.sidebar.text_input("User ID", st.session_state.test_user_id)
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["1h", "24h", "7d", "30d"],
        index=1
    )
    
    # Main Content
    if not st.session_state.api_connected:
        st.warning("⚠️ Please connect to the API first using the sidebar.")
        st.info(f"Make sure the API is running at {API_BASE_URL}")
        st.code("python -m src.api", language="bash")
        return
    
    # Fetch data
    metrics_data = get_user_metrics(user_id, timeframe)
    profile_data = get_user_profile(user_id)
    
    # Overview Section
    st.header("📊 Overview")
    
    if profile_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Baseline Coherence",
                f"{profile_data['baseline_coherence']:.2f}",
                delta=f"{profile_data['volatility']:.2f} volatility"
            )
        
        with col2:
            trend_emoji = {"improving": "📈", "declining": "📉", "stable": "➡️"}
            st.metric(
                "Trend",
                profile_data['trend'],
                delta=trend_emoji.get(profile_data['trend'], "")
            )
        
        with col3:
            st.metric(
                "Assessment Count",
                profile_data['assessment_count']
            )
        
        with col4:
            st.metric(
                "Risk Factors",
                len(profile_data.get('risk_factors', [])),
                delta=f"{len(profile_data.get('protective_factors', []))} protective"
            )
        
        # Alerts
        if profile_data.get('risk_factors'):
            st.markdown(
                f"<div class='crisis-alert'>⚠️ Risk Factors: {', '.join(profile_data['risk_factors'])}</div>",
                unsafe_allow_html=True
            )
        
        if profile_data.get('protective_factors'):
            st.markdown(
                f"<div class='success-alert'>✅ Protective Factors: {', '.join(profile_data['protective_factors'])}</div>",
                unsafe_allow_html=True
            )
    
    # Real-time Metrics
    st.header("🔴 Real-time Metrics")
    
    if metrics_data and metrics_data.get('status') == 'success' and metrics_data.get('data'):
        # Convert to DataFrame
        df = pd.DataFrame(metrics_data['data'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Current state
        if not df.empty:
            latest = df.iloc[-1]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Coherence gauge
                fig_gauge = create_coherence_gauge(
                    latest['coherence_score'],
                    "Current Coherence Score"
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                # Variables radar
                variables = {
                    'psi': latest['psi'],
                    'rho': latest['rho'],
                    'q': latest['q'],
                    'f': latest['f']
                }
                fig_radar = create_variables_radar(variables)
                st.plotly_chart(fig_radar, use_container_width=True)
            
            # Time series
            st.subheader("📈 Coherence Timeline")
            fig_timeline = create_time_series_chart(df)
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # State distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("State Distribution")
                state_counts = df['coherence_state'].value_counts()
                fig_pie = px.pie(
                    values=state_counts.values,
                    names=state_counts.index,
                    title="Time Spent in Each State",
                    color_discrete_map={
                        'crisis': '#ff4b4b',
                        'low': '#ffa500',
                        'medium': '#ffff00',
                        'high': '#90ee90',
                        'optimal': '#008000'
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.subheader("Intervention Analysis")
                if 'intervention_applied' in df.columns:
                    intervention_rate = df['intervention_applied'].mean()
                    st.metric("Intervention Rate", f"{intervention_rate:.1%}")
                if 'response_optimized' in df.columns:
                    optimization_rate = df['response_optimized'].mean()
                    st.metric("Response Optimization Rate", f"{optimization_rate:.1%}")
    else:
        st.info("No data available for this user and timeframe.")
    
    # Interactive Testing
    st.header("🧪 Interactive Testing")
    
    with st.expander("Test Coherence Assessment", expanded=False):
        test_message = st.text_area(
            "Enter a test message:",
            "I'm feeling overwhelmed with work but trying to stay positive."
        )
        
        if st.button("Assess Coherence"):
            with st.spinner("Assessing coherence..."):
                result = assess_coherence(test_message, user_id)
                
                if result:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Coherence Score", f"{result['coherence_score']:.2f}")
                        st.metric("State", result['state'])
                    
                    with col2:
                        st.write("**Variables:**")
                        for var, value in result['variables'].items():
                            st.write(f"{var}: {value:.2f}")
                    
                    with col3:
                        st.write("**Intervention Priorities:**")
                        for var, priority in result['intervention_priorities'].items():
                            if priority > 0.1:
                                st.write(f"{var}: {priority:.2f}")
                else:
                    st.error("Failed to assess coherence")
    
    # System Health
    st.header("🏥 System Health")
    
    health_cols = st.columns(3)
    
    with health_cols[0]:
        try:
            health_response = requests.get(f"{API_BASE_URL}/health")
            if health_response.status_code == 200:
                st.success("✅ API Status: Healthy")
            else:
                st.error("❌ API Status: Unhealthy")
        except:
            st.error("❌ API Status: Unreachable")
    
    with health_cols[1]:
        st.info(f"📊 Monitoring User: {user_id}")
    
    with health_cols[2]:
        st.info(f"🕐 Last Update: {datetime.now().strftime('%H:%M:%S')}")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(st.session_state.refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()