# src/preloader.py

import streamlit as st
import time

def load_css():
    """Load custom CSS for the preloader"""
    st.markdown("""
    <style>
    .preloader-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 9999;
        flex-direction: column;
    }
    
    .preloader-content {
        text-align: center;
        color: white;
    }
    
    .preloader-title {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
        animation: pulse 2s infinite;
    }
    
    .preloader-subtitle {
        font-size: 1.2rem;
        margin-bottom: 2rem;
        opacity: 0.9;
    }
    
    .spinner {
        border: 4px solid rgba(255,255,255,0.3);
        border-radius: 50%;
        border-top: 4px solid white;
        width: 60px;
        height: 60px;
        animation: spin 1s linear infinite;
        margin: 0 auto 2rem auto;
    }
    
    .progress-bar {
        width: 300px;
        height: 6px;
        background-color: rgba(255,255,255,0.3);
        border-radius: 3px;
        overflow: hidden;
        margin: 0 auto;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
        border-radius: 3px;
        animation: loading 3s ease-in-out;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    @keyframes loading {
        0% { width: 0%; }
        100% { width: 100%; }
    }
    
    .feature-item {
        background: rgba(255,255,255,0.1);
        padding: 8px 16px;
        margin: 4px;
        border-radius: 20px;
        font-size: 0.9rem;
        opacity: 0;
        animation: fadeInUp 0.5s ease forwards;
    }
    
    .feature-item:nth-child(1) { animation-delay: 0.5s; }
    .feature-item:nth-child(2) { animation-delay: 0.7s; }
    .feature-item:nth-child(3) { animation-delay: 0.9s; }
    .feature-item:nth-child(4) { animation-delay: 1.1s; }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    </style>
    """, unsafe_allow_html=True)

def show_preloader(app_title="InsightMD", subtitle="Interpretable Diabetes Prediction System", loading_time=3.5):
    """
    Display an animated preloader screen
    
    Args:
        app_title (str): The main title to display
        subtitle (str): The subtitle to display
        loading_time (float): How long to show the preloader in seconds
    """
    load_css()
    
    preloader_placeholder = st.empty()
    
    with preloader_placeholder.container():
        st.markdown(f"""
        <div class="preloader-container">
            <div class="preloader-content">
                <div class="preloader-title">🩺 {app_title}</div>
                <div class="preloader-subtitle">{subtitle}</div>
                <div class="spinner"></div>
                <div class="progress-bar">
                    <div class="progress-fill"></div>
                </div>
                <div style="margin-top: 2rem;">
                    <div class="feature-item">🤖 Loading AI Models</div>
                    <div class="feature-item">📊 Preparing SHAP Explainer</div>
                    <div class="feature-item">🔬 Initializing Prediction Engine</div>
                    <div class="feature-item">✨ Almost Ready...</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Simulate loading time
    time.sleep(loading_time)
    
    # Clear the preloader
    preloader_placeholder.empty()

def initialize_preloader():
    """
    Initialize the preloader session state and show it if needed
    """
    if 'app_loaded' not in st.session_state:
        st.session_state.app_loaded = False
    
    if not st.session_state.app_loaded:
        show_preloader()
        st.session_state.app_loaded = True