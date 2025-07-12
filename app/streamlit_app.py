import streamlit as st
import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import re
from pathlib import Path
from dotenv import load_dotenv

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Import necessary modules
from app.langraph.main import get_medical_diagnosis, get_medical_diagnosis_with_translation
try:
    from app.agents.translation_agent import SUPPORTED_LANGUAGES
except ImportError:
    # Fallback if translation module is not available
    SUPPORTED_LANGUAGES = {
        "spanish": "Spanish",
        "french": "French", 
        "german": "German",
        "italian": "Italian",
        "portuguese": "Portuguese",
        "russian": "Russian",
        "chinese": "Chinese (Simplified)",
        "japanese": "Japanese",
        "korean": "Korean",
        "arabic": "Arabic",
        "vietnamese": "Vietnamese"
    }

# Page configuration
st.set_page_config(
    page_title="Cancer Consensus AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
/* Main elements */
.main {
    background-color: #f8f9fa;
}

/* Headers */
h1, h2, h3 {
    font-family: 'Helvetica Neue', sans-serif;
    color: #2c3e50;
}

/* Cards */
.card {
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    border: 1px solid rgba(0,0,0,0.05);
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    background-color: white;
}

.primary-card {
    border-left: 5px solid #3498db;
}

.success-card {
    border-left: 5px solid #2ecc71;
}

.warning-card {
    border-left: 5px solid #f39c12;
}

.danger-card {
    border-left: 5px solid #e74c3c;
}

/* Tags */
.tag {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    border-radius: 50px;
    font-size: 0.85rem;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}

.tag-blue {
    background-color: rgba(52, 152, 219, 0.15);
    color: #3498db;
}

.tag-green {
    background-color: rgba(46, 204, 113, 0.15);
    color: #2ecc71;
}

.tag-yellow {
    background-color: rgba(243, 156, 18, 0.15);
    color: #f39c12;
}

.tag-red {
    background-color: rgba(231, 76, 60, 0.15);
    color: #e74c3c;
}

/* Source citation */
.source-citation {
    background-color: #f8f9fa;
    border-radius: 5px;
    padding: 0.5rem;
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
}

/* Improved readability for results */
.results-container p {
    line-height: 1.6;
    margin-bottom: 1rem;
}

/* Conclusion highlight */
.conclusion {
    background-color: rgba(46, 204, 113, 0.1);
    border-left: 5px solid #2ecc71;
    padding: 1rem;
    margin: 1rem 0;
    line-height: 1.6;
}

/* Reasoning section */
.reasoning {
    background-color: rgba(52, 152, 219, 0.1);
    border-left: 5px solid #3498db;
    padding: 1rem;
    margin: 1rem 0;
    line-height: 1.6;
}

/* Markdown enhancements */
p {
    margin-bottom: 1rem;
    line-height: 1.5;
}

strong {
    font-weight: 600;
}

em {
    font-style: italic;
}

hr {
    margin: 1.5rem 0;
    border: 0;
    border-top: 1px solid rgba(0,0,0,0.1);
}

/* Numbered list styling */
ol {
    padding-left: 1.5rem;
    margin-bottom: 1rem;
}

ol li {
    margin-bottom: 0.5rem;
}

/* Unordered list styling */
ul {
    padding-left: 1.5rem;
    margin-bottom: 1rem;
}

ul li {
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True) 

# Helper functions
def format_consensus_text(consensus_text):
    """Format the consensus text with proper styling."""
    import re
    
    # Handle markdown formatting for bold text
    consensus_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', consensus_text)
    
    # Handle markdown formatting for italic text
    consensus_text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', consensus_text)
    
    # Handle horizontal rules (---) in markdown
    consensus_text = re.sub(r'\n---\n', r'<hr/>', consensus_text)
    
    # Unified formatting for all content types
    # First, remove ### from the beginning of lines
    consensus_text = re.sub(r'^###\s+', '', consensus_text, flags=re.MULTILINE)
    
    # Handle special sections with consistent formatting
    # ONCOLOGICAL REASONING
    consensus_text = re.sub(r'ONCOLOGICAL REASONING', '<h3 style="color: #e74c3c; margin: 15px 0 10px 0;">ONCOLOGICAL REASONING</h3>', consensus_text)
    
    # CONSENSUS CANCER DIAGNOSIS
    consensus_text = re.sub(r'CONSENSUS CANCER DIAGNOSIS', '<h3 style="color: #2ecc71; margin: 15px 0 10px 0;">CONSENSUS CANCER DIAGNOSIS</h3>', consensus_text)
    
    # COMPREHENSIVE CANCER CARE PLAN
    consensus_text = re.sub(r'COMPREHENSIVE CANCER CARE PLAN', '<h3 style="color: #f39c12; margin: 15px 0 10px 0;">COMPREHENSIVE CANCER CARE PLAN</h3>', consensus_text)
    
    # PATIENT GUIDANCE
    consensus_text = re.sub(r'PATIENT GUIDANCE', '<h3 style="color: #9b59b6; margin: 15px 0 10px 0;">PATIENT GUIDANCE</h3>', consensus_text)
    
    # CONCLUSION
    consensus_text = re.sub(r'Conclusion:', '<h3 style="color: #2ecc71; margin: 15px 0 10px 0;">Conclusion:</h3>', consensus_text)
    
    # Handle Step headers
    consensus_text = re.sub(r'(Step \d+:)', r'<h4 style="margin: 20px 0 10px 0; color: #3498db;">\1</h4>', consensus_text)
    
    # Handle bullet points
    consensus_text = re.sub(r'- ', r'• ', consensus_text)
    
    # Handle numbered lists
    consensus_text = re.sub(r'(\d+)\. ', r'<strong>\1.</strong> ', consensus_text)
    
    # Handle key-value pairs
    consensus_text = re.sub(r'(\w+):\s*-\s*', r'<strong>\1:</strong> ', consensus_text)
    
    # Split into paragraphs and format
    paragraphs = consensus_text.split('\n\n')
    formatted_text = ""
    
    for para in paragraphs:
        if para.strip():
            # Skip if it's already a header
            if para.strip().startswith('<h3') or para.strip().startswith('<h4'):
                formatted_text += para.strip()
            # Handle lists within paragraphs
            elif para.strip().startswith('•') or para.strip().startswith('<strong>'):
                formatted_text += f'<div style="margin: 5px 0;">{para.strip()}</div>'
            else:
                formatted_text += f'<p style="margin: 8px 0; line-height: 1.6;">{para.strip()}</p>'
    
    return formatted_text
    
    return formatted_text

def display_progress_steps(current_step=0):
    """Display a progress indicator for the analysis workflow."""
    # Create a container for the steps
    cols = st.columns(5)
    
    steps = ["Research", "Verification", "Analysis", "Recommendations", "Consensus"]
    
    # Display each step in its own column
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            if i < current_step:
                st.markdown(f'<div style="text-align: center;"><div style="background-color: #2ecc71; color: white; width: 30px; height: 30px; border-radius: 50%; line-height: 30px; margin: 0 auto;">✓</div><div>{step}</div></div>', unsafe_allow_html=True)
            elif i == current_step:
                st.markdown(f'<div style="text-align: center;"><div style="background-color: #3498db; color: white; width: 30px; height: 30px; border-radius: 50%; line-height: 30px; margin: 0 auto;">{i+1}</div><div>{step}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="text-align: center;"><div style="background-color: white; color: black; width: 30px; height: 30px; border-radius: 50%; line-height: 30px; border: 2px solid #e0e0e0; margin: 0 auto;">{i+1}</div><div>{step}</div></div>', unsafe_allow_html=True)

def display_tags(tags, style="blue"):
    """Display a list of tags with appropriate styling."""
    html_tags = ""
    for tag in tags:
        html_tags += f'<span class="tag tag-{style}">{tag}</span>'
    
    st.markdown(html_tags, unsafe_allow_html=True) 

def display_results(result):
    """Display the analysis results with modern UI."""
    
    # Main header based on topic
    st.title(f"{result['topic'].title()} Analysis Results")
    
    # Show translation status if applicable
    if "translation_info" in result:
        translation_info = result["translation_info"]
        target_lang = translation_info.get("target_language", "Unknown")
        lang_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang)
        
        st.markdown(f"""
        <div style="background-color: #e8f4fd; border-left: 5px solid #3498db; padding: 15px; margin-bottom: 20px; border-radius: 5px;">
            <h4 style="margin: 0; color: #3498db;">🌐 Translation Information</h4>
            <p style="margin: 5px 0 0 0;">Results translated to: <strong>{lang_name}</strong></p>
            <p style="margin: 5px 0 0 0; font-size: 0.9em; color: #666;">Translation Agent: {translation_info.get('translation_agent', 'Unknown')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Show translation error if any
    if "translation_error" in result:
        st.error(f"Translation failed: {result['translation_error']}")
        st.info("Displaying results in original language.")
    
    # Create tabs for organization
    tabs = st.tabs([
        "📊 Summary", 
        "🔎 Detailed Analysis",
        "💊 Treatment Plan", 
        "🏥 Clinical Data"
    ])
    
    # Tab 1: Summary
    with tabs[0]:
        # Simple header without box
        st.markdown("""
        <h3 style="color: #3498db; margin-bottom: 15px;">
            <span style="margin-right: 10px;">🔍</span>Consensus Analysis
        </h3>
        """, unsafe_allow_html=True)
        
        formatted_consensus = format_consensus_text(result["consensus"])
        st.markdown(formatted_consensus, unsafe_allow_html=True)
            
    # Tab 2: Detailed Analysis
    with tabs[1]:
        # Check if we have specialized lung cancer analysis
        if "lung_cancer_analysis" in result and result["lung_cancer_analysis"]:
            lung_analysis = result["lung_cancer_analysis"]
            classification = lung_analysis["classification"]
            staging = lung_analysis["staging"]
            
            # Cancer type and staging
            col1, col2 = st.columns(2)
                
            # Classification card
            with col1:
                st.markdown('<div class="card primary-card">', unsafe_allow_html=True)
                st.subheader("Cancer Classification")
                st.markdown(f"**Main Type:** {classification['main_type']}")
                st.markdown(f"**Subtype:** {classification['subtype']}")
                st.markdown(f"**Differentiation:** {classification['differentiation']}")
                
                if classification['genetic_markers']:
                    st.markdown("**Genetic Markers:**")
                    display_tags(classification['genetic_markers'], "green")
                    
                st.markdown('</div>', unsafe_allow_html=True)
                
            # Staging card
            with col2:
                st.markdown('<div class="card warning-card">', unsafe_allow_html=True)
                st.subheader("Cancer Staging")
                st.markdown(f"**Stage:** {staging['stage']}")
                
                if "tnm" in staging and staging['tnm'] != "Not applicable for SCLC":
                    st.markdown(f"**TNM Classification:** {staging['tnm']}")
                
                st.markdown(f"**Description:** {staging['description']}")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Standard diagnostic information
            st.subheader("Diagnostic Assessment")
            
            if isinstance(result['diagnoses'], list) and result['diagnoses']:
                # Group diagnoses by categories
                primary_diagnosis = None
                molecular_profile = None
                other_diagnoses = []
                
                for diag in result['diagnoses']:
                    if ":" in diag and ("cancer" in diag.lower()):
                        primary_diagnosis = diag
                    elif diag.startswith("Molecular profile:"):
                        molecular_profile = diag
                    else:
                        other_diagnoses.append(diag)
                    
                # Create cards for different diagnosis components
                if primary_diagnosis:
                    st.markdown(f"""
                    <div style="background-color: white; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 15px; border-left: 5px solid #e74c3c;">
                        <h3>Primary Diagnosis</h3>
                        <p>{primary_diagnosis}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Molecular profile section - full width since histopathology is removed
                if molecular_profile:
                    st.markdown(f"""
                    <div style="background-color: white; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 15px; border-left: 5px solid #f39c12;">
                        <h3>Molecular Profile</h3>
                        <p>{molecular_profile.replace('Molecular profile: ', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Other important findings
                if other_diagnoses:
                    st.subheader("Additional Findings")
                    for diag in other_diagnoses:
                        st.markdown(f"- {diag}")
            else:
                st.info(result["diagnoses"] if isinstance(result["diagnoses"], str) else "No diagnostic information available")
    
    # Tab 3: Treatment Plan
    with tabs[2]:
        st.header("Treatment Plan")
        
        # Check if we have specialized lung cancer analysis
        if "lung_cancer_analysis" in result and result["lung_cancer_analysis"]:
            treatment = result["lung_cancer_analysis"]["treatment_recommendations"]
            
            # Create a treatment timeline visualization
            st.markdown("""
            <div style="background-color: white; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 20px;">
                <h3 style="margin-top: 0; color: #2c3e50;">Treatment Timeline</h3>
                <div style="display: flex; overflow-x: auto; padding-bottom: 15px; margin: 20px 0;">
                    <div style="display: flex; min-width: 100%;">
                        <div style="flex: 1; text-align: center; position: relative; padding-top: 25px;">
                            <div style="height: 30px; width: 30px; background-color: #3498db; border-radius: 50%; margin: 0 auto; position: relative; z-index: 2; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">1</div>
                            <div style="position: absolute; top: 55px; width: 100%; text-align: center;">
                                <h4 style="margin: 0; color: #3498db;">Initial Treatment</h4>
                                <p style="margin: 5px 0; font-size: 14px; color: #666;">First-line therapy</p>
                            </div>
                        </div>
                        <div style="flex: 1; text-align: center; position: relative; padding-top: 25px;">
                            <div style="height: 30px; width: 30px; background-color: #f39c12; border-radius: 50%; margin: 0 auto; position: relative; z-index: 2; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">2</div>
                            <div style="position: absolute; top: 55px; width: 100%; text-align: center;">
                                <h4 style="margin: 0; color: #f39c12;">Response Assessment</h4>
                                <p style="margin: 5px 0; font-size: 14px; color: #666;">Imaging & biomarkers</p>
                            </div>
                        </div>
                        <div style="flex: 1; text-align: center; position: relative; padding-top: 25px;">
                            <div style="height: 30px; width: 30px; background-color: #2ecc71; border-radius: 50%; margin: 0 auto; position: relative; z-index: 2; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">3</div>
                            <div style="position: absolute; top: 55px; width: 100%; text-align: center;">
                                <h4 style="margin: 0; color: #2ecc71;">Maintenance</h4>
                                <p style="margin: 5px 0; font-size: 14px; color: #666;">Long-term care</p>
                            </div>
                        </div>
                        <div style="flex: 1; text-align: center; position: relative; padding-top: 25px;">
                            <div style="height: 30px; width: 30px; background-color: #9b59b6; border-radius: 50%; margin: 0 auto; position: relative; z-index: 2; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">4</div>
                            <div style="position: absolute; top: 55px; width: 100%; text-align: center;">
                                <h4 style="margin: 0; color: #9b59b6;">Follow-up Care</h4>
                                <p style="margin: 5px 0; font-size: 14px; color: #666;">Surveillance</p>
                            </div>
                        </div>
                    </div>
                </div>
                <div style="height: 5px; background-color: #e0e0e0; margin-top: -50px; position: relative; z-index: 1;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Treatment approaches section with icons
            st.subheader("Treatment Approaches")
            
            # Create a grid layout for treatment cards
            col1, col2 = st.columns(2)
            
            # Primary treatment card
            with col1:
                if "primary_treatment" in treatment:
                    st.markdown("""
                    <div style="background-color: white; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 20px; height: 100%;">
                        <div style="display: flex; align-items: center; margin-bottom: 15px;">
                            <div style="background-color: rgba(52, 152, 219, 0.1); border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; margin-right: 15px;">
                                <span style="font-size: 20px;">💊</span>
                            </div>
                            <h3 style="margin: 0; color: #3498db;">Primary Treatment</h3>
                        </div>
                        <ul style="list-style-type: none; padding-left: 0; margin-top: 15px;">
                    """, unsafe_allow_html=True)
                    
                    for item in treatment['primary_treatment']:
                        st.markdown(f'<li style="margin-bottom: 10px; padding-left: 20px; position: relative;"><span style="position: absolute; left: 0; color: #3498db;">▸</span> {item}</li>', unsafe_allow_html=True)
        
                    st.markdown("</ul></div>", unsafe_allow_html=True)
        
            # Targeted therapy card
            with col2:
                if "targeted_therapy" in treatment and treatment["targeted_therapy"]:
                    st.markdown("""
                    <div style="background-color: white; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 20px; height: 100%;">
                        <div style="display: flex; align-items: center; margin-bottom: 15px;">
                            <div style="background-color: rgba(46, 204, 113, 0.1); border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; margin-right: 15px;">
                                <span style="font-size: 20px;">🎯</span>
                            </div>
                            <h3 style="margin: 0; color: #2ecc71;">Targeted Therapy</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    for therapy in treatment["targeted_therapy"]:
                        marker = therapy.get("marker", "")
                        st.markdown(f'<div style="margin-top: 15px;"><span style="font-weight: bold; color: #2ecc71;">{marker}</span></div>', unsafe_allow_html=True)
                        
                        if "first_line" in therapy:
                            st.markdown('<div style="margin-top: 5px;"><span style="font-weight: bold;">First-line:</span></div>', unsafe_allow_html=True)
                            st.markdown(f'<div style="margin-left: 15px;">{", ".join(therapy["first_line"])}</div>', unsafe_allow_html=True)
                        
                        if "subsequent" in therapy:
                            st.markdown('<div style="margin-top: 5px;"><span style="font-weight: bold;">Subsequent:</span></div>', unsafe_allow_html=True)
                            st.markdown(f'<div style="margin-left: 15px;">{", ".join(therapy["subsequent"])}</div>', unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # Second row
            col1, col2 = st.columns(2)
            
            # Immunotherapy card
            with col1:
                if "immunotherapy" in treatment and treatment["immunotherapy"]:
                    st.markdown("""
                    <div style="background-color: white; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 20px; margin-top: 20px; height: 100%;">
                        <div style="display: flex; align-items: center; margin-bottom: 15px;">
                            <div style="background-color: rgba(243, 156, 18, 0.1); border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; margin-right: 15px;">
                                <span style="font-size: 20px;">🛡️</span>
                            </div>
                            <h3 style="margin: 0; color: #f39c12;">Immunotherapy</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if "first_line" in treatment["immunotherapy"]:
                        st.markdown('<div style="margin-top: 10px;"><span style="font-weight: bold;">First-line options:</span></div>', unsafe_allow_html=True)
                        st.markdown(f'<ul style="padding-left: 20px; margin-top: 5px;">', unsafe_allow_html=True)
                        for item in treatment["immunotherapy"]["first_line"]:
                            st.markdown(f'<li>{item}</li>', unsafe_allow_html=True)
                        st.markdown('</ul>', unsafe_allow_html=True)
                    
                    if "combination" in treatment["immunotherapy"]:
                        st.markdown('<div style="margin-top: 10px;"><span style="font-weight: bold;">Combination approaches:</span></div>', unsafe_allow_html=True)
                        st.markdown(f'<ul style="padding-left: 20px; margin-top: 5px;">', unsafe_allow_html=True)
                        for item in treatment["immunotherapy"]["combination"]:
                            st.markdown(f'<li>{item}</li>', unsafe_allow_html=True)
                        st.markdown('</ul>', unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # Clinical considerations card
            with col2:
                if "clinical_considerations" in treatment and treatment["clinical_considerations"]:
                    st.markdown("""
                    <div style="background-color: white; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 20px; margin-top: 20px; height: 100%;">
                        <div style="display: flex; align-items: center; margin-bottom: 15px;">
                            <div style="background-color: rgba(155, 89, 182, 0.1); border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; margin-right: 15px;">
                                <span style="font-size: 20px;">📋</span>
                            </div>
                            <h3 style="margin: 0; color: #9b59b6;">Clinical Considerations</h3>
                        </div>
                        <ul style="list-style-type: none; padding-left: 0; margin-top: 15px;">
                    """, unsafe_allow_html=True)
                    
                    for consideration in treatment["clinical_considerations"]:
                        st.markdown(f'<li style="margin-bottom: 10px; padding-left: 20px; position: relative;"><span style="position: absolute; left: 0; color: #9b59b6;">•</span> {consideration}</li>', unsafe_allow_html=True)
                    
                    st.markdown("</ul></div>", unsafe_allow_html=True)
                    
            # Prognosis information if available
            if "prognosis" in result["lung_cancer_analysis"]:
                prognosis = result["lung_cancer_analysis"]["prognosis"]
                
                st.subheader("Prognosis & Survival")
                
                # Create a better visualization for survival rates
                base_rate = prognosis.get("base_5yr_survival_rate", 0)
                adjusted_rate = prognosis.get("adjusted_5yr_survival_rate", 0)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Visual representation of survival rates
                    st.markdown(f"""
                    <div style="background-color: white; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 20px; text-align: center;">
                        <h4 style="margin-top: 0;">5-Year Survival Rate</h4>
                        <div style="position: relative; width: 150px; height: 150px; margin: 0 auto;">
                            <svg width="150" height="150" viewBox="0 0 100 100">
                                <circle cx="50" cy="50" r="45" fill="none" stroke="#f0f0f0" stroke-width="10" />
                                <circle cx="50" cy="50" r="45" fill="none" stroke="#3498db" stroke-width="10"
                                    stroke-dasharray="{adjusted_rate} 100" stroke-dashoffset="25" />
                                <text x="50" y="50" font-size="20" text-anchor="middle" dominant-baseline="middle" fill="#2c3e50" font-weight="bold">{adjusted_rate}%</text>
                            </svg>
                        </div>
                        <div style="margin-top: 10px; font-size: 14px; color: #666;">
                            Base rate: {base_rate}% | Adjusted: {adjusted_rate}%
                        </div>
                        <div style="margin-top: 5px; font-size: 14px; color: {('#2ecc71' if adjusted_rate > base_rate else '#e74c3c')};">
                            {('+' if adjusted_rate > base_rate else '')}{adjusted_rate-base_rate}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Prognosis description and recommendations
                    st.markdown(f"""
                    <div style="background-color: white; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 20px; height: 100%;">
                        <h4 style="margin-top: 0; color: #2c3e50;">Prognosis Assessment</h4>
                        <p style="margin-bottom: 15px;">{prognosis["prognosis_description"]}</p>
                    """, unsafe_allow_html=True)
                    
                    if "recommendations" in prognosis and prognosis["recommendations"]:
                        st.markdown('<h5 style="margin-bottom: 10px; color: #3498db;">Recommendations to Improve Outcomes:</h5>', unsafe_allow_html=True)
                        st.markdown('<ul style="padding-left: 20px; margin-top: 0;">', unsafe_allow_html=True)
                        
                        for rec in prognosis["recommendations"]:
                            st.markdown(f'<li style="margin-bottom: 5px;">{rec}</li>', unsafe_allow_html=True)
                            
                        st.markdown('</ul>', unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            # Standard treatment display with better styling
            if isinstance(result['treatments'], list) and result['treatments']:
                # Create a more attractive way to display treatments
                for i, treatment in enumerate(result['treatments']):
                    bg_color = "#3498db" if i == 0 else "#2ecc71" if i == 1 else "#f39c12"
                    icon = "💊" if "chemotherapy" in treatment.lower() else "🔬" if "targeted" in treatment.lower() else "🛡️" if "immuno" in treatment.lower() else "🏥"
                    
                    st.markdown(f"""
                    <div style="background-color: white; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 15px; border-left: 5px solid {bg_color};">
                        <div style="display: flex; align-items: center;">
                            <div style="background-color: rgba({','.join(map(str, [int(bg_color[1:3], 16), int(bg_color[3:5], 16), int(bg_color[5:7], 16)]))}, 0.1); border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; margin-right: 15px;">
                                <span style="font-size: 20px;">{icon}</span>
                            </div>
                            <h3 style="margin: 0; color: {bg_color};">Treatment Option {i+1}</h3>
                        </div>
                        <div style="margin-top: 15px; margin-left: 55px;">
                            {treatment}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info(result["treatments"] if isinstance(result["treatments"], str) else "No treatment recommendations available")
    
    # Tab 4: Clinical Data
    with tabs[3]:
        # Clinical trials section
        st.subheader("Clinical Trials")
        
        if "lung_cancer_analysis" in result and result["lung_cancer_analysis"] and "clinical_trials" in result["lung_cancer_analysis"]:
            trials = result["lung_cancer_analysis"]["clinical_trials"]
            
            if trials['matching_trials_count'] > 0:
                st.success(f"Found **{trials['matching_trials_count']}** potentially matching clinical trials")
                
                # Create a dataframe for trials
                trial_data = []
                for trial in trials['matching_trials']:
                    trial_data.append({
                        "ID": trial['id'],
                        "Title": trial['title'],
                        "Phase": trial['phase'],
                        "Status": trial['status'],
                        "URL": trial['url']
                    })
                
                trial_df = pd.DataFrame(trial_data)
                
                # Display trials in an interactive table
                st.dataframe(trial_df, use_container_width=True, hide_index=True)
                
                # Show detailed information for top trials
                st.subheader("Top Trial Recommendations")
                
                for i, trial in enumerate(trials['matching_trials'][:3]):
                    with st.expander(f"{trial['title']}"):
                        st.markdown(f"**ID:** {trial['id']}")
                        st.markdown(f"**Phase:** {trial['phase']}")
                        st.markdown(f"**Status:** {trial['status']}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Conditions:**")
                            for condition in trial['conditions']:
                                st.markdown(f"- {condition}")
                        
                        with col2:
                            st.markdown("**Interventions:**")
                            for intervention in trial['interventions']:
                                st.markdown(f"- {intervention}")
                    
                        st.markdown("**Eligibility:**")
                        for key, value in trial['eligibility'].items():
                            if isinstance(value, list):
                                st.markdown(f"- {key.replace('_', ' ').title()}: {', '.join(value)}")
                            else:
                                st.markdown(f"- {key.replace('_', ' ').title()}: {value}")
                    
                        st.markdown(f"**Locations:** {', '.join(trial['locations'])}")
                        st.markdown(f"[View Trial on ClinicalTrials.gov]({trial['url']})")
            else:
                st.info("No matching clinical trials found.")
        else:
            # Display sample clinical trial data when real data is not available
            st.markdown("""
            <div style="background-color: #f8f9fa; border-left: 5px solid #3498db; padding: 15px; margin-bottom: 20px; border-radius: 5px;">
                <p>Sample clinical trial data is shown below. In a real scenario, trials would be matched to the patient's specific cancer profile.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create sample trial data
            sample_trials = [
                {
                    "ID": "NCT04583995",
                    "Title": "Osimertinib With or Without Chemotherapy in EGFR Mutant NSCLC",
                    "Phase": "Phase 3",
                    "Status": "Recruiting",
                    "URL": "https://clinicaltrials.gov/ct2/show/NCT04583995"
                },
                {
                    "ID": "NCT03521154",
                    "Title": "Pembrolizumab With or Without Chemotherapy in Treating Patients With Stage I-IIIA Non-small Cell Lung Cancer",
                    "Phase": "Phase 2",
                    "Status": "Recruiting",
                    "URL": "https://clinicaltrials.gov/ct2/show/NCT03521154"
                },
                {
                    "ID": "NCT04585815",
                    "Title": "Durvalumab + Novel Oncology Therapies for NSCLC",
                    "Phase": "Phase 2",
                    "Status": "Recruiting",
                    "URL": "https://clinicaltrials.gov/ct2/show/NCT04585815"
                }
            ]
            
            # Display sample trials in a table
            sample_df = pd.DataFrame(sample_trials)
            st.dataframe(sample_df, use_container_width=True, hide_index=True)
            
            # Show detailed sample trial information
            st.subheader("Sample Trial Details")
            
            with st.expander("Osimertinib With or Without Chemotherapy in EGFR Mutant NSCLC"):
                st.markdown("**ID:** NCT04583995")
                st.markdown("**Phase:** Phase 3")
                st.markdown("**Status:** Recruiting")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Conditions:**")
                    st.markdown("- Non-Small Cell Lung Cancer")
                    st.markdown("- EGFR Mutation")
                
                with col2:
                    st.markdown("**Interventions:**")
                    st.markdown("- Osimertinib")
                    st.markdown("- Pemetrexed")
                    st.markdown("- Carboplatin")
                
                st.markdown("**Eligibility:**")
                st.markdown("- Stage: IIIB, IV")
                st.markdown("- Markers: EGFR Mutation")
                st.markdown("- Prior Treatment: No prior systemic therapy")
                st.markdown("- Performance Status: 0-1")
                st.markdown("- Brain Metastases: Allowed if stable")
                
                st.markdown("**Locations:** Multiple locations in United States, International sites")
                st.markdown("[View Trial on ClinicalTrials.gov](https://clinicaltrials.gov/ct2/show/NCT04583995)")
        
        # Sources and references section
        st.markdown("---")
        st.subheader("Sources & References")
        
        # Source credibility score visualization
        credibility = result['source_credibility']
        if credibility >= 0.7:
            color = "success"
            assessment = "High Credibility"
        elif credibility >= 0.5:
            color = "warning"
            assessment = "Medium Credibility"
        else:
            color = "danger"
            assessment = "Low Credibility"
            
        # Create a card with source credibility information
        st.markdown(f'''
        <div class="card {color}-card">
            <h3>Source Credibility Assessment</h3>
            <p>Overall credibility score: <strong>{credibility:.2f}/1.0</strong></p>
            <p>Assessment: <strong>{assessment}</strong></p>
            <div style="background: linear-gradient(to right, #e74c3c, #f39c12, #2ecc71); height: 10px; border-radius: 5px; margin: 10px 0;">
                <div style="background-color: #3498db; width: 5px; height: 15px; position: relative; left: {credibility*100}%; transform: translateX(-50%); top: -3px;"></div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # References table
        sources = result['verified_sources']
        if sources:
            # Extract credibility scores if available
            source_data = []
            for i, source in enumerate(sources):
                credibility_score = None
                if "Credibility:" in source:
                    try:
                        cred_text = source.split("Credibility:")[1].strip()
                        credibility_score = float(cred_text.split("/")[0])
                    except:
                        credibility_score = None
                
                source_data.append({
                    "Source #": i+1,
                    "Reference": source.split("(Credibility:")[0] if "Credibility:" in source else source,
                    "Credibility": credibility_score if credibility_score else "N/A"
                })
            
            # Display as a dataframe
            sources_df = pd.DataFrame(source_data)
            st.dataframe(sources_df, use_container_width=True, hide_index=True)
        else:
            st.info("No source information available.")

def main():
    # Sidebar
    with st.sidebar:
        st.title("Cancer Consensus AI")
        st.markdown("Advanced cancer analysis system powered by AI agents consensus.")
        
        st.markdown("---")
        
        # Translation settings
        st.subheader("Translation Settings")
        
        # Language selection
        language_options = ["English (Original)"] + [f"{name} ({code})" for code, name in SUPPORTED_LANGUAGES.items()]
        selected_language = st.selectbox(
            "Translate results to:",
            language_options,
            index=0
        )
        
        # Extract language code
        target_language = None
        if selected_language != "English (Original)":
            target_language = selected_language.split("(")[1].split(")")[0]
        
        # Translation status
        if target_language:
            st.info(f"🌐 Results will be translated to {SUPPORTED_LANGUAGES.get(target_language, target_language)}")
        else:
            st.info("🌐 Results will be displayed in English")
        
        # Store target_language in session state for access in main function
        st.session_state.target_language = target_language
        
        st.markdown("---")
        
        # API Status
        st.subheader("API Status")
        io_api_key = os.getenv("IOINTELLIGENCE_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        serper_api_key = os.getenv("SERPER_API_KEY")
        
        if io_api_key:
            st.success("✅ IOINTELLIGENCE_API_KEY")
        elif openai_api_key:
            st.success("✅ OPENAI_API_KEY")
        else:
            st.error("❌ LLM API KEY (Required)")
            
        if serper_api_key:
            st.success("✅ SERPER_API_KEY")
        else:
            st.warning("⚠️ SERPER_API_KEY (Optional)")
        
        # Translation API Status
        if io_api_key:
            st.success("✅ Translation Agent (IO Intelligence)")
        else:
            st.warning("⚠️ Translation Agent (IO Intelligence API key required)")
        
        st.markdown("---")
        
        # About section
        st.subheader("About")
        st.markdown("""
        **Cancer Consensus AI** analyzes cancer cases using multiple AI agents:
        
        - Research Agent
        - Source Verification Agent
        - Diagnostician Agents
        - Treatment Advisor Agents
        - Consensus Builder Agent
        - Translation Agent (IO Intelligence)
        
        Each agent contributes its expertise to create a comprehensive cancer analysis.
        """)
        
    # Main content
    st.title("Cancer Analysis System")
    st.markdown("Get consensus-based analysis for cancer cases using multiple specialized AI agents.")
    
    # Patient form
    st.subheader("Patient Information")
    
    with st.form("cancer_analysis_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            topic = st.text_input("Cancer concern", "Lung cancer")
            symptoms = st.text_area("Symptoms", "Persistent cough, shortness of breath, chest pain, weight loss")
        
        with col2:
            medical_history = st.text_area("Medical history", "60-year-old male, 40 pack-year smoking history, COPD, family history of lung cancer")
            test_results = st.text_area("Test results", "CT scan shows 3.5 cm mass in right upper lobe with mediastinal lymphadenopathy. PET scan positive for hypermetabolic activity. Biopsy confirms non-small cell lung cancer, adenocarcinoma. EGFR mutation positive. PD-L1 expression 60%. No distant metastases.")
        
        use_realtime = st.checkbox("Use real-time research", value=False)
        submit_button = st.form_submit_button("Begin Analysis")
    
    # Process when diagnosis button is clicked
    if submit_button:
        if not (os.getenv("IOINTELLIGENCE_API_KEY") or os.getenv("OPENAI_API_KEY")):
            st.error("Missing API key for LLM. Please configure OPENAI_API_KEY or IOINTELLIGENCE_API_KEY in .env file")
            return
            
        # Get target language from session state
        target_language = st.session_state.get('target_language', None)
        
        # Check translation capability
        if target_language and not os.getenv("IOINTELLIGENCE_API_KEY"):
            st.warning("Translation requires IOINTELLIGENCE_API_KEY. Results will be displayed in English.")
            target_language = None
            
        if use_realtime and not os.getenv("SERPER_API_KEY"):
            st.warning("Missing SERPER_API_KEY for web search. System will use simulated research data.")
        
        # Analysis process
        progress_container = st.empty()
        status_text = st.empty()
        result_container = st.empty()
        
        try:
            # Process stages
            stages = [
                "Cancer Research",
                "Source Verification",
                "Cancer Analysis",
                "Treatment Analysis",
                "Consensus Building"
            ]
            
            # Run the actual diagnosis workflow
            with st.spinner("Analyzing cancer information..."):
                # Show initial progress
                with progress_container.container():
                    display_progress_steps(0)
                status_text.info(f"In progress: {stages[0]}...")
                
                # Get results with optional translation
                if target_language:
                    result = get_medical_diagnosis_with_translation(
                        topic=topic,
                        symptoms=symptoms,
                        medical_history=medical_history,
                        test_results=test_results,
                        realtime=use_realtime,
                        target_language=target_language
                    )
                else:
                    result = get_medical_diagnosis(
                        topic=topic,
                        symptoms=symptoms,
                        medical_history=medical_history,
                        test_results=test_results,
                        realtime=use_realtime
                    )
                
                # Display progress for remaining steps
                for i in range(1, len(stages)):
                    with progress_container.container():
                        display_progress_steps(i)
                    status_text.info(f"In progress: {stages[i]}...")
                    time.sleep(0.5)  # Simulate progress
            
            # Complete progress indicator
            with progress_container.container():
                display_progress_steps(len(stages))
            status_text.success("Analysis complete!")
            
            # Display results
            with result_container.container():
                display_results(result)
                
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.error("Please check your configuration and API keys in the .env file")
            
            # Debug information
            st.error("Debug Information:")
            st.write(f"Target language: {target_language}")
            st.write(f"Topic: {topic}")
            st.write(f"Use realtime: {use_realtime}")
            
            # Show API key status
            io_key = os.getenv("IOINTELLIGENCE_API_KEY")
            openai_key = os.getenv("OPENAI_API_KEY")
            st.write(f"IO Intelligence API Key: {'✅ Found' if io_key else '❌ Not found'}")
            st.write(f"OpenAI API Key: {'✅ Found' if openai_key else '❌ Not found'}")

if __name__ == "__main__":
    main() 