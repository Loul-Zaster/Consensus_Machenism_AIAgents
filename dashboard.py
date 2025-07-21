#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integrated Dashboard for Consensus Mechanism AI Agents
Combines both main application (Cancer Analysis System) and evaluation tool (RAG Evaluation Dashboard)
into a single interface.
"""

import os
import sys
import streamlit as st
import importlib.util
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt

# Add root directory to sys.path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Load environment variables from .env file
load_dotenv()

# Paths to application files
main_app_path = os.path.join(current_dir, "app", "streamlit_app.py")
eval_app_path = os.path.join(current_dir, "streamlit_evaluation.py")

# Check if application files exist
if not os.path.exists(main_app_path):
    st.error(f"Error: File not found {main_app_path}")
    st.stop()

if not os.path.exists(eval_app_path):
    st.error(f"Error: File not found {eval_app_path}")
    st.stop()

# Function to import modules from file
def import_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import RAGAS evaluation module
try:
    from evaluation_ragas import (
        setup_evaluator_models,
        prepare_evaluation_dataset,
        evaluate_with_ragas,
        display_results_streamlit
    )
except ImportError:
    st.error("Could not import evaluation_ragas module. Please check your installation.")
    st.stop()

# Import required functions from app/langraph/main
try:
    sys.path.append(os.path.join(current_dir, "app"))
    from app.langraph.main import get_medical_diagnosis, get_medical_diagnosis_with_translation
except ImportError:
    st.error("Could not import app.langraph.main module. Please check your installation.")
    st.stop()

# Import support functions from the main application
try:
    # Import main module
    main_app_module = import_module_from_file("main_app", main_app_path)
    # Get necessary functions and variables from the module
    format_consensus_text = main_app_module.format_consensus_text
    display_progress_steps = main_app_module.display_progress_steps
    display_results = main_app_module.display_results
    
    try:
        SUPPORTED_LANGUAGES = main_app_module.SUPPORTED_LANGUAGES
    except AttributeError:
        # Fallback if not found
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
except Exception as e:
    st.error(f"Error when importing module from main application: {str(e)}")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="Consensus Mechanism AI Agents",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Update Custom CSS section, add center container
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 2rem;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .app-subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .mode-selector {
        text-align: center;
        padding: 1rem;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    /* Mode selection cards */
    .center-container {
        display: flex;
        justify-content: center;
        width: 100%;
        margin: 0 auto;
    }
    .cards-container {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 2rem auto;
        max-width: 900px;
        width: 100%;
    }
    .mode-card {
        background-color: white;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        cursor: pointer;
        flex: 1;
        min-width: 300px;
        max-width: 400px;
    }
    .mode-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
    }
    .mode-card.selected {
        border: 3px solid #2c3e50;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .mode-icon {
        font-size: 3.5rem;
        margin-bottom: 1rem;
    }
    .mode-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .mode-description {
        font-size: 1rem;
        color: #7f8c8d;
        margin-bottom: 1rem;
    }
    .cancer-mode {
        border-left: 5px solid #3498db;
    }
    .cancer-mode .mode-icon {
        color: #3498db;
    }
    .eval-mode {
        border-left: 5px solid #e74c3c;
    }
    .eval-mode .mode-icon {
        color: #e74c3c;
    }
    .mode-badge {
        background-color: rgba(52, 152, 219, 0.1);
        color: #3498db;
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-top: 1rem;
    }
    .eval-mode .mode-badge {
        background-color: rgba(231, 76, 60, 0.1);
        color: #e74c3c;
    }
    .hidden-buttons {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Page title
st.markdown('<h1 class="main-title">Cancer Consensus AI Agents</h1>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Cancer analysis system and performance evaluation based on AI multi-agents</p>', unsafe_allow_html=True)

# Add JavaScript to handle click events
st.markdown("""
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Find card elements
    const cancerCard = document.getElementById('cancer-mode-card');
    const evalCard = document.getElementById('eval-mode-card');
    
    // Find hidden buttons
    const cancerButton = document.querySelector('button[data-testid="baseButton-btn_cancer"]');
    const evalButton = document.querySelector('button[data-testid="baseButton-btn_eval"]');
    
    // Handle click event for cancer card
    if (cancerCard) {
        cancerCard.addEventListener('click', function() {
            if (cancerButton) cancerButton.click();
        });
    }
    
    // Handle click event for eval card
    if (evalCard) {
        evalCard.addEventListener('click', function() {
            if (evalButton) evalButton.click();
        });
    }
});

// Define function called from onclick
function selectMode(mode) {
    if (mode === 'cancer') {
        const btn = document.querySelector('button[data-testid="baseButton-btn_cancer"]');
        if (btn) btn.click();
    } else if (mode === 'eval') {
        const btn = document.querySelector('button[data-testid="baseButton-btn_eval"]');
        if (btn) btn.click();
    }
}
</script>
""", unsafe_allow_html=True)

# Check and set app_mode in session state if not already present
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "Cancer Analysis System"

# Create layout for mode selection with prominent cards
st.markdown("<h2 style='text-align: center; margin-bottom: 1.5rem;'>Select Mode</h2>", unsafe_allow_html=True)

# Center container
st.markdown('<div class="center-container">', unsafe_allow_html=True)

# Replace columns with HTML container
container_html = '<div class="cards-container">'

# Cancer analysis mode card
cancer_card_class = "mode-card cancer-mode" + (" selected" if st.session_state.app_mode == "Cancer Analysis System" else "")
cancer_card = f"""
<div class="{cancer_card_class}" id="cancer-mode-card" onclick="selectMode('cancer')">
    <div class="mode-icon">🔬</div>
    <div class="mode-title">Cancer Analysis System</div>
    <div class="mode-description">Analyze cancer cases based on symptoms, medical history, and test results using AI multi-agents.</div>
    <div class="mode-badge">Diagnostic Mode</div>
</div>
"""

# RAG evaluation mode card
eval_card_class = "mode-card eval-mode" + (" selected" if st.session_state.app_mode == "RAG Evaluation Dashboard" else "")
eval_card = f"""
<div class="{eval_card_class}" id="eval-mode-card" onclick="selectMode('eval')">
    <div class="mode-icon">📊</div>
    <div class="mode-title">RAG Evaluation Dashboard</div>
    <div class="mode-description">Evaluate the performance of RAG systems based on specialized metrics.</div>
    <div class="mode-badge">Evaluation Mode</div>
</div>
"""

# Close container
container_html += cancer_card + eval_card + '</div></div>'
st.markdown(container_html, unsafe_allow_html=True)

# Hidden buttons to handle events when JavaScript is not working
with st.container():
    st.markdown('<div class="hidden-buttons">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Select Cancer Analysis System", key="btn_cancer"):
            st.session_state.app_mode = "Cancer Analysis System"
            st.rerun()
    with col2:
        if st.button("Select RAG Evaluation Dashboard", key="btn_eval"):
            st.session_state.app_mode = "RAG Evaluation Dashboard"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Add separator
st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)

# Main content based on selected mode
app_mode = st.session_state.app_mode

# Sidebar
with st.sidebar:
    st.title("Options")
    st.markdown("---")
    
    # Display options based on selected mode
    if app_mode == "Cancer Analysis System":
        st.subheader("Translation Settings")
        
        # Language options
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
        
        # Save target_language in session state
        st.session_state.target_language = target_language
        
    else:  # RAG Evaluation Dashboard
        st.subheader("Evaluation Information")
        st.markdown("""
        The evaluation tool uses RAGAS to analyze RAG system performance based on metrics:
        - Faithfulness
        - Answer Relevancy
        - Context Precision
        - Context Recall
        """)
    
    # Display API status common to both modes
    st.markdown("---")
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
    
    # Translation Agent status
    if io_api_key:
        st.success("✅ Translation Agent (IO Intelligence)")
    else:
        st.warning("⚠️ Translation Agent (Requires IO Intelligence API key)")

# Function to run evaluation (from streamlit_evaluation.py)
def run_evaluation():
    with st.spinner("Setting up evaluation models..."):
        evaluator_llm, evaluator_embeddings = setup_evaluator_models()
    
    with st.spinner("Preparing evaluation dataset..."):
        eval_dataset = prepare_evaluation_dataset()
    
    # Display dataset information
    st.success(f"Loaded {len(eval_dataset)} evaluation samples")
    
    # Create progress bar
    progress_bar = st.progress(0, text="Evaluating...")
    
    # Progress bar handler
    class StreamlitProgressBar:
        def __init__(self):
            self.n = 0
            self.total = len(eval_dataset)
        
        def update(self, n=1):
            self.n += n
            progress_bar.progress(
                self.n / self.total,
                text=f"Evaluating sample {self.n}/{self.total}"
            )
            return True
        
        def close(self):
            pass
    
    # Evaluate with RAGAS
    with st.spinner("Evaluating with RAGAS..."):
        results = evaluate_with_ragas(
            eval_dataset, 
            evaluator_llm, 
            evaluator_embeddings,
            StreamlitProgressBar()
        )
    
    # Convert results to DataFrame
    results_df = results.to_pandas()
    
    # Save results to session state
    st.session_state.results_df = results_df
    st.session_state.evaluation_complete = True
    
    # Display success message
    st.success("Evaluation complete!")
    
    return results_df

# MAIN CONTENT - Display content based on selected mode
if app_mode == "Cancer Analysis System":
    # Display Cancer Analysis System
    st.header("Cancer Analysis System")
    st.markdown("Analyze cancer cases based on consensus mechanism from multiple specialized AI agents.")
    
    # Patient form
    st.subheader("Patient Information")
    
    with st.form("cancer_analysis_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            topic = st.text_input("Cancer Concern", "Lung cancer")
            symptoms = st.text_area("Symptoms", "Persistent cough, shortness of breath, chest pain, weight loss")
        
        with col2:
            medical_history = st.text_area("Medical History", "60-year-old male, 40 pack-year smoking history, COPD, family history of lung cancer")
            test_results = st.text_area("Test Results", "CT scan shows 3.5 cm mass in right upper lobe with mediastinal lymphadenopathy. PET scan positive for hypermetabolic activity. Biopsy confirms non-small cell lung cancer, adenocarcinoma. EGFR mutation positive. PD-L1 expression 60%. No distant metastases.")
        
        use_realtime = st.checkbox("Use real-time search", value=False)
        submit_button = st.form_submit_button("Start Analysis")
    
    # Handle analysis button click
    if submit_button:
        if not (os.getenv("IOINTELLIGENCE_API_KEY") or os.getenv("OPENAI_API_KEY")):
            st.error("Missing LLM API key. Please configure IOINTELLIGENCE_API_KEY or OPENAI_API_KEY in the .env file")
            st.stop()
            
        # Get target language from session state
        target_language = st.session_state.get('target_language', None)
        
        # Check translation capability
        if target_language and not os.getenv("IOINTELLIGENCE_API_KEY"):
            st.warning("Translation requires IOINTELLIGENCE_API_KEY. Results will be displayed in English.")
            target_language = None
            
        if use_realtime and not os.getenv("SERPER_API_KEY"):
            st.warning("Missing SERPER_API_KEY for web search. The system will use simulated research data.")
        
        # Analysis process
        progress_container = st.empty()
        status_text = st.empty()
        result_container = st.empty()
        
        try:
            # Processing stages
            stages = [
                "Cancer Research",
                "Source Verification",
                "Cancer Analysis",
                "Treatment Analysis",
                "Consensus Building"
            ]
            
            # Run the actual analysis process
            with st.spinner("Analyzing cancer information..."):
                # Display initial progress
                with progress_container.container():
                    display_progress_steps(0)
                status_text.info(f"Processing: {stages[0]}...")
                
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
                    status_text.info(f"Processing: {stages[i]}...")
                    import time
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
            st.error("Debug information:")
            st.write(f"Target language: {target_language}")
            st.write(f"Topic: {topic}")
            st.write(f"Use real-time search: {use_realtime}")
            
            # Show API key status
            io_key = os.getenv("IOINTELLIGENCE_API_KEY")
            openai_key = os.getenv("OPENAI_API_KEY")
            st.write(f"IO Intelligence API Key: {'✅ Found' if io_key else '❌ Not found'}")
            st.write(f"OpenAI API Key: {'✅ Found' if openai_key else '❌ Not found'}")

else:  # RAG Evaluation Dashboard
    # Display RAG Evaluation Dashboard
    st.header("RAG Evaluation Dashboard")
    st.markdown("""
        This is an evaluation dashboard for RAG (Retrieval-Augmented Generation) systems using RAGAS -
        a comprehensive evaluation framework for RAG systems. This dashboard displays important metrics
        to evaluate the performance of RAG systems in retrieving information and generating answers.
    """)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Overview", "Details", "Explanations"])
    
    # Overview tab
    with tab1:
        st.markdown("<h3>Evaluation Overview</h3>", unsafe_allow_html=True)
        
        # Button to start evaluation
        if 'evaluation_complete' not in st.session_state:
            st.session_state.evaluation_complete = False
        
        if not st.session_state.evaluation_complete:
            if st.button("Start Evaluation", type="primary"):
                results_df = run_evaluation()
        else:
            if st.button("Re-evaluate", type="primary"):
                st.session_state.evaluation_complete = False
                results_df = run_evaluation()
            else:
                results_df = st.session_state.results_df
        
        # Display results if evaluation is complete
        if st.session_state.evaluation_complete:
            # Create metric cards
            col1, col2, col3, col4 = st.columns(4)
            
            metrics = {
                "faithfulness": {
                    "title": "Faithfulness",
                    "description": "Truthfulness of the answer relative to the context"
                },
                "answer_relevancy": {
                    "title": "Answer Relevancy",
                    "description": "Relevance of the answer to the question"
                },
                "context_precision": {
                    "title": "Context Precision",
                    "description": "Accuracy of the retrieved context"
                },
                "context_recall": {
                    "title": "Context Recall",
                    "description": "Coverage of necessary information in the context"
                }
            }
            
            for i, (metric, info) in enumerate(metrics.items()):
                col = [col1, col2, col3, col4][i]
                with col:
                    st.markdown(f"""
                    <div style="background-color: #f0f2f6; border-radius: 10px; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                        <div style="font-size: 2.5rem; font-weight: bold; color: #1E88E5; text-align: center;">{results_df[metric].mean():.4f}</div>
                        <div style="font-size: 1.2rem; color: #424242; text-align: center; margin-top: 0.5rem;">{info["title"]}</div>
                        <div style="font-size: 0.9rem; color: #616161; text-align: center; margin-top: 0.5rem;">{info["description"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Create charts
            st.markdown("<h3>Evaluation Charts</h3>", unsafe_allow_html=True)
            
            # Create DataFrame for chart
            chart_data = pd.DataFrame({
                'Metric': list(metrics.keys()),
                'Score': [results_df[metric].mean() for metric in metrics.keys()]
            })
            
            # Draw bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(
                chart_data['Metric'], 
                chart_data['Score'],
                color=['#1E88E5', '#42A5F5', '#64B5F6', '#90CAF9']
            )
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.01,
                    f'{height:.4f}',
                    ha='center', 
                    va='bottom',
                    fontweight='bold'
                )
            
            ax.set_ylim(0, 1.1)  # Set y limits from 0 to 1.1
            ax.set_title('RAGAS Evaluation Metrics', fontsize=16, pad=20)
            ax.set_xlabel('Metrics', fontsize=12, labelpad=10)
            ax.set_ylabel('Score', fontsize=12, labelpad=10)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Display chart in Streamlit
            st.pyplot(fig)
    
    # Details tab
    with tab2:
        st.markdown("<h3>Evaluation Details</h3>", unsafe_allow_html=True)
        
        if st.session_state.get('evaluation_complete', False):
            # Display detailed table
            st.dataframe(st.session_state.results_df, use_container_width=True)
            
            # Download results option
            csv = st.session_state.results_df.to_csv(index=False)
            st.download_button(
                label="Download Results (CSV)",
                data=csv,
                file_name="ragas_evaluation_results.csv",
                mime="text/csv",
            )
        else:
            st.info("Please run the evaluation in the Overview tab before viewing details.")
    
    # Explanations tab
    with tab3:
        st.markdown("<h3>Metrics Explanations</h3>", unsafe_allow_html=True)
        
        # Detailed explanation of metrics
        metrics_explanation = {
            "faithfulness": {
                "title": "Faithfulness",
                "description": "Measures the factual accuracy of the generated answer compared to the provided context. A high score indicates the answer does not contain false information or hallucinations.",
                "importance": "This metric is crucial to ensure that the model does not generate false information or content not present in the provided context.",
                "improvement": "To improve, ensure that the retrieved context is high-quality and relevant, and adjust prompts to encourage the model to use only information from the context."
            },
            "answer_relevancy": {
                "title": "Answer Relevancy",
                "description": "Measures how relevant the answer is to the question. A high score indicates the answer directly addresses the question asked.",
                "importance": "This metric ensures that the answer actually addresses the issue the user is concerned about, even if the answer is faithful to the context.",
                "improvement": "To improve, optimize prompts to focus on the specific question and ensure that the model correctly understands the intent of the question."
            },
            "context_precision": {
                "title": "Context Precision",
                "description": "Measures the ratio of information in the retrieved context that is actually relevant to the question. A high score indicates the retrieved context contains little irrelevant information.",
                "importance": "This metric evaluates the effectiveness of the retrieval system in filtering out irrelevant information, helping the model focus on important information.",
                "improvement": "To improve, optimize the retrieval system, adjust chunk sizes, and improve similarity calculation methods."
            },
            "context_recall": {
                "title": "Context Recall",
                "description": "Measures the coverage of the retrieved context relative to the information needed to answer the question. A high score indicates the context contains most or all necessary information.",
                "importance": "This metric ensures that the retrieval system does not miss important information necessary to fully answer the question.",
                "improvement": "To improve, expand the number of documents retrieved, improve segmentation strategies, and ensure that the knowledge base is comprehensive enough."
            }
        }
        
        # Display explanation for each metric
        for metric, info in metrics_explanation.items():
            with st.expander(info["title"], expanded=True):
                st.markdown(f"**Description:** {info['description']}")
                st.markdown(f"**Importance:** {info['importance']}")
                st.markdown(f"**How to improve:** {info['improvement']}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; margin-top: 2rem;">
    <p style="font-size: 0.9rem; color: #7f8c8d;">
        Cancer Consensus AI Agents | Cancer analysis based on AI multi-agents
    </p>
</div>
""", unsafe_allow_html=True) 