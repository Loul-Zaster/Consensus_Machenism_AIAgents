import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Add root directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import RAGAS evaluation module
from evaluation_ragas import (
    setup_evaluator_models,
    prepare_evaluation_dataset,
    evaluate_with_ragas,
    display_results_streamlit
)

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
    }
    .metric-title {
        font-size: 1.2rem;
        color: #424242;
        text-align: center;
        margin-top: 0.5rem;
    }
    .metric-description {
        font-size: 0.9rem;
        color: #616161;
        text-align: center;
        margin-top: 0.5rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #424242;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Page title
st.markdown('<h1 class="main-header">RAG Evaluation Dashboard</h1>', unsafe_allow_html=True)

# Introduction
st.markdown("""
    This is an evaluation dashboard for RAG (Retrieval-Augmented Generation) systems using RAGAS - 
    a comprehensive evaluation framework for RAG systems. This dashboard displays important metrics 
    to evaluate the performance of RAG systems in retrieving information and generating answers.
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Overview", "Details", "Explanations"])

# Function to run evaluation
def run_evaluation():
    with st.spinner("Setting up evaluation models..."):
        evaluator_llm, evaluator_embeddings = setup_evaluator_models()
    
    with st.spinner("Preparing evaluation dataset..."):
        eval_dataset = prepare_evaluation_dataset()
    
    # Display dataset information
    st.success(f"Loaded {len(eval_dataset)} evaluation samples")
    
    # Create progress bar
    progress_bar = st.progress(0, text="Evaluating...")
    
    # Fix the tqdm progress bar issue
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

# Overview tab
with tab1:
    st.markdown('<h2 class="section-header">Evaluation Overview</h2>', unsafe_allow_html=True)
    
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
                st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{results_df[metric].mean():.4f}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-title">{info["title"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-description">{info["description"]}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Create charts
        st.markdown('<h3 class="section-header">Evaluation Charts</h3>', unsafe_allow_html=True)
        
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
        
        ax.set_ylim(0, 1.1)  # Set y limit from 0 to 1.1
        ax.set_title('RAGAS Evaluation Metrics', fontsize=16, pad=20)
        ax.set_xlabel('Metrics', fontsize=12, labelpad=10)
        ax.set_ylabel('Score', fontsize=12, labelpad=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Display chart in Streamlit
        st.pyplot(fig)

# Details tab
with tab2:
    st.markdown('<h2 class="section-header">Evaluation Details</h2>', unsafe_allow_html=True)
    
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
    st.markdown('<h2 class="section-header">Metrics Explanations</h2>', unsafe_allow_html=True)
    
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

# Main function
def main():
    pass

if __name__ == "__main__":
    main()