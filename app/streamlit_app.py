import streamlit as st
import os
import sys
import time
import markdown
from pathlib import Path
from dotenv import load_dotenv

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Import necessary modules
from app.langraph.main import get_medical_diagnosis

st.set_page_config(
    page_title="Consensus Mechanism AI Agents",
    page_icon="🩺",
    layout="wide",
)

def main():
    st.title("Consensus Mechanism AI Agents")
    st.subheader("Medical Diagnosis System Based on AI Consensus")

    # Sidebar for system information
    with st.sidebar:
        st.header("Information")
        st.info("""
        **Consensus Mechanism AI Agents** is a system using multiple AI agents to:
        1. Perform real-time medical research
        2. Verify source credibility
        3. Generate multiple independent diagnoses
        4. Create treatment plans
        5. Reach a consensus conclusion
        """)
        
        st.header("API Keys")
        # Display API key status
        io_api_key = os.getenv("IOINTELLIGENCE_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        serper_api_key = os.getenv("SERPER_API_KEY")
        
        if io_api_key:
            st.success("✅ IOINTELLIGENCE_API_KEY: Configured")
        elif openai_api_key:
            st.success("✅ OPENAI_API_KEY: Configured")
        else:
            st.error("❌ No API key found for LLM. OPENAI_API_KEY or IOINTELLIGENCE_API_KEY required")
            
        if serper_api_key:
            st.success("✅ SERPER_API_KEY: Configured")
        else:
            st.warning("⚠️ SERPER_API_KEY: Not configured (real-time search will not work)")

    # Input form
    with st.form("diagnosis_form"):
        st.header("Enter Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            topic = st.text_input("Medical topic to research", "Tinnitus")
            symptoms = st.text_area("Symptoms", "Ringing in ears, difficulty sleeping")
        
        with col2:
            medical_history = st.text_area("Medical history", "Recent concert attendance, no prior hearing issues")
            test_results = st.text_area("Test results", "None")
        
        use_realtime = st.checkbox("Researching", value=False)
        
        submit_button = st.form_submit_button("Begin Diagnosis")
    
    # Process when diagnosis button is clicked
    if submit_button:
        if not (os.getenv("IOINTELLIGENCE_API_KEY") or os.getenv("OPENAI_API_KEY")):
            st.error("Missing API key for LLM. Please configure OPENAI_API_KEY or IOINTELLIGENCE_API_KEY in .env file")
            return
            
        if use_realtime and not os.getenv("SERPER_API_KEY"):
            st.warning("Missing SERPER_API_KEY for web search. System will use simulated data.")
        
        # Display progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        result_container = st.empty()
        
        try:
            # Diagnosis workflow stages
            stages = [
                "Medical Research",
                "Source Verification",
                "Diagnostic Analysis",
                "Treatment Recommendations",
                "Building Consensus",
                "Completing Report"
            ]
            
            # Run the actual diagnosis workflow
            with st.spinner("Researching and analyzing..."):
                status_text.text(f"In progress: {stages[0]}...")
                progress_bar.progress(1/len(stages))
                
                result = get_medical_diagnosis(
                    topic=topic,
                    symptoms=symptoms,
                    medical_history=medical_history,
                    test_results=test_results,
                    realtime=use_realtime
                )
                
                # Display progress for remaining steps
                for i in range(1, len(stages)):
                    status_text.text(f"In progress: {stages[i]}...")
                    progress_bar.progress((i+1)/len(stages))
                    time.sleep(0.5)  # Simulate progress
            
            status_text.text("Diagnosis complete!")
            progress_bar.progress(100)
            
            # Display results
            with result_container.container():
                st.success("Diagnosis complete! See results below:")
                st.divider()
                
                st.header(f"Diagnosis for: {result['topic']}")
                
                st.subheader("Consensus Conclusion")
                st.markdown(result['consensus'])
                
                st.subheader("Diagnoses")
                if isinstance(result['diagnoses'], list):
                    for i, diag in enumerate(result['diagnoses']):
                        st.write(f"{i+1}. {diag}")
                else:
                    st.markdown(result['diagnoses'])
                
                st.subheader("Treatment Recommendations")
                if isinstance(result['treatments'], list):
                    for i, treat in enumerate(result['treatments']):
                        st.write(f"{i+1}. {treat}")
                else:
                    st.markdown(result['treatments'])
                
                # Display research information if available
                with st.expander("View Detailed Research Information"):
                    st.write(result['research_findings'])
                    
                    st.subheader("References")
                    for source in result['verified_sources']:
                        st.write(f"- {source}")
                    
                    st.write(f"**Source Credibility Score**: {result['source_credibility']:.2f}/1.0")
                
        except Exception as e:
            st.error(f"An error occurred during diagnosis: {str(e)}")
            st.error("Please check your configuration and API keys in the .env file")

if __name__ == "__main__":
    main() 