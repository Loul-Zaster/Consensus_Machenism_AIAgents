"""
LangGraph implementation of the Consensus Mechanism AI Agents system.
"""

import os
from typing import Dict, List, Any, Tuple, Optional, Annotated, TypedDict, Literal
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

from app.agents import ResearcherAgent
from app.langraph.agents import SourceVerifier, Diagnostician, TreatmentAdvisor, ConsensusBuilder

# Load environment variables
load_dotenv()

# Define state types
class MedicalDiagnosisState(TypedDict):
    """State for the medical diagnosis graph."""
    topic: str
    symptoms: str
    medical_history: str
    test_results: str
    research_findings: Optional[str]
    verified_sources: Optional[List[str]]
    source_credibility: Optional[float]
    diagnoses: List[str]
    treatments: List[str]
    consensus: Optional[str]
    current_round: int
    max_rounds: int
    next: Optional[str]
    research_attempt: int
    verification_attempt: int


def create_medical_diagnosis_graph(researcher: ResearcherAgent) -> StateGraph:
    """
    Create a graph for medical diagnosis workflow.
    
    Args:
        researcher: A ResearcherAgent instance
        
    Returns:
        A StateGraph instance representing the medical diagnosis workflow
    """
    # Create workflow graph
    workflow = StateGraph(state_schema=MedicalDiagnosisState)
    
    # Add nodes to the graph
    workflow.add_node("research", researcher.run)
    
    source_verifier = SourceVerifier()
    workflow.add_node("verify_sources", source_verifier.run)
    
    diagnostician = Diagnostician()
    workflow.add_node("diagnose", diagnostician.run)
    
    treatment_advisor = TreatmentAdvisor()
    workflow.add_node("recommend_treatment", treatment_advisor.run)
    
    consensus_builder = ConsensusBuilder()
    workflow.add_node("build_consensus", consensus_builder.run)
    
    # Add edges to connect the nodes
    workflow.add_edge("research", "verify_sources")
    workflow.add_edge("verify_sources", "diagnose")
    workflow.add_edge("diagnose", "recommend_treatment")
    workflow.add_edge("recommend_treatment", "build_consensus")
    
    # Add conditional edges for multi-round consensus
    workflow.add_conditional_edges(
        "build_consensus",
        lambda x: x["next"],
        {
            "research": "research",
            None: END
        }
    )
    
    # Set entry point
    workflow.set_entry_point("research")
    
    # Compile the graph
    return workflow.compile()


def run_medical_diagnosis(
    topic: str,
    symptoms: str = "No symptoms provided.",
    medical_history: str = "No medical history provided.",
    test_results: str = "No test results provided.",
    realtime: bool = False
) -> Dict[str, Any]:
    """
    Run the medical diagnosis workflow.
    
    Args:
        topic: The medical topic to research
        symptoms: Patient symptoms
        medical_history: Patient medical history
        test_results: Patient test results
        realtime: Whether to use real-time web search
        
    Returns:
        Dictionary with diagnosis results
    """
    try:
        # Setup timeout toàn cục cho các requests
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        # Cấu hình retry và timeout cho requests
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        # Không thiết lập requests.defaults.timeout vì thuộc tính này không tồn tại
        # Thay vào đó, sử dụng timeout trong mỗi request
        
        # Setup researcher agent
        print(f"Set realtime={realtime} for ResearcherAgent")
        researcher = ResearcherAgent(realtime=realtime)
        researcher.session = session  # Truyền session đã cấu hình
        
        # Create and run the graph
        graph = create_medical_diagnosis_graph(researcher)
        
        # Define input state
        input_state = {
            "topic": topic,
            "symptoms": symptoms,
            "medical_history": medical_history,
            "test_results": test_results,
            "diagnoses": [],
            "treatments": [],
            "research_findings": None,
            "verified_sources": None,
            "source_credibility": None,
            "consensus": None,
            "current_round": 1,
            "max_rounds": 1,
            "next": None,
            "research_attempt": 0,  # Initialize research attempt counter
            "verification_attempt": 0  # Initialize verification attempt counter
        }
        
        print(f"Starting medical diagnosis for {topic}")
        
        # Run the workflow with timeout - sử dụng threading.Timer thay vì signal
        # vì signal.SIGALRM không được hỗ trợ trên Windows
        import threading
        import time
        
        timeout_seconds = 300  # 5 phút
        timeout_happened = False
        
        def timeout_handler():
            nonlocal timeout_happened
            timeout_happened = True
            print("Medical diagnosis workflow timed out")
        
        timer = threading.Timer(timeout_seconds, timeout_handler)
        
        try:
            timer.start()
            start_time = time.time()
            result = graph.invoke(input_state)
            timer.cancel()
            
            if timeout_happened:
                print("Medical diagnosis workflow timed out. Returning partial results.")
                return {
                    "research_findings": "Diagnosis process timed out. Please try again later.",
                    "diagnoses": ["Diagnosis could not be completed due to timeout."],
                    "treatments": ["Treatment recommendations could not be generated."],
                    "consensus": "The diagnosis workflow timed out before completion.",
                    "verified_sources": [],
                    "source_credibility": 0.0
                }
            
            print(f"Workflow completed in {time.time() - start_time:.2f} seconds.")
            return result
            
        finally:
            timer.cancel()
        
    except Exception as e:
        print(f"Error during diagnosis: {str(e)}")
        return {
            "research_findings": f"Error during research: {str(e)}",
            "diagnoses": [],
            "treatments": [],
            "consensus": "Unable to complete diagnosis due to technical issues.",
            "verified_sources": [],
            "source_credibility": 0.0
        } 