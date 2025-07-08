"""
Main entry point for the LangGraph implementation of the Consensus Mechanism AI Agents system.
"""

import os
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple
import json
from dotenv import load_dotenv

from langgraph.graph import StateGraph
from langchain_core.runnables import Runnable, RunnableConfig
from app.langraph.graph import run_medical_diagnosis
from app.langraph.agents import ResearcherAgent, SourceVerifier, Diagnostician, TreatmentAdvisor, ConsensusBuilder

# Load environment variables
load_dotenv()

def format_source_credibility(verified_sources: List[Dict[str, Any]], source_credibility: float) -> str:
    """
    Format the source credibility information for the report.
    
    Args:
        verified_sources: List of verified sources with credibility scores
        source_credibility: Overall source credibility score
        
    Returns:
        Formatted string with source credibility information
    """
    if not verified_sources:
        return "No source verification information available."
    
    result = "### Source Credibility Assessment\n\n"
    result += f"**Overall Credibility Score**: {source_credibility * 10:.1f}/10\n\n"
    
    result += "#### Verified Sources\n\n"
    for i, source in enumerate(verified_sources, 1):
        result += f"**Source {i}: {source.get('name', 'Unknown')}**\n"
        result += f"- Type: {source.get('type', 'Unknown')}\n"
        result += f"- Credibility Score: {source.get('credibility_score', 'N/A')}/10\n"
        result += f"- Assessment: {source.get('reasoning', 'No assessment available.')}\n\n"
    
    return result

def get_medical_diagnosis(
    topic: str,
    symptoms: str = "No symptoms provided.",
    medical_history: str = "No medical history provided.",
    test_results: str = "No test results provided.",
    realtime: bool = False
) -> Dict[str, Any]:
    """
    Get medical diagnosis for a given topic and symptoms.
    
    Args:
        topic: Medical topic to research
        symptoms: Patient symptoms
        medical_history: Patient medical history
        test_results: Patient test results
        realtime: Whether to use real-time web search
        
    Returns:
        Dictionary with diagnosis results
    """
    try:
        # Run the medical diagnosis workflow
        result = run_medical_diagnosis(
            topic=topic,
            symptoms=symptoms,
            medical_history=medical_history,
            test_results=test_results,
            realtime=realtime
        )
        
        # Extract results
        consensus = result.get("consensus", "No consensus reached.")
        diagnoses = result.get("diagnoses", [])
        treatments = result.get("treatments", [])
        research_findings = result.get("research_findings", "No research findings.")
        verified_sources = result.get("verified_sources", [])
        source_credibility = result.get("source_credibility", 0.0)
        
        return {
            "topic": topic,
            "consensus": consensus,
            "diagnoses": diagnoses,
            "treatments": treatments,
            "research_findings": research_findings,
            "verified_sources": verified_sources,
            "source_credibility": source_credibility
        }
    except Exception as e:
        print(f"Error during medical diagnosis: {str(e)}")
        return {
            "topic": topic,
            "error": str(e),
            "consensus": "Could not generate a consensus due to an error.",
            "diagnoses": [],
            "treatments": [],
            "research_findings": "Error during research.",
            "verified_sources": [],
            "source_credibility": 0.0
        }

if __name__ == "__main__":
    # Example usage
    result = get_medical_diagnosis(
        topic="diabetes type 2",
        symptoms="Frequent urination, increased thirst, unexplained weight loss, fatigue",
        medical_history="Family history of diabetes, overweight, hypertension",
        test_results="Fasting blood glucose: 180 mg/dL, HbA1c: 7.5%",
        realtime=True
    )
    
    print("\nMEDICAL DIAGNOSIS RESULTS:")
    print(f"Topic: {result['topic']}")
    print("\nConsensus:")
    print(result['consensus'])
    
    print("\nDiagnoses:")
    for i, diag in enumerate(result['diagnoses']):
        print(f"{i+1}. {diag}")
    
    print("\nTreatment Recommendations:")
    for i, treat in enumerate(result['treatments']):
        print(f"{i+1}. {treat}")
    
    print("\nSource Credibility:", result['source_credibility']) 