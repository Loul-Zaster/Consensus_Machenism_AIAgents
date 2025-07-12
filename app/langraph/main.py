"""
Main entry point for the LangGraph implementation of the Cancer Consensus AI Agents system.
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
from app.agents.translation_agent import translate_medical_consensus, SUPPORTED_LANGUAGES

# Load environment variables
load_dotenv()

def format_source_credibility(verified_sources: List[Dict[str, Any]], source_credibility: float) -> str:
    """
    Format the oncology source credibility information for the report.
    
    Args:
        verified_sources: List of verified sources with credibility scores
        source_credibility: Overall source credibility score
        
    Returns:
        Formatted string with source credibility information
    """
    if not verified_sources:
        return "No oncology source verification information available."
    
    result = "### Oncology Source Credibility Assessment\n\n"
    result += f"**Overall Credibility Score**: {source_credibility * 10:.1f}/10\n\n"
    
    result += "#### Verified Cancer Information Sources\n\n"
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
    realtime: bool = False,
    min_sources: int = 10
) -> Dict[str, Any]:
    """
    Get cancer diagnosis for a given topic and symptoms.
    
    Args:
        topic: Cancer type or concern to research
        symptoms: Patient cancer-related symptoms
        medical_history: Patient medical history relevant to cancer
        test_results: Cancer-related test results
        realtime: Whether to use real-time web search
        min_sources: Minimum number of sources to include in research
        
    Returns:
        Dictionary with cancer diagnosis results
    """
    try:
        # Run the cancer diagnosis workflow
        result = run_medical_diagnosis(
            topic=topic,
            symptoms=symptoms,
            medical_history=medical_history,
            test_results=test_results,
            realtime=realtime,
            min_sources=min_sources
    )
    
        # Extract results
        consensus = result.get("consensus", "No cancer consensus reached.")
        diagnoses = result.get("diagnoses", [])
        treatments = result.get("treatments", [])
        research_findings = result.get("research_findings", "No cancer research findings.")
        verified_sources = result.get("verified_sources", [])
        source_credibility = result.get("source_credibility", 0.0)
        
        # Ensure we have enough sources
        if verified_sources and len(verified_sources) < min_sources:
            print(f"Warning: Only found {len(verified_sources)} sources, which is less than the minimum {min_sources}")
        
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
        print(f"Error during cancer diagnosis: {str(e)}")
        return {
            "topic": topic,
            "error": str(e),
            "consensus": "Could not generate a cancer consensus due to an error.",
            "diagnoses": [],
            "treatments": [],
            "research_findings": "Error during cancer research.",
            "verified_sources": [],
            "source_credibility": 0.0
        }

def get_medical_diagnosis_with_translation(
    topic: str,
    symptoms: str = "No symptoms provided.",
    medical_history: str = "No medical history provided.",
    test_results: str = "No test results provided.",
    realtime: bool = False,
    min_sources: int = 10,
    target_language: str = None
) -> Dict[str, Any]:
    """
    Get cancer diagnosis with optional translation.
    
    Args:
        topic: Cancer type or concern to research
        symptoms: Patient cancer-related symptoms
        medical_history: Patient medical history relevant to cancer
        test_results: Cancer-related test results
        realtime: Whether to use real-time web search
        min_sources: Minimum number of sources to include in research
        target_language: Target language for translation (optional)
        
    Returns:
        Dictionary with cancer diagnosis results (translated if target_language specified)
    """
    # Get the original diagnosis
    result = get_medical_diagnosis(
        topic=topic,
        symptoms=symptoms,
        medical_history=medical_history,
        test_results=test_results,
        realtime=realtime,
        min_sources=min_sources
    )
    
    # Translate if target language is specified
    if target_language and target_language in SUPPORTED_LANGUAGES:
        try:
            print(f"Attempting to translate to {target_language}...")
            translated_result = translate_medical_consensus(result, target_language)
            print(f"Translation completed successfully")
            return translated_result
        except Exception as e:
            print(f"Translation failed: {e}")
            result["translation_error"] = str(e)
            return result
    elif target_language:
        print(f"Target language '{target_language}' not supported. Available: {list(SUPPORTED_LANGUAGES.keys())}")
        result["translation_error"] = f"Language '{target_language}' not supported"
        return result
    
    return result

if __name__ == "__main__":
    # Example usage
    result = get_medical_diagnosis(
        topic="breast cancer",
        symptoms="Lump in breast, skin changes, nipple discharge",
        medical_history="Family history of breast cancer, no prior cancer diagnosis",
        test_results="Mammogram shows suspicious mass, awaiting biopsy",
        realtime=True,
        min_sources=10
    )
    
    print("\nCANCER DIAGNOSIS RESULTS:")
    print(f"Topic: {result['topic']}")
    print("\nConsensus:")
    print(result['consensus'])
    
    print("\nCancer Diagnoses:")
    for i, diag in enumerate(result['diagnoses']):
        print(f"{i+1}. {diag}")
    
    print("\nCancer Treatment Recommendations:")
    for i, treat in enumerate(result['treatments']):
        print(f"{i+1}. {treat}")
    
    print("\nOncology Source Credibility:", result['source_credibility'])
    print(f"\nNumber of sources: {len(result['verified_sources'])}")
    print("\nSources:")
    for i, source in enumerate(result['verified_sources']):
        print(f"{i+1}. {source}") 