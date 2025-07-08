#!/usr/bin/env python
"""
Main entry point for the Medical Diagnosis System using LangGraph.
"""

import sys
import argparse
import warnings
from datetime import datetime

from app.langraph.graph import run_medical_diagnosis

# Ignore warnings from pysbd (sentence boundary detection library)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run(topic, symptoms=None, medical_history=None, test_results=None):
    """
    Run the Medical Diagnosis System with the given inputs.
    
    Args:
        topic: The medical topic to research
        symptoms: Patient symptoms (optional)
        medical_history: Patient medical history (optional)
        test_results: Patient test results (optional)
        
    Returns:
        The results from the system execution
    """
    try:
        # Run the medical diagnosis workflow
        results = run_medical_diagnosis(
            topic=topic,
            symptoms=symptoms or "No symptoms provided.",
            medical_history=medical_history or "No medical history provided.",
            test_results=test_results or "No test results provided."
        )
        return results
    except Exception as e:
        print(f"Error running the Medical Diagnosis System: {e}")
        raise

def save_results(results, topic):
    """
    Save the results to a file.
    
    Args:
        results: Results from the medical diagnosis workflow
        topic: Medical topic
        
    Returns:
        Path to the saved file
    """
    # Create a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"medical_diagnosis_{timestamp}.md"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Medical Diagnosis for {topic}\n\n")
        
        f.write("## Research Findings\n\n")
        f.write(results.get("research_findings", "No research findings available."))
        
        f.write("\n\n## Diagnoses\n\n")
        for i, diagnosis in enumerate(results.get("diagnoses", []), 1):
            f.write(f"### Expert {i}\n\n")
            f.write(diagnosis)
            f.write("\n\n")
        
        f.write("\n\n## Treatment Recommendations\n\n")
        for i, treatment in enumerate(results.get("treatments", []), 1):
            f.write(f"### Expert {i}\n\n")
            f.write(treatment)
            f.write("\n\n")
        
        f.write("\n\n## Consensus\n\n")
        f.write(results.get("consensus", "No consensus available."))
    
    return filename

def main():
    """
    Main function to run the application from the command line.
    """
    parser = argparse.ArgumentParser(
        description="Run the Medical Diagnosis System with Consensus Mechanism."
    )
    
    parser.add_argument(
        "topic", 
        help="The medical topic to research"
    )
    
    parser.add_argument(
        "--symptoms", "-s",
        help="Patient symptoms",
        default=None
    )
    
    parser.add_argument(
        "--medical-history", "-m",
        help="Patient medical history",
        default=None
    )
    
    parser.add_argument(
        "--test-results", "-t",
        help="Patient test results",
        default=None
    )
    
    args = parser.parse_args()
    
    # Run the medical diagnosis workflow
    print(f"Running medical diagnosis for topic: {args.topic}")
    results = run(
        topic=args.topic,
        symptoms=args.symptoms,
        medical_history=args.medical_history,
        test_results=args.test_results
    )
    
    # Save the results
    filename = save_results(results, args.topic)
    
    print(f"\nResults saved to {filename}")
    
    # Print the consensus
    print("\n=== CONSENSUS ===\n")
    print(results.get("consensus", "No consensus available."))

if __name__ == "__main__":
    main() 