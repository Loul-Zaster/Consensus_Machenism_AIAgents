#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to test RAGAS installation

Usage:
    python test_ragas.py
"""

import os
import json
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Load environment variables from .env file
load_dotenv()

def test_ragas_installation():
    """Check RAGAS installation"""
    print("Checking RAGAS installation...")
    
    # Check OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY is not set in environment variables.")
        print("Please add OPENAI_API_KEY to your .env file")
        return False
    
    # Create simple test data
    test_data = {
        "question": ["What are the main subtypes of NSCLC?"],
        "answer": ["The main subtypes of Non-Small Cell Lung Cancer are adenocarcinoma, squamous cell carcinoma, and large cell carcinoma."],
        "contexts": [[
            "Non-Small Cell Lung Cancer (NSCLC) is the most common type of lung cancer... The main subtypes of NSCLC include adenocarcinoma, squamous cell carcinoma, and large cell carcinoma.",
            "Treatment for SCLC is different from NSCLC."
        ]],
        "ground_truths": [[
            "The main subtypes of Non-Small Cell Lung Cancer are adenocarcinoma, squamous cell carcinoma, and large cell carcinoma."
        ]]
    }
    
    # Create dataset
    try:
        test_dataset = Dataset.from_dict(test_data)
        print("✓ Dataset created successfully")
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return False
    
    # Initialize evaluation models
    try:
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo", temperature=0))
        evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
        print("✓ Evaluation models initialized successfully")
    except Exception as e:
        print(f"Error initializing evaluation models: {e}")
        return False
    
    # Evaluate with RAGAS
    try:
        print("Evaluating with RAGAS (this may take a few minutes)...")
        result = evaluate(
            test_dataset, 
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall], 
            llm=evaluator_llm, 
            embeddings=evaluator_embeddings
        )
        print("✓ RAGAS evaluation completed successfully")
        
        # Display results
        results_df = result.to_pandas()
        print("\nEvaluation Results:")
        for metric in results_df.columns:
            print(f" - {metric}: {results_df[metric].mean():.4f}")
        
        return True
    except Exception as e:
        print(f"Error during RAGAS evaluation: {e}")
        return False

def main():
    print("=== RAGAS Installation Test ===")
    success = test_ragas_installation()
    
    if success:
        print("\n✅ RAGAS installation successful! You can now use RAGAS to evaluate RAG systems.")
        print("To run a full evaluation, use the command:")
        print("  python run_evaluation.py")
        print("To run the evaluation with a Streamlit interface, use:")
        print("  python run_evaluation.py --streamlit")
        print("  or: streamlit run streamlit_evaluation.py")
    else:
        print("\n❌ RAGAS installation check failed. Please check the errors and try again.")

if __name__ == "__main__":
    main()