import os
import time
import threading
import json
from dotenv import load_dotenv
from prompts import PERSONAS, EXPERT_ANALYSIS_PROMPT_TEMPLATE, CONSENSUS_SYNTHESIS_PROMPT_TEMPLATE, DRAFT_PROMPT_TEMPLATE, CRITIC_PROMPT_TEMPLATE, REFINE_PROMPT_TEMPLATE
# Comment out these imports for testing
# import openai
# import google.generativeai as genai
from langchain_community.vectorstores import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
# from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
import warnings
warnings.filterwarnings("ignore")

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Mock LLM client for testing
class MockChatModel:
    def invoke(self, messages):
        query = messages[0].content if messages else "No query provided"
        # Create a mock response
        class MockResponse:
            def __init__(self, content):
                self.content = content
        
        # Simple logic to generate mock content based on query
        if "lung cancer" in query.lower():
            return MockResponse("This is a mock response about lung cancer for testing purposes.")
        else:
            return MockResponse("This is a generic mock response for testing purposes.")

# Mock embedding model
class MockEmbeddingsModel:
    def embed_documents(self, texts):
        # Return random embeddings for testing
        import numpy as np
        return [np.random.rand(384) for _ in texts]
    
    def embed_query(self, text):
        # Return random embeddings for testing
        import numpy as np
        return np.random.rand(384)

# Mock tokenizer
class MockTokenizer:
    def __call__(self, pairs, **kwargs):
        # Return a simple dictionary that can be used by the mock model
        return {"input_ids": torch.ones((len(pairs), 10)), "attention_mask": torch.ones((len(pairs), 10))}

# Mock model
class MockModel:
    def __call__(self, **kwargs):
        # Return a simple output with logits
        class MockOutput:
            def __init__(self):
                batch_size = kwargs.get("input_ids", torch.ones((1, 10))).shape[0]
                self.logits = torch.rand(batch_size)
        return MockOutput()

load_dotenv()
# client = openai.OpenAI(
#     api_key="$IOINTELLIGENCE_API_KEY",
#     base_url="https://api.intelligence.io.solutions/api/v1/",
# )

def sanitize_text(text: str) -> str:
    return text.encode('utf-8', 'replace').decode('utf-8')

class FileIngestionHandler(FileSystemEventHandler):
    """Handles the event when a new file is created."""
    def __init__(self, agent):
        self.agent = agent

    def on_created(self, event):
        if not event.is_directory and (event.src_path.endswith('.txt') or event.src_path.endswith('.pdf')):
            print(f"\n[Watcher] New file detected: {event.src_path}. Starting ingestion...")
            self.agent.ingest_file(event.src_path)
            print("[Watcher] Ingestion complete. Agent is updated.")

class AdvancedRAGAgent:
    def __init__(self, data_path="data", chroma_path="chroma_db", k_retrieval=10, k_rerank_context=3):
        self.data_path = data_path
        self.chroma_path = chroma_path
        self.k_retrieval = k_retrieval
        self.k_rerank_context = k_rerank_context
        self.disclaimer = "\n\n*Disclaimer: This information is for educational purposes only and is not a substitute for professional medical advice. Always consult a healthcare provider for any health concerns.*"
        
        print("Initializing Advanced Lung Cancer RAG Agent (TEST MODE)...")

        # --- 1. KHỞI TẠO MÔ HÌNH (MOCK FOR TESTING) ---
        print(" - Loading mock models for testing...")
        # Use mock models for testing
        self.generation_model = MockChatModel()
        self.embedding_model = MockEmbeddingsModel()
        
        # Mock tokenizer and model for testing
        print(" - Loading mock reranker model...")
        self.rerank_tokenizer = MockTokenizer()
        self.rerank_model = MockModel()

        # Khởi tạo Vector Store
        print(f" - Setting up mock Vector Store for testing...")
        try:
            self.vectorstore = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=self.embedding_model
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={'k': self.k_retrieval})
        except Exception as e:
            print(f" - Error setting up vector store: {e}")
            print(" - Creating a simple mock retriever...")
            # Create a very simple mock retriever
            class MockRetriever:
                def invoke(self, query):
                    from langchain_core.documents import Document
                    docs = [
                        Document(
                            page_content="Lung cancer is one of the most common cancers worldwide.",
                            metadata={"source": "mock_document.pdf"}
                        ),
                        Document(
                            page_content="Early detection of lung cancer is crucial for successful treatment.",
                            metadata={"source": "mock_document.pdf"}
                        ),
                        Document(
                            page_content="Symptoms of lung cancer may include persistent cough and shortness of breath.",
                            metadata={"source": "mock_document.pdf"}
                        )
                    ]
                    return docs
            self.retriever = MockRetriever()
        
        try:
            self.initial_ingest()
            self.start_file_watcher()
        except Exception as e:
            print(f" - Error during initialization: {e}")
            print(" - Continuing in test mode with limited functionality...")

    def initial_ingest(self):
        print(" - Scanning data folder for initial ingestion...")
        try:
            for filename in os.listdir(self.data_path):
                file_path = os.path.join(self.data_path, filename)
                if os.path.isfile(file_path) and (file_path.endswith('.txt') or file_path.endswith('.pdf')):
                    self.ingest_file(file_path)
        except Exception as e:
            print(f" - Error scanning data folder: {e}")
            print(" - Skipping initial ingestion...")
    
    def ingest_file(self, file_path):
        try:
            filename = os.path.basename(file_path)
            print(f"   - Processing file: '{filename}' (test mode)")
            # In test mode, we don't actually ingest the file
        except Exception as e:
            print(f"   - Error processing file: {e}")

    def start_file_watcher(self):
        print(" - Starting file watcher service (test mode)...")
        # In test mode, we don't actually start a file watcher
        pass

    def stop_file_watcher(self):
        print("\nFile watcher service stopped (test mode).")

    def retrieve_and_rerank(self, query: str):
        print("\n1. [Retrieval] Searching for relevant documents (test mode)...")
        try:
            retrieved_docs = self.retriever.invoke(query)
                
            print(f"   => Found {len(retrieved_docs)} documents. Skipping [Rerank] in test mode...")
            
            # In test mode, we simply return the documents without reranking
            final_docs = retrieved_docs
            final_contexts = [doc.page_content for doc in final_docs[:self.k_rerank_context]]
            return final_docs, final_contexts
        except Exception as e:
            print(f"   => Error during retrieval: {e}")
            # Return empty results in case of error
            return [], []

    def self_correcting_generate(self, query: str, contexts: list):
        context_str = "\n\n".join(contexts) if contexts else "No context available for testing."

        # === Round 1: Generate Draft Answer ===
        print("2. [Generation - Draft] Generating draft answer (test mode)...")
        draft_prompt = f"""
        You are a specialized AI assistant providing information about lung cancer.
        Based ONLY on the provided context below, answer the user's question.
        If the context does not contain the answer, state that the information is not available in the provided documents. Do not use outside knowledge.

        CONTEXT:
        ---
        {context_str}
        ---
        QUESTION: {query}

        DRAFT ANSWER:
        """
        try:
            response = self.generation_model.invoke([HumanMessage(content=draft_prompt)])
            draft_answer = response.content
        except Exception as e:
            return f"Error during draft generation (test mode): {e}", contexts

        # In test mode, we skip the critique and refinement steps
        print("   => Test mode: Skipping critique and refinement steps.")
        return draft_answer, contexts

    def run_medical_council_consensus(self, query: str, contexts: list):
        context_str = "\n\n".join(contexts) if contexts else "No context available for testing."

        print("2. [Medical Council] Test mode: Simulating specialist consultation...")
        
        # In test mode, we generate a simple mock consensus
        mock_consensus = f"TEST MODE RESPONSE: This is a simulated consensus report about '{query}' for testing purposes."
        print("   => Test mode: Generated mock consensus.")
        
        return mock_consensus, contexts

    def ask(self, query: str):
        """Main function to run the full pipeline and answer a query."""
        print("\n--- TEST MODE ENABLED ---")
        sanitized_query = sanitize_text(query)
        if not sanitized_query:
            return "Your query could not be processed.", []
        
        retrieved_docs, contexts = self.retrieve_and_rerank(query)

        if not contexts:
            return "I'm sorry, I could not find any relevant documents for your question." + self.disclaimer, []
        
        final_answer, source_contexts = self.self_correcting_generate(query, contexts)

        # Ensure the disclaimer is always added to the final output
        if self.disclaimer not in final_answer:
            final_answer += self.disclaimer
            
        return final_answer, source_contexts