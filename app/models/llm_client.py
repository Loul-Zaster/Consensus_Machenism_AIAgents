"""
Client for interacting with IO.net Intelligence API.
"""

import os
import openai
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import ChatMessage

# Load environment variables
load_dotenv()

def get_llm():
    """
    Get a LangChain LLM client.
    
    Returns:
        ChatOpenAI instance
    """
    # Check for API keys
    if os.getenv("OPENAI_API_KEY"):
        print("Using OpenAI API key")
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            temperature=0.7
        )
    elif os.getenv("IOINTELLIGENCE_API_KEY"):
        print("Using IO.net Intelligence API key with DeepSeek-R1-0528")
        return ChatOpenAI(
            api_key=os.getenv("IOINTELLIGENCE_API_KEY"),
            base_url=os.getenv("IOINTELLIGENCE_BASE_URL", "https://api.intelligence.io.solutions/api/v1/"),
            model="deepseek-ai/DeepSeek-R1-0528",
            temperature=0.7
        )
    else:
        # Fallback to a dummy LLM for development
        print("Warning: No API keys found, using a simulated LLM")
        return SimulatedLLM()

class SimulatedLLM:
    """A simulated LLM for development purposes when no API keys are available."""
    
    def invoke(self, inputs):
        """Simulate an LLM response."""
        if isinstance(inputs, str):
            query = inputs
        elif isinstance(inputs, dict) and "text" in inputs:
            query = inputs["text"]
        elif isinstance(inputs, dict) and all(k in inputs for k in ["topic", "symptoms"]):
            # Medical diagnosis query
            query = f"Diagnosis for {inputs['topic']} with symptoms: {inputs['symptoms']}"
        else:
            query = "Unknown query"
        
        return ChatMessage(
            role="assistant",
            content=f"This is a simulated response for: {query[:50]}...\n\n"
                   f"In production, this would be a real AI response from a language model."
        )

class IOIntelligenceClient:
    """
    Client for interacting with IO.net Intelligence API.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None
    ):
        """
        Initialize the IO.net Intelligence client.
        
        Args:
            api_key: API key for IO.net Intelligence API
            base_url: Base URL for the API
            default_model: Default model to use
        """
        self.api_key = api_key or os.getenv("IOINTELLIGENCE_API_KEY")
        if not self.api_key:
            raise ValueError("IO.net Intelligence API key not provided and not found in environment variables.")
        
        self.base_url = base_url or os.getenv("IOINTELLIGENCE_BASE_URL", "https://api.intelligence.io.solutions/api/v1/")
        self.default_model = default_model or os.getenv("IOINTELLIGENCE_DEFAULT_MODEL", "DeepSeek-R1-0528")
        
        # Initialize the OpenAI client with IO.net Intelligence API
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models.
        
        Returns:
            List of available models
        """
        try:
            response = self.client.models.list()
            return [{"id": model.id, "created": model.created, "owned_by": model.owned_by} for model in response.data]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a chat completion.
        
        Args:
            messages: List of messages in the conversation
            model: Model to use for completion
            temperature: Temperature for sampling
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Chat completion response
        """
        try:
            response = self.client.chat.completions.create(
                model=model or self.default_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            if stream:
                return response  # Return the stream object
            
            return {
                "id": response.id,
                "object": response.object,
                "created": response.created,
                "model": response.model,
                "choices": [
                    {
                        "index": choice.index,
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content
                        },
                        "finish_reason": choice.finish_reason
                    }
                    for choice in response.choices
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            print(f"Error generating chat completion: {e}")
            return {"error": str(e)}
    
    def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """
        Get embedding for text.
        
        Args:
            text: Text to embed
            model: Model to use for embedding
            
        Returns:
            Embedding vector
        """
        try:
            # For IO.net Intelligence API, use the embedding model
            embedding_model = model or "BAAI/bge-multilingual-gemma2"
            response = self.client.embeddings.create(
                model=embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return [] 