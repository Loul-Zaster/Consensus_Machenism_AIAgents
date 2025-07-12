"""
Translation agent using IO Intelligence framework for medical consensus results.
"""

import os
import asyncio
from typing import Dict, Any, List, Optional, Union
from dotenv import load_dotenv

try:
    from iointel import Agent, Workflow
    IOINTEL_AVAILABLE = True
except ImportError:
    IOINTEL_AVAILABLE = False
    print("Warning: IO Intelligence framework not available. Translation features will be disabled.")

# Load environment variables
load_dotenv()

class MedicalTranslationAgent:
    """
    Medical translation agent using IO Intelligence framework.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.intelligence.io.solutions/api/v1"):
        """
        Initialize the translation agent.
        
        Args:
            api_key: API key for IO Intelligence (defaults to IOINTELLIGENCE_API_KEY from env)
            base_url: Base URL for the IO Intelligence API
        """
        self.api_key = api_key or os.getenv("IOINTELLIGENCE_API_KEY")
        self.base_url = base_url
        self.agent = None
        self.workflow = None
        
        if IOINTEL_AVAILABLE and self.api_key:
            self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the translation agent."""
        try:
            self.agent = Agent(
                name="Medical Translation Agent",
                instructions="""You are a specialized medical translation assistant. 
                Your role is to translate medical consensus reports, diagnoses, and treatment plans 
                while maintaining medical accuracy and terminology. 
                
                Important guidelines:
                1. Preserve all medical terms and their accuracy
                2. Maintain the professional tone of medical documents
                3. Keep the structure and formatting of the original text
                4. Ensure medical abbreviations and terms are correctly translated
                5. Preserve numerical values, percentages, and measurements exactly
                6. Maintain the hierarchical structure of medical reports
                
                When translating medical content, prioritize accuracy over fluency.""",
                model="meta-llama/Llama-3.3-70B-Instruct",
                api_key=self.api_key,
                base_url=self.base_url
            )
        except Exception as e:
            print(f"Warning: Failed to initialize IO Intelligence agent: {e}")
            self.agent = None
    
    def _extract_translated_text(self, result: Any) -> str:
        """
        Extract translated text from various response formats.
        
        Args:
            result: Translation result from IO Intelligence API
            
        Returns:
            Extracted translated text as string
        """
        # If already a string, return as is
        if isinstance(result, str):
            return result
            
        # Handle dictionary formats
        if isinstance(result, dict):
            # IO Intelligence API format
            if "translate_text" in result:
                return result["translate_text"]
                
            # Other possible formats
            for key in ["text", "translation", "content", "result", "translated_text"]:
                if key in result:
                    return result[key]
                    
            # If we have results key, try to extract from it
            if "results" in result:
                return self._extract_translated_text(result["results"])
                
            # Last resort: convert dict to string
            return str(result)
            
        # Handle other types
        return str(result)
    
    async def translate_medical_text(self, text: str, target_language: str) -> str:
        """
        Translate medical text to the target language.
        
        Args:
            text: Medical text to translate
            target_language: Target language code (e.g., 'spanish', 'french', 'german')
            
        Returns:
            Translated text
        """
        # Validate input
        if not isinstance(text, str):
            print(f"Warning: Expected string, got {type(text)}. Converting to string.")
            text = str(text)
        
        if not IOINTEL_AVAILABLE:
            return f"[Translation not available - IO Intelligence framework not installed]\n\n{text}"
        
        if not self.agent:
            return f"[Translation not available - Agent not initialized]\n\n{text}"
        
        try:
            print(f"Translating text of length {len(text)} to {target_language}")
            self.workflow = Workflow(objective=text, client_mode=False)
            
            workflow_result = await self.workflow.translate_text(
                target_language=target_language,
                agents=[self.agent]
            ).run_tasks()
            
            # Extract the translated text from the result
            translated_text = self._extract_translated_text(workflow_result)
            
            print(f"Translation completed successfully, result type: {type(translated_text)}")
            return translated_text
            
        except Exception as e:
            print(f"Translation error: {e}")
            return f"[Translation failed: {str(e)}]\n\n{text}"
    
    def translate_medical_text_sync(self, text: str, target_language: str) -> str:
        """
        Synchronous wrapper for translate_medical_text.
        
        Args:
            text: Medical text to translate
            target_language: Target language code
            
        Returns:
            Translated text
        """
        try:
            return asyncio.run(self.translate_medical_text(text, target_language))
        except Exception as e:
            print(f"Sync translation error: {e}")
            return f"[Translation failed: {str(e)}]\n\n{text}"
    
    async def translate_consensus_report(self, consensus_data: Dict[str, Any], target_language: str) -> Dict[str, Any]:
        """
        Translate an entire consensus report.
        
        Args:
            consensus_data: Dictionary containing consensus report data
            target_language: Target language code
            
        Returns:
            Dictionary with translated consensus report
        """
        try:
            print(f"Starting translation of consensus report to {target_language}")
            print(f"Consensus data keys: {list(consensus_data.keys())}")
            
            translated_data = consensus_data.copy()
            
            # Translate main consensus text
            if "consensus" in consensus_data:
                print("Translating consensus text...")
                translated_text = await self.translate_medical_text(
                    consensus_data["consensus"], target_language
                )
                translated_data["consensus"] = translated_text
            
            # Translate diagnoses
            if "diagnoses" in consensus_data and isinstance(consensus_data["diagnoses"], list):
                print(f"Translating {len(consensus_data['diagnoses'])} diagnoses...")
                translated_data["diagnoses"] = []
                for i, diagnosis in enumerate(consensus_data["diagnoses"]):
                    print(f"Translating diagnosis {i+1}...")
                    translated_diagnosis = await self.translate_medical_text(diagnosis, target_language)
                    translated_data["diagnoses"].append(translated_diagnosis)
            
            # Translate treatments
            if "treatments" in consensus_data and isinstance(consensus_data["treatments"], list):
                print(f"Translating {len(consensus_data['treatments'])} treatments...")
                translated_data["treatments"] = []
                for i, treatment in enumerate(consensus_data["treatments"]):
                    print(f"Translating treatment {i+1}...")
                    translated_treatment = await self.translate_medical_text(treatment, target_language)
                    translated_data["treatments"].append(translated_treatment)
            
            # Translate research findings
            if "research_findings" in consensus_data:
                print("Translating research findings...")
                translated_findings = await self.translate_medical_text(
                    consensus_data["research_findings"], target_language
                )
                translated_data["research_findings"] = translated_findings
            
            # Add translation metadata
            translated_data["translation_info"] = {
                "target_language": target_language,
                "translated_at": asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else None,
                "translation_agent": "IO Intelligence Medical Translation Agent"
            }
            
            print("Translation completed successfully")
            return translated_data
            
        except Exception as e:
            print(f"Error in translate_consensus_report: {e}")
            # Return original data with error info
            consensus_data["translation_error"] = str(e)
            return consensus_data
    
    def translate_consensus_report_sync(self, consensus_data: Dict[str, Any], target_language: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for translate_consensus_report.
        
        Args:
            consensus_data: Dictionary containing consensus report data
            target_language: Target language code
            
        Returns:
            Dictionary with translated consensus report
        """
        try:
            return asyncio.run(self.translate_consensus_report(consensus_data, target_language))
        except Exception as e:
            print(f"Sync consensus translation error: {e}")
            return consensus_data

# Global translation agent instance
_translation_agent = None

def get_translation_agent() -> MedicalTranslationAgent:
    """
    Get or create a global translation agent instance.
    
    Returns:
        MedicalTranslationAgent instance
    """
    global _translation_agent
    if _translation_agent is None:
        _translation_agent = MedicalTranslationAgent()
    return _translation_agent

def translate_medical_consensus(consensus_data: Dict[str, Any], target_language: str) -> Dict[str, Any]:
    """
    Convenience function to translate medical consensus data.
    
    Args:
        consensus_data: Dictionary containing consensus report data
        target_language: Target language code
        
    Returns:
        Dictionary with translated consensus report
    """
    agent = get_translation_agent()
    return agent.translate_consensus_report_sync(consensus_data, target_language)

def translate_text(text: str, target_language: str) -> str:
    """
    Convenience function to translate medical text.
    
    Args:
        text: Text to translate
        target_language: Target language code
        
    Returns:
        Translated text
    """
    agent = get_translation_agent()
    return agent.translate_medical_text_sync(text, target_language)

# Supported languages
SUPPORTED_LANGUAGES = {
    "arabic": "Arabic",
    "bengali": "Bengali",
    "bulgarian": "Bulgarian",
    "chinese": "Chinese (Simplified)",
    "czech": "Czech",
    "danish": "Danish",
    "dutch": "Dutch",
    "finnish": "Finnish",
    "french": "French",
    "german": "German",
    "greek": "Greek",
    "gujarati": "Gujarati",
    "hebrew": "Hebrew",
    "hindi": "Hindi",
    "hungarian": "Hungarian",
    "italian": "Italian",
    "japanese": "Japanese",
    "kannada": "Kannada",
    "korean": "Korean",
    "malayalam": "Malayalam",
    "marathi": "Marathi",
    "norwegian": "Norwegian",
    "persian": "Persian",
    "polish": "Polish",
    "portuguese": "Portuguese",
    "punjabi": "Punjabi",
    "romanian": "Romanian",
    "russian": "Russian",
    "spanish": "Spanish",
    "swedish": "Swedish",
    "tamil": "Tamil",
    "telugu": "Telugu",
    "thai": "Thai",
    "turkish": "Turkish",
    "urdu": "Urdu",
    "vietnamese": "Vietnamese"
} 