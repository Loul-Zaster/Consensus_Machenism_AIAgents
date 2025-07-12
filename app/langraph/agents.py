"""
Agents for the Consensus Mechanism AI Agents system.
"""

import os
import json
import re
from typing import Dict, Any, List, Tuple, Optional, Callable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool, StructuredTool

from app.tools.web_search import GoogleSearchTool, WebScraper, SerpApiSearchTool, web_search
from app.models.llm_client import get_llm
from app.models.lung_cancer_classifier import LungCancerClassifier
from app.models.lung_cancer_stager import LungCancerStager
from app.models.lung_cancer_treatment_advisor import LungCancerTreatmentAdvisor
from app.models.lung_cancer_prognosis import LungCancerPrognosisPredictor
from app.models.clinical_trial_finder import ClinicalTrialFinder

def filter_thinking_tags(content: str) -> str:
    """
    Filter out content within <think>...</think> tags or similar formats.
    
    Args:
        content: String to filter
        
    Returns:
        String with thinking content removed
    """
    # Filter content within <think>...</think> tags
    pattern = r'<think>.*?</think>'
    content = re.sub(pattern, '', content, flags=re.DOTALL)
    
    # Also filter similar tags like [thinking], [thought], etc.
    patterns = [
        r'\[thinking\].*?\[/thinking\]',
        r'\[thought\].*?\[/thought\]',
        r'\[reasoning\].*?\[/reasoning\]'
    ]
    
    for pattern in patterns:
        content = re.sub(pattern, '', content, flags=re.DOTALL)
    
    # Remove excess empty lines
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = content.strip()
    
    return content

class BaseAgent:
    """Base agent class for all agents in the system."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7
    ):
        """
        Initialize the agent.
        
        Args:
            api_key: API key for the LLM
            base_url: Base URL for the LLM API
            model: Model to use
            temperature: Temperature for sampling
        """
        self.api_key = api_key or os.getenv("IOINTELLIGENCE_API_KEY")
        if not self.api_key:
            raise ValueError("API key not provided and not found in environment variables.")
        
        self.base_url = base_url or os.getenv("IOINTELLIGENCE_BASE_URL", "https://api.intelligence.io.solutions/api/v1/")
        self.model = model or os.getenv("IOINTELLIGENCE_DEFAULT_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=self.model,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
            temperature=temperature
        )
    
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent.
        
        Args:
            inputs: Inputs for the agent
            
        Returns:
            Agent outputs
        """
        raise NotImplementedError("Subclasses must implement run method.")


class ResearcherAgent:
    """Agent for researching cancer-related topics using web search."""
    
    def __init__(self, realtime: bool = False, min_sources: int = 10):
        """
        Initialize the researcher agent.
        
        Args:
            realtime: Whether to use real-time web search
            min_sources: Minimum number of sources to include in research
        """
        self.realtime = realtime
        self.min_sources = min_sources
        self.session = None  # Will be set from outside
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the researcher agent to collect information about a cancer topic.
        
        Args:
            state: Dictionary with current workflow state
            
        Returns:
            Updated state with research findings
        """
        topic = state["topic"]
        symptoms = state["symptoms"]
        medical_history = state.get("medical_history", "")
        test_results = state.get("test_results", "")
        attempt = state.get("research_attempt", 0)
        min_sources = state.get("min_sources", self.min_sources)  # Get min_sources from state or use default
        
        # Increment attempt count
        attempt += 1
        state["research_attempt"] = attempt
        
        if attempt > 3:
            print(f"Warning: Research attempt {attempt} for topic {topic}. Adjusting strategy.")
            # If multiple attempts have been made, use a different search strategy
            query = f"scientific oncology information {topic} cancer treatment diagnosis evidence based medicine"
            use_trusted_domains = True  # Always use trusted domains after multiple attempts
        else:
            # Create cancer-specific search query
            if symptoms and len(symptoms.strip()) > 0 and symptoms.lower() != "no symptoms provided.":
                query = f"{topic} cancer {symptoms} causes diagnosis treatment oncology"
            else:
                query = f"{topic} cancer oncology diagnosis treatment research"
            
            # Add information from medical history and test results
            if medical_history and medical_history.lower() != "no medical history provided.":
                relevant_history = medical_history[:100]  # Take first 100 characters
                query += f" with {relevant_history}"
                
            if test_results and test_results.lower() != "no test results provided.":
                relevant_tests = test_results[:100]  # Take first 100 characters
                query += f" cancer markers {relevant_tests}"
            
            use_trusted_domains = True  # Default to using trusted cancer domains
        
        print(f"Cancer research query: {query}")
        print(f"Use trusted domains: {use_trusted_domains}")
        print(f"Target minimum sources: {min_sources}")
        
        if not self.realtime:
            # Simulate research results if not using realtime
            print("Using simulated cancer research results")
            findings = self._simulate_research(topic, symptoms, min_sources)
            
            return {**state, "research_findings": findings, "next": "verify_sources"}
        
        try:
            # Calculate number of results to request based on min_sources
            # Request more than needed to account for filtering and duplicates
            base_results = max(min_sources, 12)
            
            # Perform actual web search
            search_results = web_search(query, num_results=base_results, use_trusted_domains=use_trusted_domains)
            
            if not search_results:
                print("No search results found. Using simulated cancer research.")
                findings = self._simulate_research(topic, symptoms, min_sources)
                return {**state, "research_findings": findings, "next": "verify_sources"}
            
            # If we don't have enough results yet, try additional queries
            if len(search_results) < min_sources:
                # Try different query formulations to get more diverse results
                additional_queries = [
                    f"{topic} cancer latest research treatment options clinical trials",
                    f"{topic} cancer diagnosis guidelines oncology",
                    f"{topic} cancer prognosis survival rates statistics",
                    f"{topic} cancer genetic factors biomarkers",
                    f"{topic} cancer supportive care management"
                ]
                
                # Keep track of links we've already seen
                seen_links = {result["link"] for result in search_results}
                
                # Try each additional query until we have enough results
                for additional_query in additional_queries:
                    if len(search_results) >= min_sources:
                        break
                        
                    print(f"Searching with additional query: {additional_query}")
                    additional_results = web_search(additional_query, num_results=base_results, use_trusted_domains=use_trusted_domains)
                    
                    # Add new results that we haven't seen before
                    for result in additional_results:
                        if result["link"] not in seen_links:
                            search_results.append(result)
                            seen_links.add(result["link"])
                            
                            if len(search_results) >= min_sources:
                                break
            
            # Report how many sources we found
            print(f"Found {len(search_results)} sources for cancer research")
            
            # Summarize results
            summary = f"Cancer research findings for {topic}:\n\n"
            
            for i, result in enumerate(search_results):
                title = result.get("title", "No title")
                snippet = result.get("snippet", "No snippet")
                link = result.get("link", "No link")
                
                summary += f"{i+1}. {title}\n"
                summary += f"   Summary: {snippet}\n"
                summary += f"   Source: {link}\n\n"
            
            return {**state, "research_findings": summary, "next": "verify_sources"}
            
        except Exception as e:
            print(f"Error during cancer research: {str(e)}")
            # Use simulated results in case of error
            findings = self._simulate_research(topic, symptoms, min_sources)
            return {**state, "research_findings": findings, "next": "verify_sources"}
    
    def _simulate_research(self, topic: str, symptoms: str, min_sources: int = 10) -> str:
        """
        Generate simulated cancer research findings for offline testing.
        
        Args:
            topic: Cancer topic to research
            symptoms: Patient symptoms
            min_sources: Minimum number of sources to include
            
        Returns:
            Simulated research findings with at least min_sources sources
        """
        # Ensure we have at least min_sources number of sources
        sources = [
            "National Cancer Institute (2023)",
            "American Society of Clinical Oncology (ASCO) Guidelines",
            "National Comprehensive Cancer Network (NCCN)",
            "American Cancer Society",
            "Memorial Sloan Kettering Cancer Center",
            "MD Anderson Cancer Center",
            "Dana-Farber Cancer Institute",
            "Mayo Clinic Cancer Center",
            "Cancer Research UK",
            "World Health Organization (WHO)",
            "Journal of Clinical Oncology",
            "New England Journal of Medicine",
            "The Lancet Oncology",
            "European Society for Medical Oncology (ESMO)",
            "International Agency for Research on Cancer (IARC)"
        ]
        
        # Ensure we have at least min_sources
        while len(sources) < min_sources:
            sources.append(f"Cancer Research Journal Vol. {len(sources) + 1}")
        
        # Take at least min_sources sources
        used_sources = sources[:min_sources]
        
        return f"""Cancer Research Findings for {topic}:

1. Definition and Overview:
   {topic.capitalize()} cancer is a malignancy that affects thousands of patients each year.
   Common symptoms include {symptoms}.

2. Cancer Classification:
   - Histological type: Carcinoma/Sarcoma/Leukemia/Lymphoma/Myeloma
   - Stage: Early/Advanced/Metastatic
   - Molecular profile: Key genetic mutations and biomarkers
   
3. Diagnostic Approach:
   - Clinical evaluation and physical examination
   - Imaging studies (CT, MRI, PET scan)
   - Biopsy and pathological confirmation
   - Molecular testing for targeted therapy selection
   
4. Treatment Options:
   - Surgical interventions
   - Radiation therapy approaches
   - Chemotherapy regimens
   - Targeted therapies based on molecular profile
   - Immunotherapy options
   - Clinical trials availability
   
5. Prognosis:
   - 5-year survival rates based on stage and molecular features
   - Prognostic factors and biomarkers
   - Recurrence risk assessment
   
6. Latest Research:
   - Novel targeted therapies in development
   - Immunotherapy advancements
   - Precision medicine approaches
   - Ongoing clinical trials

7. Genetic Considerations:
   - Hereditary risk factors
   - Genetic testing recommendations
   - Implications for family members

8. Supportive Care:
   - Managing treatment side effects
   - Quality of life considerations
   - Psychological support resources
   - Nutrition and exercise guidelines

9. Survivorship:
   - Long-term follow-up protocols
   - Late effects of treatment
   - Cancer rehabilitation options
   
10. Prevention Strategies:
    - Risk reduction measures
    - Screening recommendations
    - Lifestyle modifications
   
{chr(10).join([f"Source: {source}" for source in used_sources])}
"""


class Diagnostician:
    """Agent for diagnosing cancer conditions."""
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate potential cancer diagnoses based on research and symptoms.
        
        Args:
            state: The current state
            
        Returns:
            Updated state with diagnoses
        """
        topic = state["topic"]
        symptoms = state["symptoms"]
        medical_history = state.get("medical_history", "")
        test_results = state.get("test_results", "")
        findings = state.get("research_findings", "")
        
        llm = get_llm()
        
        # Prepare the diagnostic prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a highly skilled oncologist specializing in cancer diagnosis. Based on the patient information and research findings provided, 
             suggest the most likely cancer diagnoses. Focus on evidence-based oncology.
             
             FORMAT YOUR RESPONSE AS FOLLOWS:
             
             Present 1-3 potential cancer diagnoses in clear sections with headings. For each diagnosis:
             
             ## [CANCER DIAGNOSIS NAME]
             
             **Likelihood:** High/Medium/Low
             
             **Cancer Type:** Specify the exact type and subtype of cancer
             
             **Stage Assessment:** Preliminary assessment of potential staging (if possible from information)
             
             **Reasoning:** Provide a clear, concise paragraph explaining the evidence supporting this diagnosis. 
             Reference symptoms, history, risk factors, and research that align with this diagnosis.
             
             **Key Indicators:** List 2-3 bullet points of the most important symptoms or findings supporting this diagnosis.
             
             **Recommended Confirmatory Tests:** List the specific tests needed to confirm this diagnosis.
             
             Use clear, professional oncology terminology but ensure it's also understandable."""),
            ("human", """Topic: {topic}
            Patient Symptoms: {symptoms}
            Medical History: {medical_history}
            Test Results: {test_results}
            
            Research Findings:
            {findings}
            
            Based on the above information, what are the most likely cancer diagnoses? Format as instructed.""")
        ])
            
        try:
            chain = prompt | llm
            result = chain.invoke({
                "topic": topic,
                "symptoms": symptoms,
                "medical_history": medical_history,
                "test_results": test_results,
                "findings": findings
            })
            
            diagnoses = filter_thinking_tags(result.content)
            # Keep as string to preserve formatting
            
            return {**state, "diagnoses": diagnoses, "next": "recommend_treatment"}
            
        except Exception as e:
            print(f"Error during cancer diagnosis: {str(e)}")
            return {**state, "diagnoses": f"Unable to generate cancer diagnosis: {str(e)}", "next": "recommend_treatment"}


class TreatmentAdvisor:
    """Agent for recommending cancer treatments."""
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend cancer treatments based on diagnoses.
        
        Args:
            state: The current state
            
        Returns:
            Updated state with treatment recommendations
        """
        diagnoses = state.get("diagnoses", [])
        if not diagnoses:
            return {**state, "treatments": ["No cancer diagnoses provided to base treatments on"], "next": "build_consensus"}
        
        # Keep as string if it's already a string
        if isinstance(diagnoses, list):
            diagnoses_str = "\n".join(diagnoses)
        else:
            diagnoses_str = str(diagnoses)
            
        symptoms = state.get("symptoms", "")
        medical_history = state.get("medical_history", "")
        findings = state.get("research_findings", "")
        
        llm = get_llm()
        
        # Prepare the treatment prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a medical oncologist specializing in cancer treatment. Based on the diagnoses and patient information, 
             recommend appropriate evidence-based cancer treatments.
             
             FORMAT YOUR RESPONSE AS FOLLOWS:
             
             Present your cancer treatment recommendations in clear, organized sections:
             
             ## Primary Cancer Interventions
             
             Present the primary treatment recommendations for each cancer type mentioned in the diagnosis, formatted as:
             
             ### [TREATMENT MODALITY]
             
             **Purpose:** Brief explanation of what this treatment targets in the cancer
             
             **Details:** Clear instructions on implementation (dosage if medication, frequency, duration, etc.)
             
             **Evidence:** Brief note on the evidence supporting this approach for this specific cancer type
             
             **Sequencing:** Recommended order of treatments (neoadjuvant, adjuvant, etc.)
             
             ## Supportive Care & Symptom Management
             
             List specific supportive care measures to manage cancer symptoms and treatment side effects.
             
             ## Follow-up & Monitoring
             
             Specify cancer-specific monitoring protocols, including:
             - Imaging frequency and type
             - Blood tests and tumor markers
             - Surveillance timeline
             - Signs of recurrence to monitor
             
             ## Clinical Trial Opportunities
             
             Note any relevant clinical trial categories that might be appropriate.
             
             ## Precautions & Contraindications
             
             Note any important contraindications or warnings specific to the recommended cancer treatments.
             
             Use clear, practical oncology language that healthcare providers can easily communicate to patients."""),
            ("human", """Cancer Diagnoses:
            {diagnoses}
            
            Patient Symptoms: {symptoms}
            Medical History: {medical_history}
            
            Research Findings:
            {findings}
            
            Based on these cancer diagnoses and patient information, what treatments would you recommend? Format as instructed.""")
        ])
        
        try:
            chain = prompt | llm
            result = chain.invoke({
                "diagnoses": diagnoses_str,
            "symptoms": symptoms,
                "medical_history": medical_history,
                "findings": findings
            })
            
            treatments = filter_thinking_tags(result.content)
            # Keep as string to preserve formatting
            
            return {**state, "treatments": treatments, "next": "build_consensus"}
            
        except Exception as e:
            print(f"Error generating cancer treatment recommendations: {str(e)}")
            return {**state, "treatments": f"Unable to generate cancer treatment recommendations: {str(e)}", "next": "build_consensus"}


class ConsensusBuilder:
    """Agent for building consensus among multiple cancer diagnoses and treatments."""
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build consensus from the various cancer diagnoses and treatments.
        
        Args:
            state: The current state
            
        Returns:
            Updated state with consensus
        """
        topic = state.get("topic", "")
        diagnoses = state.get("diagnoses", [])
        treatments = state.get("treatments", [])
        findings = state.get("research_findings", "")
        sources = state.get("verified_sources", [])
        credibility = state.get("source_credibility", 0.0)
        
        # Handle different data types
        if isinstance(diagnoses, list):
            diagnoses_str = "\n".join(diagnoses)
        else:
            diagnoses_str = str(diagnoses)
            
        if isinstance(treatments, list):
            treatments_str = "\n".join(treatments)
        else:
            treatments_str = str(treatments)
            
        if isinstance(sources, list):
            sources_str = "\n".join(sources)
        else:
            sources_str = str(sources)
        
        llm = get_llm()
        
        # Prepare the consensus prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a tumor board chairperson responsible for synthesizing multiple expert opinions on cancer cases. 
             Your task is to analyze the cancer diagnoses and treatments provided, and create a unified assessment that represents 
             the most likely scenario based on available evidence.
             
             STRUCTURE YOUR RESPONSE IN THE EXACT FOLLOWING FORMAT:
             
             Step 1: ONCOLOGICAL REASONING (Labeled "ONCOLOGICAL REASONING")
             Explain in detail your medical reasoning process, weighing the different cancer diagnoses, evidence strength, biomarkers, 
             staging considerations, and treatment rationales. This section shows your critical thinking and evaluation of conflicting information.
             
             Step 2: CONSENSUS CANCER DIAGNOSIS (Labeled "CONSENSUS CANCER DIAGNOSIS")
             Provide a clear statement of the most likely cancer diagnosis including:
             - Specific cancer type and subtype
             - Preliminary staging assessment (TNM if applicable)
             - Key molecular/genetic features (if mentioned)
             - Confidence level in diagnosis
             
             Step 3: COMPREHENSIVE CANCER CARE PLAN (Labeled "COMPREHENSIVE CANCER CARE PLAN")
             Present a clear, practical, and concise cancer treatment plan that includes:
             1. First-line treatment recommendations with rationale
             2. Sequencing of multimodal therapy if applicable (surgery, radiation, systemic therapy)
             3. Supportive care needs
             4. Surveillance and monitoring protocol
             5. Potential clinical trial considerations
             
             Step 4: PATIENT GUIDANCE (Labeled "PATIENT GUIDANCE")
             Provide clear guidance for the patient regarding:
             - What to expect during treatment
             - Important symptoms to report immediately
             - Lifestyle recommendations during cancer treatment
             - Resources for cancer support
             
             Be clear, evidence-based, and patient-centered in all sections."""),
            ("human", """Cancer Topic: {topic}
            
            Cancer Diagnoses:
{diagnoses}

            Recommended Cancer Treatments:
{treatments}

            Cancer Research Findings Summary:
            {findings}
            
            Sources (Credibility Score: {credibility}):
            {sources}
            
            Based on all this information, provide your oncological reasoning, consensus cancer diagnosis, comprehensive cancer care plan, and patient guidance.""")
        ])
        
        try:
            chain = prompt | llm
            result = chain.invoke({
                "topic": topic,
                "diagnoses": diagnoses_str,
                "treatments": treatments_str,
                "findings": findings,
                "sources": sources_str,
                "credibility": f"{credibility:.1f}"
            })
            
            consensus = filter_thinking_tags(result.content)
            
            # Handle rounds if needed
            current_round = state.get("current_round", 1)
            max_rounds = state.get("max_rounds", 1)
            
            if current_round < max_rounds:
                # Continue with another round
                next_step = "research"  # Start another cycle
                current_round += 1
            else:
                # End the process
                next_step = None
            
            return {
                **state, 
                "consensus": consensus, 
                "current_round": current_round,
                "next": next_step
            }
            
        except Exception as e:
            print(f"Error building cancer consensus: {str(e)}")
            return {**state, "consensus": f"Unable to build cancer consensus: {str(e)}", "next": None}


class SourceVerifier:
    """Agent for verifying sources and assessing credibility."""
    
    def __init__(self):
        """Initialize the source verifier agent."""
        self.session = None  # Will be set from outside
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the source verifier agent to verify sources and assess credibility.
        
        Args:
            state: Dictionary with current workflow state
            
        Returns:
            Updated state with verified sources and credibility assessment
        """
        research_findings = state["research_findings"]
        sources = self._extract_sources(research_findings)
        
        print(f"Verifying {len(sources)} sources")
        
        # Assess credibility of each source
        verified_sources = []
        total_credibility = 0
        
        for source in sources:
            credibility = self._assess_source_credibility(source)
            # Only include sources with credibility score of 6.0 or higher
            if credibility >= 6.0:
                source_with_credibility = f"{source} (Credibility: {credibility}/10)"
                verified_sources.append(source_with_credibility)
                total_credibility += credibility
        
        # Calculate average credibility (normalized to 0-1 range)
        avg_credibility = total_credibility / len(verified_sources) / 10 if verified_sources else 0
        
        # Determine next step based on verification results
        next_step = "diagnose"  # Default next step
        verification_attempt = state.get("verification_attempt", 0) + 1
        
        if not verified_sources:
            print("No credible sources found, retrying research")
            next_step = "research"
            
            # If we've tried verification multiple times without success, proceed anyway
            if verification_attempt >= 3:
                next_step = "diagnose"
                verified_sources = ["No highly credible sources found after multiple attempts"]
                avg_credibility = 0.3  # Low credibility score
        elif "lung" in state["topic"].lower() and "cancer" in state["topic"].lower():
            next_step = "lung_cancer_analysis"
        
        print(f"Source verification complete. Average credibility: {avg_credibility:.2f}")
        
        return {
            **state,
            "verified_sources": verified_sources,
            "source_credibility": avg_credibility,
            "next": next_step,
            "verification_attempt": verification_attempt
        }
    
    def _extract_sources(self, findings: str) -> List[str]:
        """
        Extract sources from research findings.
        
        Args:
            findings: Research findings text
            
        Returns:
            List of extracted sources
        """
        sources = set()  # Use set to avoid duplicates
        
        # Look for "Source:" in the text
        lines = findings.split('\n')
        for line in lines:
            if 'Source:' in line or 'source:' in line:
                source = line.split(':', 1)[1].strip()
                sources.add(source)
        
            # Also look for URLs
            if 'http://' in line or 'https://' in line:
                words = line.split()
                for word in words:
                    if word.startswith('http'):
                        sources.add(word.strip('.,()[]{}'))
        
        return list(sources)  # Convert to list when returning
    
    def _assess_source_credibility(self, source: str) -> float:
        """
        Evaluate the credibility of a cancer information source.
        
        Args:
            source: The source to evaluate
            
        Returns:
            Credibility score (0.0-10.0)
        """
        # List of highly credible cancer information sources
        top_cancer_sources = [
            'cancer.gov', 'nci.nih.gov', 'cancer.org', 'asco.org', 'nccn.org',
            'cancerresearchuk.org', 'esmo.org', 'mskcc.org', 'mdanderson.org',
            'dana-farber.org', 'nejm.org', 'thelancet.com', 'jco.org'
        ]
        
        # List of credible but more general medical sources
        credible_medical_sources = [
            'mayoclinic.org', 'nih.gov', 'who.int', 'pubmed.gov', 'medlineplus.gov',
            'hopkinsmedicine.org', 'clevelandclinic.org', 'jamanetwork.com'
        ]
        
        # Check if source is from a top cancer source
        for top_source in top_cancer_sources:
            if top_source.lower() in source.lower():
                return 9.0 + (1.0 * (top_source == 'cancer.gov' or top_source == 'nci.nih.gov'))  # NCI gets highest score
        
        # Check if source is from a credible medical source
        for med_source in credible_medical_sources:
            if med_source.lower() in source.lower():
                return 7.5
        
        # Check for academic or research indicators
        if any(indicator in source.lower() for indicator in ['journal', 'study', 'research', 'trial', 'publication']):
            return 7.0
        
        # Check for medical professional organizations
        if any(indicator in source.lower() for indicator in ['association', 'society', 'college', 'foundation']):
            return 6.5
        
        # Default moderate score for unrecognized sources
        return 5.0 


class LungCancerSpecialistAgent:
    """Agent specialized in lung cancer diagnosis, staging, treatment, and prognosis."""
    
    def __init__(self, realtime: bool = False, min_sources: int = 10):
        """
        Initialize the lung cancer specialist agent.
        
        Args:
            realtime: Whether to use real-time web search
            min_sources: Minimum number of sources to include in research
        """
        self.realtime = realtime
        self.min_sources = min_sources
        self.session = None  # Will be set from outside
        
        # Initialize specialized lung cancer modules
        self.classifier = LungCancerClassifier()
        self.stager = LungCancerStager()
        self.treatment_advisor = LungCancerTreatmentAdvisor()
        self.prognosis_predictor = LungCancerPrognosisPredictor()
        self.trial_finder = ClinicalTrialFinder()
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the lung cancer specialist agent to analyze lung cancer case.
        
        Args:
            state: Dictionary with current workflow state
            
        Returns:
            Updated state with lung cancer analysis
        """
        topic = state["topic"]
        symptoms = state["symptoms"]
        medical_history = state.get("medical_history", "")
        test_results = state.get("test_results", "")
        research_findings = state.get("research_findings", "")
        
        # Ensure the topic is related to lung cancer
        if "lung" not in topic.lower() and "cancer" not in topic.lower():
            topic = "lung cancer: " + topic
        
        print(f"Analyzing lung cancer case: {topic}")
        
        # Step 1: Classify lung cancer type
        classification = self.classifier.classify(
            symptoms=symptoms,
            test_results=test_results,
            medical_history=medical_history
        )
        
        cancer_type = classification["main_type"]
        cancer_subtype = classification["subtype"]
        genetic_markers = classification["genetic_markers"]
        smoking_status = classification["smoking_status"]
        
        print(f"Classified as: {cancer_type}, Subtype: {cancer_subtype}")
        print(f"Genetic markers: {', '.join(genetic_markers) if genetic_markers else 'None identified'}")
        
        # Step 2: Determine cancer stage
        staging = self.stager.stage(
            test_results=test_results,
            cancer_type=cancer_type,
            additional_info=research_findings + " " + medical_history
        )
        
        cancer_stage = staging["stage"]
        tnm_classification = staging["tnm"]
        
        print(f"Stage: {cancer_stage}, TNM: {tnm_classification}")
        
        # Step 3: Recommend treatment
        # Extract patient age if available
        age_match = re.search(r'(\d+)[- ]year[s]?[- ]old', medical_history)
        patient_age = int(age_match.group(1)) if age_match else None
        
        # Extract performance status if available
        ps_match = re.search(r'ECOG (?:PS|performance status)[: ]*(\d)', medical_history + " " + test_results)
        performance_status = int(ps_match.group(1)) if ps_match else 0  # Default to 0 if not specified
        
        # Extract comorbidities
        comorbidities = []
        common_comorbidities = ["diabetes", "hypertension", "copd", "heart disease", "kidney disease", "liver disease"]
        for comorbidity in common_comorbidities:
            if comorbidity in medical_history.lower():
                comorbidities.append(comorbidity)
        
        # Determine PD-L1 expression if mentioned
        pd_l1_expression = None
        if "pd-l1" in test_results.lower() or "pd l1" in test_results.lower():
            if "high" in test_results.lower() or "≥ 50%" in test_results or ">= 50%" in test_results:
                pd_l1_expression = "high"
            elif "low" in test_results.lower() or "1-49%" in test_results:
                pd_l1_expression = "low"
            elif "negative" in test_results.lower() or "< 1%" in test_results or "<1%" in test_results:
                pd_l1_expression = "negative"
        
        treatment_recommendations = self.treatment_advisor.recommend_treatment(
            cancer_type=cancer_type,
            cancer_stage=cancer_stage,
            genetic_markers=genetic_markers,
            pd_l1_expression=pd_l1_expression,
            patient_age=patient_age,
            performance_status=performance_status,
            comorbidities=comorbidities
        )
        
        # Step 4: Predict prognosis
        # Determine gender if mentioned
        gender = None
        if re.search(r'\b(male|man)\b', medical_history.lower()):
            gender = "male"
        elif re.search(r'\b(female|woman)\b', medical_history.lower()):
            gender = "female"
        
        # Check for weight loss
        weight_loss = "weight loss" in symptoms.lower() or "weight loss" in medical_history.lower()
        
        # Check for metastasis sites
        metastasis_sites = []
        potential_sites = ["brain", "liver", "bone", "adrenal", "lung"]
        for site in potential_sites:
            metastasis_pattern = f"{site} (metastasis|metastases|lesion)"
            if re.search(metastasis_pattern, test_results.lower() + " " + medical_history.lower()):
                metastasis_sites.append(site)
        
        prognosis = self.prognosis_predictor.predict_prognosis(
            cancer_type=cancer_type,
            cancer_stage=cancer_stage,
            genetic_markers=genetic_markers,
            patient_age=patient_age,
            gender=gender,
            performance_status=performance_status,
            weight_loss=weight_loss,
            metastasis_sites=metastasis_sites
        )
        
        # Step 5: Find clinical trials
        brain_metastases = "brain" in " ".join(metastasis_sites).lower()
        
        clinical_trials = self.trial_finder.find_trials(
            cancer_type=cancer_type,
            cancer_stage=cancer_stage,
            genetic_markers=genetic_markers,
            prior_treatment="No prior treatment" if "treatment" not in medical_history.lower() else "Prior treatment",
            performance_status=performance_status,
            brain_metastases=brain_metastases
        )
        
        # Create detailed diagnoses
        detailed_diagnoses = self._create_detailed_diagnoses(
            classification=classification,
            staging=staging,
            symptoms=symptoms,
            test_results=test_results,
            medical_history=medical_history,
            genetic_markers=genetic_markers
        )
        
        # Compile comprehensive lung cancer report
        lung_cancer_report = self._compile_report(
            classification=classification,
            staging=staging,
            treatment_recommendations=treatment_recommendations,
            prognosis=prognosis,
            clinical_trials=clinical_trials
        )
        
        # Update state with lung cancer analysis
        return {
            **state,
            "lung_cancer_analysis": {
                "classification": classification,
                "staging": staging,
                "treatment_recommendations": treatment_recommendations,
                "prognosis": prognosis,
                "clinical_trials": clinical_trials
            },
            "diagnoses": detailed_diagnoses,
            "treatments": self._format_treatments(treatment_recommendations),
            "consensus": lung_cancer_report,
            "next": "verify_sources"
        }
    
    def _create_detailed_diagnoses(self, 
                                  classification: Dict[str, Any],
                                  staging: Dict[str, Any],
                                  symptoms: str,
                                  test_results: str,
                                  medical_history: str,
                                  genetic_markers: List[str]) -> List[str]:
        """Create detailed diagnoses based on all available information."""
        detailed_diagnoses = []
        
        # Primary diagnosis with cancer type, subtype and stage
        primary_diagnosis = f"{classification['main_type']}: {classification['subtype']}, Stage {staging['stage']}"
        if "tnm" in staging and staging['tnm'] != "Not applicable for SCLC":
            primary_diagnosis += f" ({staging['tnm']})"
        detailed_diagnoses.append(primary_diagnosis)
        
        # Add histopathological features
        histo_features = []
        if "well" in classification['differentiation'].lower():
            histo_features.append("well-differentiated")
        elif "moderately" in classification['differentiation'].lower():
            histo_features.append("moderately differentiated")
        elif "poorly" in classification['differentiation'].lower():
            histo_features.append("poorly differentiated")
        elif "undifferentiated" in classification['differentiation'].lower():
            histo_features.append("undifferentiated")
        
        # Check for specific histological patterns in test results
        histology_patterns = ["acinar", "papillary", "lepidic", "solid", "micropapillary", "mucinous", "keratinizing", "non-keratinizing"]
        for pattern in histology_patterns:
            if pattern in test_results.lower():
                histo_features.append(pattern + " pattern")
        
        if histo_features:
            detailed_diagnoses.append(f"Histopathology: {', '.join(histo_features)}")
        
        # Add molecular profile
        if genetic_markers:
            detailed_diagnoses.append(f"Molecular profile: {', '.join(genetic_markers)}")
        
        # Add PD-L1 status if mentioned
        pd_l1_match = re.search(r'PD-L1[:\s]*(\d+)%', test_results)
        if pd_l1_match:
            pd_l1_value = pd_l1_match.group(1)
            detailed_diagnoses.append(f"PD-L1 expression: {pd_l1_value}%")
        
        # Add smoking status
        if classification['smoking_status'] != "Unknown":
            detailed_diagnoses.append(f"Smoking status: {classification['smoking_status']}")
        
        # Add metastasis information
        metastasis_sites = []
        potential_sites = ["brain", "liver", "bone", "adrenal", "contralateral lung"]
        for site in potential_sites:
            metastasis_pattern = f"{site} (metastasis|metastases|lesion)"
            if re.search(metastasis_pattern, test_results.lower() + " " + medical_history.lower()):
                metastasis_sites.append(site)
        
        if metastasis_sites:
            detailed_diagnoses.append(f"Metastases: {', '.join(metastasis_sites)}")
        
        # Add lymph node involvement
        if "N0" in staging.get("tnm", ""):
            detailed_diagnoses.append("Lymph nodes: No regional lymph node involvement")
        elif "N1" in staging.get("tnm", ""):
            detailed_diagnoses.append("Lymph nodes: Ipsilateral peribronchial/hilar involvement")
        elif "N2" in staging.get("tnm", ""):
            detailed_diagnoses.append("Lymph nodes: Ipsilateral mediastinal/subcarinal involvement")
        elif "N3" in staging.get("tnm", ""):
            detailed_diagnoses.append("Lymph nodes: Contralateral mediastinal/hilar or supraclavicular involvement")
        
        # Add key comorbidities that affect treatment decisions
        comorbidities = []
        if "copd" in medical_history.lower() or "chronic obstructive pulmonary disease" in medical_history.lower():
            comorbidities.append("COPD")
        if "heart" in medical_history.lower() or "cardiac" in medical_history.lower():
            comorbidities.append("Cardiovascular disease")
        if "diabetes" in medical_history.lower():
            comorbidities.append("Diabetes")
        if "kidney" in medical_history.lower() or "renal" in medical_history.lower():
            comorbidities.append("Renal impairment")
        
        if comorbidities:
            detailed_diagnoses.append(f"Relevant comorbidities: {', '.join(comorbidities)}")
        
        # Add performance status if available
        ps_match = re.search(r'ECOG (?:PS|performance status)[: ]*(\d)', medical_history + " " + test_results)
        if ps_match:
            ps_value = ps_match.group(1)
            detailed_diagnoses.append(f"ECOG Performance Status: {ps_value}")
        
        return detailed_diagnoses
    
    def _format_treatments(self, treatment_recommendations: Dict[str, Any]) -> List[str]:
        """Format treatment recommendations as a list of strings."""
        treatments = []
        
        if "primary_treatment" in treatment_recommendations:
            treatments.append(f"Primary: {', '.join(treatment_recommendations['primary_treatment'])}")
        
        if "chemotherapy" in treatment_recommendations:
            treatments.append(f"Chemotherapy: {', '.join(treatment_recommendations['chemotherapy'])}")
        
        if "targeted_therapy" in treatment_recommendations and treatment_recommendations["targeted_therapy"]:
            for therapy in treatment_recommendations["targeted_therapy"]:
                marker = therapy.get("marker", "")
                first_line = ", ".join(therapy.get("first_line", []))
                treatments.append(f"Targeted therapy for {marker}: {first_line}")
        
        if "immunotherapy" in treatment_recommendations and treatment_recommendations["immunotherapy"]:
            if "first_line" in treatment_recommendations["immunotherapy"]:
                immuno = ", ".join(treatment_recommendations["immunotherapy"]["first_line"])
                treatments.append(f"Immunotherapy: {immuno}")
        
        if "radiation_therapy" in treatment_recommendations:
            treatments.append(f"Radiation: {', '.join(treatment_recommendations['radiation_therapy'])}")
        
        if "additional_treatments" in treatment_recommendations:
            treatments.append(f"Additional: {', '.join(treatment_recommendations['additional_treatments'])}")
        
        return treatments
    
    def _compile_report(self, 
                       classification: Dict[str, Any],
                       staging: Dict[str, Any],
                       treatment_recommendations: Dict[str, Any],
                       prognosis: Dict[str, Any],
                       clinical_trials: Dict[str, Any]) -> str:
        """Compile a comprehensive lung cancer report."""
        report = f"# Comprehensive Lung Cancer Assessment\n\n"
        
        # Classification section
        report += f"## Cancer Classification\n"
        report += f"- **Type**: {classification['main_type']}\n"
        report += f"- **Subtype**: {classification['subtype']}\n"
        
        if classification['genetic_markers']:
            report += f"- **Genetic Markers**: {', '.join(classification['genetic_markers'])}\n"
        
        report += f"- **Smoking Status**: {classification['smoking_status']}\n"
        report += f"- **Differentiation**: {classification['differentiation']}\n\n"
        
        # Staging section
        report += f"## Cancer Staging\n"
        report += f"- **Stage**: {staging['stage']}\n"
        
        if "tnm" in staging and staging['tnm'] != "Not applicable for SCLC":
            report += f"- **TNM Classification**: {staging['tnm']}\n"
        
        report += f"- **Description**: {staging['description']}\n\n"
        
        # Treatment section
        report += f"## Treatment Recommendations\n"
        
        if "primary_treatment" in treatment_recommendations:
            report += f"### Primary Treatment\n"
            for treatment in treatment_recommendations['primary_treatment']:
                report += f"- {treatment}\n"
            report += "\n"
        
        if "targeted_therapy" in treatment_recommendations and treatment_recommendations["targeted_therapy"]:
            report += f"### Targeted Therapy Options\n"
            for therapy in treatment_recommendations["targeted_therapy"]:
                marker = therapy.get("marker", "")
                report += f"**For {marker}**:\n"
                if "first_line" in therapy:
                    report += f"- First-line: {', '.join(therapy['first_line'])}\n"
                if "subsequent" in therapy:
                    report += f"- Subsequent: {', '.join(therapy['subsequent'])}\n"
            report += "\n"
        
        if "immunotherapy" in treatment_recommendations and treatment_recommendations["immunotherapy"]:
            report += f"### Immunotherapy Options\n"
            if "first_line" in treatment_recommendations["immunotherapy"]:
                report += f"- First-line: {', '.join(treatment_recommendations['immunotherapy']['first_line'])}\n"
            if "combination" in treatment_recommendations["immunotherapy"]:
                report += f"- Combination: {', '.join(treatment_recommendations['immunotherapy']['combination'])}\n"
            report += "\n"
        
        if "clinical_considerations" in treatment_recommendations and treatment_recommendations["clinical_considerations"]:
            report += f"### Clinical Considerations\n"
            for consideration in treatment_recommendations["clinical_considerations"]:
                report += f"- {consideration}\n"
            report += "\n"
        
        # Prognosis section
        report += f"## Prognosis\n"
        report += f"- **5-Year Survival Rate**: Approximately {prognosis['adjusted_5yr_survival_rate']}% (range: {prognosis['adjusted_5yr_survival_range']})\n"
        report += f"- **Assessment**: {prognosis['prognosis_description']}\n\n"
        
        if "recommendations" in prognosis:
            report += f"### Recommendations to Improve Outcomes\n"
            for recommendation in prognosis["recommendations"]:
                report += f"- {recommendation}\n"
            report += "\n"
        
        # Clinical trials section
        report += f"## Clinical Trials\n"
        report += f"Found {clinical_trials['matching_trials_count']} potentially matching clinical trials.\n\n"
        
        if clinical_trials['matching_trials_count'] > 0:
            report += f"### Top Clinical Trial Options\n"
            for i, trial in enumerate(clinical_trials['matching_trials'][:3]):  # Show top 3 trials
                report += f"**{i+1}. {trial['title']}**\n"
                report += f"- ID: {trial['id']}\n"
                report += f"- Phase: {trial['phase']}\n"
                report += f"- Status: {trial['status']}\n"
                report += f"- URL: {trial['url']}\n\n"
        
        # Summary section
        report += f"## Summary\n"
        report += f"This patient has {classification['main_type']} ({classification['subtype']}), Stage {staging['stage']}. "
        
        if classification['genetic_markers']:
            report += f"Genetic testing reveals {', '.join(classification['genetic_markers'])}. "
        
        report += f"Recommended treatment includes {', '.join(treatment_recommendations['primary_treatment'][:2])}. "
        report += f"The estimated 5-year survival rate is approximately {prognosis['adjusted_5yr_survival_rate']}%. "
        
        if clinical_trials['matching_trials_count'] > 0:
            report += f"There are {clinical_trials['matching_trials_count']} potentially matching clinical trials that may be considered."
        
        return report 