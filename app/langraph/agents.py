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

def filter_thinking_tags(content: str) -> str:
    """
    Lọc bỏ nội dung nằm trong thẻ <think>...</think> hoặc các định dạng tương tự.
    
    Args:
        content: Chuỗi cần lọc
        
    Returns:
        Chuỗi đã được lọc bỏ phần thinking
    """
    # Lọc bỏ nội dung trong thẻ <think>...</think>
    pattern = r'<think>.*?</think>'
    content = re.sub(pattern, '', content, flags=re.DOTALL)
    
    # Cũng lọc bỏ các thẻ tương tự khác như [thinking], [thought], v.v.
    patterns = [
        r'\[thinking\].*?\[/thinking\]',
        r'\[thought\].*?\[/thought\]',
        r'\[reasoning\].*?\[/reasoning\]'
    ]
    
    for pattern in patterns:
        content = re.sub(pattern, '', content, flags=re.DOTALL)
    
    # Loại bỏ các dòng trống dư thừa
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
    """Agent for researching medical topics using web search."""
    
    def __init__(self, realtime: bool = False):
        """Initialize the researcher agent."""
        self.realtime = realtime
        self.session = None  # Sẽ được set từ bên ngoài
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the researcher agent to collect information about a medical topic.
        
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
        
        # Tăng số lần thử
        attempt += 1
        state["research_attempt"] = attempt
        
        if attempt > 3:
            print(f"Warning: Research attempt {attempt} for topic {topic}. Adjusting strategy.")
            # Nếu đã thử nhiều lần, sử dụng chiến lược tìm kiếm khác
            query = f"scientific medical information {topic} treatment diagnosis evidence based medicine"
            use_trusted_domains = True  # Luôn sử dụng domain tin cậy sau nhiều lần thử
        else:
            # Tạo truy vấn tìm kiếm
            if symptoms and len(symptoms.strip()) > 0 and symptoms.lower() != "no symptoms provided.":
                query = f"{topic} {symptoms} causes diagnosis treatment medical information"
            else:
                query = f"{topic} medical information diagnosis treatment"
            
            # Bổ sung thông tin từ lịch sử y tế và kết quả xét nghiệm
            if medical_history and medical_history.lower() != "no medical history provided.":
                relevant_history = medical_history[:100]  # Lấy 100 ký tự đầu tiên
                query += f" with {relevant_history}"
                
            if test_results and test_results.lower() != "no test results provided.":
                relevant_tests = test_results[:100]  # Lấy 100 ký tự đầu tiên
                query += f" test results {relevant_tests}"
            
            use_trusted_domains = True  # Mặc định sử dụng các domain y tế tin cậy
        
        print(f"Research query: {query}")
        print(f"Use trusted domains: {use_trusted_domains}")
        
        if not self.realtime:
            # Giả lập kết quả nghiên cứu nếu không cần realtime
            print("Using simulated research results")
            findings = self._simulate_research(topic, symptoms)
            
            return {**state, "research_findings": findings, "next": "verify_sources"}
        
        try:
            # Thực hiện tìm kiếm web thực sự
            search_results = web_search(query, num_results=5, use_trusted_domains=use_trusted_domains)
            
            if not search_results:
                print("No search results found. Using simulated research.")
                findings = self._simulate_research(topic, symptoms)
                return {**state, "research_findings": findings, "next": "verify_sources"}
            
            # Tổng hợp kết quả
            summary = f"Research findings for {topic}:\n\n"
            
            for i, result in enumerate(search_results):
                title = result.get("title", "No title")
                snippet = result.get("snippet", "No snippet")
                link = result.get("link", "No link")
                
                summary += f"{i+1}. {title}\n"
                summary += f"   Summary: {snippet}\n"
                summary += f"   Source: {link}\n\n"
            
            return {**state, "research_findings": summary, "next": "verify_sources"}
            
        except Exception as e:
            print(f"Error during research: {str(e)}")
            # Sử dụng kết quả giả lập trong trường hợp lỗi
            findings = self._simulate_research(topic, symptoms)
            return {**state, "research_findings": findings, "next": "verify_sources"}
    
    def _simulate_research(self, topic: str, symptoms: str) -> str:
        """Generate simulated research findings for offline testing."""
        return f"""Research findings for {topic}:

1. Definition and Overview:
   {topic.capitalize()} is a medical condition that affects thousands of patients each year.
   Common symptoms include {symptoms}.

2. Potential Causes:
   - Genetic factors
   - Environmental triggers
   - Lifestyle factors
   
3. Diagnostic Approach:
   - Clinical evaluation
   - Laboratory tests
   - Imaging studies
   
4. Treatment Options:
   - Medication management
   - Lifestyle modifications
   - Surgical interventions when necessary
   
5. Prognosis:
   The prognosis varies depending on the severity and individual patient factors.
   
Source: Medical Encyclopedia (2023)
"""


class Diagnostician:
    """Agent for diagnosing medical conditions."""
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate potential diagnoses based on research and symptoms.
        
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
            ("system", """You are a highly skilled medical diagnostician. Based on the patient information and research findings provided, 
             suggest the most likely diagnoses. Focus on evidence-based medicine.
             
             FORMAT YOUR RESPONSE AS FOLLOWS:
             
             Present 1-3 potential diagnoses in clear sections with headings. For each diagnosis:
             
             ## [DIAGNOSIS NAME]
             
             **Likelihood:** High/Medium/Low
             
             **Reasoning:** Provide a clear, concise paragraph explaining the evidence supporting this diagnosis. 
             Reference symptoms, history, and research that align with this diagnosis.
             
             **Key Indicators:** List 2-3 bullet points of the most important symptoms or findings supporting this diagnosis.
             
             Use clear, professional medical language but ensure it's also understandable."""),
            ("human", """Topic: {topic}
            Patient Symptoms: {symptoms}
            Medical History: {medical_history}
            Test Results: {test_results}
            
            Research Findings:
            {findings}
            
            Based on the above information, what are the most likely diagnoses? Format as instructed.""")
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
            print(f"Error during diagnosis: {str(e)}")
            return {**state, "diagnoses": f"Unable to generate diagnosis: {str(e)}", "next": "recommend_treatment"}


class TreatmentAdvisor:
    """Agent for recommending treatments."""
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend treatments based on diagnoses.
        
        Args:
            state: The current state
            
        Returns:
            Updated state with treatment recommendations
        """
        diagnoses = state.get("diagnoses", [])
        if not diagnoses:
            return {**state, "treatments": ["No diagnoses provided to base treatments on"], "next": "build_consensus"}
        
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
            ("system", """You are a medical treatment specialist. Based on the diagnoses and patient information, 
             recommend appropriate evidence-based treatments.
             
             FORMAT YOUR RESPONSE AS FOLLOWS:
             
             Present your treatment recommendations in clear, organized sections:
             
             ## Primary Interventions
             
             Present 2-4 primary treatment recommendations, each formatted as:
             
             ### [TREATMENT NAME]
             
             **Purpose:** Brief explanation of what this treatment addresses
             
             **Details:** Clear instructions on implementation (dosage if medication, frequency, duration, etc.)
             
             **Evidence:** Brief note on the evidence supporting this approach
             
             ## Lifestyle & Supportive Measures
             
             List 2-3 lifestyle modifications or supportive treatments that complement primary interventions.
             
             ## Follow-up & Monitoring
             
             Specify when the patient should follow up and what should be monitored.
             
             ## Precautions
             
             Note any important contraindications or warnings.
             
             Use clear, practical language that healthcare providers can easily communicate to patients."""),
            ("human", """Diagnoses:
            {diagnoses}
            
            Patient Symptoms: {symptoms}
            Medical History: {medical_history}
            
            Research Findings:
            {findings}
            
            Based on these diagnoses and patient information, what treatments would you recommend? Format as instructed.""")
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
            print(f"Error generating treatment recommendations: {str(e)}")
            return {**state, "treatments": f"Unable to generate treatment recommendations: {str(e)}", "next": "build_consensus"}


class ConsensusBuilder:
    """Agent for building consensus among multiple diagnoses and treatments."""
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build consensus from the various diagnoses and treatments.
        
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
            ("system", """You are a medical consensus builder. Your task is to analyze the diagnoses and treatments provided,
             and create a unified assessment that represents the most likely scenario based on available evidence.
             
             STRUCTURE YOUR RESPONSE IN THE EXACT FOLLOWING FORMAT:
             
             Step 1: REASONING (Labeled "REASONING")
             Explain in detail your medical reasoning process, weighing the different diagnoses, evidence strength, and treatment rationales.
             This section shows your critical thinking and evaluation of conflicting information.
             
             Step 2: CONSENSUS DIAGNOSIS (Labeled "CONSENSUS DIAGNOSIS")
             Provide a clear statement of the most likely diagnosis based on your reasoning, with a brief explanation.
             
             Step 3: PATIENT ACTION PLAN (Labeled "PATIENT ACTION PLAN")
             Present a clear, practical, and concise action plan that a patient can easily follow.
             Use accessible language and organize by priority (immediate actions first).
             Include specific steps, recommended timeframes, and clear guidance on when to seek further medical help.
             
             Be clear, evidence-based, and patient-centered in all sections."""),
            ("human", """Topic: {topic}
            
            Diagnoses:
{diagnoses}

            Recommended Treatments:
{treatments}

            Research Findings Summary:
            {findings}
            
            Sources (Credibility Score: {credibility}):
            {sources}
            
            Based on all this information, provide your reasoning, consensus diagnosis, and patient action plan.""")
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
            print(f"Error building consensus: {str(e)}")
            return {**state, "consensus": f"Unable to build consensus: {str(e)}", "next": None}


class SourceVerifier:
    """Agent for verifying the credibility of medical sources."""
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify the credibility of the research sources.
        
        Args:
            state: The current state
            
        Returns:
            Updated state with verified sources and credibility score
        """
        attempt = state.get("verification_attempt", 0)
        attempt += 1
        state["verification_attempt"] = attempt
        
        if attempt > 3:
            print(f"Warning: Verification attempt {attempt}. Using default verification.")
            verified_sources = ["Unable to verify sources after multiple attempts"]
            credibility = 0.7  # Assign a moderate credibility score
            return {**state, "verified_sources": verified_sources, "source_credibility": credibility, "next": "diagnose"}
        
        findings = state.get("research_findings", "")
        
        # Extract sources from the research findings
        sources = self._extract_sources(findings)
        
        if not sources:
            sources = ["No explicit sources found in the research findings"]
        
        # In a real implementation, we would verify each source
        # For simplicity, we'll simulate verification
        verified_sources = sources
        
        # Calculate a credibility score (0.0 - 1.0)
        # In a real implementation, this would be based on source reputation, etc.
        credibility = min(0.7 + (len(verified_sources) * 0.05), 1.0)
        
        return {**state, "verified_sources": verified_sources, "source_credibility": credibility, "next": "diagnose"}
    
    def _extract_sources(self, findings: str) -> List[str]:
        """Extract sources from research findings."""
        sources = set()  # Sử dụng set để tránh trùng lặp
        
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
        
        return list(sources)  # Chuyển về list khi trả về 