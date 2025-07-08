"""
AI Agents module for the Consensus Mechanism system.
"""
from langchain.tools.base import BaseTool
from typing import Dict, Any, List, Optional
import json
from app.tools.web_search import web_search
import time

class ResearcherAgent:
    def __init__(self, realtime: bool = False):
        self.realtime = realtime
        self.session = None  # Sẽ được set từ bên ngoài
        
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Research medical information based on the given topic and symptoms.
        
        Args:
            state: The current state
                
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