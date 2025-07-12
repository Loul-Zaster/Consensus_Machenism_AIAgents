"""
Lung Cancer Classifier Module

This module provides functionality to classify lung cancer based on patient data,
including symptoms, test results, and medical history.
"""

from typing import Dict, Any, List, Tuple, Optional
import re

class LungCancerClassifier:
    """
    Classifier for lung cancer types and subtypes based on patient data.
    """
    
    def __init__(self):
        """Initialize the lung cancer classifier."""
        # Define key terms for different lung cancer types
        self.sclc_terms = [
            "small cell", "sclc", "oat cell", "neuroendocrine", 
            "extensive-stage", "limited-stage"
        ]
        
        self.nsclc_terms = [
            "non-small cell", "nsclc", "non small cell", "nonsmall cell"
        ]
        
        self.adenocarcinoma_terms = [
            "adenocarcinoma", "acinar", "papillary", "bronchioloalveolar",
            "lepidic", "egfr mutation", "alk rearrangement", "ros1", "ground glass"
        ]
        
        self.squamous_terms = [
            "squamous", "epidermoid", "keratinizing", "squamous cell carcinoma", "scc"
        ]
        
        self.large_cell_terms = [
            "large cell", "large-cell", "undifferentiated", "anaplastic", "pleomorphic"
        ]
        
        # Define key genetic mutations
        self.genetic_markers = {
            "EGFR": ["egfr", "epidermal growth factor receptor"],
            "ALK": ["alk", "anaplastic lymphoma kinase"],
            "ROS1": ["ros1", "ros-1"],
            "BRAF": ["braf", "b-raf"],
            "KRAS": ["kras", "k-ras"],
            "MET": ["met", "c-met", "met exon 14"],
            "RET": ["ret", "ret fusion"],
            "NTRK": ["ntrk", "neurotrophic receptor tyrosine kinase"],
            "HER2": ["her2", "erbb2"],
            "PD-L1": ["pd-l1", "programmed death-ligand 1"]
        }
    
    def classify(self, 
                 symptoms: str, 
                 test_results: str, 
                 medical_history: str = "") -> Dict[str, Any]:
        """
        Classify lung cancer type based on patient data.
        
        Args:
            symptoms: Patient symptoms
            test_results: Test results including pathology, imaging, etc.
            medical_history: Patient medical history
            
        Returns:
            Dictionary with classification results
        """
        combined_text = f"{symptoms} {test_results} {medical_history}".lower()
        
        # Determine main cancer type
        cancer_type = self._determine_main_type(combined_text)
        
        # Determine subtype
        cancer_subtype = self._determine_subtype(combined_text, cancer_type)
        
        # Identify genetic markers
        genetic_markers = self._identify_genetic_markers(combined_text)
        
        # Determine smoking status
        smoking_status = self._determine_smoking_status(medical_history.lower())
        
        # Assess differentiation level
        differentiation = self._assess_differentiation(combined_text)
        
        return {
            "main_type": cancer_type,
            "subtype": cancer_subtype,
            "genetic_markers": genetic_markers,
            "smoking_status": smoking_status,
            "differentiation": differentiation,
            "confidence": self._calculate_confidence(cancer_type, cancer_subtype, combined_text)
        }
    
    def _determine_main_type(self, text: str) -> str:
        """Determine the main lung cancer type (SCLC vs NSCLC)."""
        sclc_score = sum(1 for term in self.sclc_terms if term in text)
        nsclc_score = sum(1 for term in self.nsclc_terms if term in text)
        
        # Check for explicit mentions of types
        if sclc_score > nsclc_score:
            return "Small Cell Lung Cancer (SCLC)"
        elif nsclc_score > 0 or any(term in text for term in self.adenocarcinoma_terms + self.squamous_terms + self.large_cell_terms):
            return "Non-Small Cell Lung Cancer (NSCLC)"
        else:
            # Default to NSCLC as it's more common (85% of cases)
            return "Likely Non-Small Cell Lung Cancer (NSCLC)"
    
    def _determine_subtype(self, text: str, main_type: str) -> str:
        """Determine the lung cancer subtype."""
        if "Small Cell" in main_type:
            # SCLC subtypes
            if "combined" in text:
                return "Combined Small Cell Carcinoma"
            elif "pure" in text:
                return "Pure Small Cell Carcinoma"
            else:
                return "Small Cell Carcinoma"
        else:
            # NSCLC subtypes
            adenocarcinoma_score = sum(1 for term in self.adenocarcinoma_terms if term in text)
            squamous_score = sum(1 for term in self.squamous_terms if term in text)
            large_cell_score = sum(1 for term in self.large_cell_terms if term in text)
            
            if adenocarcinoma_score > squamous_score and adenocarcinoma_score > large_cell_score:
                return "Adenocarcinoma"
            elif squamous_score > adenocarcinoma_score and squamous_score > large_cell_score:
                return "Squamous Cell Carcinoma"
            elif large_cell_score > 0:
                return "Large Cell Carcinoma"
            else:
                return "Unspecified NSCLC"
    
    def _identify_genetic_markers(self, text: str) -> List[str]:
        """Identify genetic markers mentioned in the text."""
        found_markers = []
        
        for marker, terms in self.genetic_markers.items():
            if any(term in text for term in terms):
                # Try to determine if it's a mutation, fusion, etc.
                if f"{marker} mutation" in text or "mutated" in text:
                    found_markers.append(f"{marker} Mutation")
                elif f"{marker} fusion" in text:
                    found_markers.append(f"{marker} Fusion")
                elif f"{marker} rearrangement" in text:
                    found_markers.append(f"{marker} Rearrangement")
                elif f"{marker} amplification" in text:
                    found_markers.append(f"{marker} Amplification")
                elif f"{marker} positive" in text or f"{marker}+" in text:
                    found_markers.append(f"{marker} Positive")
                else:
                    found_markers.append(marker)
        
        return found_markers
    
    def _determine_smoking_status(self, medical_history: str) -> str:
        """Determine patient smoking status from medical history."""
        if not medical_history:
            return "Unknown"
            
        if "never smok" in medical_history or "non-smoker" in medical_history or "nonsmoker" in medical_history:
            return "Never Smoker"
        elif "former smoker" in medical_history or "ex-smoker" in medical_history or "quit smoking" in medical_history:
            return "Former Smoker"
        elif "smoker" in medical_history or "pack-year" in medical_history or "smoking" in medical_history:
            # Try to extract pack-years if available
            pack_years_match = re.search(r'(\d+)\s*pack[\s-]years?', medical_history)
            if pack_years_match:
                return f"Current Smoker ({pack_years_match.group(1)} pack-years)"
            else:
                return "Current Smoker"
        else:
            return "Unknown"
    
    def _assess_differentiation(self, text: str) -> str:
        """Assess the differentiation level of the tumor."""
        if "well differentiated" in text or "grade 1" in text:
            return "Well Differentiated"
        elif "moderately differentiated" in text or "grade 2" in text:
            return "Moderately Differentiated"
        elif "poorly differentiated" in text or "grade 3" in text:
            return "Poorly Differentiated"
        elif "undifferentiated" in text or "grade 4" in text:
            return "Undifferentiated"
        else:
            return "Unknown Differentiation"
    
    def _calculate_confidence(self, cancer_type: str, cancer_subtype: str, text: str) -> float:
        """Calculate confidence level in the classification."""
        confidence = 0.5  # Base confidence
        
        # Adjust based on specificity of classification
        if "Likely" not in cancer_type:
            confidence += 0.1
        
        if cancer_subtype != "Unspecified NSCLC":
            confidence += 0.1
        
        # Adjust based on presence of specific diagnostic terms
        diagnostic_terms = [
            "biopsy confirmed", "pathology report", "histologically confirmed",
            "immunohistochemistry", "histopathology", "cytology"
        ]
        
        for term in diagnostic_terms:
            if term in text:
                confidence += 0.05
                break
        
        # Cap confidence at 0.95
        return min(confidence, 0.95) 