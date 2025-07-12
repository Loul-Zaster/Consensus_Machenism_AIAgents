"""
Lung Cancer Staging Module

This module provides functionality to determine the stage of lung cancer
based on the TNM classification system (8th edition).
"""

from typing import Dict, Any, List, Tuple, Optional
import re

class LungCancerStager:
    """
    Stager for lung cancer based on TNM classification system (8th edition).
    """
    
    def __init__(self):
        """Initialize the lung cancer stager."""
        # TNM patterns to look for in text
        self.t_patterns = {
            "TX": ["tx", "tumor cannot be assessed", "primary tumor cannot be assessed"],
            "T0": ["t0", "no evidence of primary tumor"],
            "Tis": ["tis", "carcinoma in situ"],
            "T1": ["t1", "tumor ≤ 3 cm", "tumor <= 3 cm", "tumor 3 cm or less"],
            "T1a": ["t1a", "tumor ≤ 1 cm", "tumor <= 1 cm", "tumor 1 cm or less"],
            "T1b": ["t1b", "tumor > 1 cm but ≤ 2 cm", "tumor > 1 cm but <= 2 cm", "tumor between 1 and 2 cm"],
            "T1c": ["t1c", "tumor > 2 cm but ≤ 3 cm", "tumor > 2 cm but <= 3 cm", "tumor between 2 and 3 cm"],
            "T2": ["t2", "tumor > 3 cm but ≤ 5 cm", "tumor > 3 cm but <= 5 cm", "tumor between 3 and 5 cm"],
            "T2a": ["t2a", "tumor > 3 cm but ≤ 4 cm", "tumor > 3 cm but <= 4 cm", "tumor between 3 and 4 cm"],
            "T2b": ["t2b", "tumor > 4 cm but ≤ 5 cm", "tumor > 4 cm but <= 5 cm", "tumor between 4 and 5 cm"],
            "T3": ["t3", "tumor > 5 cm but ≤ 7 cm", "tumor > 5 cm but <= 7 cm", "tumor between 5 and 7 cm", "invasion of chest wall"],
            "T4": ["t4", "tumor > 7 cm", "tumor more than 7 cm", "invasion of mediastinum", "invasion of diaphragm", "invasion of heart", "invasion of great vessels"]
        }
        
        self.n_patterns = {
            "NX": ["nx", "lymph nodes cannot be assessed"],
            "N0": ["n0", "no regional lymph node metastasis"],
            "N1": ["n1", "metastasis in ipsilateral peribronchial", "ipsilateral hilar lymph nodes"],
            "N2": ["n2", "metastasis in ipsilateral mediastinal", "subcarinal lymph nodes"],
            "N3": ["n3", "metastasis in contralateral mediastinal", "contralateral hilar", "ipsilateral or contralateral scalene", "supraclavicular lymph nodes"]
        }
        
        self.m_patterns = {
            "M0": ["m0", "no distant metastasis"],
            "M1": ["m1", "distant metastasis"],
            "M1a": ["m1a", "separate tumor nodule(s) in a contralateral lobe", "pleural nodules", "malignant pleural effusion", "malignant pericardial effusion"],
            "M1b": ["m1b", "single extrathoracic metastasis", "single distant metastasis"],
            "M1c": ["m1c", "multiple extrathoracic metastases", "multiple distant metastases"]
        }
        
        # Stage grouping based on TNM
        self.stage_groups = {
            "IA1": [("T1a", "N0", "M0")],
            "IA2": [("T1b", "N0", "M0")],
            "IA3": [("T1c", "N0", "M0")],
            "IB": [("T2a", "N0", "M0")],
            "IIA": [("T2b", "N0", "M0")],
            "IIB": [("T1a", "N1", "M0"), ("T1b", "N1", "M0"), ("T1c", "N1", "M0"), 
                   ("T2a", "N1", "M0"), ("T2b", "N1", "M0"), ("T3", "N0", "M0")],
            "IIIA": [("T1a", "N2", "M0"), ("T1b", "N2", "M0"), ("T1c", "N2", "M0"), 
                    ("T2a", "N2", "M0"), ("T2b", "N2", "M0"), ("T3", "N1", "M0"), 
                    ("T4", "N0", "M0"), ("T4", "N1", "M0")],
            "IIIB": [("T1a", "N3", "M0"), ("T1b", "N3", "M0"), ("T1c", "N3", "M0"), 
                    ("T2a", "N3", "M0"), ("T2b", "N3", "M0"), ("T3", "N2", "M0"), 
                    ("T4", "N2", "M0")],
            "IIIC": [("T3", "N3", "M0"), ("T4", "N3", "M0")],
            "IVA": [("Any T", "Any N", "M1a"), ("Any T", "Any N", "M1b")],
            "IVB": [("Any T", "Any N", "M1c")]
        }
        
        # SCLC staging is simpler - just Limited vs Extensive
        self.sclc_limited_patterns = [
            "limited stage", "limited-stage", "confined to hemithorax", "confined to one hemithorax",
            "confined to ipsilateral hemithorax", "can be encompassed in a radiation field"
        ]
        
        self.sclc_extensive_patterns = [
            "extensive stage", "extensive-stage", "beyond one hemithorax", "distant metastasis",
            "beyond radiation field", "metastatic", "metastases"
        ]
    
    def stage(self, 
              test_results: str, 
              cancer_type: str,
              additional_info: str = "") -> Dict[str, Any]:
        """
        Determine lung cancer stage based on test results and cancer type.
        
        Args:
            test_results: Test results including imaging, pathology, etc.
            cancer_type: Type of lung cancer (SCLC or NSCLC)
            additional_info: Any additional information
            
        Returns:
            Dictionary with staging results
        """
        combined_text = f"{test_results} {additional_info}".lower()
        
        if "small cell" in cancer_type.lower() or "sclc" in cancer_type.lower():
            # SCLC staging is simpler - just Limited vs Extensive
            return self._stage_sclc(combined_text)
        else:
            # NSCLC uses TNM staging
            return self._stage_nsclc(combined_text)
    
    def _stage_sclc(self, text: str) -> Dict[str, Any]:
        """Stage Small Cell Lung Cancer as Limited or Extensive."""
        # Check for explicit mentions of stage
        limited_score = sum(1 for pattern in self.sclc_limited_patterns if pattern in text)
        extensive_score = sum(1 for pattern in self.sclc_extensive_patterns if pattern in text)
        
        if extensive_score > limited_score:
            stage = "Extensive-Stage SCLC"
            description = "Cancer has spread beyond one lung or to distant parts of the body"
            confidence = 0.8 if extensive_score > 1 else 0.6
        elif limited_score > 0:
            stage = "Limited-Stage SCLC"
            description = "Cancer is confined to one lung and regional lymph nodes"
            confidence = 0.8 if limited_score > 1 else 0.6
        else:
            # Look for metastasis indicators if stage not explicitly mentioned
            metastasis_terms = ["metastasis", "metastases", "metastatic", "distant spread", "spread to liver", 
                              "spread to brain", "spread to bone", "spread to adrenal"]
            
            if any(term in text for term in metastasis_terms):
                stage = "Extensive-Stage SCLC"
                description = "Cancer has spread beyond one lung or to distant parts of the body"
                confidence = 0.7
            else:
                stage = "Unknown Stage SCLC"
                description = "Insufficient information to determine SCLC stage"
                confidence = 0.3
        
        return {
            "stage": stage,
            "description": description,
            "tnm": "Not applicable for SCLC",
            "confidence": confidence
        }
    
    def _stage_nsclc(self, text: str) -> Dict[str, Any]:
        """Stage Non-Small Cell Lung Cancer using TNM system."""
        # Determine T, N, and M classifications
        t_class = self._determine_t_classification(text)
        n_class = self._determine_n_classification(text)
        m_class = self._determine_m_classification(text)
        
        # Determine overall stage based on TNM
        stage, confidence = self._determine_stage_group(t_class, n_class, m_class, text)
        
        # Generate description
        description = self._generate_stage_description(stage)
        
        return {
            "stage": stage,
            "description": description,
            "tnm": f"{t_class} {n_class} {m_class}",
            "t_classification": t_class,
            "n_classification": n_class,
            "m_classification": m_class,
            "confidence": confidence
        }
    
    def _determine_t_classification(self, text: str) -> str:
        """Determine T classification from text."""
        # First check for explicit T classifications
        for t_class, patterns in self.t_patterns.items():
            if any(pattern in text for pattern in patterns):
                return t_class
        
        # If no explicit classification, try to infer from tumor size
        size_pattern = r'tumor\s+(?:size|measures|measuring|of)\s+(\d+(?:\.\d+)?)\s*(?:cm|centimeter)'
        size_match = re.search(size_pattern, text)
        
        if size_match:
            try:
                size = float(size_match.group(1))
                if size <= 1:
                    return "T1a"
                elif size <= 2:
                    return "T1b"
                elif size <= 3:
                    return "T1c"
                elif size <= 4:
                    return "T2a"
                elif size <= 5:
                    return "T2b"
                elif size <= 7:
                    return "T3"
                else:
                    return "T4"
            except ValueError:
                pass
        
        # Check for invasion terms that would indicate higher T stages
        if any(term in text for term in ["invades", "invasion", "invading", "extends into"]):
            if any(term in text for term in ["chest wall", "parietal pleura", "phrenic nerve"]):
                return "T3"
            elif any(term in text for term in ["mediastinum", "heart", "great vessels", "trachea", "carina", "esophagus", "vertebra", "diaphragm"]):
                return "T4"
        
        return "TX"  # Cannot be assessed
    
    def _determine_n_classification(self, text: str) -> str:
        """Determine N classification from text."""
        # Check for explicit N classifications
        for n_class, patterns in self.n_patterns.items():
            if any(pattern in text for pattern in patterns):
                return n_class
        
        # Check for lymph node involvement terms
        if "no lymph node" in text or "lymph nodes negative" in text or "no nodal involvement" in text:
            return "N0"
        elif "ipsilateral hilar" in text or "peribronchial" in text:
            return "N1"
        elif "ipsilateral mediastinal" in text or "subcarinal" in text:
            return "N2"
        elif "contralateral" in text or "supraclavicular" in text or "scalene" in text:
            return "N3"
        
        return "NX"  # Cannot be assessed
    
    def _determine_m_classification(self, text: str) -> str:
        """Determine M classification from text."""
        # Check for explicit M classifications
        for m_class, patterns in self.m_patterns.items():
            if any(pattern in text for pattern in patterns):
                return m_class
        
        # Check for metastasis terms
        if "no metastasis" in text or "no distant metastasis" in text or "no evidence of metastatic disease" in text:
            return "M0"
        elif "pleural nodules" in text or "pleural effusion" in text or "pericardial effusion" in text or "separate tumor nodule" in text and "contralateral" in text:
            return "M1a"
        elif "single" in text and any(term in text for term in ["metastasis", "metastatic lesion"]) and any(term in text for term in ["brain", "liver", "adrenal", "bone"]):
            return "M1b"
        elif "multiple" in text and any(term in text for term in ["metastases", "metastatic lesions"]) and any(term in text for term in ["brain", "liver", "adrenal", "bone"]):
            return "M1c"
        elif any(term in text for term in ["metastasis", "metastases", "metastatic"]) and any(term in text for term in ["brain", "liver", "adrenal", "bone"]):
            return "M1"  # Metastasis present but details unclear
        
        return "M0"  # Default to M0 if no evidence of metastasis
    
    def _determine_stage_group(self, t_class: str, n_class: str, m_class: str, text: str) -> Tuple[str, float]:
        """Determine stage group based on TNM classification."""
        # First check for explicit stage mentions
        stage_pattern = r'stage\s+(I+V?[ABC]?|I+[ABC]?\d?)'
        stage_match = re.search(stage_pattern, text, re.IGNORECASE)
        
        if stage_match:
            explicit_stage = stage_match.group(1).upper()
            # Normalize stage format
            if explicit_stage == "I":
                explicit_stage = "IA"
            elif explicit_stage == "II":
                explicit_stage = "IIA"
            elif explicit_stage == "III":
                explicit_stage = "IIIA"
            elif explicit_stage == "IV":
                explicit_stage = "IVA"
            return explicit_stage, 0.9
        
        # If M1b or M1c, it's stage IV regardless of T and N
        if m_class == "M1c":
            return "IVB", 0.9
        elif m_class == "M1b" or m_class == "M1a":
            return "IVA", 0.9
        elif m_class == "M1":
            return "IV", 0.8  # Just "IV" without A/B if details unclear
        
        # Otherwise, look up the stage based on TNM combination
        for stage, tnm_combinations in self.stage_groups.items():
            for t, n, m in tnm_combinations:
                if (t == t_class or t == "Any T") and (n == n_class or n == "Any N") and (m == m_class or m == "Any M"):
                    confidence = 0.8
                    # Reduce confidence if any classification is X
                    if "X" in t_class or "X" in n_class:
                        confidence -= 0.2
                    return stage, confidence
        
        # If no match found, make a best guess based on available info
        if t_class in ["T1a", "T1b", "T1c"] and n_class == "N0":
            return "IA", 0.6
        elif t_class in ["T2a", "T2b"] and n_class == "N0":
            return "IB", 0.6
        elif n_class in ["N1"]:
            return "II", 0.5
        elif n_class in ["N2", "N3"] or t_class == "T4":
            return "III", 0.5
        
        return "Unknown", 0.3
    
    def _generate_stage_description(self, stage: str) -> str:
        """Generate a description of the cancer stage."""
        stage_descriptions = {
            "IA1": "Very early cancer confined to lung tissue. Tumor is 1 cm or less.",
            "IA2": "Very early cancer confined to lung tissue. Tumor is between 1-2 cm.",
            "IA3": "Very early cancer confined to lung tissue. Tumor is between 2-3 cm.",
            "IA": "Very early cancer confined to lung tissue. Tumor is 3 cm or less.",
            "IB": "Early cancer confined to lung tissue. Tumor is between 3-4 cm.",
            "IIA": "Early cancer confined to lung tissue. Tumor is between 4-5 cm.",
            "IIB": "Locally advanced cancer that may have spread to nearby lymph nodes or chest structures.",
            "II": "Early cancer that may have spread to nearby lymph nodes.",
            "IIIA": "Locally advanced cancer that has spread to lymph nodes on the same side of the chest.",
            "IIIB": "Locally advanced cancer that has spread to lymph nodes above the collarbone or on the opposite side.",
            "IIIC": "Locally advanced cancer with extensive lymph node involvement.",
            "III": "Locally advanced cancer that has spread to nearby structures or lymph nodes.",
            "IVA": "Advanced cancer that has spread within the chest cavity or to a single area outside the chest.",
            "IVB": "Advanced cancer that has spread to multiple areas outside the chest.",
            "IV": "Advanced cancer that has spread to distant parts of the body.",
            "Limited-Stage SCLC": "Cancer is confined to one lung and regional lymph nodes, and can be safely treated with radiation therapy.",
            "Extensive-Stage SCLC": "Cancer has spread beyond one lung, to the other lung, to lymph nodes on the other side, or to distant organs.",
            "Unknown": "Insufficient information to determine the cancer stage accurately."
        }
        
        return stage_descriptions.get(stage, "Stage information not available.") 