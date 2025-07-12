"""
Lung Cancer Treatment Advisor Module

This module provides treatment recommendations for lung cancer based on
cancer type, stage, genetic markers, and patient characteristics.
"""

from typing import Dict, Any, List, Tuple, Optional

class LungCancerTreatmentAdvisor:
    """
    Treatment advisor for lung cancer based on NCCN guidelines.
    """
    
    def __init__(self):
        """Initialize the lung cancer treatment advisor."""
        # Define treatment options by cancer type and stage
        self.nsclc_treatments = {
            "IA": {
                "primary": ["Surgical resection (lobectomy preferred)"],
                "alternative": ["Stereotactic Body Radiation Therapy (SBRT) if medically inoperable"],
                "adjuvant": ["Observation", "Consider adjuvant chemotherapy for high-risk features"]
            },
            "IB": {
                "primary": ["Surgical resection (lobectomy preferred)"],
                "alternative": ["Stereotactic Body Radiation Therapy (SBRT) if medically inoperable"],
                "adjuvant": ["Consider adjuvant chemotherapy for high-risk features"]
            },
            "IIA": {
                "primary": ["Surgical resection (lobectomy preferred)"],
                "alternative": ["Definitive radiation therapy if medically inoperable"],
                "adjuvant": ["Adjuvant chemotherapy"]
            },
            "IIB": {
                "primary": ["Surgical resection (lobectomy preferred)"],
                "alternative": ["Definitive radiation therapy if medically inoperable"],
                "adjuvant": ["Adjuvant chemotherapy"]
            },
            "IIIA": {
                "primary": ["Multidisciplinary evaluation"],
                "options": [
                    "Surgery followed by adjuvant chemotherapy",
                    "Concurrent chemoradiation therapy",
                    "Induction chemotherapy followed by surgery"
                ]
            },
            "IIIB": {
                "primary": ["Concurrent chemoradiation therapy"],
                "alternative": ["Sequential chemoradiation therapy if poor performance status"],
                "additional": ["Consider durvalumab after chemoradiation if no progression"]
            },
            "IIIC": {
                "primary": ["Concurrent chemoradiation therapy"],
                "alternative": ["Sequential chemoradiation therapy if poor performance status"],
                "additional": ["Consider durvalumab after chemoradiation if no progression"]
            },
            "IVA": {
                "primary": ["Systemic therapy based on biomarker testing"],
                "options": [
                    "Targeted therapy for actionable mutations",
                    "Immunotherapy for PD-L1 positive tumors",
                    "Chemotherapy",
                    "Consider local therapy for oligometastatic disease"
                ]
            },
            "IVB": {
                "primary": ["Systemic therapy based on biomarker testing"],
                "options": [
                    "Targeted therapy for actionable mutations",
                    "Immunotherapy for PD-L1 positive tumors",
                    "Chemotherapy",
                    "Best supportive care"
                ]
            }
        }
        
        self.sclc_treatments = {
            "Limited-Stage": {
                "primary": ["Concurrent chemoradiation therapy"],
                "chemotherapy": ["Platinum-based chemotherapy (cisplatin or carboplatin) + etoposide"],
                "radiation": ["Thoracic radiation therapy (preferably concurrent with chemotherapy)"],
                "additional": ["Prophylactic cranial irradiation (PCI) if good response to initial therapy"]
            },
            "Extensive-Stage": {
                "primary": ["Systemic therapy"],
                "first_line": ["Platinum-based chemotherapy (cisplatin or carboplatin) + etoposide + atezolizumab/durvalumab"],
                "alternative": ["Platinum-based chemotherapy (cisplatin or carboplatin) + etoposide"],
                "additional": ["Consider prophylactic cranial irradiation (PCI) if good response to chemotherapy"],
                "subsequent": ["Topotecan", "Lurbinectedin", "Clinical trial", "Best supportive care"]
            }
        }
        
        # Define targeted therapy options by genetic marker
        self.targeted_therapies = {
            "EGFR Mutation": {
                "first_line": ["Osimertinib", "Erlotinib", "Gefitinib", "Afatinib", "Dacomitinib"],
                "resistance": ["Osimertinib (if not used first-line)", "Chemotherapy", "Clinical trial"]
            },
            "ALK Rearrangement": {
                "first_line": ["Alectinib", "Brigatinib", "Lorlatinib"],
                "subsequent": ["Lorlatinib", "Ceritinib", "Chemotherapy"]
            },
            "ROS1 Fusion": {
                "first_line": ["Entrectinib", "Crizotinib"],
                "subsequent": ["Lorlatinib", "Chemotherapy"]
            },
            "BRAF Mutation": {
                "first_line": ["Dabrafenib + Trametinib"],
                "subsequent": ["Immunotherapy", "Chemotherapy"]
            },
            "KRAS Mutation": {
                "G12C": ["Sotorasib", "Adagrasib"],
                "other": ["Immunotherapy", "Chemotherapy"]
            },
            "MET Exon 14": {
                "first_line": ["Tepotinib", "Capmatinib"],
                "subsequent": ["Chemotherapy"]
            },
            "RET Fusion": {
                "first_line": ["Selpercatinib", "Pralsetinib"],
                "subsequent": ["Cabozantinib", "Chemotherapy"]
            },
            "NTRK Fusion": {
                "first_line": ["Larotrectinib", "Entrectinib"],
                "subsequent": ["Chemotherapy"]
            },
            "HER2 Mutation": {
                "first_line": ["Trastuzumab deruxtecan", "Chemotherapy"],
                "subsequent": ["Clinical trial", "Chemotherapy"]
            }
        }
        
        # Define immunotherapy options by PD-L1 expression
        self.immunotherapy_options = {
            "high_expression": {  # PD-L1 ≥ 50%
                "first_line": ["Pembrolizumab", "Cemiplimab", "Atezolizumab"],
                "combination": ["Pembrolizumab + chemotherapy", "Atezolizumab + chemotherapy + bevacizumab"]
            },
            "low_expression": {  # PD-L1 1-49%
                "first_line": ["Pembrolizumab + chemotherapy", "Atezolizumab + chemotherapy + bevacizumab"],
                "alternative": ["Pembrolizumab monotherapy", "Chemotherapy"]
            },
            "negative": {  # PD-L1 < 1%
                "first_line": ["Chemotherapy + immunotherapy", "Chemotherapy"],
                "subsequent": ["Nivolumab", "Atezolizumab", "Pembrolizumab"]
            }
        }
    
    def recommend_treatment(self, 
                           cancer_type: str,
                           cancer_stage: str,
                           genetic_markers: List[str] = None,
                           pd_l1_expression: str = None,
                           patient_age: int = None,
                           performance_status: int = None,
                           comorbidities: List[str] = None) -> Dict[str, Any]:
        """
        Recommend treatment based on cancer characteristics and patient factors.
        
        Args:
            cancer_type: Type of lung cancer (e.g., "NSCLC", "SCLC")
            cancer_stage: Stage of cancer (e.g., "IA", "IIIB", "Limited-Stage")
            genetic_markers: List of genetic markers (e.g., ["EGFR Mutation", "ALK Rearrangement"])
            pd_l1_expression: PD-L1 expression level (e.g., "high", "low", "negative")
            patient_age: Patient age in years
            performance_status: ECOG performance status (0-4)
            comorbidities: List of patient comorbidities
            
        Returns:
            Dictionary with treatment recommendations
        """
        if genetic_markers is None:
            genetic_markers = []
        
        if comorbidities is None:
            comorbidities = []
        
        # Normalize cancer type
        cancer_type = cancer_type.upper()
        
        # Determine if this is NSCLC or SCLC
        if "SCLC" in cancer_type or "SMALL CELL" in cancer_type:
            return self._recommend_sclc_treatment(cancer_stage, patient_age, performance_status, comorbidities)
        else:
            return self._recommend_nsclc_treatment(cancer_stage, genetic_markers, pd_l1_expression, 
                                                 patient_age, performance_status, comorbidities)
    
    def _recommend_nsclc_treatment(self, 
                                  stage: str, 
                                  genetic_markers: List[str],
                                  pd_l1_expression: str,
                                  patient_age: int,
                                  performance_status: int,
                                  comorbidities: List[str]) -> Dict[str, Any]:
        """Recommend treatment for NSCLC."""
        # Normalize stage format
        if stage.startswith("I") or stage.startswith("II") or stage.startswith("III") or stage.startswith("IV"):
            normalized_stage = stage
        else:
            # Extract Roman numeral part if stage is in format like "Stage IIIA"
            import re
            stage_match = re.search(r'(I+V?[ABC]?|I+[ABC]?\d?)', stage)
            if stage_match:
                normalized_stage = stage_match.group(1)
            else:
                normalized_stage = "Unknown"
        
        # Get base recommendations for this stage
        if normalized_stage in self.nsclc_treatments:
            base_recommendations = self.nsclc_treatments[normalized_stage]
        else:
            # Find closest stage if exact match not found
            if normalized_stage.startswith("I"):
                base_recommendations = self.nsclc_treatments["IA"]
            elif normalized_stage.startswith("II"):
                base_recommendations = self.nsclc_treatments["IIA"]
            elif normalized_stage.startswith("III"):
                base_recommendations = self.nsclc_treatments["IIIA"]
            elif normalized_stage.startswith("IV"):
                base_recommendations = self.nsclc_treatments["IVA"]
            else:
                base_recommendations = {
                    "primary": ["Treatment recommendations require accurate staging"],
                    "note": ["Please consult with a multidisciplinary tumor board"]
                }
        
        # Create initial recommendation
        recommendations = {
            "cancer_type": "Non-Small Cell Lung Cancer (NSCLC)",
            "stage": stage,
            "primary_treatment": base_recommendations.get("primary", []),
            "alternative_treatments": base_recommendations.get("alternative", []),
            "additional_treatments": base_recommendations.get("adjuvant", []) + base_recommendations.get("additional", []),
            "targeted_therapy": [],
            "immunotherapy": [],
            "clinical_considerations": []
        }
        
        # Add stage-specific options if available
        if "options" in base_recommendations:
            recommendations["treatment_options"] = base_recommendations["options"]
        
        # Add targeted therapy recommendations if genetic markers are present
        for marker in genetic_markers:
            for marker_type, therapies in self.targeted_therapies.items():
                if marker_type.lower() in marker.lower():
                    recommendations["targeted_therapy"].append({
                        "marker": marker,
                        "first_line": therapies.get("first_line", []),
                        "subsequent": therapies.get("subsequent", [])
                    })
                    
                    # If stage IV, prioritize targeted therapy
                    if normalized_stage.startswith("IV"):
                        recommendations["primary_treatment"] = [f"Targeted therapy for {marker}"]
                        recommendations["primary_treatment"].extend(therapies.get("first_line", []))
        
        # Add immunotherapy recommendations based on PD-L1 expression
        if pd_l1_expression:
            if pd_l1_expression.lower() == "high" or "≥ 50%" in pd_l1_expression or ">= 50%" in pd_l1_expression:
                recommendations["immunotherapy"] = self.immunotherapy_options["high_expression"]
                
                # If stage IV and no targetable mutations, prioritize immunotherapy
                if normalized_stage.startswith("IV") and not recommendations["targeted_therapy"]:
                    recommendations["primary_treatment"] = ["Immunotherapy (PD-L1 high expression)"]
                    recommendations["primary_treatment"].extend(self.immunotherapy_options["high_expression"]["first_line"])
                    
            elif pd_l1_expression.lower() == "low" or "1-49%" in pd_l1_expression:
                recommendations["immunotherapy"] = self.immunotherapy_options["low_expression"]
            else:
                recommendations["immunotherapy"] = self.immunotherapy_options["negative"]
        
        # Add clinical considerations based on patient factors
        if patient_age and patient_age >= 75:
            recommendations["clinical_considerations"].append("Consider less intensive therapy due to advanced age")
            
        if performance_status and performance_status >= 2:
            recommendations["clinical_considerations"].append("Consider less intensive therapy due to poor performance status")
            recommendations["clinical_considerations"].append("Evaluate for palliative care referral")
        
        for comorbidity in comorbidities:
            if "heart" in comorbidity.lower() or "cardiac" in comorbidity.lower():
                recommendations["clinical_considerations"].append("Cardiac evaluation recommended before treatment")
            if "pulmonary" in comorbidity.lower() or "copd" in comorbidity.lower():
                recommendations["clinical_considerations"].append("Pulmonary function testing recommended before surgery")
        
        # Add general recommendations
        recommendations["general_recommendations"] = [
            "Smoking cessation counseling if currently smoking",
            "Multidisciplinary tumor board discussion recommended",
            "Consider clinical trial participation",
            "Palliative care integration throughout treatment course"
        ]
        
        return recommendations
    
    def _recommend_sclc_treatment(self, 
                                 stage: str, 
                                 patient_age: int,
                                 performance_status: int,
                                 comorbidities: List[str]) -> Dict[str, Any]:
        """Recommend treatment for SCLC."""
        # Determine if Limited or Extensive stage
        if "limited" in stage.lower():
            normalized_stage = "Limited-Stage"
            base_recommendations = self.sclc_treatments["Limited-Stage"]
        elif "extensive" in stage.lower():
            normalized_stage = "Extensive-Stage"
            base_recommendations = self.sclc_treatments["Extensive-Stage"]
        else:
            normalized_stage = "Unknown Stage"
            base_recommendations = {
                "primary": ["Treatment recommendations require accurate staging"],
                "note": ["Please consult with a multidisciplinary tumor board"]
            }
        
        # Create initial recommendation
        recommendations = {
            "cancer_type": "Small Cell Lung Cancer (SCLC)",
            "stage": normalized_stage,
            "primary_treatment": base_recommendations.get("primary", []),
            "chemotherapy": base_recommendations.get("chemotherapy", []) + base_recommendations.get("first_line", []),
            "radiation_therapy": base_recommendations.get("radiation", []),
            "additional_treatments": base_recommendations.get("additional", []),
            "subsequent_therapy": base_recommendations.get("subsequent", []),
            "clinical_considerations": []
        }
        
        # Add clinical considerations based on patient factors
        if patient_age and patient_age >= 75:
            recommendations["clinical_considerations"].append("Consider carboplatin instead of cisplatin due to advanced age")
            recommendations["clinical_considerations"].append("Careful assessment of benefit vs. risk for prophylactic cranial irradiation")
            
        if performance_status and performance_status >= 2:
            recommendations["clinical_considerations"].append("Consider less intensive therapy due to poor performance status")
            recommendations["clinical_considerations"].append("Evaluate for palliative care referral")
            
            if performance_status >= 3:
                recommendations["clinical_considerations"].append("Consider best supportive care instead of aggressive treatment")
        
        for comorbidity in comorbidities:
            if "heart" in comorbidity.lower() or "cardiac" in comorbidity.lower():
                recommendations["clinical_considerations"].append("Cardiac evaluation recommended before treatment")
                recommendations["clinical_considerations"].append("Consider carboplatin instead of cisplatin")
            if "renal" in comorbidity.lower() or "kidney" in comorbidity.lower():
                recommendations["clinical_considerations"].append("Renal function assessment required")
                recommendations["clinical_considerations"].append("Consider carboplatin instead of cisplatin if renal impairment")
            if "hearing" in comorbidity.lower() or "neuropathy" in comorbidity.lower():
                recommendations["clinical_considerations"].append("Consider carboplatin instead of cisplatin to reduce neurotoxicity")
        
        # Add general recommendations
        recommendations["general_recommendations"] = [
            "Smoking cessation counseling if currently smoking",
            "Multidisciplinary tumor board discussion recommended",
            "Consider clinical trial participation",
            "Early integration of palliative care recommended",
            "Close monitoring for treatment response (typically after 2-3 cycles)"
        ]
        
        return recommendations 