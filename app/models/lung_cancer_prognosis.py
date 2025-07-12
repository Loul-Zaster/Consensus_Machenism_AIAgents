"""
Lung Cancer Prognosis Module

This module provides functionality to predict prognosis and survival rates
for lung cancer patients based on cancer type, stage, and patient factors.
"""

from typing import Dict, Any, List, Tuple, Optional

class LungCancerPrognosisPredictor:
    """
    Predictor for lung cancer prognosis and survival rates.
    """
    
    def __init__(self):
        """Initialize the lung cancer prognosis predictor."""
        # 5-year survival rates by stage for NSCLC (based on SEER database and AJCC 8th edition)
        self.nsclc_survival_rates = {
            "IA1": 92,
            "IA2": 83,
            "IA3": 77,
            "IA": 84,  # Average for all IA
            "IB": 68,
            "IIA": 60,
            "IIB": 53,
            "II": 56,  # Average for all II
            "IIIA": 36,
            "IIIB": 26,
            "IIIC": 13,
            "III": 30,  # Average for all III
            "IVA": 10,
            "IVB": 1,
            "IV": 7   # Average for all IV
        }
        
        # 5-year survival rates for SCLC
        self.sclc_survival_rates = {
            "Limited-Stage": 27,
            "Extensive-Stage": 3,
            "Unknown": 7  # Average for all SCLC
        }
        
        # Prognostic factors that affect survival
        self.prognostic_factors = {
            "positive": [
                "good_performance", "younger_age", "female_gender", 
                "no_weight_loss", "normal_ldh", "limited_metastases",
                "egfr_mutation", "alk_rearrangement", "ros1_fusion"
            ],
            "negative": [
                "poor_performance", "older_age", "male_gender", 
                "weight_loss", "elevated_ldh", "extensive_metastases",
                "kras_mutation", "brain_metastases", "liver_metastases", 
                "bone_metastases", "adrenal_metastases"
            ]
        }
    
    def predict_prognosis(self, 
                         cancer_type: str,
                         cancer_stage: str,
                         genetic_markers: List[str] = None,
                         patient_age: int = None,
                         gender: str = None,
                         performance_status: int = None,
                         weight_loss: bool = None,
                         metastasis_sites: List[str] = None) -> Dict[str, Any]:
        """
        Predict prognosis based on cancer characteristics and patient factors.
        
        Args:
            cancer_type: Type of lung cancer (e.g., "NSCLC", "SCLC")
            cancer_stage: Stage of cancer (e.g., "IA", "IIIB", "Limited-Stage")
            genetic_markers: List of genetic markers (e.g., ["EGFR Mutation", "ALK Rearrangement"])
            patient_age: Patient age in years
            gender: Patient gender
            performance_status: ECOG performance status (0-4)
            weight_loss: Whether patient has significant weight loss
            metastasis_sites: List of metastasis sites
            
        Returns:
            Dictionary with prognosis information
        """
        if genetic_markers is None:
            genetic_markers = []
        
        if metastasis_sites is None:
            metastasis_sites = []
        
        # Normalize cancer type
        cancer_type = cancer_type.upper()
        
        # Determine base survival rate based on cancer type and stage
        if "SCLC" in cancer_type or "SMALL CELL" in cancer_type:
            survival_rate, survival_range = self._get_sclc_survival_rate(cancer_stage)
            cancer_type_normalized = "Small Cell Lung Cancer (SCLC)"
        else:
            survival_rate, survival_range = self._get_nsclc_survival_rate(cancer_stage)
            cancer_type_normalized = "Non-Small Cell Lung Cancer (NSCLC)"
        
        # Adjust survival rate based on prognostic factors
        adjusted_survival_rate = survival_rate
        adjustment_factors = []
        
        # Adjust for age
        if patient_age is not None:
            if patient_age < 50:
                adjusted_survival_rate += 5
                adjustment_factors.append({"factor": "Younger age (<50)", "impact": "Positive", "adjustment": "+5%"})
            elif patient_age >= 70:
                adjusted_survival_rate -= 5
                adjustment_factors.append({"factor": "Older age (≥70)", "impact": "Negative", "adjustment": "-5%"})
        
        # Adjust for gender
        if gender is not None:
            if gender.lower() == "female":
                adjusted_survival_rate += 3
                adjustment_factors.append({"factor": "Female gender", "impact": "Positive", "adjustment": "+3%"})
            elif gender.lower() == "male":
                adjusted_survival_rate -= 1
                adjustment_factors.append({"factor": "Male gender", "impact": "Negative", "adjustment": "-1%"})
        
        # Adjust for performance status
        if performance_status is not None:
            if performance_status <= 1:
                adjusted_survival_rate += 5
                adjustment_factors.append({"factor": "Good performance status (0-1)", "impact": "Positive", "adjustment": "+5%"})
            elif performance_status >= 2:
                adjusted_survival_rate -= 10
                adjustment_factors.append({"factor": "Poor performance status (≥2)", "impact": "Negative", "adjustment": "-10%"})
        
        # Adjust for weight loss
        if weight_loss is not None:
            if weight_loss:
                adjusted_survival_rate -= 5
                adjustment_factors.append({"factor": "Significant weight loss", "impact": "Negative", "adjustment": "-5%"})
        
        # Adjust for genetic markers
        positive_markers = ["EGFR Mutation", "ALK Rearrangement", "ROS1 Fusion"]
        negative_markers = ["KRAS Mutation"]
        
        for marker in genetic_markers:
            if any(pos_marker.lower() in marker.lower() for pos_marker in positive_markers):
                if "IV" in cancer_stage:  # Most beneficial in advanced disease
                    adjusted_survival_rate += 10
                    adjustment_factors.append({"factor": marker, "impact": "Positive", "adjustment": "+10%"})
                else:
                    adjusted_survival_rate += 5
                    adjustment_factors.append({"factor": marker, "impact": "Positive", "adjustment": "+5%"})
            elif any(neg_marker.lower() in marker.lower() for neg_marker in negative_markers):
                adjusted_survival_rate -= 3
                adjustment_factors.append({"factor": marker, "impact": "Negative", "adjustment": "-3%"})
        
        # Adjust for metastasis sites
        for site in metastasis_sites:
            if "brain" in site.lower():
                adjusted_survival_rate -= 10
                adjustment_factors.append({"factor": "Brain metastases", "impact": "Negative", "adjustment": "-10%"})
            elif "liver" in site.lower():
                adjusted_survival_rate -= 8
                adjustment_factors.append({"factor": "Liver metastases", "impact": "Negative", "adjustment": "-8%"})
            elif "bone" in site.lower():
                adjusted_survival_rate -= 5
                adjustment_factors.append({"factor": "Bone metastases", "impact": "Negative", "adjustment": "-5%"})
            elif "adrenal" in site.lower():
                adjusted_survival_rate -= 3
                adjustment_factors.append({"factor": "Adrenal metastases", "impact": "Negative", "adjustment": "-3%"})
        
        # Ensure survival rate is within realistic bounds
        adjusted_survival_rate = max(1, min(99, adjusted_survival_rate))
        
        # Calculate adjusted survival range
        lower_bound = max(1, adjusted_survival_rate - 7)
        upper_bound = min(99, adjusted_survival_rate + 7)
        adjusted_survival_range = (lower_bound, upper_bound)
        
        # Generate prognosis description
        prognosis_description = self._generate_prognosis_description(
            cancer_type_normalized, cancer_stage, adjusted_survival_rate, adjustment_factors
        )
        
        # Generate recommendations for improving prognosis
        recommendations = self._generate_recommendations(
            cancer_type_normalized, cancer_stage, genetic_markers, metastasis_sites
        )
        
        return {
            "cancer_type": cancer_type_normalized,
            "cancer_stage": cancer_stage,
            "base_5yr_survival_rate": survival_rate,
            "base_5yr_survival_range": f"{survival_range[0]}% to {survival_range[1]}%",
            "adjusted_5yr_survival_rate": adjusted_survival_rate,
            "adjusted_5yr_survival_range": f"{adjusted_survival_range[0]}% to {adjusted_survival_range[1]}%",
            "adjustment_factors": adjustment_factors,
            "prognosis_description": prognosis_description,
            "recommendations": recommendations
        }
    
    def _get_nsclc_survival_rate(self, stage: str) -> Tuple[int, Tuple[int, int]]:
        """Get base 5-year survival rate for NSCLC by stage."""
        # Normalize stage format
        normalized_stage = stage.upper().replace("STAGE ", "").strip()
        
        # Get survival rate for this stage
        if normalized_stage in self.nsclc_survival_rates:
            survival_rate = self.nsclc_survival_rates[normalized_stage]
        else:
            # Find closest stage if exact match not found
            if normalized_stage.startswith("I"):
                survival_rate = self.nsclc_survival_rates["IA"]
            elif normalized_stage.startswith("II"):
                survival_rate = self.nsclc_survival_rates["II"]
            elif normalized_stage.startswith("III"):
                survival_rate = self.nsclc_survival_rates["III"]
            elif normalized_stage.startswith("IV"):
                survival_rate = self.nsclc_survival_rates["IV"]
            else:
                survival_rate = 50  # Default if stage is unknown
        
        # Calculate range (±5%)
        lower_bound = max(1, survival_rate - 5)
        upper_bound = min(99, survival_rate + 5)
        
        return survival_rate, (lower_bound, upper_bound)
    
    def _get_sclc_survival_rate(self, stage: str) -> Tuple[int, Tuple[int, int]]:
        """Get base 5-year survival rate for SCLC by stage."""
        # Normalize stage format
        normalized_stage = stage.upper().replace("STAGE ", "").strip()
        
        # Get survival rate for this stage
        if "LIMITED" in normalized_stage:
            survival_rate = self.sclc_survival_rates["Limited-Stage"]
        elif "EXTENSIVE" in normalized_stage:
            survival_rate = self.sclc_survival_rates["Extensive-Stage"]
        else:
            survival_rate = self.sclc_survival_rates["Unknown"]
        
        # Calculate range (±5%)
        lower_bound = max(1, survival_rate - 5)
        upper_bound = min(99, survival_rate + 5)
        
        return survival_rate, (lower_bound, upper_bound)
    
    def _generate_prognosis_description(self, 
                                       cancer_type: str, 
                                       cancer_stage: str, 
                                       survival_rate: int,
                                       adjustment_factors: List[Dict[str, str]]) -> str:
        """Generate a description of the prognosis."""
        # Base description by survival rate
        if survival_rate >= 70:
            outlook = "favorable"
        elif survival_rate >= 40:
            outlook = "intermediate"
        elif survival_rate >= 15:
            outlook = "guarded"
        else:
            outlook = "poor"
        
        description = f"The overall prognosis for {cancer_type} at {cancer_stage} is {outlook}. "
        description += f"The estimated 5-year survival rate is approximately {survival_rate}%. "
        
        # Add information about positive prognostic factors
        positive_factors = [factor for factor in adjustment_factors if factor["impact"] == "Positive"]
        if positive_factors:
            description += "Positive factors improving the prognosis include "
            description += ", ".join([factor["factor"] for factor in positive_factors]) + ". "
        
        # Add information about negative prognostic factors
        negative_factors = [factor for factor in adjustment_factors if factor["impact"] == "Negative"]
        if negative_factors:
            description += "Factors that may negatively affect the prognosis include "
            description += ", ".join([factor["factor"] for factor in negative_factors]) + ". "
        
        # Add general caveat
        description += "It's important to note that these statistics are based on population averages and individual outcomes may vary significantly."
        
        return description
    
    def _generate_recommendations(self, 
                                 cancer_type: str, 
                                 cancer_stage: str, 
                                 genetic_markers: List[str],
                                 metastasis_sites: List[str]) -> List[str]:
        """Generate recommendations for improving prognosis."""
        recommendations = [
            "Adhere to treatment plan and follow-up schedule",
            "Maintain good nutrition and stay physically active as tolerated",
            "Quit smoking if currently smoking",
            "Join a support group or seek psychological support"
        ]
        
        # Add specific recommendations based on cancer type and stage
        if "SCLC" in cancer_type:
            recommendations.append("Consider prophylactic cranial irradiation if recommended")
            recommendations.append("Prompt reporting of new symptoms due to risk of rapid progression")
        else:  # NSCLC
            if any(marker in " ".join(genetic_markers).lower() for marker in ["egfr", "alk", "ros1", "braf"]):
                recommendations.append("Adhere to targeted therapy regimen to maximize benefit")
                recommendations.append("Regular monitoring for treatment resistance")
            
            if "IV" in cancer_stage and not metastasis_sites:
                recommendations.append("Consider comprehensive genomic testing if not already done")
        
        # Add recommendations for metastasis management
        if any("brain" in site.lower() for site in metastasis_sites):
            recommendations.append("Be alert for neurological symptoms and report them promptly")
            recommendations.append("Follow neurological monitoring schedule")
        
        if any("bone" in site.lower() for site in metastasis_sites):
            recommendations.append("Consider bone-strengthening medications")
            recommendations.append("Take precautions to prevent falls and fractures")
        
        # Add clinical trial recommendation
        recommendations.append("Discuss clinical trial options with your oncologist")
        
        # Add palliative care recommendation for advanced disease
        if "III" in cancer_stage or "IV" in cancer_stage or "EXTENSIVE" in cancer_stage:
            recommendations.append("Consider early integration of palliative care for symptom management")
        
        return recommendations 