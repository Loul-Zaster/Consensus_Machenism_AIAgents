"""
Clinical Trial Finder Module for Lung Cancer

This module provides functionality to find relevant clinical trials
for lung cancer patients based on cancer characteristics and patient factors.
"""

from typing import Dict, Any, List, Tuple, Optional
import re

class ClinicalTrialFinder:
    """
    Finder for lung cancer clinical trials based on patient and cancer characteristics.
    """
    
    def __init__(self):
        """Initialize the clinical trial finder."""
        # Sample clinical trials database (would be replaced with API call to clinicaltrials.gov in production)
        self.sample_trials = [
            {
                "id": "NCT04583995",
                "title": "Osimertinib With or Without Chemotherapy in EGFR Mutant NSCLC",
                "phase": "Phase 3",
                "conditions": ["Non-Small Cell Lung Cancer", "EGFR Mutation"],
                "interventions": ["Osimertinib", "Pemetrexed", "Carboplatin"],
                "eligibility": {
                    "stage": ["IIIB", "IV"],
                    "markers": ["EGFR Mutation"],
                    "prior_treatment": "No prior systemic therapy",
                    "performance_status": "0-1",
                    "brain_metastases": "Allowed if stable"
                },
                "locations": ["Multiple locations in United States", "International sites"],
                "status": "Recruiting",
                "url": "https://clinicaltrials.gov/ct2/show/NCT04583995"
            },
            {
                "id": "NCT03706625",
                "title": "Pembrolizumab in Combination With Etoposide and Platinum for Extensive Stage Small Cell Lung Cancer",
                "phase": "Phase 3",
                "conditions": ["Small Cell Lung Cancer", "Extensive Stage"],
                "interventions": ["Pembrolizumab", "Etoposide", "Cisplatin", "Carboplatin"],
                "eligibility": {
                    "stage": ["Extensive Stage"],
                    "markers": [],
                    "prior_treatment": "No prior systemic therapy",
                    "performance_status": "0-1",
                    "brain_metastases": "Allowed if treated and stable"
                },
                "locations": ["Multiple locations in United States", "International sites"],
                "status": "Active, not recruiting",
                "url": "https://clinicaltrials.gov/ct2/show/NCT03706625"
            },
            {
                "id": "NCT03829332",
                "title": "Neoadjuvant Chemo-Immunotherapy for Resectable Stage IIIA NSCLC",
                "phase": "Phase 2",
                "conditions": ["Non-Small Cell Lung Cancer", "Stage IIIA"],
                "interventions": ["Atezolizumab", "Carboplatin", "Nab-paclitaxel", "Surgery"],
                "eligibility": {
                    "stage": ["IIIA"],
                    "markers": [],
                    "prior_treatment": "No prior systemic therapy",
                    "performance_status": "0-1",
                    "brain_metastases": "Not allowed"
                },
                "locations": ["Multiple locations in United States"],
                "status": "Recruiting",
                "url": "https://clinicaltrials.gov/ct2/show/NCT03829332"
            },
            {
                "id": "NCT04585815",
                "title": "Sotorasib for Advanced KRAS G12C Mutant NSCLC",
                "phase": "Phase 2",
                "conditions": ["Non-Small Cell Lung Cancer", "KRAS G12C Mutation"],
                "interventions": ["Sotorasib"],
                "eligibility": {
                    "stage": ["IIIB", "IV"],
                    "markers": ["KRAS G12C Mutation"],
                    "prior_treatment": "At least one prior systemic therapy",
                    "performance_status": "0-2",
                    "brain_metastases": "Allowed if stable"
                },
                "locations": ["Multiple locations in United States", "International sites"],
                "status": "Recruiting",
                "url": "https://clinicaltrials.gov/ct2/show/NCT04585815"
            },
            {
                "id": "NCT04209843",
                "title": "Durvalumab + Novel Agents in First-Line Treatment of Metastatic NSCLC",
                "phase": "Phase 2",
                "conditions": ["Non-Small Cell Lung Cancer", "Metastatic"],
                "interventions": ["Durvalumab", "Oleclumab", "Monalizumab"],
                "eligibility": {
                    "stage": ["IV"],
                    "markers": ["PD-L1 ≥ 1%"],
                    "prior_treatment": "No prior systemic therapy",
                    "performance_status": "0-1",
                    "brain_metastases": "Allowed if treated and stable"
                },
                "locations": ["Multiple locations in United States", "International sites"],
                "status": "Recruiting",
                "url": "https://clinicaltrials.gov/ct2/show/NCT04209843"
            },
            {
                "id": "NCT04085315",
                "title": "Stereotactic Body Radiation Therapy (SBRT) in Early Stage Non-Small Cell Lung Cancer",
                "phase": "Phase 3",
                "conditions": ["Non-Small Cell Lung Cancer", "Early Stage"],
                "interventions": ["Stereotactic Body Radiation Therapy", "Standard Radiation Therapy"],
                "eligibility": {
                    "stage": ["I", "II"],
                    "markers": [],
                    "prior_treatment": "No prior therapy for lung cancer",
                    "performance_status": "0-2",
                    "brain_metastases": "Not allowed"
                },
                "locations": ["Multiple locations in United States"],
                "status": "Recruiting",
                "url": "https://clinicaltrials.gov/ct2/show/NCT04085315"
            },
            {
                "id": "NCT03976375",
                "title": "Prophylactic Cranial Irradiation (PCI) vs MRI Surveillance in Extensive Stage SCLC",
                "phase": "Phase 3",
                "conditions": ["Small Cell Lung Cancer", "Extensive Stage"],
                "interventions": ["Prophylactic Cranial Irradiation", "MRI Surveillance"],
                "eligibility": {
                    "stage": ["Extensive Stage"],
                    "markers": [],
                    "prior_treatment": "Completed first-line therapy with response",
                    "performance_status": "0-2",
                    "brain_metastases": "Not allowed"
                },
                "locations": ["Multiple locations in United States", "International sites"],
                "status": "Recruiting",
                "url": "https://clinicaltrials.gov/ct2/show/NCT03976375"
            },
            {
                "id": "NCT04268550",
                "title": "Targeted Therapy Directed by Genetic Testing in Advanced NSCLC",
                "phase": "Phase 2",
                "conditions": ["Non-Small Cell Lung Cancer", "Advanced Stage"],
                "interventions": ["Multiple targeted therapies based on genetic testing"],
                "eligibility": {
                    "stage": ["IIIB", "IV"],
                    "markers": ["Any actionable mutation"],
                    "prior_treatment": "No restriction",
                    "performance_status": "0-2",
                    "brain_metastases": "Allowed"
                },
                "locations": ["Multiple locations in United States"],
                "status": "Recruiting",
                "url": "https://clinicaltrials.gov/ct2/show/NCT04268550"
            },
            {
                "id": "NCT03088813",
                "title": "Brigatinib in ALK-Positive NSCLC",
                "phase": "Phase 3",
                "conditions": ["Non-Small Cell Lung Cancer", "ALK-Positive"],
                "interventions": ["Brigatinib", "Crizotinib"],
                "eligibility": {
                    "stage": ["IIIB", "IV"],
                    "markers": ["ALK Rearrangement"],
                    "prior_treatment": "No prior ALK inhibitor",
                    "performance_status": "0-2",
                    "brain_metastases": "Allowed"
                },
                "locations": ["Multiple locations in United States", "International sites"],
                "status": "Active, not recruiting",
                "url": "https://clinicaltrials.gov/ct2/show/NCT03088813"
            },
            {
                "id": "NCT04619797",
                "title": "Immunotherapy and SBRT for Early Stage NSCLC",
                "phase": "Phase 1/2",
                "conditions": ["Non-Small Cell Lung Cancer", "Stage I", "Stage II"],
                "interventions": ["Atezolizumab", "Stereotactic Body Radiation Therapy"],
                "eligibility": {
                    "stage": ["I", "II"],
                    "markers": [],
                    "prior_treatment": "No prior systemic therapy",
                    "performance_status": "0-1",
                    "brain_metastases": "Not allowed"
                },
                "locations": ["Multiple locations in United States"],
                "status": "Recruiting",
                "url": "https://clinicaltrials.gov/ct2/show/NCT04619797"
            },
            {
                "id": "NCT04640272",
                "title": "Lurbinectedin in Combination With Immunotherapy in SCLC",
                "phase": "Phase 1/2",
                "conditions": ["Small Cell Lung Cancer", "Recurrent"],
                "interventions": ["Lurbinectedin", "Atezolizumab"],
                "eligibility": {
                    "stage": ["Extensive Stage", "Recurrent"],
                    "markers": [],
                    "prior_treatment": "Progressed after platinum-based chemotherapy",
                    "performance_status": "0-1",
                    "brain_metastases": "Allowed if stable"
                },
                "locations": ["Multiple locations in United States"],
                "status": "Recruiting",
                "url": "https://clinicaltrials.gov/ct2/show/NCT04640272"
            },
            {
                "id": "NCT04581824",
                "title": "Mobocertinib in EGFR Exon 20 Insertion Mutant NSCLC",
                "phase": "Phase 3",
                "conditions": ["Non-Small Cell Lung Cancer", "EGFR Exon 20 Insertion Mutation"],
                "interventions": ["Mobocertinib", "Platinum-based Chemotherapy"],
                "eligibility": {
                    "stage": ["IIIB", "IV"],
                    "markers": ["EGFR Exon 20 Insertion Mutation"],
                    "prior_treatment": "No prior systemic therapy",
                    "performance_status": "0-1",
                    "brain_metastases": "Allowed if asymptomatic and stable"
                },
                "locations": ["Multiple locations in United States", "International sites"],
                "status": "Recruiting",
                "url": "https://clinicaltrials.gov/ct2/show/NCT04581824"
            }
        ]
    
    def find_trials(self, 
                   cancer_type: str,
                   cancer_stage: str,
                   genetic_markers: List[str] = None,
                   prior_treatment: str = None,
                   performance_status: int = None,
                   brain_metastases: bool = None) -> Dict[str, Any]:
        """
        Find clinical trials based on patient and cancer characteristics.
        
        Args:
            cancer_type: Type of lung cancer (e.g., "NSCLC", "SCLC")
            cancer_stage: Stage of cancer (e.g., "IA", "IIIB", "Limited-Stage")
            genetic_markers: List of genetic markers (e.g., ["EGFR Mutation", "ALK Rearrangement"])
            prior_treatment: Description of prior treatments
            performance_status: ECOG performance status (0-4)
            brain_metastases: Whether patient has brain metastases
            
        Returns:
            Dictionary with matching clinical trials
        """
        if genetic_markers is None:
            genetic_markers = []
        
        # Normalize inputs
        cancer_type_normalized = cancer_type.upper()
        cancer_stage_normalized = cancer_stage.upper().replace("STAGE ", "").strip()
        
        # Convert performance status to string range for matching
        ps_range = None
        if performance_status is not None:
            if performance_status <= 1:
                ps_range = "0-1"
            elif performance_status <= 2:
                ps_range = "0-2"
            else:
                ps_range = "0-3"
        
        # Find matching trials
        matching_trials = []
        
        for trial in self.sample_trials:
            # Check cancer type match
            type_match = False
            if "SCLC" in cancer_type_normalized or "SMALL CELL" in cancer_type_normalized:
                if any("Small Cell" in condition for condition in trial["conditions"]):
                    type_match = True
            else:  # NSCLC
                if any("Non-Small Cell" in condition for condition in trial["conditions"]):
                    type_match = True
            
            if not type_match:
                continue
            
            # Check stage match
            stage_match = False
            
            # Handle special cases for SCLC
            if "SCLC" in cancer_type_normalized or "SMALL CELL" in cancer_type_normalized:
                if "LIMITED" in cancer_stage_normalized and any("Limited Stage" in condition for condition in trial["conditions"]):
                    stage_match = True
                elif "EXTENSIVE" in cancer_stage_normalized and any("Extensive Stage" in condition for condition in trial["conditions"]):
                    stage_match = True
            else:
                # For NSCLC, check if the trial's stage requirements include the patient's stage
                if "stage" in trial["eligibility"]:
                    # Extract the major stage (I, II, III, IV)
                    major_stage = cancer_stage_normalized[0]
                    
                    for trial_stage in trial["eligibility"]["stage"]:
                        if trial_stage == cancer_stage_normalized or trial_stage[0] == major_stage:
                            stage_match = True
                            break
                        
                        # Handle advanced/metastatic designations
                        if cancer_stage_normalized.startswith("IV") and trial_stage in ["Advanced", "Metastatic", "IV"]:
                            stage_match = True
                            break
                        
                        if cancer_stage_normalized.startswith("III") and trial_stage in ["Advanced", "III"]:
                            stage_match = True
                            break
            
            if not stage_match:
                continue
            
            # Check genetic marker match if the trial specifies markers
            marker_match = True
            if trial["eligibility"]["markers"]:
                marker_match = False
                
                for trial_marker in trial["eligibility"]["markers"]:
                    if trial_marker == "Any actionable mutation" and genetic_markers:
                        marker_match = True
                        break
                        
                    for patient_marker in genetic_markers:
                        if trial_marker.lower() in patient_marker.lower():
                            marker_match = True
                            break
                    
                    if marker_match:
                        break
            
            if not marker_match:
                continue
            
            # Check performance status match if specified
            ps_match = True
            if ps_range and "performance_status" in trial["eligibility"]:
                trial_ps = trial["eligibility"]["performance_status"]
                
                # Extract maximum allowed PS in trial
                trial_max_ps = int(trial_ps.split("-")[1])
                patient_ps = performance_status
                
                if patient_ps > trial_max_ps:
                    ps_match = False
            
            if not ps_match:
                continue
            
            # Check brain metastases eligibility if specified
            brain_met_match = True
            if brain_metastases is not None and "brain_metastases" in trial["eligibility"]:
                trial_brain_met = trial["eligibility"]["brain_metastases"]
                
                if brain_metastases and "Not allowed" in trial_brain_met:
                    brain_met_match = False
            
            if not brain_met_match:
                continue
            
            # If all criteria match, add to matching trials
            matching_trials.append(trial)
        
        # Sort trials by phase (higher phases first) and status (recruiting first)
        def trial_sort_key(trial):
            # Phase sorting
            phase = trial["phase"]
            if "3" in phase:
                phase_score = 3
            elif "2" in phase:
                phase_score = 2
            elif "1/2" in phase:
                phase_score = 1.5
            elif "1" in phase:
                phase_score = 1
            else:
                phase_score = 0
                
            # Status sorting
            status_score = 1 if trial["status"] == "Recruiting" else 0
                
            return (-phase_score, -status_score)
        
        matching_trials.sort(key=trial_sort_key)
        
        # Prepare response
        result = {
            "cancer_type": cancer_type,
            "cancer_stage": cancer_stage,
            "matching_trials_count": len(matching_trials),
            "matching_trials": matching_trials,
            "search_criteria": {
                "cancer_type": cancer_type,
                "cancer_stage": cancer_stage,
                "genetic_markers": genetic_markers,
                "prior_treatment": prior_treatment,
                "performance_status": performance_status,
                "brain_metastases": brain_metastases
            }
        }
        
        return result
    
    def get_trial_details(self, trial_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific clinical trial.
        
        Args:
            trial_id: The ID of the clinical trial
            
        Returns:
            Dictionary with trial details or None if not found
        """
        for trial in self.sample_trials:
            if trial["id"] == trial_id:
                return trial
        
        return None 