import pandas as pd
import networkx as nx
from dowhy import gcm
from typing import Dict, Any, List, Tuple
import json
import os
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from scipy.stats import norm

class ManufacturingRCAAnalyzer:
    """Comprehensive Root Cause Analysis for Manufacturing Processes"""
    
    def __init__(self, azure_openai_client):
        self.common_causal_graph = self._build_common_causal_graph()
        self.scm = gcm.StructuralCausalModel(self.common_causal_graph)
        self.client = azure_openai_client
        self.potential_root_causes = set()

    def _build_common_causal_graph(self) -> nx.DiGraph:
        """Constructs the manufacturing process causal graph"""
        return nx.DiGraph([
            ('Document_Version_Control', 'Work_Instruction_Accuracy'),
            ('Document_Version_Control', 'Visual_Aid_Accuracy'),
            ('BOM_Accuracy', 'Work_Instruction_Accuracy'),
            ('Setup_Sheet_Accuracy', 'Work_Instruction_Accuracy'),
            ('Visual_Aid_Accuracy', 'Work_Instruction_Accuracy'),
            ('Operator_Training', 'Correct_Part_Usage'),
            ('Operator_Training', 'Part_Verification_Process'),
            ('Work_Instruction_Accuracy', 'Correct_Part_Usage'),
            ('Part_Verification_Process', 'Correct_Part_Usage'),
            ('Line_Stoppage_Protocol', 'Production_Impact'),
            ('Correct_Part_Usage', 'Production_Impact'),
            ('Work_Instruction_Accuracy', 'Production_Impact'),
            ('Equipment_Condition', 'Production_Impact'),
            ('Material_Quality', 'Production_Impact')
        ])

    def _generate_realistic_synthetic_data(self, case_details: Dict[str, Any], num_samples=500) -> pd.DataFrame:
        """Generates synthetic data matching case characteristics"""
        base_values = {
            'Document_Version_Control': case_details.get('document_version_issue', 0),
            'BOM_Accuracy': case_details.get('bom_accurate', 1),
            'Setup_Sheet_Accuracy': case_details.get('setup_sheet_accurate', 1),
            'Visual_Aid_Accuracy': case_details.get('visual_aid_accurate', 0),
            'Operator_Training': case_details.get('operator_trained', 1),
            'Part_Verification_Process': case_details.get('part_verification_done', 0),
            'Line_Stoppage_Protocol': case_details.get('line_stopped_correctly', 1),
            'Equipment_Condition': case_details.get('equipment_condition', 0.8),
            'Material_Quality': case_details.get('material_quality', 0.9),
            'Correct_Part_Usage': case_details.get('correct_part_used', 0),
            'Work_Instruction_Accuracy': 1,
            'Production_Impact': case_details.get('production_impact', 1)
        }
        
        rng = np.random.default_rng(42)
        data = pd.DataFrame()
        
        # Generate base variables with appropriate distributions
        for col in base_values:
            if col in ['Document_Version_Control', 'Visual_Aid_Accuracy', 'Operator_Training',
                     'Part_Verification_Process', 'Line_Stoppage_Protocol', 'Correct_Part_Usage']:
                data[col] = rng.binomial(1, base_values[col], num_samples)
            else:
                data[col] = np.clip(rng.normal(base_values[col], 0.1, num_samples), 0, 1)
        
        # Generate intermediate nodes with causal relationships
        data['Work_Instruction_Accuracy'] = np.clip(
            0.3 * data['Document_Version_Control'] +
            0.3 * data['BOM_Accuracy'] +
            0.3 * data['Setup_Sheet_Accuracy'] +
            0.1 * data['Visual_Aid_Accuracy'] +
            rng.normal(0, 0.05, num_samples),
            0, 1
        )
        
        # Generate target node with impact relationships
        data['Production_Impact'] = np.clip(
            0.5 * (1 - data['Correct_Part_Usage']) +
            0.2 * (1 - data['Work_Instruction_Accuracy']) +
            0.2 * (1 - data['Equipment_Condition']) +
            0.1 * (1 - data['Material_Quality']) +
            rng.normal(0, 0.05, num_samples),
            0, 1
        )
        
        return data

    def analyze_impact_paths(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Analyzes all impact pathways to target node"""
        gcm.auto.assign_causal_mechanisms(self.scm, data)
        gcm.fit(self.scm, data)
        
        impact_metrics = {}
        num_samples = len(data)
        
        # 1. Direct causal influence
        try:
            influence_results = gcm.intrinsic_causal_influence(
                self.scm, 
                target_node='Production_Impact',
                prediction_model='approx'
            )
            impact_metrics['direct_influence'] = {
                node: float(score) if not callable(score) else 0.0
                for node, score in influence_results.items()
            }
        except Exception as e:
            print(f"Error calculating direct influence: {str(e)}")
            impact_metrics['direct_influence'] = {}

        # 2. Path-specific effects
        impact_metrics['path_effects'] = {}
        for node in self.common_causal_graph.nodes():
            if node != 'Production_Impact' and node in data.columns:
                if data[node].isnull().any() or not np.issubdtype(data[node].dtype, np.number):
                    continue
                    
                node_values = data[node].values.astype(float)
                ref_value = float(np.nanmean(node_values))
                
                def alt_intervention(_): return node_values.copy()
                def ref_intervention(_): return np.full(num_samples, ref_value)
                
                try:
                    effect = gcm.average_causal_effect(
                        causal_model=self.scm,
                        target_node='Production_Impact',
                        interventions_alternative={node: alt_intervention},
                        interventions_reference={node: ref_intervention},
                        num_samples_to_draw=num_samples
                    )
                    if effect is not None and not np.isnan(effect):
                        impact_metrics['path_effects'][node] = float(effect)
                except Exception as e:
                    print(f"Error calculating effect for {node}: {str(e)}")
                    continue
    
        # 3. Counterfactual impact
        impact_metrics['counterfactual'] = {}
        try:
            normal_samples = data.mean(numeric_only=True).to_frame().T
            anomaly_samples = data.iloc[:1]
            
            for node in self.common_causal_graph.nodes():
                if node != 'Production_Impact' and node in normal_samples.columns:
                    try:
                        effect = gcm.counterfactual.distribute_causal_effect(
                            self.scm,
                            'Production_Impact',
                            node,
                            normal_samples,
                            anomaly_samples
                        )
                        if effect is not None:
                            impact_metrics['counterfactual'][node] = float(effect)
                    except Exception as e:
                        print(f"Error calculating counterfactual for {node}: {str(e)}")
                        continue
        except Exception as e:
            print(f"Error in counterfactual setup: {str(e)}")

        # 4. Statistical significance
        impact_metrics['significance'] = self._calculate_impact_significance(data)
        
        return impact_metrics

    def _calculate_impact_significance(self, data: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """Calculates statistical significance of impacts"""
        significance = {}
        num_iterations = 100
        
        try:
            baseline = float(data['Production_Impact'].mean())
        except:
            return significance
        
        for node in self.common_causal_graph.nodes():
            if node != 'Production_Impact' and node in data.columns:
                if data[node].isnull().any() or not np.issubdtype(data[node].dtype, np.number):
                    continue
                    
                effects = []
                node_mean = float(data[node].mean())
                
                for _ in range(num_iterations):
                    try:
                        pred = gcm.interventional_samples(
                            self.scm,
                            interventions={node: lambda x: np.full(x.shape[0], node_mean)},
                            num_samples_to_draw=100
                        )['Production_Impact']
                        
                        effects.append(baseline - float(pred.mean()))
                    except:
                        continue
                
                if len(effects) > 0:
                    mean_effect = float(np.nanmean(effects))
                    std_effect = float(np.nanstd(effects)) if len(effects) > 1 else 0.0
                    
                    if std_effect > 0:
                        try:
                            z_score = mean_effect / (std_effect / np.sqrt(len(effects)))
                            p_value = 2 * (1 - norm.cdf(abs(z_score)))
                        except:
                            p_value = 1.0
                    else:
                        p_value = 1.0
                    
                    significance[node] = (mean_effect, float(p_value))
        
        return significance

    def analyze_case(self, case_text: str, num_samples=500) -> Dict[str, Any]:
        """Complete root cause analysis workflow"""
        case_details = self._extract_case_details(case_text)
        self._handle_potential_root_causes(case_details)
        
        data = self._generate_realistic_synthetic_data(case_details, num_samples)
        
        gcm.auto.assign_causal_mechanisms(self.scm, data)
        gcm.fit(self.scm, data)
        
        attributions = gcm.attribute_anomalies(
            self.scm, 
            target_node='Production_Impact',
            anomaly_samples=data.iloc[:1]
        )
        
        influence = self.analyze_impact_paths(data)
        
        return {
            'case_details': case_details,
            'causal_attributions': {
                k: float(v[0]) if isinstance(v, (np.ndarray, list)) else float(v)
                for k, v in attributions.items()
            },
            'impact_metrics': influence,
            'recommendations': self._generate_recommendations(case_details, influence)
        }

    # [Additional helper methods remain unchanged...]

def initialize_azure_client() -> AzureOpenAI:
    """Initializes Azure OpenAI client"""
    load_dotenv()
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

if __name__ == "__main__":
    try:
        client = initialize_azure_client()
        analyzer = ManufacturingRCAAnalyzer(client)
        
        case_text = """On Jan 12, 2021, the MESE1 Manufacturing Line was stopped..."""
        
        results = analyzer.analyze_case(case_text)
        
        print("Analysis completed successfully")
        print(f"Root Cause: {results['case_details'].get('root_cause_hypothesis', 'Unknown')}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")



*******************
*****************************

def analyze_impact_paths(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Comprehensive analysis of impact pathways to target node."""
    gcm.auto.assign_causal_mechanisms(self.scm, data)
    gcm.fit(self.scm, data)
    
    impact_metrics = {}
    num_samples = len(data)
    
    # 1. Direct causal influence (total contribution)
    try:
        influence_results = gcm.intrinsic_causal_influence(
            self.scm, 
            target_node='Production_Impact',
            prediction_model='approx'
        )
        impact_metrics['direct_influence'] = {
            node: float(score) if not callable(score) else 0.0
            for node, score in influence_results.items()
        }
    except Exception as e:
        print(f"Error calculating direct influence: {str(e)}")
        impact_metrics['direct_influence'] = {}

    # 2. Path-specific effects (direct paths)
    impact_metrics['path_effects'] = {}
    for node in self.common_causal_graph.nodes():
        if node != 'Production_Impact' and node in data.columns:
            # Skip if node has null values or isn't numeric
            if data[node].isnull().any() or not np.issubdtype(data[node].dtype, np.number):
                continue
                
            # Get concrete values first
            node_values = data[node].values.astype(float)
            ref_value = float(np.nanmean(node_values))
            
            # Define intervention functions
            def alt_intervention(_): 
                return node_values.copy()
                
            def ref_intervention(_): 
                return np.full(num_samples, ref_value)
            
            try:
                effect = gcm.average_causal_effect(
                    causal_model=self.scm,
                    target_node='Production_Impact',
                    interventions_alternative={node: alt_intervention},
                    interventions_reference={node: ref_intervention},
                    num_samples_to_draw=num_samples
                )
                if effect is not None and not np.isnan(effect):
                    impact_metrics['path_effects'][node] = float(effect)
            except Exception as e:
                print(f"Error calculating effect for {node}: {str(e)}")
                continue
    
    # 3. Counterfactual impact (what-if analysis)
    impact_metrics['counterfactual'] = {}
    try:
        normal_samples = data.mean(numeric_only=True).to_frame().T
        anomaly_samples = data.iloc[:1]
        
        for node in self.common_causal_graph.nodes():
            if node != 'Production_Impact' and node in normal_samples.columns:
                try:
                    effect = gcm.counterfactual.distribute_causal_effect(
                        self.scm,
                        'Production_Impact',
                        node,
                        normal_samples,
                        anomaly_samples
                    )
                    if effect is not None:
                        impact_metrics['counterfactual'][node] = float(effect)
                except Exception as e:
                    print(f"Error calculating counterfactual for {node}: {str(e)}")
                    continue
    except Exception as e:
        print(f"Error in counterfactual setup: {str(e)}")

    # 4. Statistical significance testing
    impact_metrics['significance'] = self._calculate_impact_significance(data)
    
    return impact_metrics

def _calculate_impact_significance(self, data: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """Calculates statistical significance of each factor's impact."""
    significance = {}
    num_iterations = 100
    
    try:
        baseline = float(data['Production_Impact'].mean())
    except:
        return significance
    
    for node in self.common_causal_graph.nodes():
        if node != 'Production_Impact' and node in data.columns:
            # Skip if node has null values or isn't numeric
            if data[node].isnull().any() or not np.issubdtype(data[node].dtype, np.number):
                continue
                
            effects = []
            node_mean = float(data[node].mean())
            
            for _ in range(num_iterations):
                try:
                    # Predict outcome under intervention
                    pred = gcm.interventional_samples(
                        self.scm,
                        interventions={node: lambda x: np.full(x.shape[0], node_mean)},
                        num_samples_to_draw=100
                    )['Production_Impact']
                    
                    effects.append(baseline - float(pred.mean()))
                except Exception as e:
                    continue
            
            if len(effects) > 0:
                mean_effect = float(np.nanmean(effects))
                std_effect = float(np.nanstd(effects)) if len(effects) > 1 else 0.0
                
                if std_effect > 0:
                    try:
                        z_score = mean_effect / (std_effect / np.sqrt(len(effects)))
                        p_value = 2 * (1 - norm.cdf(abs(z_score)))
                    except:
                        p_value = 1.0
                else:
                    p_value = 1.0
                
                significance[node] = (mean_effect, float(p_value))
    
    return significance
    
    
    ****************************
        ********************************
    def analyze_impact_paths(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Comprehensive analysis of impact pathways to target node."""
    gcm.auto.assign_causal_mechanisms(self.scm, data)
    gcm.fit(self.scm, data)
    
    impact_metrics = {}
    num_samples = len(data)
    
    # 1. Direct causal influence (total contribution)
    impact_metrics['direct_influence'] = {
        node: float(score) 
        for node, score in gcm.intrinsic_causal_influence(
            self.scm, 
            target_node='Production_Impact',
            prediction_model='approx'
        ).items()
    }
    
    # 2. Path-specific effects (direct paths)
    impact_metrics['path_effects'] = {}
    for node in self.common_causal_graph.nodes():
        if node != 'Production_Impact' and node in data.columns:
            # Skip if node has null values
            if data[node].isnull().any():
                continue
                
            # Create intervention functions with proper dimension handling
            node_values = data[node].values
            ref_value = float(data[node].mean())
            
            def alt_intervention(_): 
                return node_values.copy()
                
            def ref_intervention(_): 
                return np.full(num_samples, ref_value)
            
            try:
                effect = gcm.average_causal_effect(
                    causal_model=self.scm,
                    target_node='Production_Impact',
                    interventions_alternative={node: alt_intervention},
                    interventions_reference={node: ref_intervention},
                    num_samples_to_draw=num_samples
                )
                if effect is not None and not np.isnan(effect):
                    impact_metrics['path_effects'][node] = float(effect)
            except Exception as e:
                print(f"Error calculating effect for {node}: {str(e)}")
                continue
    
    # 3. Counterfactual impact (what-if analysis)
    impact_metrics['counterfactual'] = {}
    normal_samples = data.mean().to_frame().T
    anomaly_samples = data.iloc[:1]
    
    for node in self.common_causal_graph.nodes():
        if node != 'Production_Impact' and node in data.columns:
            try:
                effect = gcm.counterfactual.distribute_causal_effect(
                    self.scm,
                    'Production_Impact',
                    node,
                    normal_samples,
                    anomaly_samples
                )
                if effect is not None:
                    impact_metrics['counterfactual'][node] = float(effect)
            except Exception as e:
                print(f"Error calculating counterfactual for {node}: {str(e)}")
                continue
    
    # 4. Statistical significance testing
    impact_metrics['significance'] = self._calculate_impact_significance(data)
    
    return impact_metrics

def _calculate_impact_significance(self, data: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """Calculates statistical significance of each factor's impact."""
    significance = {}
    num_iterations = 100
    baseline = float(data['Production_Impact'].mean())
    
    for node in self.common_causal_graph.nodes():
        if node != 'Production_Impact' and node in data.columns:
            # Skip if node has null values
            if data[node].isnull().any():
                continue
                
            effects = []
            node_mean = float(data[node].mean())
            
            for _ in range(num_iterations):
                try:
                    # Predict outcome under intervention
                    pred = gcm.interventional_samples(
                        self.scm,
                        interventions={node: lambda x: np.full(x.shape[0], node_mean)},
                        num_samples_to_draw=100
                    )['Production_Impact']
                    
                    effects.append(baseline - float(pred.mean()))
                except Exception as e:
                    print(f"Error in significance calculation for {node}: {str(e)}")
                    continue
            
            if len(effects) > 0:
                mean_effect = float(np.mean(effects))
                std_effect = float(np.std(effects)) if len(effects) > 1 else 0.0
                
                if std_effect > 0:
                    z_score = mean_effect / (std_effect / np.sqrt(len(effects)))
                    p_value = 2 * (1 - norm.cdf(abs(z_score)))
                else:
                    p_value = 0.0
                
                significance[node] = (mean_effect, float(p_value))
    
    return significance


************************

def analyze_impact_paths(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Comprehensive analysis of impact pathways to target node."""
    gcm.auto.assign_causal_mechanisms(self.scm, data)
    gcm.fit(self.scm, data)
    
    impact_metrics = {}
    
    # 1. Direct causal influence (total contribution)
    impact_metrics['direct_influence'] = gcm.intrinsic_causal_influence(
        self.scm, 
        target_node='Production_Impact',
        prediction_model='approx'
    )
    
    # 2. Path-specific effects (direct paths)
    impact_metrics['path_effects'] = {}
    for node in self.common_causal_graph.nodes():
        if node != 'Production_Impact':
            # Create intervention distributions
            def alt_intervention(_): return data[node].values
            def ref_intervention(_): return np.full(len(data), data[node].mean())
            
            effect = gcm.average_causal_effect(
                causal_model=self.scm,
                target_node='Production_Impact',
                interventions_alternative={node: alt_intervention},
                interventions_reference={node: ref_intervention},
                observed_data=data,
                num_samples_to_draw=100
            )
            if not np.isnan(effect):
                impact_metrics['path_effects'][node] = float(effect)
    
    # 3. Counterfactual impact (what-if analysis)
    impact_metrics['counterfactual'] = {}
    normal_samples = data.mean().to_frame().T
    anomaly_samples = data.iloc[:1]
    
    for node in self.common_causal_graph.nodes():
        if node != 'Production_Impact':
            effect = gcm.counterfactual.distribute_causal_effect(
                self.scm,
                'Production_Impact',
                node,
                normal_samples,
                anomaly_samples
            )
            if effect is not None:
                impact_metrics['counterfactual'][node] = float(effect)
    
    # 4. Statistical significance testing
    impact_metrics['significance'] = self._calculate_impact_significance(data)
    
    return impact_metrics

++++++++++++++++++++++

class ManufacturingRCAAnalyzer:
    """Enhanced RCA analyzer with comprehensive impact analysis on target node."""
    
    def __init__(self, azure_openai_client):
        self.common_causal_graph = self._build_common_causal_graph()
        self.scm = gcm.StructuralCausalModel(self.common_causal_graph)
        self.client = azure_openai_client
        self.potential_root_causes = set()

    def analyze_impact_paths(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Comprehensive analysis of impact pathways to target node."""
        gcm.auto.assign_causal_mechanisms(self.scm, data)
        gcm.fit(self.scm, data)
        
        impact_metrics = {}
        
        # 1. Direct causal influence (total contribution)
        impact_metrics['direct_influence'] = gcm.intrinsic_causal_influence(
            self.scm, 
            target_node='Production_Impact',
            prediction_model='approx'
        )
        
        # 2. Path-specific effects (direct paths)
        impact_metrics['path_effects'] = {}
        for node in self.common_causal_graph.nodes():
            if node != 'Production_Impact':
                # Create intervention distributions
                intervention_data = data[node].values
                reference_data = np.full_like(intervention_data, data[node].mean())
                
                effect = gcm.average_causal_effect(
                    self.scm,
                    node,
                    'Production_Impact',
                    intervention_data=intervention_data,
                    interventions_reference=reference_data
                )
                if not np.isnan(effect):
                    impact_metrics['path_effects'][node] = effect
        
        # 3. Counterfactual impact (what-if analysis)
        impact_metrics['counterfactual'] = {}
        normal_samples = data.mean().to_frame().T
        anomaly_samples = data.iloc[:1]
        
        for node in self.common_causal_graph.nodes():
            if node != 'Production_Impact':
                effect = gcm.counterfactual.distribute_causal_effect(
                    self.scm,
                    'Production_Impact',
                    node,
                    normal_samples,
                    anomaly_samples
                )
                if effect is not None:
                    impact_metrics['counterfactual'][node] = float(effect)
        
        # 4. Statistical significance testing
        impact_metrics['significance'] = self._calculate_impact_significance(data)
        
        return impact_metrics

    # [Rest of the class implementation remains exactly the same as before]


*****************************************
**********************************************
***********************************************

import pandas as pd
import networkx as nx
from dowhy import gcm
from typing import Dict, Any, List, Tuple
import json
import os
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from scipy.stats import norm

class ManufacturingRCAAnalyzer:
    """Enhanced RCA analyzer with comprehensive impact analysis on target node."""
    
    def __init__(self, azure_openai_client):
        self.common_causal_graph = self._build_common_causal_graph()
        self.scm = gcm.StructuralCausalModel(self.common_causal_graph)
        self.client = azure_openai_client
        self.potential_root_causes = set()

    def _build_common_causal_graph(self) -> nx.DiGraph:
        """Constructs the manufacturing process causal graph with enhanced relationships."""
        return nx.DiGraph([
            ('Document_Version_Control', 'Work_Instruction_Accuracy'),
            ('Document_Version_Control', 'Visual_Aid_Accuracy'),
            ('BOM_Accuracy', 'Work_Instruction_Accuracy'),
            ('Setup_Sheet_Accuracy', 'Work_Instruction_Accuracy'),
            ('Visual_Aid_Accuracy', 'Work_Instruction_Accuracy'),
            ('Operator_Training', 'Correct_Part_Usage'),
            ('Operator_Training', 'Part_Verification_Process'),
            ('Work_Instruction_Accuracy', 'Correct_Part_Usage'),
            ('Part_Verification_Process', 'Correct_Part_Usage'),
            ('Line_Stoppage_Protocol', 'Production_Impact'),
            ('Correct_Part_Usage', 'Production_Impact'),
            ('Work_Instruction_Accuracy', 'Production_Impact'),
            ('Equipment_Condition', 'Production_Impact'),
            ('Material_Quality', 'Production_Impact')
        ])

    def _generate_realistic_synthetic_data(self, case_details: Dict[str, Any], num_samples=500) -> pd.DataFrame:
        """Generates synthetic data with realistic distributions and relationships."""
        base_values = {
            'Document_Version_Control': case_details.get('document_version_issue', 0),
            'BOM_Accuracy': case_details.get('bom_accurate', 1),
            'Setup_Sheet_Accuracy': case_details.get('setup_sheet_accurate', 1),
            'Visual_Aid_Accuracy': case_details.get('visual_aid_accurate', 0),
            'Operator_Training': case_details.get('operator_trained', 1),
            'Part_Verification_Process': case_details.get('part_verification_done', 0),
            'Line_Stoppage_Protocol': case_details.get('line_stopped_correctly', 1),
            'Equipment_Condition': case_details.get('equipment_condition', 0.8),
            'Material_Quality': case_details.get('material_quality', 0.9),
            'Correct_Part_Usage': case_details.get('correct_part_used', 0),
            'Work_Instruction_Accuracy': 1,
            'Production_Impact': case_details.get('production_impact', 1)
        }
        
        rng = np.random.default_rng(42)
        data = pd.DataFrame()
        
        # Generate base variables with appropriate distributions
        for col in ['Document_Version_Control', 'BOM_Accuracy', 'Setup_Sheet_Accuracy',
                   'Visual_Aid_Accuracy', 'Operator_Training', 'Part_Verification_Process',
                   'Line_Stoppage_Protocol', 'Equipment_Condition', 'Material_Quality']:
            if col in ['Document_Version_Control', 'Visual_Aid_Accuracy', 'Operator_Training',
                      'Part_Verification_Process', 'Line_Stoppage_Protocol']:
                data[col] = rng.binomial(1, base_values[col], num_samples)
            else:
                data[col] = np.clip(rng.normal(base_values[col], 0.1, num_samples), 0, 1)
        
        # Generate intermediate nodes with causal relationships
        data['Work_Instruction_Accuracy'] = np.clip(
            0.3 * data['Document_Version_Control'] +
            0.3 * data['BOM_Accuracy'] +
            0.3 * data['Setup_Sheet_Accuracy'] +
            0.1 * data['Visual_Aid_Accuracy'] +
            rng.normal(0, 0.05, num_samples),
            0, 1
        )
        
        data['Correct_Part_Usage'] = np.clip(
            0.4 * data['Operator_Training'] +
            0.3 * data['Work_Instruction_Accuracy'] +
            0.3 * data['Part_Verification_Process'] +
            rng.normal(0, 0.1, num_samples),
            0, 1
        ).round()
        
        # Generate target node with enhanced impact relationships
        data['Production_Impact'] = np.clip(
            0.5 * (1 - data['Correct_Part_Usage']) +
            0.2 * (1 - data['Work_Instruction_Accuracy']) +
            0.2 * (1 - data['Equipment_Condition']) +
            0.1 * (1 - data['Material_Quality']) +
            rng.normal(0, 0.05, num_samples),
            0, 1
        )
        
        return data

    def analyze_impact_paths(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Comprehensive analysis of impact pathways to target node."""
        gcm.auto.assign_causal_mechanisms(self.scm, data)
        gcm.fit(self.scm, data)
        
        impact_metrics = {}
        
        # 1. Direct causal influence (total contribution)
        impact_metrics['direct_influence'] = gcm.intrinsic_causal_influence(
            self.scm, 
            target_node='Production_Impact',
            prediction_model='approx'
        )
        
        # 2. Path-specific effects (direct paths)
        impact_metrics['path_effects'] = {}
        for node in self.common_causal_graph.nodes():
            if node != 'Production_Impact':
                effect = gcm.average_causal_effect(
                    self.scm,
                    node,
                    'Production_Impact'
                )
                if not np.isnan(effect):
                    impact_metrics['path_effects'][node] = effect
        
        # 3. Counterfactual impact (what-if analysis)
        impact_metrics['counterfactual'] = {}
        normal_samples = data.mean().to_frame().T
        anomaly_samples = data.iloc[:1]
        
        for node in self.common_causal_graph.nodes():
            if node != 'Production_Impact':
                effect = gcm.counterfactual.distribute_causal_effect(
                    self.scm,
                    'Production_Impact',
                    node,
                    normal_samples,
                    anomaly_samples
                )
                if effect is not None:
                    impact_metrics['counterfactual'][node] = float(effect)
        
        # 4. Statistical significance testing
        impact_metrics['significance'] = self._calculate_impact_significance(data)
        
        return impact_metrics

    def _calculate_impact_significance(self, data: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """Calculates statistical significance of each factor's impact."""
        significance = {}
        num_iterations = 100
        baseline = data['Production_Impact'].mean()
        
        for node in self.common_causal_graph.nodes():
            if node != 'Production_Impact':
                effects = []
                for _ in range(num_iterations):
                    # Create intervention (set to mean value)
                    intervened_data = data.copy()
                    intervened_data[node] = data[node].mean()
                    
                    # Predict outcome under intervention
                    pred = gcm.interventional_samples(
                        self.scm,
                        interventions={node: lambda x: np.full(x.shape[0], data[node].mean())},
                        num_samples_to_draw=100
                    )['Production_Impact']
                    
                    effects.append(baseline - pred.mean())
                
                # Calculate p-value
                mean_effect = np.mean(effects)
                std_effect = np.std(effects)
                if std_effect > 0:
                    z_score = mean_effect / (std_effect / np.sqrt(num_iterations))
                    p_value = 2 * (1 - norm.cdf(abs(z_score)))
                else:
                    p_value = 0.0
                
                significance[node] = (mean_effect, p_value)
        
        return significance

    def visualize_impact_breakdown(self, impact_metrics: Dict) -> str:
        """Creates detailed impact breakdown visualization."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        # Direct influence plot
        direct = impact_metrics['direct_influence']
        ax1.barh(list(direct.keys()), list(direct.values()), color='skyblue')
        ax1.set_title('Total Causal Influence', fontsize=12)
        ax1.set_xlabel('Impact Score', fontsize=10)
        
        # Path effects plot
        path_df = pd.DataFrame(impact_metrics['path_effects'].items(), 
                             columns=['Factor', 'Effect'])
        path_df.plot.barh(x='Factor', y='Effect', ax=ax2, color='lightgreen')
        ax2.set_title('Average Causal Effects (Direct Paths)', fontsize=12)
        ax2.set_xlabel('Effect Size', fontsize=10)
        
        # Counterfactual plot
        counter_df = pd.DataFrame(impact_metrics['counterfactual'].items(),
                                columns=['Factor', 'Effect'])
        counter_df.plot.barh(x='Factor', y='Effect', ax=ax3, color='salmon')
        ax3.set_title('Counterfactual Impact (What-If)', fontsize=12)
        ax3.set_xlabel('Change in Production Impact', fontsize=10)
        
        plt.tight_layout()
        path = f"impact_breakdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path

    def visualize_causal_graph(self, save_path=None) -> str:
        """Visualizes the causal graph with enhanced formatting."""
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(self.common_causal_graph, seed=42, k=0.5)
        
        # Draw nodes with different styles for root nodes
        root_nodes = [node for node in self.common_causal_graph.nodes() 
                     if self.common_causal_graph.in_degree(node) == 0]
        non_root_nodes = [node for node in self.common_causal_graph.nodes() 
                         if node not in root_nodes]
        
        nx.draw_networkx_nodes(self.common_causal_graph, pos, nodelist=root_nodes,
                              node_size=2500, node_color='lightblue', node_shape='s')
        nx.draw_networkx_nodes(self.common_causal_graph, pos, nodelist=non_root_nodes,
                              node_size=2000, node_color='lightgreen', node_shape='o')
        
        # Draw edges with weights
        nx.draw_networkx_edges(self.common_causal_graph, pos, 
                              arrowstyle='->', arrowsize=20, width=1.5,
                              edge_color='gray')
        
        # Draw labels
        nx.draw_networkx_labels(self.common_causal_graph, pos, 
                               font_size=10, font_weight='bold')
        
        plt.title("Manufacturing Process Causal Graph", fontsize=14)
        plt.axis('off')
        
        if save_path is None:
            save_path = f"causal_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

    def _generate_wrapped_paragraph(self, text: str, style) -> List[Paragraph]:
        """Generates wrapped text paragraphs for PDF reports."""
        lines = text.split('\n')
        paragraphs = []
        for line in lines:
            if line.strip():
                paragraphs.append(Paragraph(line.strip().replace('\n', '<br/>'), style))
                paragraphs.append(Spacer(1, 6))
        return paragraphs

    def export_synthetic_data(self, data: pd.DataFrame) -> str:
        """Exports synthetic data to Excel with comprehensive statistics."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = f"synthetic_data_{timestamp}.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            data.to_excel(writer, sheet_name='Synthetic_Data', index=False)
            
            # Add descriptive statistics
            stats = data.describe().transpose()
            stats.to_excel(writer, sheet_name='Statistics')
            
            # Add correlation matrix
            corr = data.corr()
            corr.to_excel(writer, sheet_name='Correlations')
            
            # Formatting
            workbook = writer.book
            worksheet = writer.sheets['Synthetic_Data']
            header_format = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC'})
            
            for col_num, value in enumerate(data.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Add conditional formatting for better visualization
            for sheet in writer.sheets.values():
                if sheet.name == 'Correlations':
                    sheet.conditional_format(
                        f'B2:{chr(65 + len(data.columns))}{len(data.columns) + 1}',
                        {'type': '3_color_scale'}
                    )
        
        return excel_path

    def _generate_impact_analysis_text(self, impact_metrics: Dict) -> str:
        """Generates detailed human-readable impact analysis."""
        text = "## Comprehensive Impact Analysis on Production Impact\n\n"
        
        # Direct influence
        text += "### Total Causal Influence\n"
        text += "Measures each factor's total contribution through all pathways:\n"
        for factor, score in sorted(impact_metrics['direct_influence'].items(), 
                                  key=lambda x: abs(x[1]), reverse=True):
            text += f"- {factor.replace('_', ' ').title()}: {score:.3f}\n"
        
        # Path effects
        text += "\n### Direct Path Effects\n"
        text += "Measures average effect through direct causal paths:\n"
        for factor, effect in sorted(impact_metrics['path_effects'].items(),
                                    key=lambda x: abs(x[1]), reverse=True):
            text += f"- {factor.replace('_', ' ').title()}: {effect:.3f}\n"
        
        # Counterfactual
        text += "\n### Counterfactual Impact (What-If Analysis)\n"
        text += "Shows expected change if factor was at normal levels:\n"
        for factor, effect in sorted(impact_metrics['counterfactual'].items(),
                                    key=lambda x: abs(x[1]), reverse=True):
            text += f"- {factor.replace('_', ' ')}: {effect:.3f} change\n"
        
        # Significance
        text += "\n### Statistical Significance (p-values)\n"
        text += "Probability that observed impact is not by chance:\n"
        for factor, (effect, p_value) in sorted(impact_metrics['significance'].items(),
                                              key=lambda x: x[1][1]):
            text += (f"- {factor.replace('_', ' ').title()}: "
                   f"effect={effect:.3f}, p={p_value:.4f} "
                   f"{'*' if p_value < 0.05 else ''}\n")
        text += "\n* indicates statistically significant (p < 0.05)\n"
        
        return text

    def generate_pdf_report(self, results: Dict[str, Any], data_path: str) -> str:
        """Generates professional PDF report with all analysis components."""
        pdf_path = f"RCA_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        doc = SimpleDocTemplate(
            pdf_path, 
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )
        
        styles = getSampleStyleSheet()
        wrapped_style = ParagraphStyle(
            'Wrapped',
            parent=styles['Normal'],
            fontSize=10,
            leading=12,
            spaceAfter=6
        )
        
        # Generate visualizations
        causal_graph_path = self.visualize_causal_graph()
        impact_path = self.visualize_impact_breakdown(results['impact_metrics'])
        
        story = []
        
        # Title
        story.append(Paragraph("Comprehensive Manufacturing Root Cause Analysis", styles['Title']))
        story.append(Spacer(1, 12))
        
        # Case Details
        story.append(Paragraph("Case Details", styles['Heading2']))
        case_data = []
        for key, value in results['case_details'].items():
            if key == 'root_cause_hypothesis':
                continue
            display_key = key.replace('_', ' ').title()
            display_value = "Yes" if isinstance(value, bool) and value else \
                          "No" if isinstance(value, bool) else str(value)
            case_data.append([display_key, display_value])
        
        case_table = Table(case_data, colWidths=[2*inch, 4*inch])
        case_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(case_table)
        story.append(Spacer(1, 12))
        
        # Root Cause Hypothesis
        hypothesis = results['case_details'].get('root_cause_hypothesis', 'Not specified')
        story.append(Paragraph("<b>Root Cause Hypothesis:</b>", styles['Heading2']))
        story.extend(self._generate_wrapped_paragraph(hypothesis, wrapped_style))
        story.append(Spacer(1, 12))
        
        # Causal Graph
        story.append(Paragraph("Causal Relationships", styles['Heading2']))
        story.append(Image(causal_graph_path, width=6*inch, height=4.5*inch))
        story.append(Spacer(1, 12))
        
        # Impact Analysis
        story.append(Paragraph("Impact Analysis on Production Impact", styles['Heading2']))
        story.append(Image(impact_path, width=6*inch, height=4.5*inch))
        story.append(Spacer(1, 12))
        
        # Detailed Impact Metrics
        impact_text = self._generate_impact_analysis_text(results['impact_metrics'])
        story.extend(self._generate_wrapped_paragraph(impact_text, wrapped_style))
        story.append(Spacer(1, 12))
        
        # Recommendations
        story.append(Paragraph("Recommended Actions", styles['Heading2']))
        story.extend(self._generate_wrapped_paragraph(results['recommendations'], wrapped_style))
        story.append(Spacer(1, 12))
        
        # Data Reference
        story.append(Paragraph("Data Reference", styles['Heading2']))
        story.append(Paragraph(f"Synthetic data exported to: {data_path}", styles['Normal']))
        
        doc.build(story)
        return pdf_path

    def _call_azure_openai(self, prompt: str, require_json: bool = False) -> str:
        """Executes Azure OpenAI API call with proper formatting."""
        messages = [{"role": "user", "content": prompt}]
        
        if require_json:
            messages[0]["content"] = f"Return the response as a valid JSON object.\n{prompt}"
            response = self.client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                messages=messages,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
        else:
            response = self.client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                messages=messages,
                temperature=0.3
            )
            
        return response.choices[0].message.content

    def _extract_case_details(self, case_text: str) -> Dict[str, Any]:
        """Extracts structured case details from incident description using AI."""
        prompt = f"""Analyze this manufacturing CAPA case and return a JSON object with these exact fields:
{{
    "document_version_issue": bool,
    "bom_accurate": bool,
    "setup_sheet_accurate": bool,
    "visual_aid_accurate": bool,
    "operator_trained": bool,
    "part_verification_done": bool,
    "line_stopped_correctly": bool,
    "correct_part_used": bool,
    "production_impact": int,
    "root_cause_hypothesis": str
}}

Case details:
{case_text}"""
        
        response = self._call_azure_openai(prompt, require_json=True)
        return json.loads(response)

    def _handle_potential_root_causes(self, case_details: Dict[str, Any]) -> None:
        """Identifies and tracks potential root causes not in main graph."""
        hypothesis = case_details.get('root_cause_hypothesis', '').lower()
        existing_nodes = set(self.common_causal_graph.nodes())
        
        potential_causes = {
            'calibration': 'Equipment_Calibration',
            'supplier': 'Supplier_Quality',
            'environment': 'Environmental_Factors'
        }
        
        for term, cause in potential_causes.items():
            if term in hypothesis and cause not in existing_nodes:
                self.potential_root_causes.add(cause)

    def _generate_recommendations(self, case_details: Dict[str, Any], impact_metrics: Dict) -> str:
        """Generates AI-powered recommendations based on impact analysis."""
        prompt = f"""Generate comprehensive CAPA recommendations based on this impact analysis.
Include specific actions addressing the highest impact factors.

Impact Analysis Summary:
{json.dumps({k: v for k, v in impact_metrics.items() if k != 'significance'}, indent=2)}

Case Details:
{json.dumps(case_details, indent=2)}

Structure your response with these sections:

1. Immediate Containment Actions:
- [Action 1]
- [Action 2]

2. Root Cause Correction:
- [Primary correction addressing top factor]
- [Secondary corrections]

3. Preventive Measures:
- [Process improvement 1]
- [Training enhancement]
- [System controls]

4. Verification Plan:
- [Validation method 1]
- [Testing approach]

5. Effectiveness Monitoring:
- [Metric 1 to track]
- [Review schedule]"""
        
        return self._call_azure_openai(prompt, require_json=False)

    def analyze_case(self, case_text: str, num_samples=500) -> Dict[str, Any]:
        """Complete enhanced analysis workflow with comprehensive impact assessment."""
        case_details = self._extract_case_details(case_text)
        self._handle_potential_root_causes(case_details)
        
        data = self._generate_realistic_synthetic_data(case_details, num_samples)
        
        # Comprehensive impact analysis
        impact_metrics = self.analyze_impact_paths(data)
        
        # Anomaly attribution
        attributions = gcm.attribute_anomalies(
            self.scm, 
            target_node='Production_Impact',
            anomaly_samples=data.iloc[:1]
        )
        
        return {
            'case_details': case_details,
            'impact_metrics': impact_metrics,
            'causal_attributions': {
                k: float(v[0]) if isinstance(v, (np.ndarray, list)) else float(v)
                for k, v in attributions.items()
            },
            'recommendations': self._generate_recommendations(case_details, impact_metrics)
        }

def initialize_azure_client() -> AzureOpenAI:
    """Initializes and returns authenticated Azure OpenAI client."""
    load_dotenv()
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

def format_plaintext_output(results: Dict[str, Any]) -> str:
    """Formats results to comprehensive plain text output."""
    output = []
    
    output.append("=== CASE DETAILS ===")
    for key, value in results['case_details'].items():
        display_key = key.replace('_', ' ').title()
        if isinstance(value, bool):
            output.append(f"{display_key}: {'Yes' if value else 'No'}")
        else:
            output.append(f"{display_key}: {value}")
    
    output.append("\n=== COMPREHENSIVE IMPACT ANALYSIS ===")
    output.append(results['impact_analysis_text'])
    
    output.append("\n=== CAUSAL ATTRIBUTIONS ===")
    for factor, score in results['causal_attributions'].items():
        output.append(f"- {factor.replace('_', ' ').title()}: {score:.4f}")
    
    output.append("\n=== RECOMMENDATIONS ===")
    output.append(results['recommendations'])
    
    return "\n".join(output)

if __name__ == "__main__":
    """Main execution flow with comprehensive analysis and reporting."""
    try:
        client = initialize_azure_client()
        analyzer = ManufacturingRCAAnalyzer(client)
        
        # Example case - can be replaced with user input
        case_text = """On Jan 12, 2021, the MESE1 Manufacturing Line was stopped due to discrepancy between Visual Aids and Setup Sheet, resulting in incorrect screw P/N used at Build Station 4. Work instruction OPER-WI-086 Rev C specifies P/N 5600203-01, but P/N 5600008-03 was used. The BOM and Setup Sheet are correct, but the Visual Aid rev 003 shows the wrong P/N."""
        
        # Perform complete analysis
        results = analyzer.analyze_case(case_text)
        
        # Add impact analysis text to results
        results['impact_analysis_text'] = analyzer._generate_impact_analysis_text(results['impact_metrics'])
        
        # Generate outputs
        data = analyzer._generate_realistic_synthetic_data(results['case_details'])
        data_path = analyzer.export_synthetic_data(data)
        pdf_path = analyzer.generate_pdf_report(results, data_path)
        
        # Print results
        print(format_plaintext_output(results))
        print(f"\nReport generated: {pdf_path}")
        print(f"Data exported to: {data_path}")

    except Exception as e:
        print(f"Error: {str(e)}")


+++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++


import pandas as pd
import networkx as nx
from dowhy import gcm
from typing import Dict, Any, List
import json
import os
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

class ManufacturingRCAAnalyzer:
    """Enhanced RCA analyzer with consistent outputs, impact visualization, and automatic root cause handling."""
    
    def __init__(self, azure_openai_client):
        self.common_causal_graph = self._build_common_causal_graph()
        self.scm = gcm.StructuralCausalModel(self.common_causal_graph)
        self.client = azure_openai_client
        self.potential_root_causes = set()

    def _build_common_causal_graph(self) -> nx.DiGraph:
        """Constructs the manufacturing process causal graph with enhanced relationships."""
        return nx.DiGraph([
            ('Document_Version_Control', 'Work_Instruction_Accuracy'),
            ('Document_Version_Control', 'Visual_Aid_Accuracy'),
            ('BOM_Accuracy', 'Work_Instruction_Accuracy'),
            ('Setup_Sheet_Accuracy', 'Work_Instruction_Accuracy'),
            ('Visual_Aid_Accuracy', 'Work_Instruction_Accuracy'),
            ('Operator_Training', 'Correct_Part_Usage'),
            ('Operator_Training', 'Part_Verification_Process'),
            ('Work_Instruction_Accuracy', 'Correct_Part_Usage'),
            ('Part_Verification_Process', 'Correct_Part_Usage'),
            ('Line_Stoppage_Protocol', 'Production_Impact'),
            ('Correct_Part_Usage', 'Production_Impact'),
            ('Work_Instruction_Accuracy', 'Production_Impact'),
            ('Equipment_Condition', 'Production_Impact'),
            ('Material_Quality', 'Production_Impact')
        ])

    def _generate_realistic_synthetic_data(self, case_details: Dict[str, Any], num_samples=500) -> pd.DataFrame:
        """Generates synthetic data with all required variables."""
        base_values = {
            'Document_Version_Control': case_details.get('document_version_issue', 0),
            'BOM_Accuracy': case_details.get('bom_accurate', 1),
            'Setup_Sheet_Accuracy': case_details.get('setup_sheet_accurate', 1),
            'Visual_Aid_Accuracy': case_details.get('visual_aid_accurate', 0),
            'Operator_Training': case_details.get('operator_trained', 1),
            'Part_Verification_Process': case_details.get('part_verification_done', 0),
            'Line_Stoppage_Protocol': case_details.get('line_stopped_correctly', 1),
            'Equipment_Condition': case_details.get('equipment_condition', 0.8),
            'Material_Quality': case_details.get('material_quality', 0.9),
            'Correct_Part_Usage': case_details.get('correct_part_used', 0),
            'Work_Instruction_Accuracy': 1,
            'Production_Impact': case_details.get('production_impact', 1)
        }
        
        rng = np.random.default_rng(42)
        data = pd.DataFrame()
        
        for col in ['Document_Version_Control', 'BOM_Accuracy', 'Setup_Sheet_Accuracy',
                   'Visual_Aid_Accuracy', 'Operator_Training', 'Part_Verification_Process',
                   'Line_Stoppage_Protocol', 'Equipment_Condition', 'Material_Quality']:
            if col in ['Document_Version_Control', 'Visual_Aid_Accuracy', 'Operator_Training',
                      'Part_Verification_Process', 'Line_Stoppage_Protocol']:
                data[col] = rng.binomial(1, base_values[col], num_samples)
            else:
                data[col] = np.clip(rng.normal(base_values[col], 0.1, num_samples), 0, 1)
        
        data['Work_Instruction_Accuracy'] = np.clip(
            0.3 * data['Document_Version_Control'] +
            0.3 * data['BOM_Accuracy'] +
            0.3 * data['Setup_Sheet_Accuracy'] +
            0.1 * data['Visual_Aid_Accuracy'] +
            rng.normal(0, 0.05, num_samples),
            0, 1
        )
        
        data['Correct_Part_Usage'] = np.clip(
            0.4 * data['Operator_Training'] +
            0.3 * data['Work_Instruction_Accuracy'] +
            0.3 * data['Part_Verification_Process'] +
            rng.normal(0, 0.1, num_samples),
            0, 1
        ).round()
        
        data['Production_Impact'] = np.clip(
            0.5 * (1 - data['Correct_Part_Usage']) +
            0.2 * (1 - data['Work_Instruction_Accuracy']) +
            0.2 * (1 - data['Equipment_Condition']) +
            0.1 * (1 - data['Material_Quality']) +
            rng.normal(0, 0.05, num_samples),
            0, 1
        )
        
        return data

    def analyze_causal_influence(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculates and returns intrinsic causal influence of each factor."""
        gcm.auto.assign_causal_mechanisms(self.scm, data)
        gcm.fit(self.scm, data)
        
        influence = gcm.intrinsic_causal_influence(
            self.scm, 
            target_node='Production_Impact',
            prediction_model='approx'
        )
        
        total = sum(abs(v) for v in influence.values())
        normalized_influence = {k: v/total for k, v in influence.items()}
        
        return dict(sorted(normalized_influence.items(), key=lambda item: abs(item[1]), reverse=True))

    def visualize_causal_graph(self, save_path=None):
        """Visualizes the causal graph using matplotlib and returns image path."""
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.common_causal_graph, seed=42)
        
        nx.draw_networkx_nodes(self.common_causal_graph, pos, node_size=2000, node_color='lightblue')
        nx.draw_networkx_edges(self.common_causal_graph, pos, arrowstyle='->', arrowsize=20, width=2)
        nx.draw_networkx_labels(self.common_causal_graph, pos, font_size=10, font_weight='bold')
        
        plt.title("Manufacturing Process Causal Graph", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"causal_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

    def visualize_impact_graph(self, influence: Dict[str, float]) -> str:
        """Generates impact-weighted digraph visualization."""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.common_causal_graph, seed=42)
        
        node_sizes = [5000 * (influence.get(node, 0.1)) for node in self.common_causal_graph.nodes()]
        
        nx.draw_networkx_nodes(
            self.common_causal_graph, pos, 
            node_size=node_sizes, 
            node_color='lightblue',
            alpha=0.8
        )
        
        edge_weights = [
            3 * influence.get(u, 0) * influence.get(v, 0)
            for u, v in self.common_causal_graph.edges()
        ]
        
        nx.draw_networkx_edges(
            self.common_causal_graph, pos,
            width=edge_weights,
            edge_color='gray',
            arrowstyle='->',
            arrowsize=20
        )
        
        labels = {
            node: f"{node}\n({influence.get(node, 0):.2f})" 
            for node in self.common_causal_graph.nodes()
        }
        
        nx.draw_networkx_labels(
            self.common_causal_graph, pos, 
            labels=labels,
            font_size=10
        )
        
        plt.title("Impact-Weighted Causal Graph", fontsize=14)
        plt.axis('off')
        
        impact_path = f"impact_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(impact_path, dpi=300, bbox_inches='tight')
        plt.close()
        return impact_path

    def _generate_wrapped_paragraph(self, text: str, style) -> List[Paragraph]:
        """Generates wrapped text paragraphs for PDF reports."""
        lines = text.split('\n')
        paragraphs = []
        for line in lines:
            if line.strip():
                paragraphs.append(Paragraph(line.strip().replace('\n', '<br/>'), style))
                paragraphs.append(Spacer(1, 6))
        return paragraphs

    def export_synthetic_data(self, data: pd.DataFrame):
        """Exports synthetic data to Excel file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = f"synthetic_data_{timestamp}.xlsx"
        
        writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')
        data.to_excel(writer, sheet_name='Synthetic_Data', index=False)
        
        stats = data.describe().transpose()
        stats.to_excel(writer, sheet_name='Statistics')
        
        workbook = writer.book
        worksheet = writer.sheets['Synthetic_Data']
        header_format = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC'})
        
        for col_num, value in enumerate(data.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        writer.close()
        return excel_path

    def generate_pdf_report(self, results: Dict[str, Any], graph_path: str, data_path: str) -> str:
        """Generates professional PDF report with wrapped text and impact graph."""
        pdf_path = f"RCA_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        doc = SimpleDocTemplate(
            pdf_path, 
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )
        
        styles = getSampleStyleSheet()
        wrapped_style = ParagraphStyle(
            'Wrapped',
            parent=styles['Normal'],
            fontSize=10,
            leading=12,
            spaceAfter=6
        )
        
        impact_path = self.visualize_impact_graph(results['causal_influence'])
        
        story = []
        
        story.append(Paragraph("Manufacturing Root Cause Analysis Report", styles['Title']))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("Case Details", styles['Heading2']))
        case_data = []
        for key, value in results['case_details'].items():
            display_key = key.replace('_', ' ').title()
            display_value = "Yes" if isinstance(value, bool) and value else \
                          "No" if isinstance(value, bool) else str(value)
            case_data.append([display_key, display_value])
        
        case_table = Table(case_data, colWidths=[2*inch, 4*inch])
        case_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(case_table)
        story.append(Spacer(1, 12))
        
        hypothesis = results['case_details'].get('root_cause_hypothesis', 'Not specified')
        story.append(Paragraph("<b>Root Cause Hypothesis:</b>", styles['Heading2']))
        story.extend(self._generate_wrapped_paragraph(hypothesis, wrapped_style))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("Impact-Weighted Causal Relationships", styles['Heading2']))
        story.append(Image(impact_path, width=6*inch, height=4.5*inch))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("Recommended Actions", styles['Heading2']))
        story.extend(self._generate_wrapped_paragraph(results['recommendations'], wrapped_style))
        
        doc.build(story)
        return pdf_path

    def _call_azure_openai(self, prompt: str, require_json: bool = False) -> str:
        """Executes Azure OpenAI API call with proper formatting."""
        messages = [{"role": "user", "content": prompt}]
        
        if require_json:
            messages[0]["content"] = f"Return the response as a valid JSON object.\n{prompt}"
            response = self.client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                messages=messages,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
        else:
            response = self.client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                messages=messages,
                temperature=0.3
            )
            
        return response.choices[0].message.content

    def _extract_case_details(self, case_text: str) -> Dict[str, Any]:
        """Extracts structured case details from incident description using AI."""
        prompt = f"""Analyze this manufacturing CAPA case and return a JSON object with these exact fields:
{{
    "document_version_issue": bool,
    "bom_accurate": bool,
    "setup_sheet_accurate": bool,
    "visual_aid_accurate": bool,
    "operator_trained": bool,
    "part_verification_done": bool,
    "line_stopped_correctly": bool,
    "correct_part_used": bool,
    "production_impact": int,
    "root_cause_hypothesis": str
}}

Case details:
{case_text}"""
        
        response = self._call_azure_openai(prompt, require_json=True)
        return json.loads(response)

    def _handle_potential_root_causes(self, case_details: Dict[str, Any]) -> None:
        """Identifies and tracks potential root causes not in main graph."""
        hypothesis = case_details.get('root_cause_hypothesis', '').lower()
        existing_nodes = set(self.common_causal_graph.nodes())
        
        potential_causes = {
            'calibration': 'Equipment_Calibration',
            'supplier': 'Supplier_Quality',
            'environment': 'Environmental_Factors'
        }
        
        for term, cause in potential_causes.items():
            if term in hypothesis and cause not in existing_nodes:
                self.potential_root_causes.add(cause)

    def _generate_recommendations(self, case_details: Dict[str, Any], attributions: Dict[str, Any]) -> str:
        """Generates human-readable corrective/preventive action recommendations."""
        prompt = f"""Generate clear, actionable CAPA recommendations in plain text format with these sections:
        
Root Cause Analysis:
[Explain the likely root cause in 2-3 sentences]

Corrective Actions:
1. [Immediate action 1]
2. [Immediate action 2] 
3. [Immediate action 3]

Preventive Actions:
1. [Long-term solution 1]
2. [Long-term solution 2]

Verification Methods:
- [How to verify corrective actions] 
- [How to verify preventive actions]

Test Cases:
1. [Specific test scenario 1]
2. [Specific test scenario 2]

Case Details:
{json.dumps(case_details, indent=2)}

Key Contributing Factors:
{json.dumps(attributions, indent=2)}"""
        
        return self._call_azure_openai(prompt, require_json=False)

    def analyze_case(self, case_text: str, num_samples=500) -> Dict[str, Any]:
        """Complete analysis workflow with causal influence and root cause handling."""
        case_details = self._extract_case_details(case_text)
        self._handle_potential_root_causes(case_details)
        
        data = self._generate_realistic_synthetic_data(case_details, num_samples)
        
        gcm.auto.assign_causal_mechanisms(self.scm, data)
        gcm.fit(self.scm, data)
        
        attributions = gcm.attribute_anomalies(
            self.scm, 
            target_node='Production_Impact',
            anomaly_samples=data.iloc[:1]
        )
        
        influence = self.analyze_causal_influence(data)
        
        return {
            'case_details': case_details,
            'causal_attributions': {
                k: float(v[0]) if isinstance(v, (np.ndarray, list)) else float(v)
                for k, v in attributions.items()
            },
            'causal_influence': influence,
            'recommendations': self._generate_recommendations(case_details, influence)
        }

def initialize_azure_client() -> AzureOpenAI:
    """Initializes and returns authenticated Azure OpenAI client."""
    load_dotenv()
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

def format_plaintext_output(results: Dict[str, Any]) -> str:
    """Formats results to exactly match JSON structure in plain text"""
    output = []
    
    output.append("=== CASE DETAILS ===")
    for key, value in results['case_details'].items():
        display_key = key.replace('_', ' ').title()
        if isinstance(value, bool):
            output.append(f"{display_key}: {'Yes' if value else 'No'}")
        else:
            output.append(f"{display_key}: {value}")
    
    output.append("\n=== CAUSAL ATTRIBUTIONS ===")
    for factor, score in results['causal_attributions'].items():
        output.append(f"- {factor.replace('_', ' ').title()}: {score:.4f}")
    
    return "\n".join(output)

if __name__ == "__main__":
    """Main execution flow with PDF reporting."""
    try:
        client = initialize_azure_client()
        analyzer = ManufacturingRCAAnalyzer(client)
        
        case_text = """On Jan 12, 2021, the MESE1 Manufacturing Line was stopped due to discrepancy between Visual Aids and Setup Sheet, resulting in incorrect screw P/N used at Build Station 4. Work instruction OPER-WI-086 Rev C specifies P/N 5600203-01, but P/N 5600008-03 was used. The BOM and Setup Sheet are correct, but the Visual Aid rev 003 shows the wrong P/N."""
        
        results = analyzer.analyze_case(case_text)
        
        graph_path = analyzer.visualize_causal_graph()
        data = analyzer._generate_realistic_synthetic_data(results['case_details'])
        data_path = analyzer.export_synthetic_data(data)
        pdf_path = analyzer.generate_pdf_report(results, graph_path, data_path)
        
        print("=== CASE DETAILS ===")
        print(json.dumps(results['case_details'], indent=2))
        
        print("\n🔍 TOTAL CAUSAL INFLUENCE ON PRODUCTION IMPACT:")
        for factor, score in results['causal_influence'].items():
            print(f"{factor:<25}: {score:.4f}")
            
        print("\n=== RECOMMENDATIONS ===")
        print(results['recommendations'])
        
        print(f"\nReport generated: {pdf_path}")

    except Exception as e:
        print(f"Error: {str(e)}")


************************
*****************************************************














def format_plaintext_output(results: Dict[str, Any]) -> str:
    """Formats results to exactly match JSON structure in plain text"""
    output = []
    
    # Case Details (matches JSON keys exactly)
    output.append("=== CASE DETAILS ===")
    for key, value in results['case_details'].items():
        display_key = key.replace('_', ' ').title()
        if isinstance(value, bool):
            output.append(f"{display_key}: {'Yes' if value else 'No'}")
        else:
            output.append(f"{display_key}: {value}")
    
    # Causal Attributions (matches JSON structure)
    output.append("\n=== CAUSAL ATTRIBUTIONS ===")
    for factor, score in results['causal_attributions'].items():
        output.append(f"- {factor.replace('_', ' ').title()}: {score:.4f}")
    
    # Recommendations (directly from JSON)
    output.append("\n=== RECOMMENDATIONS ===")
    output.append(results['recommendations'])
    
    return "\n".join(output)

********************
***************************
*******************************



def _generate_realistic_synthetic_data(self, case_details: Dict[str, Any], num_samples=500) -> pd.DataFrame:
    """Generates synthetic data with all required variables."""
    base_values = {
        'Document_Version_Control': case_details.get('document_version_issue', 0),
        'BOM_Accuracy': case_details.get('bom_accurate', 1),
        'Setup_Sheet_Accuracy': case_details.get('setup_sheet_accurate', 1),
        'Visual_Aid_Accuracy': case_details.get('visual_aid_accurate', 0),
        'Operator_Training': case_details.get('operator_trained', 1),
        'Part_Verification_Process': case_details.get('part_verification_done', 0),
        'Line_Stoppage_Protocol': case_details.get('line_stopped_correctly', 1),
        'Equipment_Condition': case_details.get('equipment_condition', 0.8),
        'Material_Quality': case_details.get('material_quality', 0.9),
        'Correct_Part_Usage': case_details.get('correct_part_used', 0),  # Added this line
        'Work_Instruction_Accuracy': 1,  # Will be calculated
        'Production_Impact': case_details.get('production_impact', 1)  # Will be calculated
    }
    
    rng = np.random.default_rng(42)
    data = pd.DataFrame()
    
    # Generate core variables
    for col in ['Document_Version_Control', 'BOM_Accuracy', 'Setup_Sheet_Accuracy',
               'Visual_Aid_Accuracy', 'Operator_Training', 'Part_Verification_Process',
               'Line_Stoppage_Protocol', 'Equipment_Condition', 'Material_Quality']:
        if col in ['Document_Version_Control', 'Visual_Aid_Accuracy', 'Operator_Training',
                  'Part_Verification_Process', 'Line_Stoppage_Protocol']:
            data[col] = rng.binomial(1, base_values[col], num_samples)
        else:
            data[col] = np.clip(rng.normal(base_values[col], 0.1, num_samples), 0, 1)
    
    # Calculate derived variables
    data['Work_Instruction_Accuracy'] = np.clip(
        0.3 * data['Document_Version_Control'] +
        0.3 * data['BOM_Accuracy'] +
        0.3 * data['Setup_Sheet_Accuracy'] +
        0.1 * data['Visual_Aid_Accuracy'] +
        rng.normal(0, 0.05, num_samples),
        0, 1
    )
    
    data['Correct_Part_Usage'] = np.clip(
        0.4 * data['Operator_Training'] +
        0.3 * data['Work_Instruction_Accuracy'] +
        0.3 * data['Part_Verification_Process'] +
        rng.normal(0, 0.1, num_samples),
        0, 1
    ).round()  # Make it binary
    
    data['Production_Impact'] = np.clip(
        0.5 * (1 - data['Correct_Part_Usage']) +
        0.2 * (1 - data['Work_Instruction_Accuracy']) +
        0.2 * (1 - data['Equipment_Condition']) +
        0.1 * (1 - data['Material_Quality']) +
        rng.normal(0, 0.05, num_samples),
        0, 1
    )
    
    return data
def analyze_case(self, case_text: str, num_samples=500) -> Dict[str, Any]:
    case_details = self._extract_case_details(case_text)
    data = self._generate_realistic_synthetic_data(case_details, num_samples)
    
    # Ensure all graph nodes are in data
    missing_nodes = set(self.common_causal_graph.nodes()) - set(data.columns)
    if missing_nodes:
        raise ValueError(f"Missing data for nodes: {missing_nodes}")
    
    # Rest of your analysis code...


**************************************
    ***************************************
        ***********************************


"""
Enhanced Manufacturing Root Cause Analysis (RCA) System with Causal Influence Analysis
"""

import pandas as pd
import networkx as nx
from dowhy import gcm
from typing import Dict, Any, List
import json
import os
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

class ManufacturingRCAAnalyzer:
    """Enhanced analyzer with causal influence analysis and improved data generation."""
    
    def __init__(self, azure_openai_client):
        self.common_causal_graph = self._build_common_causal_graph()
        self.scm = gcm.StructuralCausalModel(self.common_causal_graph)
        self.client = azure_openai_client
        
    def _build_common_causal_graph(self) -> nx.DiGraph:
        """Constructs the manufacturing process causal graph with enhanced relationships."""
        return nx.DiGraph([
            # Document control influences
            ('Document_Version_Control', 'Work_Instruction_Accuracy'),
            ('Document_Version_Control', 'Visual_Aid_Accuracy'),
            
            # Accuracy influences
            ('BOM_Accuracy', 'Work_Instruction_Accuracy'),
            ('Setup_Sheet_Accuracy', 'Work_Instruction_Accuracy'),
            ('Visual_Aid_Accuracy', 'Work_Instruction_Accuracy'),
            
            # Operational factors
            ('Operator_Training', 'Correct_Part_Usage'),
            ('Operator_Training', 'Part_Verification_Process'),
            ('Work_Instruction_Accuracy', 'Correct_Part_Usage'),
            ('Part_Verification_Process', 'Correct_Part_Usage'),
            
            # Production impacts
            ('Line_Stoppage_Protocol', 'Production_Impact'),
            ('Correct_Part_Usage', 'Production_Impact'),
            ('Work_Instruction_Accuracy', 'Production_Impact'),
            ('Equipment_Condition', 'Production_Impact'),
            ('Material_Quality', 'Production_Impact')
        ])
    
    def _generate_realistic_synthetic_data(self, case_details: Dict[str, Any], num_samples=500) -> pd.DataFrame:
        """Generates more realistic synthetic manufacturing data with correlations."""
        base_values = {
            'Document_Version_Control': case_details.get('document_version_issue', 0),
            'BOM_Accuracy': case_details.get('bom_accurate', 1),
            'Setup_Sheet_Accuracy': case_details.get('setup_sheet_accurate', 1),
            'Visual_Aid_Accuracy': case_details.get('visual_aid_accurate', 0),
            'Operator_Training': case_details.get('operator_trained', 1),
            'Part_Verification_Process': case_details.get('part_verification_done', 0),
            'Line_Stoppage_Protocol': case_details.get('line_stopped_correctly', 1),
            'Correct_Part_Usage': case_details.get('correct_part_used', 0),
            'Equipment_Condition': case_details.get('equipment_condition', 0.8),
            'Material_Quality': case_details.get('material_quality', 0.9),
            'Production_Impact': case_details.get('production_impact', 1)
        }
        
        # Generate correlated data
        data = pd.DataFrame()
        rng = np.random.default_rng(42)
        
        # Binary factors with some correlation
        data['Document_Version_Control'] = rng.binomial(1, base_values['Document_Version_Control'], num_samples)
        data['Visual_Aid_Accuracy'] = np.where(
            data['Document_Version_Control'] == 1,
            rng.binomial(1, 0.2, num_samples),  # If docs are bad, visual aids likely wrong
            rng.binomial(1, 0.9, num_samples)   # If docs good, visual aids likely correct
        )
        
        # Continuous factors with dependencies
        data['Work_Instruction_Accuracy'] = np.clip(
            0.7 * data['Document_Version_Control'] +
            0.6 * rng.normal(base_values['BOM_Accuracy'], 0.1, num_samples) +
            0.6 * rng.normal(base_values['Setup_Sheet_Accuracy'], 0.1, num_samples) +
            0.5 * data['Visual_Aid_Accuracy'] +
            rng.normal(0, 0.1, num_samples),
            0, 1
        )
        
        # More realistic production impact calculation
        data['Production_Impact'] = np.clip(
            0.4 * (1 - data['Correct_Part_Usage']) +
            0.3 * (1 - data['Work_Instruction_Accuracy']) +
            0.2 * (1 - rng.normal(base_values['Equipment_Condition'], 0.1, num_samples)) +
            0.1 * (1 - rng.normal(base_values['Material_Quality'], 0.1, num_samples)) +
            rng.normal(0, 0.05, num_samples),
            0, 1
        )
        
        return data
    
    def analyze_causal_influence(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculates and returns intrinsic causal influence of each factor."""
        # Assign causal mechanisms automatically
        gcm.auto.assign_causal_mechanisms(self.scm, data)
        
        # Fit the causal model
        gcm.fit(self.scm, data)
        
        # Calculate intrinsic causal influence
        influence = gcm.intrinsic_causal_influence(
            self.scm, 
            target_node='Production_Impact',
            prediction_model='approx'
        )
        
        # Normalize influence scores
        total = sum(abs(v) for v in influence.values())
        normalized_influence = {k: v/total for k, v in influence.items()}
        
        return dict(sorted(normalized_influence.items(), key=lambda item: abs(item[1]), reverse=True))

    # [Keep all your existing methods but update analyze_case to include influence analysis]

    def analyze_case(self, case_text: str, num_samples=500) -> Dict[str, Any]:
        """Enhanced analysis workflow with causal influence."""
        case_details = self._extract_case_details(case_text)
        data = self._generate_realistic_synthetic_data(case_details, num_samples)
        
        # Standard causal analysis
        gcm.auto.assign_causal_mechanisms(self.scm, data)
        gcm.fit(self.scm, data)
        
        attributions = gcm.attribute_anomalies(
            self.scm, 
            target_node='Production_Impact',
            anomaly_samples=data.iloc[:1]
        )
        
        # Causal influence analysis
        influence = self.analyze_causal_influence(data)
        
        return {
            'case_details': case_details,
            'causal_attributions': {
                k: float(v[0]) if isinstance(v, (np.ndarray, list)) else float(v) 
                for k, v in attributions.items()
            },
            'causal_influence': influence,
            'recommendations': self._generate_recommendations(case_details, influence)
        }

# [Rest of your existing helper functions and main execution block]

if __name__ == "__main__":
    """Enhanced main execution with causal influence reporting."""
    try:
        client = initialize_azure_client()
        analyzer = ManufacturingRCAAnalyzer(client)
        
        case_text = """Assembly line stopped due to incorrect torque specification being followed. 
        Work instruction was updated last week but operator used old printed copy. 
        Machine sensors detected abnormal vibration before failure."""
        
        # Perform analysis
        results = analyzer.analyze_case(case_text)
        
        # Generate outputs
        graph_path = analyzer.visualize_causal_graph()
        data = analyzer._generate_realistic_synthetic_data(results['case_details'])
        data_path = analyzer.export_synthetic_data(data)
        pdf_path = analyzer.generate_pdf_report(results, graph_path, data_path)
        
        # Console output
        print("=== CASE DETAILS ===")
        print(json.dumps(results['case_details'], indent=2))
        
        print("\n🔍 TOTAL CAUSAL INFLUENCE ON PRODUCTION IMPACT:")
        for factor, score in results['causal_influence'].items():
            print(f"{factor:<25}: {score:.4f}")
            
        print("\n=== RECOMMENDATIONS ===")
        print(results['recommendations'])
        
        print(f"\nReport generated: {pdf_path}")

    except Exception as e:
        print(f"Error: {str(e)}")



+++++++++++++++++++++++
+++++++++++++++++++++++++++++++++






"""
Manufacturing Root Cause Analysis (RCA) System with PDF Reporting
"""

import pandas as pd
import networkx as nx
from dowhy import gcm
from typing import Dict, Any
import json
import os
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

class ManufacturingRCAAnalyzer:
    """Main analyzer class that performs root cause analysis for manufacturing incidents."""
    
    def __init__(self, azure_openai_client):
        self.common_causal_graph = self._build_common_causal_graph()
        self.scm = gcm.StructuralCausalModel(self.common_causal_graph)
        self.client = azure_openai_client
        
    def _build_common_causal_graph(self) -> nx.DiGraph:
        """Constructs the manufacturing process causal graph."""
        return nx.DiGraph([
            ('Document_Version_Control', 'Work_Instruction_Accuracy'),
            ('BOM_Accuracy', 'Work_Instruction_Accuracy'),
            ('Setup_Sheet_Accuracy', 'Work_Instruction_Accuracy'),
            ('Visual_Aid_Accuracy', 'Work_Instruction_Accuracy'),
            ('Operator_Training', 'Correct_Part_Usage'),
            ('Work_Instruction_Accuracy', 'Correct_Part_Usage'),
            ('Part_Verification_Process', 'Correct_Part_Usage'),
            ('Line_Stoppage_Protocol', 'Production_Impact'),
            ('Correct_Part_Usage', 'Production_Impact'),
            ('Work_Instruction_Accuracy', 'Production_Impact')
        ])
    
    def _generate_synthetic_data(self, case_details: Dict[str, Any], num_samples=100) -> pd.DataFrame:
        """Generates synthetic manufacturing data based on case specifics."""
        base_values = {
            'Document_Version_Control': case_details.get('document_version_issue', 0),
            'BOM_Accuracy': case_details.get('bom_accurate', 1),
            'Setup_Sheet_Accuracy': case_details.get('setup_sheet_accurate', 1),
            'Visual_Aid_Accuracy': case_details.get('visual_aid_accurate', 0),
            'Operator_Training': case_details.get('operator_trained', 1),
            'Part_Verification_Process': case_details.get('part_verification_done', 0),
            'Line_Stoppage_Protocol': case_details.get('line_stopped_correctly', 1),
            'Correct_Part_Usage': case_details.get('correct_part_used', 0),
            'Production_Impact': case_details.get('production_impact', 1),
            'Work_Instruction_Accuracy': 1
        }
        
        data = {}
        for col, val in base_values.items():
            if col == 'Production_Impact':
                samples = np.random.normal(val, 0.1, num_samples)
                data[col] = samples.tolist()
            else:
                samples = np.random.normal(val, 0.1, num_samples)
                data[col] = np.clip(samples, 0, 1).round().astype(int).tolist()
                
        return pd.DataFrame(data)
    
    def visualize_causal_graph(self, save_path=None):
        """Visualizes the causal graph using matplotlib and returns image path."""
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.common_causal_graph, seed=42)
        
        # Draw nodes and edges
        nx.draw_networkx_nodes(self.common_causal_graph, pos, node_size=2000, node_color='lightblue')
        nx.draw_networkx_edges(self.common_causal_graph, pos, arrowstyle='->', arrowsize=20, width=2)
        nx.draw_networkx_labels(self.common_causal_graph, pos, font_size=10, font_weight='bold')
        
        plt.title("Manufacturing Process Causal Graph", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"causal_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
        
    def export_synthetic_data(self, data: pd.DataFrame):
        """Exports synthetic data to Excel file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = f"synthetic_data_{timestamp}.xlsx"
        
        writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')
        data.to_excel(writer, sheet_name='Synthetic_Data', index=False)
        
        stats = data.describe().transpose()
        stats.to_excel(writer, sheet_name='Statistics')
        
        workbook = writer.book
        worksheet = writer.sheets['Synthetic_Data']
        header_format = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC'})
        
        for col_num, value in enumerate(data.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        writer.close()
        return excel_path
    
    def generate_pdf_report(self, results: Dict[str, Any], graph_path: str, data_path: str):
        """Generates a professional PDF report of the analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = f"RCA_Report_{timestamp}.pdf"
        
        # Create document
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=16,
            alignment=1,
            spaceAfter=12
        )
        
        section_style = ParagraphStyle(
            'Section',
            parent=styles['Heading2'],
            fontSize=12,
            spaceBefore=12,
            spaceAfter=6
        )
        
        # Report content
        story = []
        
        # Title
        story.append(Paragraph("Manufacturing Root Cause Analysis Report", title_style))
        story.append(Spacer(1, 12))
        
        # Case Details
        story.append(Paragraph("Case Details", section_style))
        case_data = []
        for key, value in results['case_details'].items():
            if isinstance(value, bool):
                value = "Yes" if value else "No"
            case_data.append([key.replace('_', ' ').title(), str(value)])
        
        case_table = Table(case_data, colWidths=[2*inch, 3*inch])
        case_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(case_table)
        story.append(Spacer(1, 12))
        
        # Causal Graph
        story.append(Paragraph("Causal Relationships", section_style))
        story.append(Image(graph_path, width=6*inch, height=4.5*inch))
        story.append(Spacer(1, 12))
        
        # Key Findings
        story.append(Paragraph("Key Findings", section_style))
        if results['causal_attributions']:
            causal_factors = {
                k: float(v[0]) if isinstance(v, (list, np.ndarray)) else float(v)
                for k, v in results['causal_attributions'].items()
            }
            top_factors = sorted(causal_factors.items(), 
                               key=lambda x: abs(x[1]), reverse=True)[:3]
            
            findings_data = [["Factor", "Impact Score"]] + [[k.replace('_', ' ').title(), f"{v:.2f}"] for k, v in top_factors]
            
            findings_table = Table(findings_data, colWidths=[3*inch, 2*inch])
            findings_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (1, 1), (1, -1), 'CENTER'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOX', (0, 0), (-1, -1), 1, colors.black),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(findings_table)
            story.append(Spacer(1, 12))
        
        # Recommendations
        story.append(Paragraph("Recommendations", section_style))
        rec_text = results['recommendations'].replace('\n', '<br/>')
        story.append(Paragraph(rec_text, styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Data Reference
        story.append(Paragraph("Data Reference", section_style))
        story.append(Paragraph(f"Synthetic data exported to: {data_path}", styles['Normal']))
        
        # Build the PDF
        doc.build(story)
        return pdf_path
    
    # ... (keep all other existing methods the same)

if __name__ == "__main__":
    """Main execution flow with PDF reporting."""
    try:
        # Initialize clients and analyzer
        client = initialize_azure_client()
        analyzer = ManufacturingRCAAnalyzer(client)
        
        # Example manufacturing incident
        case_text = """On Jan 12, 2021, the MESE1 Manufacturing Line was stopped due to discrepancy between Visual Aids and Setup Sheet, resulting in incorrect screw P/N used at Build Station 4. Work instruction OPER-WI-086 Rev C specifies P/N 5600203-01, but P/N 5600008-03 was used. The BOM and Setup Sheet are correct, but the Visual Aid rev 003 shows the wrong P/N."""
        
        # Generate graph visualization
        graph_path = analyzer.visualize_causal_graph()
        print(f"\nGraph visualization saved to: {graph_path}")
        
        # Extract case details
        case_details = analyzer._extract_case_details(case_text)
        print("\n=== EXTRACTED CASE DETAILS ===")
        print(json.dumps(case_details, indent=2))
        
        # Generate and export synthetic data
        synthetic_data = analyzer._generate_synthetic_data(case_details, num_samples=100)
        data_path = analyzer.export_synthetic_data(synthetic_data)
        print(f"\nSynthetic data exported to: {data_path}")
        print("\n=== SYNTHETIC DATA SAMPLE (First 5 Rows) ===")
        print(synthetic_data.head())
        
        # Perform full analysis
        results = analyzer.analyze_case(case_text, num_samples=100)
        
        # Generate PDF report
        pdf_path = analyzer.generate_pdf_report(results, graph_path, data_path)
        print(f"\nPDF report generated: {pdf_path}")
        
        # Print console output
        print("\n" + "="*50)
        print(format_plaintext_output(results))
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Troubleshooting:")
        print("1. Verify .env file contains required Azure OpenAI credentials")
        print("2. Check case text contains clear incident details")
        print("3. Ensure deployment supports JSON format when required")


*******************************
**************************************
***********************************************





"""
Manufacturing Root Cause Analysis (RCA) System with Enhanced Visualization and Data Export
"""

import pandas as pd
import networkx as nx
from dowhy import gcm
from typing import Dict, Any
import json
import os
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from datetime import datetime

class ManufacturingRCAAnalyzer:
    """Main analyzer class that performs root cause analysis for manufacturing incidents."""
    
    def __init__(self, azure_openai_client):
        self.common_causal_graph = self._build_common_causal_graph()
        self.scm = gcm.StructuralCausalModel(self.common_causal_graph)
        self.client = azure_openai_client
        
    def _build_common_causal_graph(self) -> nx.DiGraph:
        """Constructs the manufacturing process causal graph."""
        return nx.DiGraph([
            ('Document_Version_Control', 'Work_Instruction_Accuracy'),
            ('BOM_Accuracy', 'Work_Instruction_Accuracy'),
            ('Setup_Sheet_Accuracy', 'Work_Instruction_Accuracy'),
            ('Visual_Aid_Accuracy', 'Work_Instruction_Accuracy'),
            ('Operator_Training', 'Correct_Part_Usage'),
            ('Work_Instruction_Accuracy', 'Correct_Part_Usage'),
            ('Part_Verification_Process', 'Correct_Part_Usage'),
            ('Line_Stoppage_Protocol', 'Production_Impact'),
            ('Correct_Part_Usage', 'Production_Impact'),
            ('Work_Instruction_Accuracy', 'Production_Impact')
        ])
    
    def _generate_synthetic_data(self, case_details: Dict[str, Any], num_samples=100) -> pd.DataFrame:
        """Generates synthetic manufacturing data based on case specifics."""
        base_values = {
            'Document_Version_Control': case_details.get('document_version_issue', 0),
            'BOM_Accuracy': case_details.get('bom_accurate', 1),
            'Setup_Sheet_Accuracy': case_details.get('setup_sheet_accurate', 1),
            'Visual_Aid_Accuracy': case_details.get('visual_aid_accurate', 0),
            'Operator_Training': case_details.get('operator_trained', 1),
            'Part_Verification_Process': case_details.get('part_verification_done', 0),
            'Line_Stoppage_Protocol': case_details.get('line_stopped_correctly', 1),
            'Correct_Part_Usage': case_details.get('correct_part_used', 0),
            'Production_Impact': case_details.get('production_impact', 1),
            'Work_Instruction_Accuracy': 1
        }
        
        data = {}
        for col, val in base_values.items():
            if col == 'Production_Impact':
                samples = np.random.normal(val, 0.1, num_samples)
                data[col] = samples.tolist()
            else:
                samples = np.random.normal(val, 0.1, num_samples)
                data[col] = np.clip(samples, 0, 1).round().astype(int).tolist()
                
        return pd.DataFrame(data)
    
    def visualize_causal_graph(self):
        """Visualizes the causal graph using matplotlib."""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.common_causal_graph, seed=42)
        
        # Draw nodes and edges
        nx.draw_networkx_nodes(self.common_causal_graph, pos, node_size=2000, node_color='lightblue')
        nx.draw_networkx_edges(self.common_causal_graph, pos, arrowstyle='->', arrowsize=20)
        nx.draw_networkx_labels(self.common_causal_graph, pos, font_size=10, font_weight='bold')
        
        plt.title("Manufacturing Process Causal Graph", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        
        # Save and show the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"causal_graph_{timestamp}.png"
        plt.savefig(image_path, dpi=300)
        print(f"\nGraph visualization saved to: {image_path}")
        plt.show()
        
    def export_synthetic_data(self, data: pd.DataFrame):
        """Exports synthetic data to Excel file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = f"synthetic_data_{timestamp}.xlsx"
        
        # Create a Pandas Excel writer
        writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')
        data.to_excel(writer, sheet_name='Synthetic_Data', index=False)
        
        # Add statistics sheet
        stats = data.describe().transpose()
        stats.to_excel(writer, sheet_name='Statistics')
        
        # Format the Excel file
        workbook = writer.book
        worksheet = writer.sheets['Synthetic_Data']
        header_format = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC'})
        
        for col_num, value in enumerate(data.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        writer.close()
        print(f"\nSynthetic data exported to: {excel_path}")
    
    # ... (keep all other existing methods the same)

def initialize_azure_client() -> AzureOpenAI:
    """Initializes and returns authenticated Azure OpenAI client."""
    load_dotenv()
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

def format_plaintext_output(results: Dict[str, Any]) -> str:
    """Formats analysis results into human-readable plain text report."""
    output = []
    
    # Case Details
    output.append("=== CASE DETAILS ===")
    for key, value in results['case_details'].items():
        if isinstance(value, bool):
            value = "Yes" if value else "No"
        output.append(f"{key.replace('_', ' ').title()}: {value}")
    
    # Key Findings
    output.append("\n=== KEY FINDINGS ===")
    if results['causal_attributions']:
        causal_factors = {
            k: float(v[0]) if isinstance(v, (list, np.ndarray)) else float(v)
            for k, v in results['causal_attributions'].items()
        }
        top_factors = sorted(causal_factors.items(), 
                           key=lambda x: abs(x[1]), reverse=True)[:3]
        for factor, score in top_factors:
            output.append(f"- {factor.replace('_', ' ').title()}: {score:.2f}")
    
    # Recommendations
    output.append("\n=== RECOMMENDATIONS ===")
    output.append(results['recommendations'])
    
    return "\n".join(output)

if __name__ == "__main__":
    """Main execution flow with enhanced visualization and data export."""
    try:
        # Initialize clients and analyzer
        client = initialize_azure_client()
        analyzer = ManufacturingRCAAnalyzer(client)
        
        # Visualize the causal graph
        analyzer.visualize_causal_graph()
        
        # Example manufacturing incident
        case_text = """On Jan 12, 2021, the MESE1 Manufacturing Line was stopped due to discrepancy between Visual Aids and Setup Sheet, resulting in incorrect screw P/N used at Build Station 4. Work instruction OPER-WI-086 Rev C specifies P/N 5600203-01, but P/N 5600008-03 was used. The BOM and Setup Sheet are correct, but the Visual Aid rev 003 shows the wrong P/N."""
        
        # Extract case details and print them
        case_details = analyzer._extract_case_details(case_text)
        print("\n=== EXTRACTED CASE DETAILS ===")
        print(json.dumps(case_details, indent=2))
        
        # Generate and export synthetic data
        synthetic_data = analyzer._generate_synthetic_data(case_details, num_samples=100)
        analyzer.export_synthetic_data(synthetic_data)
        
        # Print data sample
        print("\n=== SYNTHETIC DATA SAMPLE (First 5 Rows) ===")
        print(synthetic_data.head())
        
        # Perform full analysis and display results
        results = analyzer.analyze_case(case_text, num_samples=100)
        print("\n" + "="*50)
        print(format_plaintext_output(results))
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Troubleshooting:")
        print("1. Verify .env file contains required Azure OpenAI credentials")
        print("2. Check case text contains clear incident details")
        print("3. Ensure deployment supports JSON format when required")

******************************************
**********************************



# ... (keep all the existing imports and class definitions the same until the main block)

if __name__ == "__main__":
    """Main execution flow for command-line usage."""
    try:
        # Initialize clients and analyzer
        client = initialize_azure_client()
        analyzer = ManufacturingRCAAnalyzer(client)
        
        # Print the causal graph structure
        print("\n=== CAUSAL GRAPH STRUCTURE ===")
        print("Nodes:", analyzer.common_causal_graph.nodes())
        print("Edges:", analyzer.common_causal_graph.edges())
        
        # Visual representation of the graph
        print("\n=== GRAPH VISUALIZATION ===")
        for source, target in analyzer.common_causal_graph.edges():
            print(f"{source} -> {target}")
            
        # Example manufacturing incident
        case_text = """On Jan 12, 2021, the MESE1 Manufacturing Line was stopped due to discrepancy between Visual Aids and Setup Sheet, resulting in incorrect screw P/N used at Build Station 4. Work instruction OPER-WI-086 Rev C specifies P/N 5600203-01, but P/N 5600008-03 was used. The BOM and Setup Sheet are correct, but the Visual Aid rev 003 shows the wrong P/N."""
        
        # Extract case details and print them
        case_details = analyzer._extract_case_details(case_text)
        print("\n=== EXTRACTED CASE DETAILS ===")
        print(json.dumps(case_details, indent=2))
        
        # Generate and print synthetic data
        synthetic_data = analyzer._generate_synthetic_data(case_details, num_samples=10)  # Smaller sample for display
        print("\n=== SYNTHETIC DATA SAMPLE ===")
        print(synthetic_data.head())
        print("\nData Statistics:")
        print(synthetic_data.describe())
        
        # Perform full analysis and display results
        results = analyzer.analyze_case(case_text, num_samples=100)
        print("\n" + "="*50)
        print(format_plaintext_output(results))
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Troubleshooting:")
        print("1. Verify .env file contains required Azure OpenAI credentials")
        print("2. Check case text contains clear incident details")
        print("3. Ensure deployment supports JSON format when required")



***************************
*************************************
"""
Manufacturing Root Cause Analysis (RCA) System

This system combines causal machine learning with Azure OpenAI to:
1. Analyze manufacturing incident reports
2. Identify root causes using causal graphs
3. Generate actionable corrective/preventive actions

Key Components:
- Causal graph modeling manufacturing processes
- Synthetic data generation for scenario analysis
- Azure OpenAI integration for natural language understanding
- Plain text reporting for easy consumption
"""

import pandas as pd
import networkx as nx
from dowhy import gcm
from typing import Dict, Any
import json
import os
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv

class ManufacturingRCAAnalyzer:
    """Main analyzer class that performs root cause analysis for manufacturing incidents.
    
    Combines causal analysis (DoWhy/gcm) with Azure OpenAI for natural language 
    understanding and recommendation generation.
    
    Args:
        azure_openai_client: Authenticated Azure OpenAI client instance
    """
    
    def __init__(self, azure_openai_client):
        """Initialize the analyzer with Azure OpenAI client and build causal graph.
        
        Args:
            azure_openai_client: Configured AzureOpenAI client instance
        """
        self.common_causal_graph = self._build_common_causal_graph()
        self.scm = gcm.StructuralCausalModel(self.common_causal_graph)
        self.client = azure_openai_client
        
    def _build_common_causal_graph(self) -> nx.DiGraph:
        """Constructs the manufacturing process causal graph.
        
        Returns:
            nx.DiGraph: Directed graph representing causal relationships between 
                       manufacturing factors like documentation, training, and production.
                       
        Example Relationships:
            - Accurate Visual Aids → Correct Part Usage
            - Operator Training → Correct Part Usage 
            - Correct Part Usage → Production Impact
        """
        return nx.DiGraph([
            ('Document_Version_Control', 'Work_Instruction_Accuracy'),
            ('BOM_Accuracy', 'Work_Instruction_Accuracy'),
            ('Setup_Sheet_Accuracy', 'Work_Instruction_Accuracy'),
            ('Visual_Aid_Accuracy', 'Work_Instruction_Accuracy'),
            ('Operator_Training', 'Correct_Part_Usage'),
            ('Work_Instruction_Accuracy', 'Correct_Part_Usage'),
            ('Part_Verification_Process', 'Correct_Part_Usage'),
            ('Line_Stoppage_Protocol', 'Production_Impact'),
            ('Correct_Part_Usage', 'Production_Impact'),
            ('Work_Instruction_Accuracy', 'Production_Impact')
        ])
    
    def _generate_synthetic_data(self, case_details: Dict[str, Any], num_samples=100) -> pd.DataFrame:
        """Generates synthetic manufacturing data based on case specifics.
        
        Args:
            case_details: Dictionary containing case parameters (e.g., was visual aid accurate)
            num_samples: Number of synthetic data points to generate
            
        Returns:
            pd.DataFrame: Generated data with columns matching causal graph nodes
            
        Data Generation Logic:
            - Binary factors (doc accuracy, training): Binomial distribution
            - Continuous metrics (production impact): Normal distribution
            - Defaults to realistic manufacturing probabilities when case details missing
        """
        base_values = {
            'Document_Version_Control': case_details.get('document_version_issue', 0),
            'BOM_Accuracy': case_details.get('bom_accurate', 1),
            'Setup_Sheet_Accuracy': case_details.get('setup_sheet_accurate', 1),
            'Visual_Aid_Accuracy': case_details.get('visual_aid_accurate', 0),
            'Operator_Training': case_details.get('operator_trained', 1),
            'Part_Verification_Process': case_details.get('part_verification_done', 0),
            'Line_Stoppage_Protocol': case_details.get('line_stopped_correctly', 1),
            'Correct_Part_Usage': case_details.get('correct_part_used', 0),
            'Production_Impact': case_details.get('production_impact', 1),
            'Work_Instruction_Accuracy': 1
        }
        
        data = {}
        for col, val in base_values.items():
            if col == 'Production_Impact':
                samples = np.random.normal(val, 0.1, num_samples)
                data[col] = samples.tolist()
            else:
                samples = np.random.normal(val, 0.1, num_samples)
                data[col] = np.clip(samples, 0, 1).round().astype(int).tolist()
                
        return pd.DataFrame(data)
    
    def _call_azure_openai(self, prompt: str, require_json: bool = False) -> str:
        """Executes Azure OpenAI API call with proper formatting.
        
        Args:
            prompt: Complete prompt text for the AI model
            require_json: Whether to force JSON response format
            
        Returns:
            str: Raw response content from Azure OpenAI
            
        Note:
            - Sets temperature=0.3 for balanced creativity/consistency
            - Explicitly requests JSON when needed for structured parsing
        """
        messages = [{"role": "user", "content": prompt}]
        
        if require_json:
            messages[0]["content"] = f"Return the response as a valid JSON object.\n{prompt}"
            response = self.client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                messages=messages,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
        else:
            response = self.client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                messages=messages,
                temperature=0.3
            )
            
        return response.choices[0].message.content
    
    def _extract_case_details(self, case_text: str) -> Dict[str, Any]:
        """Extracts structured case details from incident description using AI.
        
        Args:
            case_text: Raw incident description (e.g., "Wrong part used because...")
            
        Returns:
            Dict: Structured case details including:
                - document_version_issue (bool)
                - visual_aid_accurate (bool) 
                - root_cause_hypothesis (str)
                - etc.
                
        Prompt Engineering:
            - Explicitly specifies required JSON schema
            - Provides clear field definitions
            - Uses manufacturing-specific terminology
        """
        prompt = f"""Analyze this manufacturing CAPA case and return a JSON object with these exact fields:
{{
    "document_version_issue": bool,
    "bom_accurate": bool,
    "setup_sheet_accurate": bool,
    "visual_aid_accurate": bool,
    "operator_trained": bool,
    "part_verification_done": bool,
    "line_stopped_correctly": bool,
    "correct_part_used": bool,
    "production_impact": int,
    "root_cause_hypothesis": str
}}

Case details:
{case_text}"""
        
        response = self._call_azure_openai(prompt, require_json=True)
        return json.loads(response)
    
    def analyze_case(self, case_text: str, num_samples=100) -> Dict[str, Any]:
        """Main analysis workflow - processes incident text through full RCA pipeline.
        
        Args:
            case_text: Raw incident description
            num_samples: Number of synthetic scenarios to generate
            
        Returns:
            Dict: Analysis results containing:
                - case_details: Extracted incident facts
                - causal_attributions: Quantified factor impacts
                - recommendations: AI-generated action items
                
        Workflow Steps:
            1. Extract structured case details using AI
            2. Generate synthetic production scenarios
            3. Train causal model on generated data  
            4. Calculate factor attributions
            5. Generate recommendations
        """
        case_details = self._extract_case_details(case_text)
        data = self._generate_synthetic_data(case_details, num_samples)
        
        gcm.auto.assign_causal_mechanisms(self.scm, data)
        gcm.fit(self.scm, data)
        
        attributions = gcm.attribute_anomalies(
            self.scm, 
            target_node='Production_Impact',
            anomaly_samples=data.iloc[:1]
        )
        
        converted_attributions = {
            k: float(v[0]) if isinstance(v, (np.ndarray, list)) else float(v) 
            for k, v in attributions.items()
        }
        
        return {
            'case_details': case_details,
            'causal_attributions': converted_attributions,
            'recommendations': self._generate_recommendations(case_details, converted_attributions)
        }
    
    def _generate_recommendations(self, case_details: Dict[str, Any], attributions: Dict[str, Any]) -> str:
        """Generates human-readable corrective/preventive action recommendations.
        
        Args:
            case_details: Extracted incident facts
            attributions: Quantified factor impact scores
            
        Returns:
            str: Plain text recommendations formatted with:
                - Root cause analysis
                - Corrective actions
                - Preventive actions
                - Verification methods
                - Test cases
        """
        prompt = f"""Generate clear, actionable CAPA recommendations in plain text format with these sections:
        
Root Cause Analysis:
[Explain the likely root cause in 2-3 sentences]

Corrective Actions:
1. [Immediate action 1]
2. [Immediate action 2] 
3. [Immediate action 3]

Preventive Actions:
1. [Long-term solution 1]
2. [Long-term solution 2]

Verification Methods:
- [How to verify corrective actions] 
- [How to verify preventive actions]

Test Cases:
1. [Specific test scenario 1]
2. [Specific test scenario 2]

Case Details:
{json.dumps(case_details, indent=2)}

Key Contributing Factors:
{json.dumps(attributions, indent=2)}"""
        
        return self._call_azure_openai(prompt, require_json=False)

def initialize_azure_client() -> AzureOpenAI:
    """Initializes and returns authenticated Azure OpenAI client.
    
    Reads credentials from .env file with these required variables:
        - AZURE_OPENAI_API_KEY
        - AZURE_OPENAI_ENDPOINT  
        - AZURE_OPENAI_DEPLOYMENT_NAME
        
    Returns:
        AzureOpenAI: Authenticated client instance
        
    Raises:
        EnvironmentError: If required variables are missing
    """
    load_dotenv()
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

def format_plaintext_output(results: Dict[str, Any]) -> str:
    """Formats analysis results into human-readable plain text report.
    
    Args:
        results: Raw analysis results from analyze_case()
        
    Returns:
        str: Formatted report with sections:
            - Case Details
            - Key Findings (top 3 contributing factors) 
            - Recommendations
            
    Formatting Features:
        - Converts booleans to Yes/No
        - Replaces underscores with spaces in factor names
        - Sorts factors by impact magnitude
        - Preserves original recommendation formatting
    """
    output = []
    
    # Case Details
    output.append("=== CASE DETAILS ===")
    for key, value in results['case_details'].items():
        if isinstance(value, bool):
            value = "Yes" if value else "No"
        output.append(f"{key.replace('_', ' ').title()}: {value}")
    
    # Key Findings
    output.append("\n=== KEY FINDINGS ===")
    if results['causal_attributions']:
        causal_factors = {
            k: float(v[0]) if isinstance(v, (list, np.ndarray)) else float(v)
            for k, v in results['causal_attributions'].items()
        }
        top_factors = sorted(causal_factors.items(), 
                           key=lambda x: abs(x[1]), reverse=True)[:3]
        for factor, score in top_factors:
            output.append(f"- {factor.replace('_', ' ').title()}: {score:.2f}")
    
    # Recommendations
    output.append("\n=== RECOMMENDATIONS ===")
    output.append(results['recommendations'])
    
    return "\n".join(output)

if __name__ == "__main__":
    """Main execution flow for command-line usage."""
    try:
        # Initialize clients and analyzer
        client = initialize_azure_client()
        analyzer = ManufacturingRCAAnalyzer(client)
        
        # Example manufacturing incident
        case_text = """On Jan 12, 2021, the MESE1 Manufacturing Line was stopped due to discrepancy between Visual Aids and Setup Sheet, resulting in incorrect screw P/N used at Build Station 4. Work instruction OPER-WI-086 Rev C specifies P/N 5600203-01, but P/N 5600008-03 was used. The BOM and Setup Sheet are correct, but the Visual Aid rev 003 shows the wrong P/N."""
        
        # Perform analysis and display results
        results = analyzer.analyze_case(case_text, num_samples=100)
        print(format_plaintext_output(results))
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Troubleshooting:")
        print("1. Verify .env file contains required Azure OpenAI credentials")
        print("2. Check case text contains clear incident details")
        print("3. Ensure deployment supports JSON format when required")
