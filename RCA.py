import pandas as pd
import networkx as nx
from dowhy import gcm
from typing import Dict, Any, Optional
import json
import os
import numpy as np
from scipy.stats import bernoulli, norm
from openai import AzureOpenAI
from dotenv import load_dotenv

class ManufacturingRCAAnalyzer:
    def __init__(self, azure_openai_client, real_data: Optional[pd.DataFrame] = None):
        """Initialize with optional real data injection"""
        self.common_causal_graph = self._build_common_causal_graph()
        self.scm = gcm.StructuralCausalModel(self.common_causal_graph)
        self.client = azure_openai_client
        self.real_data = real_data  # Store optional real data
        
    def _build_common_causal_graph(self) -> nx.DiGraph:
        """Build causal graph with manufacturing-specific relationships"""
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
        """Generate realistic synthetic data with proper distributions"""
        if self.real_data is not None:
            return self.real_data  # Use real data if provided
            
        # Define conditional distributions for more realistic data
        distributions = {
            'Document_Version_Control': gcm.ScipyDistribution(bernoulli, 
                                    case_details.get('document_version_issue', 0.1)),
            'BOM_Accuracy': gcm.ScipyDistribution(bernoulli, 
                              case_details.get('bom_accurate', 0.95)),
            'Visual_Aid_Accuracy': gcm.ScipyDistribution(bernoulli, 
                                 case_details.get('visual_aid_accurate', 0.8)),
            'Production_Impact': gcm.ScipyDistribution(norm, 
                               loc=case_details.get('production_impact', 1), scale=0.1)
        }
        
        data = {}
        for node in self.common_causal_graph.nodes():
            if node in distributions:
                data[node] = distributions[node].draw_samples(num_samples)
            else:
                # Default binary distribution
                val = case_details.get(node.lower().replace('_', ''), 0.5)
                data[node] = gcm.ScipyDistribution(bernoulli, val).draw_samples(num_samples)
                
        return pd.DataFrame(data)
    
    def _validate_extracted_details(self, extracted: Dict[str, Any], case_text: str) -> bool:
        """Validate LLM output against case text clues"""
        # Check if correct_part_used aligns with textual clues
        if 'incorrect' in case_text.lower() and extracted.get('correct_part_used', True):
            return False
        if 'correct' in case_text.lower() and not extracted.get('correct_part_used', False):
            return False
            
        # Check all required fields exist
        required_fields = ['document_version_issue', 'bom_accurate', 
                          'visual_aid_accurate', 'correct_part_used']
        return all(field in extracted for field in required_fields)
    
    def _extract_case_details(self, case_text: str, max_retries=3) -> Dict[str, Any]:
        """Robust extraction with validation and retries"""
        for attempt in range(max_retries):
            prompt = f"""Analyze this manufacturing CAPA case and return a JSON object with:
- document_version_issue (bool)
- bom_accurate (bool)  
- visual_aid_accurate (bool)
- correct_part_used (bool)
- production_impact (int 1-10)
- root_cause_hypothesis (str)
- confidence_score (float 0-1)

Also provide a brief summary and confidence score.

Case: {case_text}"""
            
            response = self._call_azure_openai(prompt, require_json=True)
            try:
                extracted = json.loads(response)
                if self._validate_extracted_details(extracted, case_text):
                    return extracted
            except json.JSONDecodeError:
                continue
                
        raise ValueError("Failed to extract valid case details after retries")
    
    def _compare_hypothesis_with_attributions(self, hypothesis: str, attributions: Dict[str, float]) -> Dict:
        """Compare LLM hypothesis with causal attributions"""
        prompt = f"""Compare this root cause hypothesis with causal factors:
        
Hypothesis: {hypothesis}

Causal Factors: {json.dumps(attributions, indent=2)}

Return JSON with:
- alignment_score (float 0-1)
- supporting_evidence (list of matching factors)
- conflicting_evidence (list of non-matching factors)"""
        
        return json.loads(self._call_azure_openai(prompt, require_json=True))
    
    def analyze_case(self, case_text: str, num_samples=100) -> Dict[str, Any]:
        """Enhanced RCA workflow with validation and comparison"""
        case_details = self._extract_case_details(case_text)
        data = self._generate_synthetic_data(case_details, num_samples)
        
        # Fit causal model with explicit mechanisms
        for node in self.common_causal_graph.nodes():
            if node == 'Production_Impact':
                self.scm.set_causal_mechanism(node, gcm.AdditiveNoiseModel(
                    gcm.ml.create_linear_regressor()))
            else:
                self.scm.set_causal_mechanism(node, gcm.AdditiveNoiseModel(
                    gcm.ml.create_logistic_regression_classifier()))
        
        gcm.fit(self.scm, data)
        
        # Multiple analysis methods
        attributions = gcm.attribute_anomalies(
            self.scm, target_node='Production_Impact',
            anomaly_samples=data.iloc[:1])
        
        # Convert and analyze results
        converted_attributions = {k: float(v[0]) for k, v in attributions.items()}
        hypothesis_analysis = self._compare_hypothesis_with_attributions(
            case_details['root_cause_hypothesis'], converted_attributions)
        
        return {
            'case_details': case_details,
            'causal_attributions': converted_attributions,
            'hypothesis_analysis': hypothesis_analysis,
            'recommendations': self._generate_recommendations(
                case_details, converted_attributions)
        }
    
    def _generate_recommendations(self, case_details: Dict, attributions: Dict) -> Dict:
        """Return structured CAPA recommendations"""
        prompt = f"""Generate detailed CAPA recommendations as JSON with:
- root_cause (str)
- corrective_actions (list)
- preventive_actions (list) 
- verification_methods (list)
- test_cases (list)

Context: {json.dumps({
    'case_details': case_details,
    'causal_factors': attributions
}, indent=2)}"""
        
        return json.loads(self._call_azure_openai(prompt, require_json=True))

def initialize_azure_client():
    """Initialize with enhanced error handling"""
    load_dotenv()
    try:
        return AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2023-12-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    except Exception as e:
        raise ConnectionError(f"Azure client init failed: {str(e)}")

if __name__ == "__main__":
    try:
        # Example with optional real data injection
        # real_data = pd.read_csv('manufacturing_data.csv')
        client = initialize_azure_client()
        analyzer = ManufacturingRCAAnalyzer(client)  # Pass real_data if available
        
        case_text = """On Jan 12, 2021..."""  # Your case here
        
        results = analyzer.analyze_case(case_text)
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        print(f"Analysis failed: {str(e)}")


*******************************************
*********************************************
    ****************************************************
********************************************************
    ************************************************************

from typing import Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import openai
import re
import pandas as pd
import dowhy
from dowhy import CausalModel
import networkx as nx
import matplotlib.pyplot as plt

load_dotenv()  # Load OpenAI key from .env
openai.api_key = os.getenv("OPENAI_API_KEY")

# === LLM Prompt to extract causes, effects, and graph ===
def extract_causal_graph(capa_text: str) -> Dict[str, Any]:
    prompt = f"""
You are a Root Cause Analysis expert and causal graph analyst. Analyze the CAPA scenario below and perform the following:

1. Identify the key variables (cause, intermediate steps, outcome).
2. Identify the causal relationships between variables (in plain English).
3. Represent the causal graph using DoWhy syntax.

CAPA SCENARIO:
\"\"\"
{capa_text}
\"\"\"

Respond ONLY in the following format:

Variables: <list of variables>
Causal Relationships: <list of plain English cause-effect>
Graph Syntax: <DoWhy graph as string>
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    output = response['choices'][0]['message']['content']
    return parse_llm_response(output)

# === Parse LLM output ===
def parse_llm_response(llm_response: str) -> Dict[str, Any]:
    variables = re.findall(r"Variables:\s*(.*)", llm_response)
    relationships = re.findall(r"Causal Relationships:\s*(.*)", llm_response)
    graph_syntax = re.findall(r"Graph Syntax:\s*([\s\S]*)", llm_response)

    return {
        "variables": eval(variables[0]) if variables else [],
        "relationships": relationships[0].split("->") if relationships else [],
        "graph_syntax": graph_syntax[0].strip() if graph_syntax else ""
    }

# === Create fake dataframe (since we use description not raw data) ===
def simulate_data(variables):
    import numpy as np
    df = pd.DataFrame(np.random.randint(0, 2, size=(1000, len(variables))), columns=variables)
    return df

# === Run DoWhy Causal Inference ===
def perform_root_cause_analysis(capa_text: str):
    graph_info = extract_causal_graph(capa_text)
    variables = graph_info["variables"]
    graph_syntax = graph_info["graph_syntax"]

    df = simulate_data(variables)

    model = CausalModel(
        data=df,
        treatment=[v for v in variables if "used" in v.lower() or "referenced" in v.lower()],
        outcome="line_stopped",  # assuming this is outcome
        graph=graph_syntax
    )

    identified_estimand = model.identify_effect()
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.propensity_score_matching")

    model.view_model()
    plt.show()

    return {
        "root_cause": estimate.value,
        "causal_graph": graph_syntax,
        "llm_variables": graph_info["variables"],
        "llm_relationships": graph_info["relationships"]
    }

# === Sample Test Case ===
capa_case = """
On Jan 12, 2021, the MESE1 Manufacturing Line at External Manufacturer Jabil was stopped by AUH (Auburn Hills) Manufacturing due to a discrepancy between the mechanical assembly Visual Aids and the Setup Sheet, resulting in an incorrect screw P/N used at MESE1-12-G Build Station 4. It was found that Step 13 of work instruction OPER-WI-086 Rev C states to use P/N 5600203-01 to secure the filter Canister Bracket to the Chassis. P/N 5600008-03 was used instead. The BOM and setup Sheet have the correct P/N for assembly, but the AUH Work Instruction Visual Aid rev 003 references the incorrect 5600008-03. EES was notified on Jan 14, 2021.
"""

# === Run Analysis ===
result = perform_root_cause_analysis(capa_case)
print("ROOT CAUSE VALUE (impact):", result["root_cause"])
print("Graph:\n", result["causal_graph"])
print("Variables:", result["llm_variables"])
print("Causal Flow:", result["llm_relationships"])

******************************
*******************************
*********************************
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
    def __init__(self, azure_openai_client):
        """Initialize with Azure OpenAI client and causal graph"""
        self.common_causal_graph = self._build_common_causal_graph()
        self.scm = gcm.StructuralCausalModel(self.common_causal_graph)
        self.client = azure_openai_client
        
    def _build_common_causal_graph(self) -> nx.DiGraph:
        """Build base causal graph with consistent node naming"""
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
        """Generate sufficient synthetic data for analysis"""
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
        """Make API call to Azure OpenAI with proper JSON handling"""
        messages = [{"role": "user", "content": prompt}]
        
        if require_json:
            # Explicitly instruct the model to return JSON
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
        """Extract structured details using Azure OpenAI with explicit JSON request"""
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
        """Complete RCA workflow with sufficient samples"""
        case_details = self._extract_case_details(case_text)
        data = self._generate_synthetic_data(case_details, num_samples)
        
        gcm.auto.assign_causal_mechanisms(self.scm, data)
        gcm.fit(self.scm, data)
        
        attributions = gcm.attribute_anomalies(
            self.scm, 
            target_node='Production_Impact',
            anomaly_samples=data.iloc[:1]
        )
        
        # Convert numpy types to native Python types
        converted_attributions = {
            k: v.tolist() if isinstance(v, np.ndarray) else float(v) 
            for k, v in attributions.items()
        }
        
        return {
            'case_details': case_details,
            'causal_attributions': converted_attributions,
            'recommendations': self._generate_recommendations(case_details, converted_attributions)
        }
    
    def _generate_recommendations(self, case_details: Dict[str, Any], attributions: Dict[str, Any]) -> str:
        """Generate CAPA recommendations"""
        prompt = f"""Generate CAPA recommendations in JSON format with these sections:
1. Root Cause
2. Corrective Actions 
3. Preventive Actions
4. Verification Methods
5. Test Cases

Case Analysis:
{json.dumps(case_details, indent=2)}

Causal Factors:
{json.dumps(attributions, indent=2)}"""
        
        return self._call_azure_openai(prompt, require_json=True)

def initialize_azure_client():
    """Initialize Azure OpenAI client"""
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
        
        case_text = """On Jan 12, 2021, the MESE1 Manufacturing Line was stopped due to discrepancy between Visual Aids and Setup Sheet, resulting in incorrect screw P/N used at Build Station 4. Work instruction OPER-WI-086 Rev C specifies P/N 5600203-01, but P/N 5600008-03 was used. The BOM and Setup Sheet are correct, but the Visual Aid rev 003 shows the wrong P/N."""
        
        results = analyzer.analyze_case(case_text, num_samples=100)
        print(json.dumps(results, indent=2, default=str))
        
    except Exception as e:
        print(f"Error: {str(e)}")


****************
************************
**************************
****************

import pandas as pd
import networkx as nx
from dowhy import gcm
from typing import Dict, Any
import json
import os
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotload_dotenv()

class ManufacturingRCAAnalyzer:
    def __init__(self, azure_openai_client):
        """Initialize with Azure OpenAI client and causal graph"""
        self.common_causal_graph = self._build_common_causal_graph()
        self.scm = gcm.StructuralCausalModel(self.common_causal_graph)
        self.client = azure_openai_client
        
    def _build_common_causal_graph(self) -> nx.DiGraph:
        """Build base causal graph with consistent node naming"""
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
        """Generate sufficient synthetic data for analysis"""
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
        
        # Create multiple samples with small random variations
        data = {}
        for col, val in base_values.items():
            if col == 'Production_Impact':
                # Continuous variable
                samples = np.random.normal(val, 0.1, num_samples)
                data[col] = samples.tolist()  # Convert to list immediately
            else:
                # Binary variables
                samples = np.random.normal(val, 0.1, num_samples)
                data[col] = np.clip(samples, 0, 1).round().astype(int).tolist()
                
        return pd.DataFrame(data)
    
    def _call_azure_openai(self, prompt: str) -> str:
        """Make API call to Azure OpenAI"""
        response = self.client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    
    def _extract_case_details(self, case_text: str) -> Dict[str, Any]:
        """Extract structured details using Azure OpenAI"""
        prompt = f"""Analyze this manufacturing CAPA case and return ONLY a valid JSON object:
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
        
        response = self._call_azure_openai(prompt)
        return json.loads(response)
    
    def analyze_case(self, case_text: str, num_samples=100) -> Dict[str, Any]:
        """Complete RCA workflow with sufficient samples"""
        case_details = self._extract_case_details(case_text)
        data = self._generate_synthetic_data(case_details, num_samples)
        
        gcm.auto.assign_causal_mechanisms(self.scm, data)
        gcm.fit(self.scm, data)
        
        attributions = gcm.attribute_anomalies(
            self.scm, 
            target_node='Production_Impact',
            anomaly_samples=data.iloc[:1]  # Use first sample as anomaly
        )
        
        # Convert all numpy types to native Python types
        converted_attributions = {}
        for k, v in attributions.items():
            if isinstance(v, np.ndarray):
                converted_attributions[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                converted_attributions[k] = float(v)
            else:
                converted_attributions[k] = v
        
        result = {
            'case_details': case_details,
            'causal_attributions': converted_attributions,
            'recommendations': self._generate_recommendations(case_details, converted_attributions)
        }
        
        return result
    
    def _generate_recommendations(self, case_details: Dict[str, Any], attributions: Dict[str, Any]) -> str:
        """Generate CAPA recommendations"""
        prompt = f"""Generate CAPA recommendations in this format:
1. Root Cause: [clear explanation]
2. Corrective Actions: [bullet points]
3. Preventive Actions: [bullet points]
4. Verification: [methods]
5. Test Cases: [specific validations]

Case Analysis:
{json.dumps(case_details, indent=2)}

Causal Factors:
{json.dumps(attributions, indent=2)}"""
        
        return self._call_azure_openai(prompt)

def initialize_azure_client():
    """Initialize Azure OpenAI client"""
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
        
        case_text = """On Jan 12, 2021, the MESE1 Manufacturing Line was stopped due to discrepancy between Visual Aids and Setup Sheet, resulting in incorrect screw P/N used at Build Station 4. Work instruction OPER-WI-086 Rev C specifies P/N 5600203-01, but P/N 5600008-03 was used. The BOM and Setup Sheet are correct, but the Visual Aid rev 003 shows the wrong P/N."""
        
        results = analyzer.analyze_case(case_text, num_samples=100)
        print(json.dumps(results, indent=2, default=str))  # Added default=str for safety
        
    except Exception as e:
        print(f"Error: {str(e)}")

******************
    *********************
**************************
    **************************


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
    def __init__(self, azure_openai_client):
        """Initialize with Azure OpenAI client and causal graph"""
        self.common_causal_graph = self._build_common_causal_graph()
        self.scm = gcm.StructuralCausalModel(self.common_causal_graph)
        self.client = azure_openai_client
        
    def _build_common_causal_graph(self) -> nx.DiGraph:
        """Build base causal graph with consistent node naming"""
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
        """Generate sufficient synthetic data for analysis"""
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
        
        # Create multiple samples with small random variations
        data = {col: np.random.normal(val, 0.1, num_samples) for col, val in base_values.items()}
        
        # Ensure binary fields stay between 0 and 1 and convert to native Python types
        for col in ['Document_Version_Control', 'BOM_Accuracy', 'Setup_Sheet_Accuracy', 
                   'Visual_Aid_Accuracy', 'Operator_Training', 'Part_Verification_Process',
                   'Line_Stoppage_Protocol', 'Correct_Part_Usage', 'Work_Instruction_Accuracy']:
            data[col] = np.clip(data[col], 0, 1).round().astype(int).tolist()
            
        # Convert numerical fields to native Python types
        data['Production_Impact'] = data['Production_Impact'].astype(float).tolist()
            
        return pd.DataFrame(data)
    
    def _call_azure_openai(self, prompt: str) -> str:
        """Make API call to Azure OpenAI"""
        response = self.client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    
    def _extract_case_details(self, case_text: str) -> Dict[str, Any]:
        """Extract structured details using Azure OpenAI"""
        prompt = f"""Analyze this manufacturing CAPA case and return ONLY a valid JSON object:
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
        
        response = self._call_azure_openai(prompt)
        return json.loads(response)
    
    def analyze_case(self, case_text: str, num_samples=100) -> Dict[str, Any]:
        """Complete RCA workflow with sufficient samples"""
        case_details = self._extract_case_details(case_text)
        data = self._generate_synthetic_data(case_details, num_samples)
        
        gcm.auto.assign_causal_mechanisms(self.scm, data)
        gcm.fit(self.scm, data)
        
        attributions = gcm.attribute_anomalies(
            self.scm, 
            target_node='Production_Impact',
            anomaly_samples=data.iloc[:1]  # Use first sample as anomaly
        )
        
        # Convert numpy types to native Python types for JSON serialization
        result = {
            'case_details': case_details,
            'causal_attributions': {k: float(v[0]) for k, v in attributions.items()},
            'recommendations': self._generate_recommendations(case_details, attributions)
        }
        
        return result
    
    def _generate_recommendations(self, case_details: Dict[str, Any], attributions: Dict[str, Any]) -> str:
        """Generate CAPA recommendations"""
        # Convert numpy types in attributions to native Python types
        safe_attributions = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                            for k, v in attributions.items()}
        
        prompt = f"""Generate CAPA recommendations in this format:
1. Root Cause: [clear explanation]
2. Corrective Actions: [bullet points]
3. Preventive Actions: [bullet points]
4. Verification: [methods]
5. Test Cases: [specific validations]

Case Analysis:
{json.dumps(case_details, indent=2)}

Causal Factors:
{json.dumps(safe_attributions, indent=2, default=str)}"""  # Added default=str for safety
        
        return self._call_azure_openai(prompt)

def initialize_azure_client():
    """Initialize Azure OpenAI client"""
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
        
        case_text = """On Jan 12, 2021, the MESE1 Manufacturing Line was stopped due to discrepancy between Visual Aids and Setup Sheet, resulting in incorrect screw P/N used at Build Station 4. Work instruction OPER-WI-086 Rev C specifies P/N 5600203-01, but P/N 5600008-03 was used. The BOM and Setup Sheet are correct, but the Visual Aid rev 003 shows the wrong P/N."""
        
        results = analyzer.analyze_case(case_text, num_samples=100)
        print(json.dumps(results, indent=2, default=str))  # Added default=str for safety
        
    except Exception as e:
        print(f"Error: {str(e)}")




****************************
*****************************
***************************
************************
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
    def __init__(self, azure_openai_client):
        """Initialize with Azure OpenAI client and causal graph"""
        self.common_causal_graph = self._build_common_causal_graph()
        self.scm = gcm.StructuralCausalModel(self.common_causal_graph)
        self.client = azure_openai_client
        
    def _build_common_causal_graph(self) -> nx.DiGraph:
        """Build base causal graph with consistent node naming"""
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
        """Generate sufficient synthetic data for analysis"""
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
        
        # Create multiple samples with small random variations
        data = {col: np.random.normal(val, 0.1, num_samples) for col, val in base_values.items()}
        
        # Ensure binary fields stay between 0 and 1
        for col in ['Document_Version_Control', 'BOM_Accuracy', 'Setup_Sheet_Accuracy', 
                   'Visual_Aid_Accuracy', 'Operator_Training', 'Part_Verification_Process',
                   'Line_Stoppage_Protocol', 'Correct_Part_Usage', 'Work_Instruction_Accuracy']:
            data[col] = np.clip(data[col], 0, 1).round()
            
        return pd.DataFrame(data)
    
    def _call_azure_openai(self, prompt: str) -> str:
        """Make API call to Azure OpenAI"""
        response = self.client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    
    def _extract_case_details(self, case_text: str) -> Dict[str, Any]:
        """Extract structured details using Azure OpenAI"""
        prompt = f"""Analyze this manufacturing CAPA case and return ONLY a valid JSON object:
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
        
        response = self._call_azure_openai(prompt)
        return json.loads(response)
    
    def analyze_case(self, case_text: str, num_samples=100) -> Dict[str, Any]:
        """Complete RCA workflow with sufficient samples"""
        case_details = self._extract_case_details(case_text)
        data = self._generate_synthetic_data(case_details, num_samples)
        
        gcm.auto.assign_causal_mechanisms(self.scm, data)
        gcm.fit(self.scm, data)
        
        attributions = gcm.attribute_anomalies(
            self.scm, 
            target_node='Production_Impact',
            anomaly_samples=data.iloc[:1]  # Use first sample as anomaly
        )
        
        return {
            'case_details': case_details,
            'causal_attributions': {k: float(v[0]) for k, v in attributions.items()},
            'recommendations': self._generate_recommendations(case_details, attributions)
        }
    
    def _generate_recommendations(self, case_details: Dict[str, Any], attributions: Dict[str, Any]) -> str:
        """Generate CAPA recommendations"""
        prompt = f"""Generate CAPA recommendations in this format:
1. Root Cause: [clear explanation]
2. Corrective Actions: [bullet points]
3. Preventive Actions: [bullet points]
4. Verification: [methods]
5. Test Cases: [specific validations]

Case Analysis:
{json.dumps(case_details, indent=2)}

Causal Factors:
{json.dumps(attributions, indent=2)}"""
        
        return self._call_azure_openai(prompt)

def initialize_azure_client():
    """Initialize Azure OpenAI client"""
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
        
        case_text = """On Jan 12, 2021, the MESE1 Manufacturing Line was stopped due to discrepancy between Visual Aids and Setup Sheet, resulting in incorrect screw P/N used at Build Station 4. Work instruction OPER-WI-086 Rev C specifies P/N 5600203-01, but P/N 5600008-03 was used. The BOM and Setup Sheet are correct, but the Visual Aid rev 003 shows the wrong P/N."""
        
        results = analyzer.analyze_case(case_text, num_samples=100)
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        print(f"Error: {str(e)}")

*********************
********************************
*****************************************
import pandas as pd
import networkx as nx
from dowhy import gcm
from typing import Dict, Any
import json
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

class ManufacturingRCAAnalyzer:
    def __init__(self, azure_openai_client):
        """Initialize with Azure OpenAI client and causal graph"""
        self.common_causal_graph = self._build_common_causal_graph()
        self.scm = gcm.StructuralCausalModel(self.common_causal_graph)
        self.client = azure_openai_client
        
    def _build_common_causal_graph(self) -> nx.DiGraph:
        """Build base causal graph with consistent node naming"""
        return nx.DiGraph([
            # Document-related factors
            ('Document_Version_Control', 'Work_Instruction_Accuracy'),
            ('BOM_Accuracy', 'Work_Instruction_Accuracy'),
            ('Setup_Sheet_Accuracy', 'Work_Instruction_Accuracy'),
            ('Visual_Aid_Accuracy', 'Work_Instruction_Accuracy'),
            
            # Process-related factors
            ('Operator_Training', 'Correct_Part_Usage'),
            ('Work_Instruction_Accuracy', 'Correct_Part_Usage'),
            ('Part_Verification_Process', 'Correct_Part_Usage'),
            ('Line_Stoppage_Protocol', 'Production_Impact'),
            
            # Outcome
            ('Correct_Part_Usage', 'Production_Impact'),
            ('Work_Instruction_Accuracy', 'Production_Impact')
        ])
    
    def _generate_synthetic_data(self, case_details: Dict[str, Any]) -> pd.DataFrame:
        """Generate synthetic data with consistent column names"""
        return pd.DataFrame({
            'Document_Version_Control': [case_details.get('document_version_issue', 0)],
            'BOM_Accuracy': [case_details.get('bom_accurate', 1)],
            'Setup_Sheet_Accuracy': [case_details.get('setup_sheet_accurate', 1)],
            'Visual_Aid_Accuracy': [case_details.get('visual_aid_accurate', 0)],
            'Operator_Training': [case_details.get('operator_trained', 1)],
            'Part_Verification_Process': [case_details.get('part_verification_done', 0)],
            'Line_Stoppage_Protocol': [case_details.get('line_stopped_correctly', 1)],
            'Correct_Part_Usage': [case_details.get('correct_part_used', 0)],
            'Production_Impact': [case_details.get('production_impact', 1)],
            'Work_Instruction_Accuracy': [1]  # Added this missing node
        })
    
    def _call_azure_openai(self, prompt: str) -> str:
        """Make API call to Azure OpenAI"""
        response = self.client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    
    def _extract_case_details(self, case_text: str) -> Dict[str, Any]:
        """Extract structured details using Azure OpenAI"""
        prompt = f"""Analyze this manufacturing CAPA case and return ONLY a valid JSON object:
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
        
        response = self._call_azure_openai(prompt)
        return json.loads(response)
    
    def analyze_case(self, case_text: str) -> Dict[str, Any]:
        """Complete RCA workflow"""
        case_details = self._extract_case_details(case_text)
        data = self._generate_synthetic_data(case_details)
        
        gcm.auto.assign_causal_mechanisms(self.scm, data)
        gcm.fit(self.scm, data)
        
        attributions = gcm.attribute_anomalies(
            self.scm, 
            target_node='Production_Impact',
            anomaly_samples=data
        )
        
        return {
            'case_details': case_details,
            'causal_attributions': {k: float(v[0]) for k, v in attributions.items()},
            'recommendations': self._generate_recommendations(case_details, attributions)
        }
    
    def _generate_recommendations(self, case_details: Dict[str, Any], attributions: Dict[str, Any]) -> str:
        """Generate CAPA recommendations"""
        prompt = f"""Generate CAPA recommendations in this format:
1. Root Cause: [clear explanation]
2. Corrective Actions: [bullet points]
3. Preventive Actions: [bullet points]
4. Verification: [methods]
5. Test Cases: [specific validations]

Case Analysis:
{json.dumps(case_details, indent=2)}

Causal Factors:
{json.dumps(attributions, indent=2)}"""
        
        return self._call_azure_openai(prompt)

def initialize_azure_client():
    """Initialize Azure OpenAI client"""
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
        
        case_text = """On Jan 12, 2021, the MESE1 Manufacturing Line was stopped due to discrepancy between Visual Aids and Setup Sheet, resulting in incorrect screw P/N used at Build Station 4. Work instruction OPER-WI-086 Rev C specifies P/N 5600203-01, but P/N 5600008-03 was used. The BOM and Setup Sheet are correct, but the Visual Aid rev 003 shows the wrong P/N."""
        
        results = analyzer.analyze_case(case_text)
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        print(f"Error: {str(e)}")
        





        ********************************
        *************************************
        ***************************************
        ************************************************
        import pandas as pd
import networkx as nx
from dowhy import gcm
from typing import Dict, Any
import json
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

class ManufacturingRCAAnalyzer:
    def __init__(self, azure_openai_client):
        """Initialize with Azure OpenAI client and causal graph"""
        self.common_causal_graph = self._build_common_causal_graph()
        self.scm = gcm.StructuralCausalModel(self.common_causal_graph)
        self.client = azure_openai_client
        
    def _build_common_causal_graph(self) -> nx.DiGraph:
        """Build base causal graph for manufacturing issues"""
        return nx.DiGraph([
            ('Document Version Control', 'Work Instruction Accuracy'),
            ('BOM Accuracy', 'Work Instruction Accuracy'),
            ('Setup Sheet Accuracy', 'Work Instruction Accuracy'),
            ('Visual Aid Accuracy', 'Work Instruction Accuracy'),
            ('Operator Training', 'Correct Part Usage'),
            ('Work Instruction Accuracy', 'Correct Part Usage'),
            ('Part Verification Process', 'Correct Part Usage'),
            ('Line Stoppage Protocol', 'Production Impact'),
            ('Correct Part Usage', 'Production Impact'),
            ('Work Instruction Accuracy', 'Production Impact')
        ])
    
    def _generate_synthetic_data(self, case_details: Dict[str, Any]) -> pd.DataFrame:
        """Generate synthetic data based on case details"""
        return pd.DataFrame({
            'Document Version Control': [case_details.get('document_version_issue', 0)],
            'BOM Accuracy': [case_details.get('bom_accurate', 1)],
            'Setup Sheet Accuracy': [case_details.get('setup_sheet_accurate', 1)],
            'Visual Aid Accuracy': [case_details.get('visual_aid_accurate', 0)],
            'Operator Training': [case_details.get('operator_trained', 1)],
            'Part Verification Process': [case_details.get('part_verification_done', 0)],
            'Line Stoppage Protocol': [case_details.get('line_stopped_correctly', 1)],
            'Correct Part Usage': [case_details.get('correct_part_used', 0)],
            'Production Impact': [case_details.get('production_impact', 1)]
        })
    
    def _call_azure_openai(self, prompt: str) -> str:
        """Make API call to Azure OpenAI"""
        response = self.client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    
    def _extract_case_details(self, case_text: str) -> Dict[str, Any]:
        """Extract structured details using Azure OpenAI"""
        prompt = f"""Analyze this manufacturing CAPA case and return ONLY a valid JSON object:
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
        
        response = self._call_azure_openai(prompt)
        return json.loads(response)
    
    def analyze_case(self, case_text: str) -> Dict[str, Any]:
        """Complete RCA workflow"""
        case_details = self._extract_case_details(case_text)
        data = self._generate_synthetic_data(case_details)
        
        gcm.auto.assign_causal_mechanisms(self.scm, data)
        gcm.fit(self.scm, data)
        
        attributions = gcm.attribute_anomalies(
            self.scm, 
            target_node='Production Impact',
            anomaly_samples=data
        )
        
        return {
            'case_details': case_details,
            'causal_attributions': {k: float(v[0]) for k, v in attributions.items()},
            'recommendations': self._generate_recommendations(case_details, attributions)
        }
    
    def _generate_recommendations(self, case_details: Dict[str, Any], attributions: Dict[str, Any]) -> str:
        """Generate CAPA recommendations"""
        prompt = f"""Generate CAPA recommendations in this format:
1. Root Cause: [clear explanation]
2. Corrective Actions: [bullet points]
3. Preventive Actions: [bullet points]
4. Verification: [methods]
5. Test Cases: [specific validations]

Case Analysis:
{json.dumps(case_details, indent=2)}

Causal Factors:
{json.dumps(attributions, indent=2)}"""
        
        return self._call_azure_openai(prompt)

def initialize_azure_client():
    """Initialize Azure OpenAI client"""
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
        
        case_text = """On Jan 12, 2021, the MESE1 Manufacturing Line was stopped due to discrepancy between Visual Aids and Setup Sheet, resulting in incorrect screw P/N used at Build Station 4. Work instruction OPER-WI-086 Rev C specifies P/N 5600203-01, but P/N 5600008-03 was used. The BOM and Setup Sheet are correct, but the Visual Aid rev 003 shows the wrong P/N."""
        
        results = analyzer.analyze_case(case_text)
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        print(f"Error: {str(e)}")
        
 **************************
*********************************
************************************
**************************************
        import pandas as pd
import networkx as nx
from dowhy import gcm
from typing import Dict, Any
import json
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

class ManufacturingRCAAnalyzer:
    def __init__(self, azure_openai_client):
        self.common_causal_graph = self._build_common_causal_graph()
        self.scm = gcm.StructuralCausalModel(self.common_causal_graph)
        self.client = azure_openai_client
        
    # [Previous methods remain the same until _extract_case_details]
    
    def _extract_case_details(self, case_text: str) -> Dict[str, Any]:
        """Use Azure OpenAI to extract structured details from case text"""
        prompt = f"""
        Analyze this manufacturing CAPA case and extract relevant details as a valid JSON object:
        
        {case_text}
        
        Return ONLY a valid JSON object with these exact fields:
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
        """
        
        try:
            response = self.client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"}  # Ensure JSON response
            )
            
            # Extract the content and validate JSON
            json_str = response.choices[0].message.content
            if not json_str.strip():
                raise ValueError("Empty response from Azure OpenAI")
                
            return json.loads(json_str)
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Response content: {json_str}")
            raise
        except Exception as e:
            print(f"Error calling Azure OpenAI: {e}")
            raise

    # [Rest of the class remains the same]

def initialize_azure_client():
    """Initialize and return Azure OpenAI client"""
    load_dotenv()  # Load environment variables
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

if __name__ == "__main__":
    try:
        client = initialize_azure_client()
        analyzer = ManufacturingRCAAnalyzer(client)
        
        case_text = """
        On Jan 12, 2021, the MESE1 Manufacturing Line at External Manufacturer Jabil was stopped by AUH (Auburn Hills) Manufacturing due to discrepancy between the mechanical assembly Visual Aids and the Setup Sheet, resulting in incorrect screw P/N used at MESE1-12-G Build Station 4. It was found that Step 13 of work instruction OPER-WI-086 Rev C states to use P/N 5600203-01 to secure the filter Canister Bracket to the Chasis. P/N 5600008-03 was used instead. The BOM and setup Sheet have the correct P/N for assembly, but the AUH Work Instruction Visual Aid rev 003 references the incorrect 5600008-03. EES was notified on Jan 14, 2021.
        """
        
        results = analyzer.analyze_case(case_text)
        print("Root Cause Analysis Results:")
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")


************
****************************
*********************************


import pandas as pd
import networkx as nx
from dowhy import gcm
from typing import Dict, Any
import json
import os
from openai import AzureOpenAI

class ManufacturingRCAAnalyzer:
    def __init__(self, azure_openai_client):
        """Initialize with Azure OpenAI client and causal graph"""
        self.common_causal_graph = self._build_common_causal_graph()
        self.scm = gcm.StructuralCausalModel(self.common_causal_graph)
        self.client = azure_openai_client
        
    def _build_common_causal_graph(self) -> nx.DiGraph:
        """Build a base causal graph for manufacturing issues"""
        causal_graph = nx.DiGraph([
            ('Document Version Control', 'Work Instruction Accuracy'),
            ('BOM Accuracy', 'Work Instruction Accuracy'),
            ('Setup Sheet Accuracy', 'Work Instruction Accuracy'),
            ('Visual Aid Accuracy', 'Work Instruction Accuracy'),
            ('Operator Training', 'Correct Part Usage'),
            ('Work Instruction Accuracy', 'Correct Part Usage'),
            ('Part Verification Process', 'Correct Part Usage'),
            ('Line Stoppage Protocol', 'Production Impact'),
            ('Correct Part Usage', 'Production Impact'),
            ('Work Instruction Accuracy', 'Production Impact')
        ])
        return causal_graph
    
    def _generate_synthetic_data(self, case_details: Dict[str, Any]) -> pd.DataFrame:
        """Generate synthetic data based on case details"""
        data = {
            'Document Version Control': [case_details.get('document_version_issue', 0)],
            'BOM Accuracy': [case_details.get('bom_accurate', 1)],
            'Setup Sheet Accuracy': [case_details.get('setup_sheet_accurate', 1)],
            'Visual Aid Accuracy': [case_details.get('visual_aid_accurate', 0)],
            'Operator Training': [case_details.get('operator_trained', 1)],
            'Part Verification Process': [case_details.get('part_verification_done', 0)],
            'Line Stoppage Protocol': [case_details.get('line_stopped_correctly', 1)],
            'Correct Part Usage': [case_details.get('correct_part_used', 0)],
            'Production Impact': [case_details.get('production_impact', 1)]
        }
        return pd.DataFrame(data)
    
    def _call_azure_openai(self, prompt: str) -> str:
        """Make API call to Azure OpenAI"""
        response = self.client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    
    def _extract_case_details(self, case_text: str) -> Dict[str, Any]:
        """Extract structured details using Azure OpenAI"""
        prompt = f"""Analyze this manufacturing CAPA case and extract relevant details in JSON format:
        
        {case_text}
        
        Return JSON with these fields:
        - document_version_issue (bool)
        - bom_accurate (bool)
        - setup_sheet_accurate (bool)
        - visual_aid_accurate (bool)
        - operator_trained (bool)
        - part_verification_done (bool)
        - line_stopped_correctly (bool)
        - correct_part_used (bool)
        - production_impact (int)
        - root_cause_hypothesis (str)"""
        
        response = self._call_azure_openai(prompt)
        return json.loads(response)
    
    def analyze_case(self, case_text: str) -> Dict[str, Any]:
        """End-to-end case analysis"""
        case_details = self._extract_case_details(case_text)
        data = self._generate_synthetic_data(case_details)
        
        gcm.auto.assign_causal_mechanisms(self.scm, data)
        gcm.fit(self.scm, data)
        
        attributions = gcm.attribute_anomalies(
            self.scm, 
            target_node='Production Impact',
            anomaly_samples=data
        )
        
        recommendations = self._generate_recommendations(case_details, attributions)
        
        return {
            'case_details': case_details,
            'causal_attributions': attributions,
            'recommendations': recommendations
        }
    
    def _generate_recommendations(self, case_details: Dict[str, Any], attributions: Dict[str, Any]) -> str:
        """Generate CAPA recommendations"""
        prompt = f"""Based on this analysis, generate specific CAPA recommendations:
        
        Case Details:
        {json.dumps(case_details, indent=2)}
        
        Causal Attributions:
        {json.dumps({k: float(v[0]) for k, v in attributions.items()}, indent=2)}
        
        Provide:
        1. Root cause confirmation
        2. Corrective actions
        3. Preventive actions
        4. Verification methods
        5. Test cases"""
        
        return self._call_azure_openai(prompt)

def initialize_azure_client():
    """Initialize Azure OpenAI client"""
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    try:
        # Initialize client and analyzer
        client = initialize_azure_client()
        analyzer = ManufacturingRCAAnalyzer(client)
        
        # Example case
        case_text = """On Jan 12, 2021..."""  # Your case text here
        
        # Perform analysis
        results = analyzer.analyze_case(case_text)
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        print(f"Error: {str(e)}")
        
        *****************************************
        *************************************
        
        
        import pandas as pd
import networkx as nx
from dowhy import gcm
from typing import Dict, Any
import openai  # or your preferred LLM provider

class ManufacturingRCAAnalyzer:
    def __init__(self):
        # Initialize with common manufacturing causal knowledge
        self.common_causal_graph = self._build_common_causal_graph()
        self.scm = gcm.StructuralCausalModel(self.common_causal_graph)
        
    def _build_common_causal_graph(self) -> nx.DiGraph:
        """Build a base causal graph for manufacturing issues"""
        causal_graph = nx.DiGraph([
            # Document-related factors
            ('Document Version Control', 'Work Instruction Accuracy'),
            ('BOM Accuracy', 'Work Instruction Accuracy'),
            ('Setup Sheet Accuracy', 'Work Instruction Accuracy'),
            ('Visual Aid Accuracy', 'Work Instruction Accuracy'),
            
            # Process-related factors
            ('Operator Training', 'Correct Part Usage'),
            ('Work Instruction Accuracy', 'Correct Part Usage'),
            ('Part Verification Process', 'Correct Part Usage'),
            ('Line Stoppage Protocol', 'Production Impact'),
            
            # Outcome
            ('Correct Part Usage', 'Production Impact'),
            ('Work Instruction Accuracy', 'Production Impact')
        ])
        return causal_graph
    
    def _generate_synthetic_data(self, case_details: Dict[str, Any]) -> pd.DataFrame:
        """Generate synthetic data based on case details for analysis"""
        # This would be enhanced with real data in production
        data = {
            'Document Version Control': [case_details.get('document_version_issue', 0)],
            'BOM Accuracy': [case_details.get('bom_accurate', 1)],
            'Setup Sheet Accuracy': [case_details.get('setup_sheet_accurate', 1)],
            'Visual Aid Accuracy': [case_details.get('visual_aid_accurate', 0)],
            'Operator Training': [case_details.get('operator_trained', 1)],
            'Part Verification Process': [case_details.get('part_verification_done', 0)],
            'Line Stoppage Protocol': [case_details.get('line_stopped_correctly', 1)],
            'Correct Part Usage': [case_details.get('correct_part_used', 0)],
            'Production Impact': [case_details.get('production_impact', 1)]
        }
        return pd.DataFrame(data)
    
    def _extract_case_details(self, case_text: str) -> Dict[str, Any]:
        """Use LLM to extract structured details from case text"""
        prompt = f"""
        Analyze this manufacturing CAPA case and extract relevant details in JSON format:
        
        {case_text}
        
        Return JSON with these fields:
        - document_version_issue (bool): Was there a document version control issue?
        - bom_accurate (bool): Was the BOM accurate?
        - setup_sheet_accurate (bool): Was the setup sheet accurate?
        - visual_aid_accurate (bool): Was the visual aid accurate?
        - operator_trained (bool): Was the operator properly trained?
        - part_verification_done (bool): Was part verification performed?
        - line_stopped_correctly (bool): Was the line stopped correctly when issue found?
        - correct_part_used (bool): Was the correct part used?
        - production_impact (int): Severity of production impact (0-10)
        - root_cause_hypothesis (str): Initial hypothesis of root cause
        """
        
        # Call to LLM (pseudo-code - implement with your preferred LLM API)
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return json.loads(response.choices[0].message.content)
    
    def analyze_case(self, case_text: str) -> Dict[str, Any]:
        """End-to-end analysis of a manufacturing case"""
        # Step 1: Extract structured data from case text using LLM
        case_details = self._extract_case_details(case_text)
        
        # Step 2: Generate synthetic data based on case details
        data = self._generate_synthetic_data(case_details)
        
        # Step 3: Fit the causal model
        gcm.auto.assign_causal_mechanisms(self.scm, data)
        gcm.fit(self.scm, data)
        
        # Step 4: Perform root cause analysis
        attributions = gcm.attribute_anomalies(
            self.scm, 
            target_node='Production Impact',
            anomaly_samples=data
        )
        
        # Step 5: Generate recommendations
        recommendations = self._generate_recommendations(case_details, attributions)
        
        return {
            'case_details': case_details,
            'causal_attributions': attributions,
            'recommendations': recommendations
        }
    
    def _generate_recommendations(self, case_details: Dict[str, Any], attributions: Dict[str, Any]) -> str:
        """Generate actionable recommendations based on analysis"""
        prompt = f"""
        Based on this manufacturing case analysis, generate specific CAPA recommendations:
        
        Case Details:
        {json.dumps(case_details, indent=2)}
        
        Causal Attributions:
        {json.dumps({k: float(v[0]) for k, v in attributions.items()}, indent=2)}
        
        Provide:
        1. Root cause confirmation
        2. Immediate corrective actions
        3. Preventive actions
        4. Verification methods
        5. Test cases to validate the fix
        """
        
        # Call to LLM
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content

##Example Usage


# Initialize the analyzer
analyzer = ManufacturingRCAAnalyzer()

# Example case text
case_text = """
On Jan 12, 2021, the MESE1 Manufacturing Line at External Manufacturer Jabil was stopped by AUH (Auburn Hills) Manufacturing due to discrepancy between the mechanical assembly Visual Aids and the Setup Sheet, resulting in incorrect screw P/N used at MESE1-12-G Build Station 4. It was found that Step 13 of work instruction OPER-WI-086 Rev C states to use P/N 5600203-01 to secure the filter Canister Bracket to the Chasis. P/N 5600008-03 was used instead. The BOM and setup Sheet have the correct P/N for assembly, but the AUH Work Instruction Visual Aid rev 003 references the incorrect 5600008-03. EES was notified on Jan 14, 2021.
"""

# Perform analysis
results = analyzer.analyze_case(case_text)

# Print results
print("Root Cause Analysis Results:")
print(json.dumps(results, indent=2))
