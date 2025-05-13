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
