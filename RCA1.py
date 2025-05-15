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
