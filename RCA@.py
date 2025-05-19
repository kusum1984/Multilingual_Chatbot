
**************************


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
    """Enhanced RCA analyzer with improved reporting and visualization."""
    
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
    """Generates professional PDF report with consistent factor presentation."""
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
    impact_path = self.visualize_impact_graph(results['causal_influence'])
    
    story = []
    
    # Title
    story.append(Paragraph("Manufacturing Root Cause Analysis Report", styles['Title']))
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
    
    # Impact Graph
    story.append(Paragraph("System-Wide Causal Influence", styles['Heading2']))
    story.append(Image(impact_path, width=6*inch, height=4.5*inch))
    story.append(Spacer(1, 12))
    
    # Key Contributing Factors - Anomaly Attribution
    story.append(Paragraph("Incident-Specific Factor Contributions", styles['Heading2']))
    if results['causal_attributions']:
        causal_factors = {
            k: float(v[0]) if isinstance(v, (list, np.ndarray)) else float(v)
            for k, v in results['causal_attributions'].items()
        }
        top_factors = sorted(causal_factors.items(), 
                           key=lambda x: abs(x[1]), reverse=True)[:5]  # Show top 5
        
        factors_data = [["Factor", "Contribution Score"]] + [
            [k.replace('_', ' ').title(), f"{v:.4f}"] 
            for k, v in top_factors
        ]
        
        factors_table = Table(factors_data, colWidths=[3*inch, 2*inch])
        factors_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(factors_table)
        story.append(Spacer(1, 6))
        
        # Add explanation
        explanation = """
        <b>Interpretation:</b> Contribution scores indicate how much each factor contributed 
        to this specific incident. Scores range from -1 to 1, where positive values indicate 
        the factor increased the likelihood of the issue, and negative values indicate it 
        had a protective effect.
        """
        story.extend(self._generate_wrapped_paragraph(explanation, wrapped_style))
    story.append(Spacer(1, 12))
    
    # System Influence Summary
    story.append(Paragraph("System Influence Summary", styles['Heading2']))
    influence_text = """
    The graph above shows each factor's normalized influence on Production Impact across 
    the entire manufacturing system. Larger nodes indicate factors with greater systemic 
    influence, regardless of their role in this specific incident.
    """
    story.extend(self._generate_wrapped_paragraph(influence_text, wrapped_style))
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
        
        # Recommendations with wrapping
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
    
    output.append("\n=== TOTAL CAUSAL INFLUENCE ON PRODUCTION IMPACT ===")
    for factor, score in results['causal_influence'].items():
        output.append(f"{factor.replace('_', ' ').title():<25}: {score:.4f}")
    
    output.append("\n=== RECOMMENDATIONS ===")
    output.append(results['recommendations'])
    
    return "\n".join(output)

if __name__ == "__main__":
    """Main execution flow with PDF reporting."""
    try:
        client = initialize_azure_client()
        analyzer = ManufacturingRCAAnalyzer(client)
        
        case_text = """On Jan 12, 2021, the MESE1 Manufacturing Line was stopped due to discrepancy between Visual Aids and Setup Sheet, resulting in incorrect screw P/N used at Build Station 4. Work instruction OPER-WI-086 Rev C specifies P/N 5600203-01, but P/N 5600008-03 was used. The BOM and Setup Sheet are correct, but the Visual Aid rev 003 shows the wrong P/N."""
        
        results = analyzer.analyze_case(case_text)
        
        data = analyzer._generate_realistic_synthetic_data(results['case_details'])
        data_path = analyzer.export_synthetic_data(data)
        pdf_path = analyzer.generate_pdf_report(results, "", data_path)
        
        print(format_plaintext_output(results))
        print(f"\nReport generated: {pdf_path}")

    except Exception as e:
        print(f"Error: {str(e)}")
