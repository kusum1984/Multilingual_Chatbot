import os
import json
import threading
import traceback
import urllib.parse
from typing import List, Dict, Any, Literal, Optional
from dataclasses import dataclass, field
from concurrent.futures import TimeoutError
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool, Tool
from langgraph.graph import END, StateGraph
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel as PydanticBaseModel
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_sql_agent
from functools import lru_cache
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from sqlalchemy import create_engine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
class Config:
    def __init__(self):
        # Elasticsearch configuration
        self.elastic_index_data_from = 0
        self.elastic_index_data_size = 10
        self.elastic_index_data_max_size = 50
        self.aggs_limit = 5
        self.max_search_retries = 3
        self.token_limit = 3000
        self.tool_timeout = 60  # seconds
       
        # Initialize Elasticsearch client with API key
        self.es = Elasticsearch(
            os.getenv("ELASTIC_ENDPOINT"),
            api_key=os.getenv("ELASTIC_API_KEY"),
            verify_certs=False
        )
       
        # Initialize LLM - using Azure
        self.llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name="gpt-4",
            model="gpt-4",
            temperature=0
        )
       
        # SQL Server connection setup
        driver = '{ODBC Driver 18 for SQL Server}'
        server = os.getenv("SQL_SERVER")
        database = os.getenv("SQL_DATABASE")
        user = os.getenv("SQL_USER")
        password = os.getenv("SQL_PASSWORD")
        
        conn = f"Driver={driver};Server=tcp:{server},1433;Database={database};Uid={user};Pwd={password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
        params = urllib.parse.quote_plus(conn)
        conn_str = f'mssql+pyodbc:///?autocommit=true&odbc_connect={params}'
        self.engine = create_engine(conn_str, echo=True)
        
        # Other configurations
        self.EXCEL_FILE_PATH = os.getenv("EXCEL_FILE_PATH", "data.xlsx")
        self.langchain_verbose = True

    @property
    def sql_db(self):
        """Create SQLDatabase instance with the configured engine"""
        return SQLDatabase(self.engine, schema="comp")

cfg = Config()

# Input Models
class ExcelSheetInput(BaseModel):
    sheet_name: Optional[str] = Field(None, description="Name of the Excel sheet to read")

class ListIndicesInput(BaseModel):
    separator: str = Field(", ", description="Separator for the list of indices")

class IndexDetailsInput(BaseModel):
    index_name: str = Field(..., description="Name of the Elasticsearch index")

class IndexDataInput(BaseModel):
    index_name: str = Field(..., description="Name of the Elasticsearch index")
    size: int = Field(5, description="Number of documents to retrieve")

class SearchToolInput(BaseModel):
    index_name: str = Field(..., description="Name of the Elasticsearch index")
    query: str = Field(..., description="Elasticsearch JSON query")
    from_: int = Field(0, description="Starting record index")
    size: int = Field(10, description="Number of records to retrieve")

# Agent State
@dataclass
class AgentState:
    question: str
    plan: Optional[str] = None
    data_sources: Optional[List[Literal["elasticsearch", "sql", "excel"]]] = None
    retrieved_data: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None
    final_answer: Optional[str] = None
    follow_up_questions: Optional[List[str]]] = None
    conversation_history: List[BaseMessage] = field(default_factory=list)
    raw_plan_output: Optional[str] = None
    errors: List[str] = field(default_factory=list)

    def update_from_planner(self, raw_output: str):
        self.raw_plan_output = raw_output
        try:
            # Extract JSON from markdown code block if present
            json_str = raw_output.split("```json")[1].split("```")[0].strip() if "```json" in raw_output else raw_output
            plan_data = json.loads(json_str)
            
            # Validate data_sources
            valid_sources = {"elasticsearch", "sql", "excel"}
            self.data_sources = [
                src for src in plan_data.get("data_sources", []) 
                if src in valid_sources
            ]
            
            if not self.data_sources:
                self.data_sources = list(valid_sources)
                self.errors.append("No valid data sources specified in plan - using all sources")
            
            self.plan = plan_data.get("plan", "No specific execution plan provided")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse planner output: {raw_output}")
            self.errors.append(f"Plan parsing error: {str(e)}")
            self.plan = "Fallback plan: 1. Search Elasticsearch\n2. Query SQL\n3. Check Excel"
            self.data_sources = ["elasticsearch", "sql", "excel"]
        except Exception as e:
            logger.error(f"Unexpected error parsing plan: {str(e)}")
            self.errors.append(f"Unexpected planning error: {str(e)}")
            self.plan = "Fallback plan: 1. Search Elasticsearch\n2. Query SQL\n3. Check Excel"
            self.data_sources = ["elasticsearch", "sql", "excel"]

# Complete Elasticsearch Tools Implementation (4 Tools)
class ElasticsearchTools:
    def __init__(self, es_client):
        self.es_client = es_client
        self.cfg = cfg

    def list_indices(self, separator: str = ", ") -> str:
        """Lists all available Elasticsearch indices"""
        try:
            indices = list(self.es_client.indices.get(index="*").keys())
            return f"Available indices:{separator}{separator.join(indices)}"
        except Exception as e:
            logger.error(f"Error listing indices: {str(e)}")
            return f"Error listing indices: {str(e)}"

    def get_index_details(self, index_name: str) -> str:
        """Gets details about a specific index including mappings and settings"""
        try:
            if not self.es_client.indices.exists(index=index_name):
                return f"Index '{index_name}' does not exist"
            
            details = {
                "aliases": self.es_client.indices.get_alias(index=index_name).get(index_name, {}).get("aliases", {}),
                "mappings": self.es_client.indices.get_mapping(index=index_name).get(index_name, {}).get("mappings", {}),
                "settings": self.es_client.indices.get_settings(index=index_name).get(index_name, {}).get("settings", {})
            }
            return json.dumps(details, indent=2)
        except Exception as e:
            logger.error(f"Error getting index details: {str(e)}")
            return f"Error getting index details: {str(e)}"

    def get_index_data(self, index_name: str, size: int = 5) -> str:
        """Gets sample documents from an index to understand its structure"""
        try:
            if not self.es_client.indices.exists(index=index_name):
                return f"Index '{index_name}' does not exist"
            
            result = self.es_client.search(
                index=index_name,
                body={"query": {"match_all": {}}},
                size=min(size, self.cfg.elastic_index_data_max_size)
            )
            hits = result.get('hits', {}).get('hits', [])
            
            if not hits:
                return f"No documents found in index '{index_name}'"
            
            return json.dumps([hit['_source'] for hit in hits[:size]], indent=2)
        except Exception as e:
            logger.error(f"Error getting index data: {str(e)}")
            return f"Error getting index data: {str(e)}"

    def elastic_search(self, index_name: str, query: str, from_: int = 0, size: int = 10) -> str:
        """Executes a search or aggregation query on an Elasticsearch index"""
        try:
            if not self.es_client.indices.exists(index=index_name):
                return f"Index '{index_name}' does not exist"
            
            size = min(self.cfg.elastic_index_data_max_size, size)
            
            try:
                query_dict = json.loads(query)
            except json.JSONDecodeError:
                return "Invalid query format - must be valid JSON"
            
            # Determine if this is a search or aggregation
            is_aggregation = "aggs" in query_dict or "aggregations" in query_dict
            
            if is_aggregation:
                size = self.cfg.aggs_limit
            
            result = self.es_client.search(
                index=index_name,
                body=query_dict,
                from_=from_,
                size=size
            )
            
            if is_aggregation:
                return json.dumps(result.get('aggregations', {}), indent=2)
            else:
                return json.dumps(result.get('hits', {}), indent=2)
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return f"Search error: {str(e)}"

# Excel Tools Implementation
class ExcelTools:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_excel_sheet(self, sheet_name: Optional[str] = None) -> str:
        try:
            loader = UnstructuredExcelLoader(
                file_path=self.file_path,
                mode="elements",
                sheet_name=sheet_name
            )
            docs = loader.load()
            
            if not docs:
                return "No data found in the Excel sheet"
            
            output = [f"Data from Excel sheet '{sheet_name or 'default'}':"]
            for doc in docs[:3]:  # Show first 3 rows
                content = doc.page_content.replace("\n", " | ")
                output.append(f"- {content[:200]}{'...' if len(content) > 200 else ''}")
            
            if len(docs) > 3:
                output.append(f"\n... and {len(docs)-3} more rows")
            
            return "\n".join(output)
        except Exception as e:
            logger.error(f"Error reading Excel file: {str(e)}")
            return f"Error reading Excel file: {str(e)}"

# Shared Memory Manager
class MemoryManager:
    def __init__(self):
        self.memory = ConversationBufferMemory(return_messages=True)
        self.lock = threading.Lock()

    def get_context(self) -> List[BaseMessage]:
        with self.lock:
            return self.memory.chat_memory.messages.copy()

    def update_context(self, user_input: str, agent_output: str):
        with self.lock:
            self.memory.save_context(
                {"input": user_input},
                {"output": agent_output}
            )

# Enhanced Prompt Manager with Detailed Agent Prompts
class PromptManager:
    _templates = {
        "planner": {
            "system": """You are an expert planning agent with deep knowledge of data systems. Your responsibilities are:

1. Analyze the user's question thoroughly:
   - Identify key entities, timeframes, and relationships
   - Detect implicit requirements and context
   - Note any technical terms that indicate specific data sources

2. Determine optimal data sources from these options:
   - Elasticsearch: For text search, log analysis, document retrieval, and unstructured data
     * Use when question contains: search, find, analyze text, logs, documents
   - SQL Database: For structured data queries, record counts, joins, and aggregations
     * Use when question contains: tables, records, count, sum, database
   - Excel: For spreadsheet data, column/row analysis, and file-based data
     * Use when question contains: spreadsheet, sheet, excel, column, row

3. Create a detailed execution plan with:
   - Clear sequencing of operations
   - Justification for each data source selection
   - Fallback options if primary sources fail

Output MUST follow this exact JSON format:
```json
{{
    "data_sources": ["elasticsearch", "sql", "excel"],
    "plan": "1. First search Elasticsearch for...\\n2. Then query SQL database for...\\n3. Finally check Excel for..."
}}
```""",
            "user": "User Question: {question}\nConversation History:\n{history}"
        },
        "elastic": {
            "system": """You are a senior Elasticsearch specialist with these capabilities:
{tools}

Operational Protocol:
1. Initial Assessment Phase:
   - Always begin by listing available indices (elastic_list_indices)
   - Review index mappings (elastic_index_details) for relevant fields

2. Investigation Phase:
   - Examine sample documents (elastic_index_data) to understand structure
   - Identify key fields that match the question requirements

3. Execution Phase:
   - Construct precise queries using:
     * query_string for text search
     * bool for complex conditions
     * aggregations for analytics
   - Validate query syntax before execution

4. Safety Checks:
   - Limit result sizes to prevent overload
   - Handle errors gracefully with clear explanations
   - Never execute unvalidated queries""",
            "user": """Task Details:
Question: {input}
Plan: {plan}

Provide raw search results with minimal transformation."""
        },
        "sql": {
            "system": """You are a lead SQL analyst with these resources:
{tools}

Execution Framework:
1. Schema Analysis:
   - First understand table relationships
   - Identify relevant columns and constraints

2. Query Construction:
   - Use proper JOIN syntax for relationships
   - Apply appropriate filtering (WHERE clauses)
   - Include necessary aggregations (GROUP BY)

3. Optimization:
   - Limit result sets for efficiency
   - Add sensible pagination
   - Include EXPLAIN for complex queries

4. Safety Protocols:
   - Parameterize all queries
   - Never concatenate raw input
   - Validate schema exists before querying""",
            "user": """Task Assignment:
Question: {input}
Plan: {plan}

Return query results in their raw format."""
        },
        "excel": {
            "system": """You are an Excel data architect with these tools:
{tools}

Workflow Guidelines:
1. File Assessment:
   - First identify which sheets contain relevant data
   - Analyze header rows for column meanings

2. Data Extraction:
   - Focus on specific ranges when possible
   - Handle merged cells carefully
   - Preserve original formatting where needed

3. Transformation:
   - Clean data as necessary (handle NA values)
   - Maintain data types during operations
   - Document any assumptions made

4. Safety Measures:
   - Never modify source files
   - Handle large files with care
   - Validate sheet names before access""",
            "user": """Task Parameters:
Question: {input}
Plan: {plan}

Provide extracted data exactly as it appears in the file."""
        },
        "summarizer": {
            "system": """You are a principal data synthesizer with these responsibilities:

1. Data Integration:
   - Correlate information from all sources
   - Resolve conflicts between sources
   - Identify patterns and insights

2. Analysis:
   - Highlight statistically significant findings
   - Note any data quality issues
   - Flag incomplete information

3. Synthesis:
   - Structure information logically
   - Maintain original meaning while simplifying
   - Preserve important technical details

4. Quality Control:
   - Verify all claims against source data
   - Maintain neutral, factual tone
   - Avoid introducing new information""",
            "user": """Synthesis Task:
Original Question: {question}
Source Data:
{retrieved_data}

Produce a comprehensive technical summary."""
        },
        "decision": {
            "system": """You are the chief response architect with these mandates:

1. Content Refinement:
   - Transform technical summary into clear response
   - Structure information for optimal understanding
   - Maintain perfect accuracy

2. Presentation:
   - Begin with direct answer to the question
   - Follow with supporting evidence
   - Use bullet points for complex information

3. Tone Management:
   - Adapt to user's apparent technical level
   - Remain professional yet approachable
   - Avoid unnecessary jargon

4. Completeness Check:
   - Ensure all question aspects are addressed
   - Note any limitations in the data
   - Provide clear next steps when appropriate""",
            "user": """Response Crafting:
Original Question: {question}
Technical Summary: {summary}

Create the final user-facing response."""
        },
        "followup": {
            "system": """You are a strategic conversation designer with these goals:

1. Question Development:
   - Create natural progressions from current discussion
   - Cover logical adjacent topics
   - Explore deeper technical dimensions

2. Quality Standards:
   - Each question must be answerable
   - Avoid yes/no questions
   - Maintain relevance to original topic

3. Formatting:
   - Present as numbered list
   - Keep each question under 15 words
   - Ensure grammatical correctness

4. Strategic Value:
   - Anticipate user needs
   - Guide toward valuable insights
   - Maintain conversation flow""",
            "user": """Follow-up Generation:
Original Question: {question}
Final Answer: {final_answer}

Produce exactly 3 high-quality follow-up questions."""
        }
    }

    @classmethod
    def get_prompt(cls, agent_type: str, tools: List[BaseTool] = None) -> ChatPromptTemplate:
        if agent_type not in cls._templates:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        template = cls._templates[agent_type]
        if tools and "{tools}" in template["system"]:
            tool_list = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)
            system_msg = template["system"].replace("{tools}", tool_list)
        else:
            system_msg = template["system"]
        
        return ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("user", template["user"])
        ])

# Initialize shared resources
memory_manager = MemoryManager()

# Cached Tool Initialization
@lru_cache(maxsize=None)
def get_elastic_tools():
    es_tools = ElasticsearchTools(cfg.es)
    return [
        Tool.from_function(
            func=es_tools.list_indices,
            name="elastic_list_indices",
            description="Lists all available Elasticsearch indices. Always call this first.",
            args_schema=ListIndicesInput
        ),
        Tool.from_function(
            func=es_tools.get_index_details,
            name="elastic_index_details",
            description="Gets details about a specific index including mappings and settings",
            args_schema=IndexDetailsInput
        ),
        Tool.from_function(
            func=es_tools.get_index_data,
            name="elastic_index_data",
            description="Gets sample documents from an index to understand its structure",
            args_schema=IndexDataInput
        ),
        Tool.from_function(
            func=es_tools.elastic_search,
            name="elastic_search",
            description="Executes search or aggregation queries on an Elasticsearch index",
            args_schema=SearchToolInput
        )
    ]

@lru_cache(maxsize=None)
def get_sql_tools():
    return SQLDatabaseToolkit(db=cfg.sql_db, llm=cfg.llm).get_tools()

@lru_cache(maxsize=None)
def get_excel_tools():
    return [
        Tool.from_function(
            func=ExcelTools(cfg.EXCEL_FILE_PATH).read_excel_sheet,
            name="excel_reader",
            description="Read data from Excel sheets",
            args_schema=ExcelSheetInput
        )
    ]

# Agent Factory
def create_agent(agent_type: str, tools: List[BaseTool]) -> AgentExecutor:
    if agent_type == "sql":
        agent = create_sql_agent(
            llm=cfg.llm,
            toolkit=SQLDatabaseToolkit(db=cfg.sql_db, llm=cfg.llm),
            verbose=cfg.langchain_verbose,
            handle_parsing_errors=True,
            max_execution_time=cfg.tool_timeout
        )
    else:
        prompt = PromptManager.get_prompt(agent_type, tools)
        agent = create_tool_calling_agent(cfg.llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=cfg.langchain_verbose,
        handle_parsing_errors=True,
        memory=memory_manager.memory,
        max_execution_time=cfg.tool_timeout
    )

# Initialize all agents
agents = {
    "planner": (PromptManager.get_prompt("planner") | cfg.llm | StrOutputParser()),
    "elastic": create_agent("elastic", get_elastic_tools()),
    "sql": create_agent("sql", get_sql_tools()),
    "excel": create_agent("excel", get_excel_tools()),
    "summarizer": (PromptManager.get_prompt("summarizer") | cfg.llm | StrOutputParser()),
    "decision": (PromptManager.get_prompt("decision") | cfg.llm | StrOutputParser()),
    "followup": (PromptManager.get_prompt("followup") | cfg.llm | StrOutputParser())
}

# Workflow Nodes
def planner_node(state: AgentState) -> AgentState:
    logger.info(f"Planning for question: {state.question}")
    history = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in state.conversation_history
    )
    
    try:
        raw_output = agents["planner"].invoke({
            "question": state.question,
            "history": history
        })
        state.update_from_planner(raw_output)
    except Exception as e:
        logger.error(f"Planner node failed: {str(e)}")
        state.errors.append(f"Planner error: {str(e)}")
        state.plan = "Fallback plan: 1. Search Elasticsearch\n2. Query SQL\n3. Check Excel"
        state.data_sources = ["elasticsearch", "sql", "excel"]
    
    return state

def executor_node(state: AgentState) -> AgentState:
    results = {}
    for source in state.data_sources or []:
        try:
            logger.info(f"Executing {source} tool for question: {state.question}")
            response = agents[source].invoke({
                "input": f"Question: {state.question}\nPlan: {state.plan}",
                "plan": state.plan
            }, config={"run_name": source})
            
            results[source] = {
                "status": "success",
                "data": response["output"] if isinstance(response, dict) else response
            }
        except TimeoutError:
            error_msg = f"{source} execution timed out after {cfg.tool_timeout} seconds"
            logger.error(error_msg)
            results[source] = {
                "status": "error",
                "error": error_msg
            }
            state.errors.append(error_msg)
        except Exception as e:
            error_msg = f"Error processing {source}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            results[source] = {
                "status": "error",
                "error": error_msg,
                "traceback": traceback.format_exc()
            }
            state.errors.append(error_msg)
    
    state.retrieved_data = results
    return state

def summarizer_node(state: AgentState) -> AgentState:
    if not state.retrieved_data:
        state.summary = "No data was retrieved from any sources"
        return state
        
    successful_data = {
        k: v["data"] for k, v in state.retrieved_data.items() 
        if v.get("status") == "success"
    }
    
    errors = {
        k: v["error"] for k, v in state.retrieved_data.items()
        if v.get("status") == "error"
    }
    
    try:
        state.summary = agents["summarizer"].invoke({
            "question": state.question,
            "retrieved_data": json.dumps({
                "successful_data": successful_data,
                "errors": errors
            }, indent=2)
        })
    except Exception as e:
        logger.error(f"Summarizer failed: {str(e)}")
        state.summary = f"Failed to generate summary: {str(e)}"
        state.errors.append(f"Summarizer error: {str(e)}")
    
    return state

def decision_node(state: AgentState) -> AgentState:
    try:
        state.final_answer = agents["decision"].invoke({
            "question": state.question,
            "summary": state.summary
        })
        
        # Include any errors in the final answer if they exist
        if state.errors:
            state.final_answer = (
                f"{state.final_answer}\n\n"
                f"Note: Some issues were encountered during processing:\n"
                + "\n".join(f"- {e}" for e in state.errors)
            )
    except Exception as e:
        logger.error(f"Decision maker failed: {str(e)}")
        state.final_answer = f"Failed to generate final answer: {str(e)}"
    
    return state

def followup_node(state: AgentState) -> AgentState:
    try:
        followups = agents["followup"].invoke({
            "question": state.question,
            "final_answer": state.final_answer
        })
        state.follow_up_questions = [
            q.strip() for q in followups.split("\n") 
            if q.strip()
        ][:3]  # Limit to 3 follow-ups
    except Exception as e:
        logger.error(f"Follow-up generator failed: {str(e)}")
        state.follow_up_questions = []
    
    return state

# Workflow Construction
def create_workflow():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("decision", decision_node)
    workflow.add_node("followup", followup_node)
    
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "summarizer")
    workflow.add_edge("summarizer", "decision")
    workflow.add_edge("decision", "followup")
    workflow.add_edge("followup", END)
    
    workflow.set_entry_point("planner")
    return workflow.compile()

# Main Application
def main():
    workflow = create_workflow()
    
    print("Data Assistant (type 'exit' to quit)")
    while True:
        try:
            question = input("\nYou: ").strip()
            if question.lower() in ('exit', 'quit'):
                break
                
            state = AgentState(
                question=question,
                conversation_history=memory_manager.get_context()
            )
            
            logger.info(f"Processing question: {question}")
            result = workflow.invoke(state)
            memory_manager.update_context(question, result.final_answer)
            
            print(f"\nAssistant: {result.final_answer}")
            if result.follow_up_questions:
                print("\nSuggested Follow-ups:")
                for i, q in enumerate(result.follow_up_questions, 1):
                    print(f"{i}. {q}")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Main loop error: {str(e)}", exc_info=True)
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()
