import os
import json
import threading
import urllib.parse
from typing import List, Dict, Any, Literal, Optional
from dataclasses import dataclass
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool, Tool
from langgraph.graph import END, StateGraph
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.document_loaders import ExcelLoader
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel as PydanticBaseModel
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_sql_agent, create_openai_functions_agent
from langchain.agents.agent_types import AgentType
from functools import lru_cache
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Load environment variables
load_dotenv()

# Configuration
class Config:
    def __init__(self):
        # Elasticsearch configuration from first example
        self.elastic_index_data_from = 0
        self.elastic_index_data_size = 10
        self.elastic_index_data_max_size = 50
        self.aggs_limit = 5
        self.max_search_retries = 3
        self.token_limit = 3000
       
        # Initialize Elasticsearch client with API key
        self.es = Elasticsearch(
            os.getenv("ELASTIC_ENDPOINT"),
            api_key=os.getenv("ELASTIC_API_KEY"),
            verify_certs=False
        )
       
        # Initialize LLM - using Azure from first example
        self.llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name="gpt-4",
            model="gpt-4",
            temperature=0
        )
       
        # SQL Server connection setup
        driver = '{ODBC Driver 18 for SQL Server}'
        server = os.getenv("SQL_SERVER")  # "kkkkk.database.windows.net"
        database = os.getenv("SQL_DATABASE")  # "wjsjsjs"
        user = os.getenv("SQL_USER")  # "SA-its-as-ssss"
        password = os.getenv("SQL_PASSWORD")  # "SSSSS"
        
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

# Enhanced AgentState
@dataclass
class AgentState:
    question: str
    plan: Optional[str] = None
    data_sources: Optional[List[Literal["elasticsearch", "sql", "excel"]]] = None
    retrieved_data: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None
    final_answer: Optional[str] = None
    follow_up_questions: Optional[List[str]] = None
    conversation_history: List[BaseMessage] = None
    raw_plan_output: Optional[str] = None

    def update_from_planner(self, raw_output: str):
        self.raw_plan_output = raw_output
        try:
            json_str = raw_output.split("```json")[1].split("```")[0].strip() if "```json" in raw_output else raw_output
            plan_data = json.loads(json_str)
            self.plan = plan_data.get("plan", "No specific execution plan")
            self.data_sources = plan_data.get("data_sources", ["elasticsearch", "sql", "excel"])
        except json.JSONDecodeError:
            self.plan = "Fallback plan: 1. Search Elasticsearch\n2. Query SQL\n3. Check Excel"
            self.data_sources = ["elasticsearch", "sql", "excel"]
        except Exception:
            self.plan = "Fallback plan: 1. Search Elasticsearch\n2. Query SQL\n3. Check Excel"
            self.data_sources = ["elasticsearch", "sql", "excel"]

# Input Models for Elasticsearch (from first example)
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

# Input model for Excel
class ExcelSheetInput(BaseModel):
    sheet_name: Optional[str] = Field(None, description="Name of the Excel sheet to read")

# Elasticsearch Tools Implementation (from first example with enhancements)
class ElasticsearchTools:
    def __init__(self, es_client):
        self.es_client = es_client
        self.cfg = cfg  # Use the global config

    def list_indices(self, separator: str = ", ") -> str:
        """Lists all available Elasticsearch indices"""
        try:
            indices = list(self.es_client.indices.get(index="*").keys())
            return f"Available indices:{separator}{separator.join(indices)}"
        except Exception as e:
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
            return f"Search error: {str(e)}"

# Excel Tools Implementation (from second example)
class ExcelTools:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_excel_sheet(self, sheet_name: Optional[str] = None) -> str:
        try:
            loader = ExcelLoader(file_path=self.file_path, sheet_name=sheet_name)
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
            return f"Error reading Excel file: {str(e)}"

# Shared Memory Manager (from second example)
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

# Centralized Prompt Management (updated with Elasticsearch focus)
class PromptManager:
    _templates = {
        "planner": {
            "system": """You are an expert data analysis planner with specialization in Elasticsearch. Analyze the question and:
1. First determine if Elasticsearch is needed (text search, logs, unstructured data)
2. Then consider SQL for structured data or Excel for spreadsheet data
3. Create a detailed execution plan with:
   - "data_sources": array of needed sources (elasticsearch first if relevant)
   - "plan": numbered steps

Elasticsearch Guidelines:
- Always check available indices first
- Examine index mappings before querying
- Use aggregations for analytics""",
            "user": """Question: {question}
Conversation History:
{history}"""
        },
        "summarizer": {
            "system": """You are a data synthesis expert with Elasticsearch knowledge. Combine the retrieved data to:
1. Answer the original question directly
2. Highlight key findings from Elasticsearch results
3. Resolve any data conflicts (prioritize Elasticsearch for search results)
4. Keep the summary concise but complete""",
            "user": """Question: {question}
Retrieved Data:
{retrieved_data}"""
        },
        "elastic": {
            "system": """You are an Elasticsearch expert with these tools:
{tools}

Guidelines:
1. FIRST list available indices using elastic_list_indices
2. THEN examine index details or sample data as needed
3. FINALLY execute specific searches when you understand the data structure

Query Tips:
- Prefer Elasticsearch DSL for complex queries
- For text search, use query_string
- For aggregations, use proper buckets""",
            "user": "{input}"
        },
        "decision": {
            "system": """You are a decision maker who evaluates the summary and provides a final answer.
1. Ensure the answer directly addresses the original question
2. Include relevant data points from the summary
3. Format the answer clearly and professionally""",
            "user": """Question: {question}
Summary: {summary}"""
        },
        "followup": {
            "system": """Generate 2-3 relevant follow-up questions based on the original question and answer.
Make them specific and actionable.""",
            "user": """Original Question: {question}
Final Answer: {final_answer}"""
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

# Cached Tool Initialization with updated Elasticsearch tools
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

# Agent Factory with updated Elasticsearch agent
def create_agent(agent_type: str, tools: List[BaseTool]) -> AgentExecutor:
    if agent_type == "sql":
        agent = create_sql_agent(
            llm=cfg.llm,
            toolkit=SQLDatabaseToolkit(db=cfg.sql_db, llm=cfg.llm),
            agent_type=AgentType.OPENAI_TOOLS,
            verbose=cfg.langchain_verbose,
            handle_parsing_errors=True
        )
    elif agent_type == "elastic":
        # Use the OpenAI functions agent for Elasticsearch as in first example
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI ElasticSearch Expert Assistant"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
       
        agent = create_openai_functions_agent(
            llm=cfg.llm,
            tools=tools,
            prompt=prompt
        )
    else:
        prompt = PromptManager.get_prompt(agent_type, tools)
        agent = create_tool_calling_agent(cfg.llm, tools, prompt)
   
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=cfg.langchain_verbose,
        handle_parsing_errors=True,
        memory=memory_manager.memory
    )

# Initialize all agents with updated Elasticsearch agent
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
    history = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in (state.conversation_history or [])
    )
   
    raw_output = agents["planner"].invoke({
        "question": state.question,
        "history": history
    })
   
    state.update_from_planner(raw_output)
    return state

def executor_node(state:
