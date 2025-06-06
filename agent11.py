from typing import Annotated, List, Dict, Optional
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import InjectedState, create_react_agent
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import Tool
from langchain_core.pydantic_v1 import BaseModel, Field
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import json
import os
import urllib.parse
from sqlalchemy import create_engine
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent

# Load environment variables
load_dotenv()

class Config:
    def __init__(self):
        # Elasticsearch config
        self.elastic_index_data_from = 0
        self.elastic_index_data_size = 10
        self.elastic_index_data_max_size = 50
        self.aggs_limit = 5
        self.max_search_retries = 3
        self.token_limit = 3000
        
        # Initialize Elasticsearch client
        self.es = Elasticsearch(
            os.getenv("ELASTIC_ENDPOINT"),
            api_key=os.getenv("ELASTIC_API_KEY"),
            verify_certs=False
        )
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
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
        engine = create_engine(conn_str, echo=True)
        self.sql_db = SQLDatabase(engine, schema="comp")
        
        self.langchain_verbose = True

cfg = Config()

# ================== Elasticsearch Tools ==================
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

def list_indices(separator: str = ", ") -> str:
    """Lists all available Elasticsearch indices"""
    try:
        indices = list(cfg.es.indices.get(index="*").keys())
        return f"Available indices:{separator}{separator.join(indices)}"
    except Exception as e:
        return f"Error listing indices: {str(e)}"

def get_index_details(index_name: str) -> str:
    """Gets details about a specific index including mappings and settings"""
    try:
        if not cfg.es.indices.exists(index=index_name):
            return f"Index '{index_name}' does not exist"
        details = {
            "aliases": cfg.es.indices.get_alias(index=index_name).get(index_name, {}).get("aliases", {}),
            "mappings": cfg.es.indices.get_mapping(index=index_name).get(index_name, {}).get("mappings", {}),
            "settings": cfg.es.indices.get_settings(index=index_name).get(index_name, {}).get("settings", {})
        }
        return json.dumps(details, indent=2)
    except Exception as e:
        return f"Error getting index details: {str(e)}"

def get_index_data(index_name: str, size: int = 5) -> str:
    """Gets sample documents from an index to understand its structure"""
    try:
        if not cfg.es.indices.exists(index=index_name):
            return f"Index '{index_name}' does not exist"
        result = cfg.es.search(
            index=index_name,
            body={"query": {"match_all": {}}},
            size=min(size, cfg.elastic_index_data_max_size)
        )
        hits = result.get('hits', {}).get('hits', [])
        if not hits:
            return f"No documents found in index '{index_name}'"
        return json.dumps([hit['_source'] for hit in hits[:size]], indent=2)
    except Exception as e:
        return f"Error getting index data: {str(e)}"

def elastic_search(index_name: str, query: str, from_: int = 0, size: int = 10) -> str:
    """Executes a search or aggregation query on an Elasticsearch index"""
    try:
        if not cfg.es.indices.exists(index=index_name):
            return f"Index '{index_name}' does not exist"
        size = min(cfg.elastic_index_data_max_size, size)
        try:
            query_dict = json.loads(query)
        except json.JSONDecodeError:
            return "Invalid query format - must be valid JSON"

        # Determine if this is a search or aggregation
        is_aggregation = "aggs" in query_dict or "aggregations" in query_dict
        if is_aggregation:
            size = cfg.aggs_limit

        result = cfg.es.search(
            index=index_name,
            body=query_dict,
            from_=from_,
            size=size
        )

        if is_aggregation:
            return json.dumps(result.get('aggregations', {}), indent=2)
        else:
            return json.dumps({
                "total": result['hits']['total']['value'],
                "hits": [hit['_source'] for hit in result['hits']['hits']]
            }, indent=2)
    except Exception as e:
        return f"Search error: {str(e)}"

# ================== SQL Agent ==================
def sql_agent_executor(query: str) -> str:
    """Executes SQL queries and returns results. Use this for all database related questions."""
    try:
        toolkit = SQLDatabaseToolkit(db=cfg.sql_db, llm=cfg.llm)
        agent = create_sql_agent(
            llm=cfg.llm,
            toolkit=toolkit,
            verbose=cfg.langchain_verbose,
            handle_parsing_errors=True
        )
        result = agent.run(query)
        return str(result)
    except Exception as e:
        return f"Error executing SQL query: {str(e)}"

# ================== Agent Functions for LangGraph ==================
def get_elastic_agent_prompt():
    return """You are an expert Elasticsearch assistant. Follow these steps carefully:

1. FIRST list available indices using elastic_list_indices if you don't know which indices exist
2. THEN examine index details or sample data as needed using:
   - elastic_index_details to understand the structure
   - elastic_index_data to see example documents
3. FINALLY execute searches using elastic_search when you understand the data structure

Important Rules:
- Always verify the index exists before querying
- For search queries, construct proper Elasticsearch JSON queries
- For aggregations, use the proper aggregation syntax
- Return complete, formatted answers to the user
- Explain what you found in natural language

Current Question: {input}"""

def elastic_agent(state: Annotated[dict, InjectedState]):
    """Handles all Elasticsearch related queries"""
    try:
        # Create Elasticsearch tools
        elastic_tools = [
            Tool.from_function(
                func=list_indices,
                name="elastic_list_indices",
                description="Lists all available Elasticsearch indices. Always call this first if you don't know what indices exist.",
                args_schema=ListIndicesInput
            ),
            Tool.from_function(
                func=get_index_details,
                name="elastic_index_details",
                description="Gets details about a specific index including mappings and settings. Use to understand the index structure.",
                args_schema=IndexDetailsInput
            ),
            Tool.from_function(
                func=get_index_data,
                name="elastic_index_data",
                description="Gets sample documents from an index to understand its content and structure.",
                args_schema=IndexDataInput
            ),
            Tool.from_function(
                func=elastic_search,
                name="elastic_search",
                description="Executes search or aggregation queries on an Elasticsearch index. Input must be valid Elasticsearch JSON query.",
                args_schema=SearchToolInput
            )
        ]
        
        # Get the user input from state
        messages = state.get("messages", [])
        user_input = messages[-1].content if messages else ""
        
        # Create agent executor with enhanced prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=get_elastic_agent_prompt()),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        agent = create_openai_functions_agent(
            llm=cfg.llm,
            tools=elastic_tools,
            prompt=prompt
        )
        
        executor = AgentExecutor(
            agent=agent,
            tools=elastic_tools,
            verbose=cfg.langchain_verbose,
            max_iterations=10,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
        
        # Execute the agent
        result = executor.invoke({
            "input": user_input,
            "chat_history": messages[:-1]  # All messages except the last one
        })
        
        # Format the final answer properly
        output = result["output"]
        if not output.strip():
            return "I couldn't find any results for your query."
        
        return output
    except Exception as e:
        return f"Error processing your Elasticsearch request: {str(e)}"

def sql_agent(state: Annotated[dict, InjectedState]):
    """Handles all SQL database related queries"""
    try:
        # Get the user input from state
        messages = state.get("messages", [])
        user_input = messages[-1].content if messages else ""
        
        # Execute the SQL agent
        result = sql_agent_executor(user_input)
        return result
    except Exception as e:
        return f"Error processing your database request: {str(e)}"

# ================== Supervisor ==================
def get_supervisor_prompt():
    return """You are a supervisor agent that routes between specialized agents:

Available Agents:
1. Elasticsearch Agent - Handles all Elasticsearch queries (index listing, searching, aggregations)
2. SQL Agent - Handles all database queries (data retrieval, analysis)

Routing Rules:
- Route to Elasticsearch Agent when the question involves:
  * Searching or querying logs/data
  * Analyzing index structures
  * Creating Elasticsearch queries
  * Terms like "Elasticsearch", "index", "search query", "aggregation"
  
- Route to SQL Agent when the question involves:
  * Database records
  * SQL queries
  * Terms like "database", "SQL", "tables", "records"

Analyze the question carefully and choose the most appropriate agent.
Provide a complete, well-formatted answer to the user."""

def create_supervisor():
    tools = [
        Tool.from_function(
            func=elastic_agent,
            name="Elasticsearch_Agent",
            description="Call this for all Elasticsearch related questions including index listing, searching, and aggregations."
        ),
        Tool.from_function(
            func=sql_agent,
            name="SQL_Agent",
            description="Call this for all SQL database related questions including data retrieval and analysis."
        )
    ]
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=get_supervisor_prompt()),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    return create_react_agent(cfg.llm, tools, prompt=prompt)

supervisor = create_supervisor()

# ================== Example Usage ==================
if __name__ == "__main__":
    # Example conversation
    questions = [
        "List all Elasticsearch indices",
        "Show me some sample data from the logs index",
        "What customers do we have in the database?"
    ]
    
    state = {
        "messages": []
    }
    
    for question in questions:
        print(f"\nUser: {question}")
        state["messages"].append(HumanMessage(content=question))
        
        # Execute the supervisor
        result = supervisor.invoke(state)
        
        print(f"Assistant: {result}")
        state["messages"].append(SystemMessage(content=result))
