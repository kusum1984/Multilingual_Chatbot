# Import statements grouped by functionality
import json
import os
import urllib.parse
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Elasticsearch imports
from elasticsearch import Elasticsearch

# LangChain imports
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.tools import Tool, BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent

# SQL imports
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine

# Load environment variables
load_dotenv()

# Configuration
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
        self.llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name="gpt-4",
            model="gpt-4",
            temperature=0
        )
        
        # SQL Database config
        self.sql_driver = '{ODBC Driver 18 for SQL Server}'
        self.sql_server = os.getenv("SQL_SERVER")
        self.sql_database = os.getenv("SQL_DATABASE")
        self.sql_user = os.getenv("SQL_USER")
        self.sql_password = os.getenv("SQL_PASSWORD")
        
        # Initialize SQL Database connection
        conn = f"Driver={self.sql_driver};Server=tcp:{self.sql_server},1433;Database={self.sql_database};Uid={self.sql_user};Pwd={self.sql_password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
        params = urllib.parse.quote_plus(conn)
        conn_str = f'mssql+pyodbc:///?autocommit=true&odbc_connect={params}'
        self.sql_engine = create_engine(conn_str, echo=True)
        self.db = SQLDatabase(self.sql_engine, schema="comp")
        self.sql_toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        
        self.langchain_verbose = True

cfg = Config()

# Elasticsearch Tool Input Models
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

# Elasticsearch Tools Implementation
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
            return json.dumps(result.get('hits', {}), indent=2)
    except Exception as e:
        return f"Search error: {str(e)}"

# Create Elasticsearch Tools
elastic_tools = [
    Tool.from_function(
        func=list_indices,
        name="elastic_list_indices",
        description="Lists all available Elasticsearch indices. Always call this first.",
        args_schema=ListIndicesInput
    ),
    Tool.from_function(
        func=get_index_details,
        name="elastic_index_details",
        description="Gets details about a specific index including mappings and settings",
        args_schema=IndexDetailsInput
    ),
    Tool.from_function(
        func=get_index_data,
        name="elastic_index_data",
        description="Gets sample documents from an index to understand its structure",
        args_schema=IndexDataInput
    ),
    Tool.from_function(
        func=elastic_search,
        name="elastic_search",
        description="Executes search or aggregation queries on an Elasticsearch index",
        args_schema=SearchToolInput
    )
]

# Elasticsearch Agent Setup
def get_elastic_system_prompt(question: str) -> str:
    return f"""
    You are an Elasticsearch expert assistant. Follow these steps for every request:
    1. First list all available indices using elastic_list_indices
    2. Then examine index details or sample data as needed
    3. Finally execute specific searches when you understand the data structure
    
    Make sure you understand the index structure before querying data.
    
    Question to answer: {question}
    """

def create_elastic_agent_executor() -> AgentExecutor:
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful AI ElasticSearch Expert Assistant that answers natural language question. Follow these steps:\n"
                             "1. First understand what data is available\n"
                             "2. Determine which indices and fields are relevant\n"
                             "3. Generate appropriate queries.\n"
                             "4. Return human-readable answer\n"
                             "Always provide context about where the information come from\n"
                             "Question: {input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    elastic_agent = create_react_agent(
        llm=cfg.llm,
        tools=elastic_tools,
        prompt=prompt,
        name="elastic_agent"
    )
    
    return AgentExecutor(agent=elastic_agent, tools=elastic_tools, verbose=cfg.langchain_verbose)

# SQL Agent Setup
def create_sql_agent():
    prompt_template = """You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.
    Only use the below tools. Only use the information returned by the below tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    To start you should ALWAYS look at the tables in the database to see what you can query.
    Do NOT skip this step.
    Then you should query the schema of the most relevant tables."""
    
    system_message = prompt_template.format(dialect="MSSQL", top_k=5)
    
    sql_agent = create_react_agent(
        llm=cfg.llm,
        tools=cfg.sql_toolkit.get_tools(),
        prompt=system_message,
        name='sql_agent'
    )
    
    return AgentExecutor(agent=sql_agent, tools=cfg.sql_toolkit.get_tools(), verbose=cfg.langchain_verbose)

# Supervisor Agent
def create_supervisor_agent():
    research_agent = create_elastic_agent_executor()
    math_agent = create_sql_agent()
    
    supervisor = create_supervisor(
        model=cfg.llm,
        agents=[research_agent, math_agent],
        prompt=(
            "You are a supervisor managing two agents:\n"
            "- a sql agent. Assign database query-related tasks to this agent\n"
            "- a elastic agent. Assign elasticsearch-related tasks to this agent\n"
            "Assign work to one agent at a time, do not call agents in parallel.\n"
            "Do not do any work yourself."
        ),
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile()
    
    return supervisor

# Main execution
if __name__ == "__main__":
    supervisor = create_supervisor_agent()
    query = "What type of data..."
    
    events = supervisor.stream(
        {"messages": [("user", query)]},
        stream_mode="values",
    )
    
    for event in events:
        event["messages"][-1].pretty_print()
        +++++++++++++++++++++++++++++++++


# Import statements grouped by functionality
import json
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Elasticsearch imports
from elasticsearch import Elasticsearch

# LangChain imports
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.tools import Tool, BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent

# SQL imports
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

# Load environment variables
load_dotenv()

# Configuration
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
        self.llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name="gpt-4",
            model="gpt-4",
            temperature=0
        )
        
        self.langchain_verbose = True

cfg = Config()

# Elasticsearch Tool Input Models
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

# Elasticsearch Tools Implementation
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
            return json.dumps(result.get('hits', {}), indent=2)
    except Exception as e:
        return f"Search error: {str(e)}"

# Create Elasticsearch Tools
elastic_tools = [
    Tool.from_function(
        func=list_indices,
        name="elastic_list_indices",
        description="Lists all available Elasticsearch indices. Always call this first.",
        args_schema=ListIndicesInput
    ),
    Tool.from_function(
        func=get_index_details,
        name="elastic_index_details",
        description="Gets details about a specific index including mappings and settings",
        args_schema=IndexDetailsInput
    ),
    Tool.from_function(
        func=get_index_data,
        name="elastic_index_data",
        description="Gets sample documents from an index to understand its structure",
        args_schema=IndexDataInput
    ),
    Tool.from_function(
        func=elastic_search,
        name="elastic_search",
        description="Executes search or aggregation queries on an Elasticsearch index",
        args_schema=SearchToolInput
    )
]

# Elasticsearch Agent Setup
def get_elastic_system_prompt(question: str) -> str:
    return f"""
    You are an Elasticsearch expert assistant. Follow these steps for every request:
    1. First list all available indices using elastic_list_indices
    2. Then examine index details or sample data as needed
    3. Finally execute specific searches when you understand the data structure
    
    Make sure you understand the index structure before querying data.
    
    Question to answer: {question}
    """

def create_elastic_agent_executor() -> AgentExecutor:
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful AI ElasticSearch Expert Assistant that answers natural language question. Follow these steps:\n"
                             "1. First understand what data is available\n"
                             "2. Determine which indices and fields are relevant\n"
                             "3. Generate appropriate queries.\n"
                             "4. Return human-readable answer\n"
                             "Always provide context about where the information come from\n"
                             "Question: {input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    elastic_agent = create_react_agent(
        llm=cfg.llm,
        tools=elastic_tools,
        prompt=prompt,
        name="elastic_agent"
    )
    
    return AgentExecutor(agent=elastic_agent, tools=elastic_tools, verbose=cfg.langchain_verbose)

# SQL Agent Setup
def setup_sql_connection():
    driver = '{ODBC Driver 18 for SQL Server}'
    server = os.getenv("SQL_SERVER")
    database = os.getenv("SQL_DATABASE")
    user = os.getenv("SQL_USER")
    password = os.getenv("SQL_PASSWORD")
    
    conn = f"Driver={driver};Server=tcp:{server},1433;Database={database};Uid={user};Pwd={password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
    params = urllib.parse.quote_plus(conn)
    conn_str = f'mssql+pyodbc:///?autocommit=true&odbc_connect={params}'
    engine = create_engine(conn_str, echo=True)
    return SQLDatabase(engine, schema="comp")

def create_sql_agent():
    db = setup_sql_connection()
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=cfg.llm)
    
    prompt_template = """You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.
    Only use the below tools. Only use the information returned by the below tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    To start you should ALWAYS look at the tables in the database to see what you can query.
    Do NOT skip this step.
    Then you should query the schema of the most relevant tables."""
    
    system_message = prompt_template.format(dialect="MSSQL", top_k=5)
    
    sql_agent = create_react_agent(
        llm=cfg.llm,
        tools=sql_toolkit.get_tools(),
        prompt=system_message,
        name='sql_agent'
    )
    
    return AgentExecutor(agent=sql_agent, tools=sql_toolkit.get_tools(), verbose=cfg.langchain_verbose)

# Supervisor Agent
def create_supervisor_agent():
    research_agent = create_elastic_agent_executor()
    math_agent = create_sql_agent()
    
    supervisor = create_supervisor(
        model=cfg.llm,
        agents=[research_agent, math_agent],
        prompt=(
            "You are a supervisor managing two agents:\n"
            "- a sql agent. Assign database query-related tasks to this agent\n"
            "- a elastic agent. Assign elasticsearch-related tasks to this agent\n"
            "Assign work to one agent at a time, do not call agents in parallel.\n"
            "Do not do any work yourself."
        ),
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile()
    
    return supervisor

# Main execution
if __name__ == "__main__":
    supervisor = create_supervisor_agent()
    query = "What type of data..."
    
    events = supervisor.stream(
        {"messages": [("user", query)]},
        stream_mode="values",
    )
    
    for event in events:
        event["messages"][-1].pretty_print()
