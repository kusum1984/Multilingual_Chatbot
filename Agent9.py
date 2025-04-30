import os
from typing import Annotated, Sequence, TypedDict
import operator
from langchain_core.messages import BaseMessage, HumanMessage, FunctionMessage
from langchain_core.tools import Tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
import json
from dotenv import load_dotenv
from typing import List, Dict, Optional
from elasticsearch import Elasticsearch
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from sqlalchemy import create_engine
import urllib.parse

# Load environment variables
load_dotenv()

# Configuration class
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

# Create SQL Tools
sql_toolkit = SQLDatabaseToolkit(db=cfg.sql_db, llm=cfg.llm)
sql_tools = sql_toolkit.get_tools()

# Combine all tools
all_tools = elastic_tools + sql_tools
tool_executor = ToolExecutor(all_tools)

# Define the Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

# Create agent nodes
def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)

# Elasticsearch Agent
elastic_agent = create_agent(
    cfg.llm,
    elastic_tools,
    system_message="You are an Elasticsearch expert. Your job is to help with Elasticsearch queries and data retrieval."
)

# SQL Agent
sql_agent = create_agent(
    cfg.llm,
    sql_tools,
    system_message="You are a SQL expert. Your job is to help with SQL queries and database operations."
)

# Helper function to create a node for a given agent
def agent_node(state, agent, name):
    result = agent.invoke(state)
    if isinstance(result, FunctionMessage):
        pass
    else:
        result = HumanMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        "sender": name,
    }

# Router function
def router(state):
    messages = state["messages"]
    last_message = messages[-1]
    if "function_call" in last_message.additional_kwargs:
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        return "end"
    return "continue"

# Tool node
def tool_node(state):
    messages = state["messages"]
    last_message = messages[-1]
    tool_input = json.loads(
        last_message.additional_kwargs["function_call"]["arguments"]
    )
    if len(tool_input) == 1 and "__arg1" in tool_input:
        tool_input = next(iter(tool_input.values()))
    tool_name = last_message.additional_kwargs["function_call"]["name"]
    action = ToolInvocation(
        tool=tool_name,
        tool_input=tool_input,
    )
    response = tool_executor.invoke(action)
    function_message = FunctionMessage(
        content=f"{tool_name} response: {str(response)}", name=action.tool
    )
    return {"messages": [function_message]}

# Create the workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("Elasticsearch", functools.partial(agent_node, agent=elastic_agent, name="Elasticsearch"))
workflow.add_node("SQL", functools.partial(agent_node, agent=sql_agent, name="SQL"))
workflow.add_node("call_tool", tool_node)

# Add edges
workflow.add_conditional_edges(
    "Elasticsearch",
    router,
    {"continue": "SQL", "call_tool": "call_tool", "end": END},
)
workflow.add_conditional_edges(
    "SQL",
    router,
    {"continue": "Elasticsearch", "call_tool": "call_tool", "end": END},
)
workflow.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],
    {
        "Elasticsearch": "Elasticsearch",
        "SQL": "SQL",
    },
)

workflow.set_entry_point("Elasticsearch")
graph = workflow.compile()

# Example usage
def run_query(question: str):
    for s in graph.stream(
        {
            "messages": [
                HumanMessage(content=question)
            ],
        },
        {"recursion_limit": 150},
    ):
        print(s)
        print("----")

# Example query that might require both Elasticsearch and SQL
run_query("Find all customers in Elasticsearch who have made purchases over $1000 in the SQL database")


#######################################
import os
from typing import Annotated, Sequence, TypedDict
import operator
import json
import functools  # Added missing import
from langchain_core.messages import BaseMessage, HumanMessage, FunctionMessage
from langchain_core.tools import Tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation  # Fixed import
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from sqlalchemy import create_engine
import urllib.parse

# Load environment variables
load_dotenv()

# Configuration class
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

# Create SQL Tools
sql_toolkit = SQLDatabaseToolkit(db=cfg.sql_db, llm=cfg.llm)
sql_tools = sql_toolkit.get_tools()

# Combine all tools
all_tools = elastic_tools + sql_tools
tool_executor = ToolExecutor(all_tools)

# Define the Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

# Create agent nodes
def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)

# Elasticsearch Agent
elastic_agent = create_agent(
    cfg.llm,
    elastic_tools,
    system_message="You are an Elasticsearch expert. Your job is to help with Elasticsearch queries and data retrieval."
)

# SQL Agent
sql_agent = create_agent(
    cfg.llm,
    sql_tools,
    system_message="You are a SQL expert. Your job is to help with SQL queries and database operations."
)

# Helper function to create a node for a given agent
def agent_node(state, agent, name):
    result = agent.invoke(state)
    if isinstance(result, FunctionMessage):
        pass
    else:
        result = HumanMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        "sender": name,
    }

# Router function
def router(state):
    messages = state["messages"]
    last_message = messages[-1]
    if "function_call" in last_message.additional_kwargs:
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        return "end"
    return "continue"

# Tool node
def tool_node(state):
    messages = state["messages"]
    last_message = messages[-1]
    tool_input = json.loads(
        last_message.additional_kwargs["function_call"]["arguments"]
    )
    if len(tool_input) == 1 and "__arg1" in tool_input:
        tool_input = next(iter(tool_input.values()))
    tool_name = last_message.additional_kwargs["function_call"]["name"]
    action = ToolInvocation(
        tool=tool_name,
        tool_input=tool_input,
    )
    response = tool_executor.invoke(action)
    function_message = FunctionMessage(
        content=f"{tool_name} response: {str(response)}", name=action.tool
    )
    return {"messages": [function_message]}

# Create the workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("Elasticsearch", functools.partial(agent_node, agent=elastic_agent, name="Elasticsearch"))
workflow.add_node("SQL", functools.partial(agent_node, agent=sql_agent, name="SQL"))
workflow.add_node("call_tool", tool_node)

# Add edges
workflow.add_conditional_edges(
    "Elasticsearch",
    router,
    {"continue": "SQL", "call_tool": "call_tool", "end": END},
)
workflow.add_conditional_edges(
    "SQL",
    router,
    {"continue": "Elasticsearch", "call_tool": "call_tool", "end": END},
)
workflow.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],
    {
        "Elasticsearch": "Elasticsearch",
        "SQL": "SQL",
    },
)

workflow.set_entry_point("Elasticsearch")
graph = workflow.compile()

# Example usage
def run_query(question: str):
    for s in graph.stream(
        {
            "messages": [
                HumanMessage(content=question)
            ],
        },
        {"recursion_limit": 150},
    ):
        print(s)
        print("----")

# Example query that might require both Elasticsearch and SQL
run_query("Find all customers in Elasticsearch who have made purchases over $1000 in the SQL database")


#############################################

import os
from typing import Annotated, Sequence, TypedDict, List, Dict, Any
import operator
import json
import functools
from langchain_core.messages import BaseMessage, HumanMessage, FunctionMessage
from langchain_core.tools import BaseTool, Tool
from langgraph.graph import END, StateGraph
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from langchain.agents import create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from sqlalchemy import create_engine
import urllib.parse

# Custom implementations to replace langgraph.prebuilt imports
class ToolInvocation:
    def __init__(self, tool: str, tool_input: Dict[str, Any]):
        self.tool = tool
        self.tool_input = tool_input

class ToolExecutor:
    def __init__(self, tools: List[BaseTool]):
        self.tools = {tool.name: tool for tool in tools}
    
    def invoke(self, tool_invocation: ToolInvocation) -> Any:
        tool = self.tools[tool_invocation.tool]
        return tool.invoke(tool_invocation.tool_input)

# Load environment variables
load_dotenv()

# Configuration class
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
    try:
        indices = list(cfg.es.indices.get(index="*").keys())
        return f"Available indices:{separator}{separator.join(indices)}"
    except Exception as e:
        return f"Error listing indices: {str(e)}"

def get_index_details(index_name: str) -> str:
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
    try:
        if not cfg.es.indices.exists(index=index_name):
            return f"Index '{index_name}' does not exist"
        size = min(cfg.elastic_index_data_max_size, size)
        try:
            query_dict = json.loads(query)
        except json.JSONDecodeError:
            return "Invalid query format - must be valid JSON"

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

# Create SQL Tools
sql_toolkit = SQLDatabaseToolkit(db=cfg.sql_db, llm=cfg.llm)
sql_tools = sql_toolkit.get_tools()

# Combine all tools
all_tools = elastic_tools + sql_tools
tool_executor = ToolExecutor(all_tools)

# Define the Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

# Create agent nodes
def create_agent(llm, tools, system_message: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)

# Create agents
elastic_agent = create_agent(
    cfg.llm,
    elastic_tools,
    system_message="You are an Elasticsearch expert. Your job is to help with Elasticsearch queries and data retrieval."
)

sql_agent = create_agent(
    cfg.llm,
    sql_tools,
    system_message="You are a SQL expert. Your job is to help with SQL queries and database operations."
)

# Helper function to create a node for a given agent
def agent_node(state, agent, name):
    result = agent.invoke(state)
    if isinstance(result, FunctionMessage):
        pass
    else:
        result = HumanMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        "sender": name,
    }

# Router function
def router(state):
    messages = state["messages"]
    last_message = messages[-1]
    if "function_call" in last_message.additional_kwargs:
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        return "end"
    return "continue"

# Tool node
def tool_node(state):
    messages = state["messages"]
    last_message = messages[-1]
    tool_input = json.loads(
        last_message.additional_kwargs["function_call"]["arguments"]
    )
    if len(tool_input) == 1 and "__arg1" in tool_input:
        tool_input = next(iter(tool_input.values()))
    tool_name = last_message.additional_kwargs["function_call"]["name"]
    action = ToolInvocation(
        tool=tool_name,
        tool_input=tool_input,
    )
    response = tool_executor.invoke(action)
    function_message = FunctionMessage(
        content=f"{tool_name} response: {str(response)}", name=action.tool
    )
    return {"messages": [function_message]}

# Create the workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("Elasticsearch", functools.partial(agent_node, agent=elastic_agent, name="Elasticsearch"))
workflow.add_node("SQL", functools.partial(agent_node, agent=sql_agent, name="SQL"))
workflow.add_node("call_tool", tool_node)

# Add edges
workflow.add_conditional_edges(
    "Elasticsearch",
    router,
    {"continue": "SQL", "call_tool": "call_tool", "end": END},
)
workflow.add_conditional_edges(
    "SQL",
    router,
    {"continue": "Elasticsearch", "call_tool": "call_tool", "end": END},
)
workflow.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],
    {
        "Elasticsearch": "Elasticsearch",
        "SQL": "SQL",
    },
)

workflow.set_entry_point("Elasticsearch")
graph = workflow.compile()

# Example usage
def run_query(question: str):
    for s in graph.stream(
        {
            "messages": [
                HumanMessage(content=question)
            ],
        },
        {"recursion_limit": 150},
    ):
        print(s)
        print("----")

# Example query
run_query("Find all customers in Elasticsearch who have made purchases over $1000 in the SQL database")
