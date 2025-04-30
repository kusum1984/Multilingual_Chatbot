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

# ========== CUSTOM IMPLEMENTATIONS ==========
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

# ========== CONFIGURATION ==========
class Config:
    def __init__(self):
        # Elasticsearch config
        self.es = Elasticsearch(
            os.getenv("ELASTIC_ENDPOINT"),
            api_key=os.getenv("ELASTIC_API_KEY"),
            verify_certs=False
        )
        
        # LLM config
        self.llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name="gpt-4",
            model="gpt-4",
            temperature=0,
            max_retries=3
        )
        
        # SQL config
        driver = '{ODBC Driver 18 for SQL Server}'
        conn = f"Driver={driver};Server={os.getenv('SQL_SERVER')};Database={os.getenv('SQL_DATABASE')};Uid={os.getenv('SQL_USER')};Pwd={os.getenv('SQL_PASSWORD')};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
        params = urllib.parse.quote_plus(conn)
        conn_str = f'mssql+pyodbc:///?autocommit=true&odbc_connect={params}'
        engine = create_engine(conn_str)
        self.sql_db = SQLDatabase(engine, schema="comp")

        # Execution parameters
        self.max_recursion = 30  # Reduced from 50 to prevent long runs
        self.max_docs = 5       # Reduced sample size

cfg = Config()

# ========== ELASTICSEARCH TOOLS ==========
class ListIndicesInput(BaseModel):
    separator: str = Field(", ", description="Separator between index names")

class IndexDetailsInput(BaseModel):
    index_name: str = Field(..., description="Name of the Elasticsearch index")

class SampleDataInput(BaseModel):
    index_name: str = Field(..., description="Name of the Elasticsearch index")
    size: int = Field(3, description="Number of sample documents to retrieve")  # Reduced default

class SearchInput(BaseModel):
    index_name: str = Field(..., description="Name of the Elasticsearch index")
    query: str = Field(..., description="Query in Elasticsearch JSON format")
    size: int = Field(5, description="Maximum number of results to return")  # Reduced default

def list_indices(separator: str = ", ") -> str:
    """List all available indices in Elasticsearch"""
    try:
        indices = list(cfg.es.indices.get(index="*").keys())
        return f"Available indices:{separator}{separator.join(indices)}"
    except Exception as e:
        return f"Error listing indices: {str(e)}"

def get_index_details(index_name: str) -> str:
    """Get detailed information about a specific index"""
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

def get_sample_data(index_name: str, size: int = 3) -> str:  # Reduced default size
    """Get sample documents from an index"""
    try:
        if not cfg.es.indices.exists(index=index_name):
            return f"Index '{index_name}' does not exist"
        
        result = cfg.es.search(
            index=index_name,
            body={"query": {"match_all": {}}},
            size=min(size, cfg.max_docs)
        )
        hits = result.get('hits', {}).get('hits', [])
        if not hits:
            return f"No documents found in index '{index_name}'"
        
        samples = []
        for hit in hits[:size]:
            sample = {k: v for k, v in hit['_source'].items() if not isinstance(v, (dict, list)) or not v}
            samples.append(sample)
        return json.dumps(samples, indent=2)
    except Exception as e:
        return f"Error getting sample data: {str(e)}"

def elastic_search(index_name: str, query: str, size: int = 5) -> str:  # Reduced default size
    """Execute a search query against Elasticsearch"""
    try:
        if not cfg.es.indices.exists(index=index_name):
            return f"Index '{index_name}' does not exist"
        
        try:
            query_dict = json.loads(query)
        except json.JSONDecodeError:
            return "Invalid query format - must be valid JSON"
        
        result = cfg.es.search(
            index=index_name,
            body=query_dict,
            size=min(size, cfg.max_docs)
        )
        hits = result.get('hits', {}).get('hits', [])
        simplified_results = []
        for hit in hits:
            simplified = {k: v for k, v in hit['_source'].items() if not isinstance(v, (dict, list)) or not v}
            simplified_results.append(simplified)
        return json.dumps(simplified_results, indent=2)
    except Exception as e:
        return f"Search error: {str(e)}"

# ========== AGENT SETUP ==========
elastic_tools = [
    Tool.from_function(
        func=list_indices,
        name="elastic_list_indices",
        description="List all available Elasticsearch indices. Always call this first.",
        args_schema=ListIndicesInput
    ),
    Tool.from_function(
        func=get_index_details,
        name="elastic_index_details",
        description="Get detailed information about a specific index including mappings and settings",
        args_schema=IndexDetailsInput
    ),
    Tool.from_function(
        func=get_sample_data,
        name="elastic_sample_data",
        description="Get sample documents from an index to understand its structure (limit 3 docs by default)",
        args_schema=SampleDataInput
    ),
    Tool.from_function(
        func=elastic_search,
        name="elastic_search",
        description="Execute search queries against Elasticsearch (limit 5 results by default)",
        args_schema=SearchInput
    )
]

sql_toolkit = SQLDatabaseToolkit(db=cfg.sql_db, llm=cfg.llm)
sql_tools = sql_toolkit.get_tools()

all_tools = elastic_tools + sql_tools
tool_executor = ToolExecutor(all_tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

def create_agent(llm, tools, system_message: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert assistant. Follow these rules STRICTLY:
        1. Be concise and specific in your responses
        2. Use the minimum necessary tools to answer
        3. When you have enough information to answer, say "FINAL ANSWER: [your answer]"
        4. If stuck after 2 attempts, say "FINAL ANSWER: I need human assistance"
        5. Available tools: {", ".join([t.name for t in tools])}
        {system_message}"""),
        MessagesPlaceholder(variable_name="messages")
    ])
    return prompt | llm.bind_tools(tools)

elastic_agent = create_agent(
    cfg.llm, 
    elastic_tools,
    """You are an Elasticsearch expert. Your responsibilities:
    - First list available indices if needed
    - Check index structure before querying
    - Use minimal sample data to understand structure
    - Return concise search results"""
)

sql_agent = create_agent(
    cfg.llm,
    sql_tools,
    """You are a SQL expert. Your responsibilities:
    - Verify table structure before querying
    - Use minimal sample data to understand schema
    - Return concise query results
    - Add LIMIT clauses to prevent large results"""
)

# ========== GRAPH EXECUTION ==========
def agent_node(state, agent, name):
    result = agent.invoke(state)
    if not isinstance(result, FunctionMessage):
        result = HumanMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {"messages": [result], "sender": name}

def router(state):
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check for final answer
    if "FINAL ANSWER" in last_message.content:
        return "end"
    
    # Prevent infinite loops
    if len([m for m in messages if m.type == "human"]) > 3:  # Max 3 back-and-forths
        return "end"
    
    # Route tool calls
    if "function_call" in last_message.additional_kwargs:
        return "call_tool"
    
    # Default continue
    return "continue"

def tool_node(state):
    messages = state["messages"]
    last_message = messages[-1]
    
    try:
        tool_input = json.loads(last_message.additional_kwargs["function_call"]["arguments"])
        if len(tool_input) == 1 and "__arg1" in tool_input:
            tool_input = next(iter(tool_input.values()))
        
        action = ToolInvocation(
            tool=last_message.additional_kwargs["function_call"]["name"],
            tool_input=tool_input
        )
        response = tool_executor.invoke(action)
        return {"messages": [FunctionMessage(content=str(response), name=action.tool)]}
    except Exception as e:
        return {"messages": [FunctionMessage(content=f"Tool error: {str(e)}", name="system")]}

# Build workflow
workflow = StateGraph(AgentState)
workflow.add_node("Elasticsearch", functools.partial(agent_node, agent=elastic_agent, name="Elasticsearch"))
workflow.add_node("SQL", functools.partial(agent_node, agent=sql_agent, name="SQL"))
workflow.add_node("call_tool", tool_node)

# Connect nodes
workflow.add_conditional_edges(
    "Elasticsearch",
    router,
    {"continue": "SQL", "call_tool": "call_tool", "end": END}
)
workflow.add_conditional_edges(
    "SQL",
    router,
    {"continue": "Elasticsearch", "call_tool": "call_tool", "end": END}
)
workflow.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],
    {"Elasticsearch": "Elasticsearch", "SQL": "SQL"}
)

workflow.set_entry_point("Elasticsearch")
graph = workflow.compile()

# ========== EXECUTION HANDLER ==========
def execute_query(question: str):
    try:
        print(f"\nExecuting query: {question}")
        for step in graph.stream(
            {"messages": [HumanMessage(content=question)]},
            {"recursion_limit": cfg.max_recursion}
        ):
            print("\n=== AGENT STEP ===")
            print(step)
            
            # Early termination if final answer found
            if any("FINAL ANSWER" in str(msg) for msg in step.get("messages", [])):
                print("\nFINAL ANSWER DETECTED - TERMINATING")
                return step
        
        return {"status": "completed", "messages": "No final answer within recursion limit"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    load_dotenv()
    
    # Example query with clearer requirements
    question = """Get customer contact information from Elasticsearch for customers 
    who have made purchases over $1000 in the SQL database. 
    Provide only the customer names and email addresses."""
    
    result = execute_query(question)
    print("\n=== FINAL RESULT ===")
    print(result)
