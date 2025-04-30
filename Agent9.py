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

# ===== CUSTOM IMPLEMENTATIONS =====
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

# ===== CONFIGURATION =====
class Config:
    def __init__(self):
        load_dotenv()
        
        # Elasticsearch
        self.es = Elasticsearch(
            os.getenv("ELASTIC_ENDPOINT"),
            api_key=os.getenv("ELASTIC_API_KEY"),
            verify_certs=False
        )
        
        # Azure OpenAI
        self.llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name="gpt-4",
            temperature=0,
            max_retries=3
        )
        
        # SQL Server
        driver = '{ODBC Driver 18 for SQL Server}'
        conn_str = (
            f"Driver={driver};"
            f"Server={os.getenv('SQL_SERVER')};"
            f"Database={os.getenv('SQL_DATABASE')};"
            f"Uid={os.getenv('SQL_USER')};"
            f"Pwd={os.getenv('SQL_PASSWORD')};"
            "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
        )
        params = urllib.parse.quote_plus(conn_str)
        engine = create_engine(f'mssql+pyodbc:///?autocommit=true&odbc_connect={params}')
        self.sql_db = SQLDatabase(engine, schema="comp")

        # Execution limits
        self.max_recursion = 25
        self.max_results = 5

cfg = Config()

# ===== ELASTICSEARCH TOOLS =====
class ListIndicesInput(BaseModel):
    separator: str = Field(", ", description="Separator between index names")

class IndexDetailsInput(BaseModel):
    index_name: str = Field(..., description="Name of the Elasticsearch index")

class SampleDataInput(BaseModel):
    index_name: str = Field(..., description="Name of the Elasticsearch index")
    size: int = Field(3, description="Number of sample documents to retrieve")

class SearchInput(BaseModel):
    index_name: str = Field(..., description="Name of the Elasticsearch index")
    query: str = Field(..., description="Query in JSON format")
    size: int = Field(5, description="Maximum results to return")

def list_indices(separator: str = ", ") -> str:
    try:
        indices = list(cfg.es.indices.get(index="*").keys())
        return f"Available indices:{separator}{separator.join(indices)}"
    except Exception as e:
        return f"Error listing indices: {str(e)}"

def get_index_details(index_name: str) -> str:
    try:
        if not cfg.es.indices.exists(index=index_name):
            return f"Index '{index_name}' doesn't exist"
        
        details = {
            "aliases": cfg.es.indices.get_alias(index=index_name).get(index_name, {}).get("aliases", {}),
            "mappings": cfg.es.indices.get_mapping(index=index_name).get(index_name, {}).get("mappings", {}),
            "settings": cfg.es.indices.get_settings(index=index_name).get(index_name, {}).get("settings", {})
        }
        return json.dumps(details, indent=2)
    except Exception as e:
        return f"Error getting details: {str(e)}"

def get_sample_data(index_name: str, size: int = 3) -> str:
    try:
        if not cfg.es.indices.exists(index=index_name):
            return f"Index '{index_name}' doesn't exist"
        
        result = cfg.es.search(
            index=index_name,
            body={"query": {"match_all": {}}},
            size=min(size, cfg.max_results)
        )
        hits = result.get('hits', {}).get('hits', [])
        if not hits:
            return f"No documents in index '{index_name}'"
        
        samples = []
        for hit in hits[:size]:
            samples.append({k: v for k, v in hit['_source'].items() if not isinstance(v, (dict, list))})
        return json.dumps(samples, indent=2)
    except Exception as e:
        return f"Error getting samples: {str(e)}"

def elastic_search(index_name: str, query: str, size: int = 5) -> str:
    try:
        if not cfg.es.indices.exists(index=index_name):
            return f"Index '{index_name}' doesn't exist"
        
        query_dict = json.loads(query)
        result = cfg.es.search(
            index=index_name,
            body=query_dict,
            size=min(size, cfg.max_results)
        )
        hits = result.get('hits', {}).get('hits', [])
        return json.dumps([{k: v for k, v in hit['_source'].items() if not isinstance(v, (dict, list))} for hit in hits], indent=2)
    except Exception as e:
        return f"Search error: {str(e)}"

# ===== TOOL SETUP =====
elastic_tools = [
    Tool.from_function(
        func=list_indices,
        name="list_es_indices",
        description="List all Elasticsearch indices",
        args_schema=ListIndicesInput
    ),
    Tool.from_function(
        func=get_index_details,
        name="get_es_index_details",
        description="Get details about an Elasticsearch index",
        args_schema=IndexDetailsInput
    ),
    Tool.from_function(
        func=get_sample_data,
        name="get_es_sample_data",
        description="Get sample documents from an index (default: 3 docs)",
        args_schema=SampleDataInput
    ),
    Tool.from_function(
        func=elastic_search,
        name="search_es_index",
        description="Execute search queries (default: 5 results)",
        args_schema=SearchInput
    )
]

sql_toolkit = SQLDatabaseToolkit(db=cfg.sql_db, llm=cfg.llm)
sql_tools = sql_toolkit.get_tools()

all_tools = elastic_tools + sql_tools
tool_executor = ToolExecutor(all_tools)

# ===== AGENT SETUP =====
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

def create_agent(llm, tools, system_message: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert assistant. Follow these rules:
        1. Be concise and specific
        2. Use minimal necessary tools
        3. When done, say "FINAL ANSWER: [answer]"
        4. If stuck, say "FINAL ANSWER: Need human help"
        5. Tools: {", ".join([t.name for t in tools])}
        {system_message}"""),
        MessagesPlaceholder(variable_name="messages")
    ])
    return prompt | llm.bind_tools(tools)

elastic_agent = create_agent(
    cfg.llm,
    elastic_tools,
    """You are an Elasticsearch expert. Your rules:
    - First check index existence
    - Use minimal sample data
    - Return concise results"""
)

sql_agent = create_agent(
    cfg.llm,
    sql_tools,
    """You are a SQL expert. Your rules:
    - Check table structure first
    - Use LIMIT in queries
    - Return concise results"""
)

# ===== GRAPH SETUP =====
def agent_node(state, agent, name):
    result = agent.invoke(state)
    if not isinstance(result, FunctionMessage):
        result = HumanMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {"messages": [result], "sender": name}

def router(state):
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check for termination conditions
    if any("FINAL ANSWER" in msg.content for msg in messages):
        return "end"
    if len(messages) > 10:
        return "end"
    
    # Route tool calls
    if hasattr(last_message, 'additional_kwargs') and last_message.additional_kwargs.get('function_call'):
        return "call_tool"
    
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
        return {"messages": [FunctionMessage(content=f"Tool error: {str(e)}", name="error")]}

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

# ===== EXECUTION =====
def execute_query(question: str):
    try:
        print(f"\nExecuting: {question}")
        final_state = None
        
        for step in graph.stream(
            {"messages": [HumanMessage(content=question)]},
            {"recursion_limit": cfg.max_recursion}
        ):
            print("\n=== STEP ===")
            print(step)
            final_state = step
            
            # Early termination
            if any("FINAL ANSWER" in str(msg) for msg in step.get("messages", [])):
                break
        
        # Format final output
        if final_state:
            answers = [msg.content for msg in final_state["messages"] if "FINAL ANSWER" in msg.content]
            return {
                "status": "success",
                "answer": answers[-1] if answers else "\n".join([msg.content for msg in final_state["messages"][-2:]]),
                "steps": len(final_state["messages"])
            }
        return {"status": "completed", "answer": "No final answer generated"}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ===== MAIN =====
if __name__ == "__main__":
    # Example query
    question = """Get active customers from Elasticsearch 'customers' index 
    who have made purchases over $1000 in the SQL database. 
    Return: customer_id, name, email, total_purchases"""
    
    result = execute_query(question)
    print("\n=== FINAL RESULT ===")
    print(json.dumps(result, indent=2))
