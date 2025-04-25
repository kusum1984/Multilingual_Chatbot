import os
import json
from typing import Literal, TypedDict, Annotated
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine
from elasticsearch import Elasticsearch
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# Configuration
class Config:
    def __init__(self):
        # Elasticsearch setup
        self.es = Elasticsearch(os.getenv("ELASTIC_ENDPOINT"), 
                              api_key=os.getenv("ELASTIC_API_KEY"))
        
        # SQL setup
        conn_str = 'mssql+pyodbc://user:pass@server/database'
        self.engine = create_engine(conn_str)
        self.sql_db = SQLDatabase(self.engine)
        
        # LLM setup
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)

cfg = Config()

# Define members and options
members = ["sql_agent", "elastic_agent"]
options = members + ["FINISH"]

# Supervisor prompt
system_prompt = f"""
You are a supervisor managing these workers: {members}.
Decide which worker should act next based on the user request.
Respond with either 'sql_agent', 'elastic_agent', or 'FINISH'.
"""

class Router(TypedDict):
    next: Literal["sql_agent", "elastic_agent", "FINISH"]

class State(TypedDict):
    messages: list
    next: str

# Elasticsearch Tools (all 4)
class ElasticsearchTools:
    def __init__(self, es_client):
        self.es = es_client

    def list_indices(self, separator: str = ", ") -> str:
        indices = list(self.es.indices.get(index="*").keys())
        return f"Available indices:{separator}{separator.join(indices)}"

    def get_index_details(self, index_name: str) -> str:
        details = {
            "mappings": self.es.indices.get_mapping(index=index_name),
            "settings": self.es.indices.get_settings(index=index_name)
        }
        return json.dumps(details, indent=2)

    def get_sample_docs(self, index_name: str, size: int = 5) -> str:
        result = self.es.search(index=index_name, body={"query": {"match_all": {}}}, size=size)
        return json.dumps([hit['_source'] for hit in result['hits']['hits']], indent=2)

    def execute_search(self, index_name: str, query: str) -> str:
        query_dict = json.loads(query)
        result = self.es.search(index=index_name, body=query_dict)
        return json.dumps(result['hits'], indent=2)

# Tool Input Schemas
class SearchToolInput(BaseModel):
    index_name: str = Field(..., description="Elasticsearch index name")
    query: str = Field(..., description="JSON query string")

class IndexDetailsInput(BaseModel):
    index_name: str = Field(..., description="Index to inspect")

# Initialize agents
def get_elastic_tools():
    es_tools = ElasticsearchTools(cfg.es)
    return [
        Tool.from_function(
            func=es_tools.list_indices,
            name="list_indices",
            description="List all available indices"
        ),
        Tool.from_function(
            func=es_tools.get_index_details,
            name="get_index_details",
            description="Get index mappings and settings",
            args_schema=IndexDetailsInput
        ),
        Tool.from_function(
            func=es_tools.get_sample_docs,
            name="get_sample_documents",
            description="Get sample documents from index"
        ),
        Tool.from_function(
            func=es_tools.execute_search,
            name="execute_search",
            description="Execute search query",
            args_schema=SearchToolInput
        )
    ]

sql_agent = create_sql_agent(
    llm=cfg.llm,
    toolkit=SQLDatabaseToolkit(db=cfg.sql_db, llm=cfg.llm),
    handle_parsing_errors=True
)

elastic_agent = create_tool_calling_agent(
    llm=cfg.llm,
    tools=get_elastic_tools(),
    prompt=ChatPromptTemplate.from_messages([
        ("system", "You are an Elasticsearch expert"),
        ("user", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])
)

# Nodes
def supervisor_node(state: State) -> Command[Literal["sql_agent", "elastic_agent", "__end__"]]:
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = cfg.llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END
    return Command(goto=goto, update={"next": goto})

def sql_node(state: State) -> Command[Literal["supervisor"]]:
    result = sql_agent.invoke({"input": state["messages"][-1].content})
    return Command(
        update={"messages": [HumanMessage(content=result["output"], name="sql_agent")]},
        goto="supervisor"
    )

def elastic_node(state: State) -> Command[Literal["supervisor"]]:
    result = elastic_agent.invoke({"input": state["messages"][-1].content})
    return Command(
        update={"messages": [HumanMessage(content=result["output"], name="elastic_agent")]},
        goto="supervisor"
    )

# Build graph
builder = StateGraph(State)
builder.add_node("supervisor", supervisor_node)
builder.add_node("sql_agent", sql_node)
builder.add_node("elastic_agent", elastic_node)

builder.add_edge(START, "supervisor")
builder.add_edge("sql_agent", "supervisor")
builder.add_edge("elastic_agent", "supervisor")

builder.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {"sql_agent": "sql_agent", "elastic_agent": "elastic_agent", "FINISH": END}
)

graph = builder.compile()

# Run the graph
def run_query(question: str):
    state = {"messages": [HumanMessage(content=question)], "next": START}
    for step in graph.stream(state):
        print(step["messages"][-1].content)
