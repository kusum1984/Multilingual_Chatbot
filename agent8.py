# === Full Working LangGraph-based Multi-Agent System ===

from langchain.agents import create_sql_agent, create_openai_functions_agent, AgentExecutor
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import Tool
from langchain_openai import AzureChatOpenAI
from elasticsearch import Elasticsearch
from langchain.tools import StructuredTool
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from sqlalchemy import create_engine
import urllib.parse
import os
import json

# === Load environment variables ===
load_dotenv()

# === Configuration ===
class Config:
    def __init__(self):
        self.elastic_index_data_max_size = 50
        self.aggs_limit = 5
        self.langchain_verbose = True

        self.es = Elasticsearch(
            os.getenv("ELASTIC_ENDPOINT"),
            api_key=os.getenv("ELASTIC_API_KEY"),
            verify_certs=False
        )

        self.llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name="gpt-4",
            model="gpt-4",
            temperature=0
        )

        driver = '{ODBC Driver 18 for SQL Server}'
        server = os.getenv("SQL_SERVER")
        database = os.getenv("SQL_DATABASE")
        user = os.getenv("SQL_USER")
        password = os.getenv("SQL_PASSWORD")

        conn = f"Driver={driver};Server=tcp:{server},1433;Database={database};Uid={user};Pwd={password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
        params = urllib.parse.quote_plus(conn)
        conn_str = f'mssql+pyodbc:///?autocommit=true&odbc_connect={params}'
        self.engine = create_engine(conn_str, echo=True)
        self.sql_db = SQLDatabase(engine=self.engine, schema="comp")

cfg = Config()

# === Elasticsearch Tools ===
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
        result = cfg.es.search(index=index_name, body={"query": {"match_all": {}}}, size=min(size, cfg.elastic_index_data_max_size))
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
        query_dict = json.loads(query)
        is_aggregation = "aggs" in query_dict or "aggregations" in query_dict
        if is_aggregation:
            size = cfg.aggs_limit
        result = cfg.es.search(index=index_name, body=query_dict, from_=from_, size=size)
        return json.dumps(result.get('aggregations' if is_aggregation else 'hits', {}), indent=2)
    except Exception as e:
        return f"Search error: {str(e)}"

# === Tools ===
tools = [
    StructuredTool.from_function(func=list_indices, name="elastic_list_indices", description="Lists all available Elasticsearch indices.", args_schema=ListIndicesInput),
    StructuredTool.from_function(func=get_index_details, name="elastic_index_details", description="Gets details about an index.", args_schema=IndexDetailsInput),
    StructuredTool.from_function(func=get_index_data, name="elastic_index_data", description="Gets sample documents from an index.", args_schema=IndexDataInput),
    StructuredTool.from_function(func=elastic_search, name="elastic_search", description="Searches data in an index.", args_schema=SearchToolInput),
]

# === Agents ===
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful ElasticSearch Expert Assistant."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

elastic_agent_executor = AgentExecutor(
    agent=create_openai_functions_agent(llm=cfg.llm, tools=tools, prompt=prompt),
    tools=tools,
    verbose=cfg.langchain_verbose,
    max_iterations=15,
    handle_parsing_errors=True,
    return_intermediate_steps=True
)

sql_agent_executor = create_sql_agent(
    llm=cfg.llm,
    toolkit=SQLDatabaseToolkit(db=cfg.sql_db, llm=cfg.llm),
    verbose=cfg.langchain_verbose,
    handle_parsing_errors=True
)

# === LangGraph Setup ===
from langgraph.graph import StateGraph
from typing import TypedDict, Union

class AgentState(TypedDict):
    question: str
    sql_result: Union[str, None]
    elastic_result: Union[str, None]
    final_answer: Union[str, None]

def sql_node(state: AgentState):
    result = sql_agent_executor.invoke({"input": state["question"]})
    return {"sql_result": result.get("output", "")}

def elastic_node(state: AgentState):
    result = elastic_agent_executor.invoke({"input": state["question"]})
    return {"elastic_result": result.get("output", "")}

def combine_node(state: AgentState):
    return {
        "final_answer": (
            f"SQL Agent Answer:\n{state['sql_result']}\n\n"
            f"Elasticsearch Agent Answer:\n{state['elastic_result']}"
        )
    }

# === Graph ===
graph = StateGraph(AgentState)
graph.add_node("sql_agent", sql_node)
graph.add_node("elastic_agent", elastic_node)
graph.add_node("combine", combine_node)
graph.set_entry_point("sql_agent")
graph.add_edge("sql_agent", "elastic_agent")
graph.add_edge("elastic_agent", "combine")
graph.add_edge("combine", END)

app = graph.compile()

# === Usage ===
if __name__ == "__main__":
    question = "What are the top selling products and their descriptions from search and database?"
    result = app.invoke({"question": question})
    print("\n=== Final Combined Answer ===")
    print(result["final_answer"])
