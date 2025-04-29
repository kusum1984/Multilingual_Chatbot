# === Full Working Multi-Agent System ===

from langchain.agents import AgentExecutor, create_openai_functions_agent, create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.tools import Tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from sqlalchemy import create_engine
import urllib.parse
import os
import json
from typing import List

# === Load environment variables ===
load_dotenv()

# === Configuration ===
class Config:
    def __init__(self):
        # Elasticsearch
        self.elastic_index_data_from = 0
        self.elastic_index_data_size = 10
        self.elastic_index_data_max_size = 50
        self.aggs_limit = 5
        self.max_search_retries = 3
        self.token_limit = 3000
        self.es = Elasticsearch(
            os.getenv("ELASTIC_ENDPOINT"),
            api_key=os.getenv("ELASTIC_API_KEY"),
            verify_certs=False
        )

        # LLM
        self.llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name="gpt-4",
            model="gpt-4",
            temperature=0
        )

        # LangChain verbosity
        self.langchain_verbose = True

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
        self.sql_db = SQLDatabase(engine=self.engine, schema="comp")

cfg = Config()

# === Elasticsearch Agent Tools ===
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

from langchain.tools import StructuredTool

tools = [
    StructuredTool.from_function(func=list_indices, name="elastic_list_indices", description="Lists all available Elasticsearch indices. Always call this first.", args_schema=ListIndicesInput),
    StructuredTool.from_function(func=get_index_details, name="elastic_index_details", description="Gets details about a specific index including mappings and settings", args_schema=IndexDetailsInput),
    StructuredTool.from_function(func=get_index_data, name="elastic_index_data", description="Gets sample documents from an index to understand its structure", args_schema=IndexDataInput),
    StructuredTool.from_function(func=elastic_search, name="elastic_search", description="Executes search or aggregation queries on an Elasticsearch index", args_schema=SearchToolInput),
]

# === Elasticsearch Agent ===
def get_system_prompt(question: str) -> str:
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
        SystemMessage(content="You are a helpful AI ElasticSearch Expert Assistant"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    agent = create_openai_functions_agent(llm=cfg.llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=cfg.langchain_verbose, max_iterations=15, handle_parsing_errors=True, return_intermediate_steps=True)

# === SQL Agent ===
sql_agent = create_sql_agent(
    llm=cfg.llm,
    toolkit=SQLDatabaseToolkit(db=cfg.sql_db, llm=cfg.llm),
    verbose=cfg.langchain_verbose,
    handle_parsing_errors=True
)

# === Decision Agent ===
def route_question(question: str) -> str:
    q_lower = question.lower()
    if any(k in q_lower for k in ["table", "sql", "database", "record", "column", "employee", "department"]):
        return "sql"
    elif any(k in q_lower for k in ["index", "elasticsearch", "search", "document", "query"]):
        return "elastic"
    else:
        return "sql"  # default fallback

# === Main Interface ===
def run_agent(question: str):
    route = route_question(question)
    if route == "sql":
        return sql_agent.invoke({"input": question})
    elif route == "elastic":
        elastic_agent = create_elastic_agent_executor()
        return elastic_agent.invoke({"input": question})
    else:
        return {"output": "Unable to determine which agent to use."}

# Example usage:
# response = run_agent("List all departments from the database")
# print(response["output"])
