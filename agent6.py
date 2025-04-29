from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.tools import Tool as StructureTool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import urllib
import os
import json

# Load environment variables
load_dotenv()

# --------------------- Configuration ---------------------
class Config:
    def __init__(self):
        self.elastic_index_data_from = 0
        self.elastic_index_data_size = 10
        self.elastic_index_data_max_size = 50
        self.aggs_limit = 5
        self.max_search_retries = 3
        self.token_limit = 3000
        self.langchain_verbose = True

        # Elasticsearch client
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

cfg = Config()

# --------------------- SQL Server Connection ---------------------
driver = '{ODBC Driver 18 for SQL Server}'
server = os.getenv("SQL_SERVER")
database = os.getenv("SQL_DATABASE")
user = os.getenv("SQL_USER")
password = os.getenv("SQL_PASSWORD")

conn = f"Driver={driver};Server=tcp:{server},1433;Database={database};Uid={user};Pwd={password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
params = urllib.parse.quote_plus(conn)
conn_str = f'mssql+pyodbc:///?autocommit=true&odbc_connect={params}'
sql_engine = create_engine(conn_str, echo=True)

# --------------------- SQL Agent ---------------------
class SQLAgent:
    def __init__(self, engine):
        self.engine = engine

    def run_query(self, query: str):
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(query))
                rows = result.fetchall()
                headers = result.keys()
                return [dict(zip(headers, row)) for row in rows] if rows else "No results found."
        except Exception as e:
            return f"SQL error: {e}"

sql_agent = SQLAgent(sql_engine)

# --------------------- Elasticsearch Tools ---------------------
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

tools = [
    StructureTool.from_function(func=list_indices, name="elastic_list_indices", description="Lists all available Elasticsearch indices. Always call this first.", args_schema=ListIndicesInput),
    StructureTool.from_function(func=get_index_details, name="elastic_index_details", description="Gets details about a specific index including mappings and settings", args_schema=IndexDetailsInput),
    StructureTool.from_function(func=get_index_data, name="elastic_index_data", description="Gets sample documents from an index", args_schema=IndexDataInput),
    StructureTool.from_function(func=elastic_search, name="elastic_search", description="Executes search or aggregation queries on an index", args_schema=SearchToolInput)
]

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful AI ElasticSearch Expert Assistant."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_openai_functions_agent(llm=cfg.llm, tools=tools, prompt=prompt)
elastic_agent = AgentExecutor(agent=agent, tools=tools, verbose=cfg.langchain_verbose, max_iterations=15, handle_parsing_errors=True, return_intermediate_steps=True)

# --------------------- Decision Router ---------------------
decision_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a router that decides whether a user query should go to the SQL database or the Elasticsearch system."),
    ("user", "Query: {query}\nRespond with only 'sql' or 'elasticsearch'.")
])
decision_chain = decision_prompt | cfg.llm

def route_query(user_query: str) -> str:
    decision = decision_chain.invoke({"query": user_query})
    route = decision.content.strip().lower()
    return route if route in ["sql", "elasticsearch"] else "sql"

# --------------------- Main Execution ---------------------
def answer_user_query(user_query: str):
    route = route_query(user_query)
    print(f"\n🧭 Routed to: {route.upper()}")

    if route == "sql":
        sql_result = sql_agent.run_query(user_query)
        print("📊 SQL Agent Result:\n", sql_result)
        return sql_result

    elif route == "elasticsearch":
        response = elastic_agent.invoke({"input": user_query})
        print("🔍 Elasticsearch Agent Result:\n", response["output"])
        return response["output"]

# Example Usage
if __name__ == "__main__":
    user_input = input("Ask a question: ")
    answer_user_query(user_input)
