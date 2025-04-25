import os
import json
import threading
import traceback
import urllib.parse
from typing import List, Dict, Any, Literal, Optional, TypedDict
from dataclasses import dataclass, field
from concurrent.futures import TimeoutError
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool, Tool
from langgraph.graph import END, StateGraph
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel as PydanticBaseModel
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_sql_agent
from functools import lru_cache
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from sqlalchemy import create_engine
import logging
from langgraph.types import Command
from typing_extensions import TypedDict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration (unchanged)
class Config:
    def __init__(self):
        # Elasticsearch configuration
        self.elastic_index_data_from = 0
        self.elastic_index_data_size = 10
        self.elastic_index_data_max_size = 50
        self.aggs_limit = 5
        self.max_search_retries = 3
        self.token_limit = 3000
        self.tool_timeout = 60  # seconds
       
        # Initialize Elasticsearch client with API key
        self.es = Elasticsearch(
            os.getenv("ELASTIC_ENDPOINT"),
            api_key=os.getenv("ELASTIC_API_KEY"),
            verify_certs=False
        )
       
        # Initialize LLM - using Azure
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
        self.engine = create_engine(conn_str, echo=True)
        
        # Other configurations
        self.EXCEL_FILE_PATH = os.getenv("EXCEL_FILE_PATH", "data.xlsx")
        self.langchain_verbose = True

    @property
    def sql_db(self):
        """Create SQLDatabase instance with the configured engine"""
        return SQLDatabase(self.engine, schema="comp")

cfg = Config()

# Define our team members (agents)
members = ["sql_agent", "elastic_agent"]
options = members + ["FINISH"]

# Supervisor prompt
system_prompt = f"""
You are a supervisor managing data retrieval operations between:
- SQL Agent: Handles structured database queries
- Elastic Agent: Handles search and document retrieval

Given a user question, decide which agent should handle it next based on:
- Use SQL Agent for database queries, record counts, and structured data
- Use Elastic Agent for text search, document retrieval, and unstructured data
- Respond with FINISH when the task is complete

Available agents: {members}
"""

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal["sql_agent", "elastic_agent", "FINISH"]

class State(TypedDict):
    messages: list
    next: str
    question: str
    results: dict
    errors: list

# Input Models (unchanged)
class ExcelSheetInput(BaseModel):
    sheet_name: Optional[str] = Field(None, description="Name of the Excel sheet to read")

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

# Tools and Agents Implementation (unchanged)
class ElasticsearchTools:
    # ... (keep your existing ElasticsearchTools implementation exactly the same)
    pass

class ExcelTools:
    # ... (keep your existing ExcelTools implementation exactly the same)
    pass

class MemoryManager:
    # ... (keep your existing MemoryManager implementation exactly the same)
    pass

memory_manager = MemoryManager()

# Cached Tool Initialization (unchanged)
@lru_cache(maxsize=None)
def get_elastic_tools():
    # ... (keep your existing get_elastic_tools implementation)
    pass

@lru_cache(maxsize=None)
def get_sql_tools():
    # ... (keep your existing get_sql_tools implementation)
    pass

@lru_cache(maxsize=None)
def get_excel_tools():
    # ... (keep your existing get_excel_tools implementation)
    pass

# Initialize all agents (unchanged)
agents = {
    "sql": create_sql_agent(
        llm=cfg.llm,
        toolkit=SQLDatabaseToolkit(db=cfg.sql_db, llm=cfg.llm),
        verbose=cfg.langchain_verbose,
        handle_parsing_errors=True,
        max_execution_time=cfg.tool_timeout
    ),
    "elastic": create_tool_calling_agent(
        llm=cfg.llm,
        tools=get_elastic_tools(),
        prompt=ChatPromptTemplate.from_messages([
            ("system", "You are an Elasticsearch expert"),
            ("user", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
    )
}

# Supervisor and Agent Nodes
def supervisor_node(state: State) -> Command[Literal["sql_agent", "elastic_agent", "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["question"]}
    ] + state.get("messages", [])
    
    # Get decision from LLM
    response = cfg.llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    
    if goto == "FINISH":
        goto = END

    return Command(goto=goto, update={"next": goto})

def sql_node(state: State) -> Command[Literal["supervisor"]]:
    try:
        result = agents["sql"].invoke({"input": state["question"]})
        return Command(
            update={
                "messages": [
                    HumanMessage(content=str(result["output"]), 
                    AIMessage(content="SQL query completed", name="sql_agent")
                ],
                "results": {**state.get("results", {}), "sql": str(result["output"])}
            },
            goto="supervisor",
        )
    except Exception as e:
        error_msg = f"SQL Agent error: {str(e)}"
        logger.error(error_msg)
        return Command(
            update={
                "messages": [AIMessage(content=error_msg, name="sql_agent")],
                "errors": state.get("errors", []) + [error_msg]
            },
            goto="supervisor",
        )

def elastic_node(state: State) -> Command[Literal["supervisor"]]:
    try:
        result = agents["elastic"].invoke({"input": state["question"]})
        return Command(
            update={
                "messages": [
                    HumanMessage(content=str(result["output"])),
                    AIMessage(content="Elasticsearch query completed", name="elastic_agent")
                ],
                "results": {**state.get("results", {}), "elastic": str(result["output"])}
            },
            goto="supervisor",
        )
    except Exception as e:
        error_msg = f"Elastic Agent error: {str(e)}"
        logger.error(error_msg)
        return Command(
            update={
                "messages": [AIMessage(content=error_msg, name="elastic_agent")],
                "errors": state.get("errors", []) + [error_msg]
            },
            goto="supervisor",
        )

# Build the graph
builder = StateGraph(State)
builder.add_node("supervisor", supervisor_node)
builder.add_node("sql_agent", sql_node)
builder.add_node("elastic_agent", elastic_node)

# Define edges
builder.add_edge(START, "supervisor")
builder.add_edge("supervisor", "sql_agent")
builder.add_edge("supervisor", "elastic_agent")
builder.add_edge("sql_agent", "supervisor")
builder.add_edge("elastic_agent", "supervisor")

# Add conditional edges
builder.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {"sql_agent": "sql_agent", "elastic_agent": "elastic_agent", "FINISH": END}
)

graph = builder.compile()

# Main Application
def main():
    print("Data Assistant (type 'exit' to quit)")
    while True:
        try:
            question = input("\nYou: ").strip()
            if question.lower() in ('exit', 'quit'):
                break
                
            # Initialize state
            state = {
                "question": question,
                "messages": [],
                "results": {},
                "errors": [],
                "next": START
            }
            
            # Execute the graph
            for step in graph.stream(state):
                if "results" in step:
                    print(f"\nIntermediate Results: {step['results']}")
                
            # Get final results
            final_results = step.get("results", {})
            errors = step.get("errors", [])
            
            # Display final answer
            if final_results:
                print("\nFinal Results:")
                for source, result in final_results.items():
                    print(f"{source.upper()}: {result[:500]}...")  # Truncate long outputs
            else:
                print("\nNo results were generated")
                
            if errors:
                print("\nErrors encountered:")
                for error in errors:
                    print(f"- {error}")
                    
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Main loop error: {str(e)}", exc_info=True)
            print(f"\nError: An unexpected error occurred. Please try again.")

if __name__ == "__main__":
    main()
