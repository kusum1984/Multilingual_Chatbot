import os
from typing import Literal, TypedDict, List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import asyncio
import json
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import Tool, BaseTool
from langgraph.graph import END, StateGraph
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from elasticsearch import Elasticsearch

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Data model for routing
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["sql_db", "elasticsearch"] = Field(
        ...,
        description="Given a user question choose to route it to SQL database or ElasticSearch.",
    )

# Create structured LLM router
structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt for router
system = """You are an expert at routing a user question to either SQL database or ElasticSearch.
The SQL database contains structured relational data with tables, rows, and columns.
ElasticSearch contains document-based data with full-text search capabilities.

Use SQL database for:
- Questions about specific records or relationships between entities
- Questions requiring aggregation, filtering, or joining of structured data
- Questions about database schema or table structures

Use ElasticSearch for:
- Full-text search queries
- Questions about unstructured or semi-structured data
- Questions requiring fuzzy matching or relevance scoring"""
route_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}"),
])

# Initialize SQL Agent
db = SQLDatabase.from_uri("your_database_uri")  # Replace with your actual DB URI
sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_agent = create_sql_agent(
    llm=llm,
    toolkit=sql_toolkit,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=False
)

# ElasticSearch Configuration and Tools
class ElasticConfig:
    def __init__(self):
        self.elastic_index_data_from = 0
        self.elastic_index_data_size = 10
        self.elastic_index_data_max_size = 50
        self.aggs_limit = 5
        self.max_search_retries = 3
        self.token_limit = 3000
        self.langchain_verbose = True

cfg = ElasticConfig()

# Initialize Elasticsearch client
es = Elasticsearch(
    os.getenv("ELASTIC_ENDPOINT"),
    api_key=os.getenv("ELASTIC_API_KEY"),
    verify_certs=False
)

# Tool Input Models
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
        indices = list(es.indices.get(index="*").keys())
        return f"Available indices:{separator}{separator.join(indices)}"
    except Exception as e:
        return f"Error listing indices: {str(e)}"

def get_index_details(index_name: str) -> str:
    """Gets details about a specific index including mappings and settings"""
    try:
        if not es.indices.exists(index=index_name):
            return f"Index '{index_name}' does not exist"
        details = {
            "aliases": es.indices.get_alias(index=index_name).get(index_name, {}).get("aliases", {}),
            "mappings": es.indices.get_mapping(index=index_name).get(index_name, {}).get("mappings", {}),
            "settings": es.indices.get_settings(index=index_name).get(index_name, {}).get("settings", {})
        }
        return json.dumps(details, indent=2)
    except Exception as e:
        return f"Error getting index details: {str(e)}"

def get_index_data(index_name: str, size: int = 5) -> str:
    """Gets sample documents from an index to understand its structure"""
    try:
        if not es.indices.exists(index=index_name):
            return f"Index '{index_name}' does not exist"
        result = es.search(
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
        if not es.indices.exists(index=index_name):
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

        result = es.search(
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
es_tools = [
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
def get_es_system_prompt(question: str) -> str:
    return f"""You are an Elasticsearch expert assistant. Follow these steps for every request:
1. First list all available indices using elastic_list_indices
2. Then examine index details or sample data as needed
3. Finally execute specific searches when you understand the data structure

Make sure you understand the index structure before querying data.
Question to answer: {question}"""

def create_es_agent_executor() -> AgentExecutor:
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful AI ElasticSearch Expert Assistant"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_openai_functions_agent(
        llm=llm,
        tools=es_tools,
        prompt=prompt
    )
    
    return AgentExecutor(
        agent=agent,
        tools=es_tools,
        verbose=cfg.langchain_verbose,
        max_iterations=15,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

es_agent_executor = create_es_agent_executor()

# Enhanced graph state with additional metadata
class EnhancedGraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    sql_result: Optional[str]
    es_result: Optional[str]
    sql_confidence: Optional[float]
    es_confidence: Optional[float]
    sql_time: Optional[float]
    es_time: Optional[float]
    routing_decision: Optional[str]
    combined_result: Optional[bool]
    error: Optional[str]

def run_parallel_queries(state: EnhancedGraphState) -> EnhancedGraphState:
    """Execute both agents in parallel and collect results with metrics."""
    question = state["question"]
    
    def run_agent(agent_func, input_data, agent_type):
        import time
        start_time = time.time()
        try:
            result = agent_func(input_data)
            elapsed = time.time() - start_time
            
            # Calculate confidence based on response length and content
            output = result.get("output", "") if isinstance(result, dict) else str(result)
            confidence = min(0.3 + (len(output) / 1000), 0.95)  # Basic confidence heuristic
            
            return {
                "result": output,
                "time": elapsed,
                "confidence": confidence,
                "error": None
            }
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "result": str(e),
                "time": elapsed,
                "confidence": 0.0,
                "error": str(e)
            }
    
    with ThreadPoolExecutor() as executor:
        sql_future = executor.submit(
            run_agent, 
            sql_agent.invoke, 
            {"input": question},
            "sql"
        )
        es_future = executor.submit(
            run_agent,
            es_agent_executor.invoke,
            {"input": question},
            "es"
        )
        
        sql_response = sql_future.result()
        es_response = es_future.result()
    
    return {
        "question": question,
        "sql_result": sql_response["result"],
        "es_result": es_response["result"],
        "sql_confidence": sql_response["confidence"],
        "es_confidence": es_response["confidence"],
        "sql_time": sql_response["time"],
        "es_time": es_response["time"],
        "sql_error": sql_response["error"],
        "es_error": es_response["error"]
    }

async def route_question(state: EnhancedGraphState) -> Dict[str, Any]:
    """Determine which data source is most appropriate for the question."""
    question = state["question"]
    try:
        source = await structured_llm_router.ainvoke({"question": question})
        return {
            "routing_decision": source.datasource,
            "question": question
        }
    except Exception as e:
        return {
            "routing_decision": "sql_db",  # Default to SQL on routing failure
            "question": question,
            "error": f"Routing error: {str(e)}"
        }

def analyze_results(state: EnhancedGraphState) -> EnhancedGraphState:
    """Analyze and combine results from both agents with sophisticated logic."""
    question = state["question"]
    sql_res = state.get("sql_result", "")
    es_res = state.get("es_result", "")
    sql_conf = state.get("sql_confidence", 0)
    es_conf = state.get("es_confidence", 0)
    routing = state.get("routing_decision", "sql_db")
    
    # Validate results
    def is_valid_result(result):
        if not result or not isinstance(result, str):
            return False
        result_lower = result.lower()
        error_indicators = [
            "error", "not found", "no results", "exception",
            "does not exist", "unable to", "failed"
        ]
        return not any(indicator in result_lower for indicator in error_indicators)
    
    sql_valid = is_valid_result(sql_res)
    es_valid = is_valid_result(es_res)
    
    # Decision logic
    if sql_valid and es_valid:
        # Both valid - combine based on confidence and routing
        if routing == "sql_db" and sql_conf >= es_conf * 0.7:
            selected = sql_res
            source = "SQL Database (primary choice)"
            combined = False
            if es_conf > 0.6:
                selected += f"\n\nElasticSearch corroborates this with {es_conf:.0%} confidence."
                combined = True
        elif routing == "elasticsearch" and es_conf >= sql_conf * 0.7:
            selected = es_res
            source = "ElasticSearch (primary choice)"
            combined = False
            if sql_conf > 0.6:
                selected += f"\n\nSQL Database corroborates this with {sql_conf:.0%} confidence."
                combined = True
        else:
            # Present both with explanation
            selected = (f"## SQL Database Response (confidence: {sql_conf:.0%}):\n{sql_res}\n\n"
                       f"## ElasticSearch Response (confidence: {es_conf:.0%}):\n{es_res}\n\n"
                       "The systems provided different answers. Please evaluate both responses.")
            source = "Both systems"
            combined = True
    elif sql_valid:
        selected = sql_res
        source = "SQL Database"
        combined = False
        if routing == "elasticsearch":
            selected += "\n\nNote: ElasticSearch didn't return valid results, showing SQL results instead."
    elif es_valid:
        selected = es_res
        source = "ElasticSearch"
        combined = False
        if routing == "sql_db":
            selected += "\n\nNote: SQL Database didn't return valid results, showing ElasticSearch results instead."
    else:
        selected = "I couldn't find relevant information in either data source."
        source = "Neither system"
        combined = False
    
    # Format final response
    generation = f"## Response from {source}:\n\n{selected}"
    if combined:
        generation += "\n\nNote: Results were combined from both data sources."
    
    return {
        "generation": generation,
        "combined_result": combined,
        "question": question,
        "routing_decision": routing
    }

def format_final_response(state: EnhancedGraphState) -> Dict[str, Any]:
    """Prepare the final response with metadata."""
    base_response = {
        "answer": state["generation"],
        "question": state["question"],
        "metadata": {
            "sources_used": {
                "sql": bool(state.get("sql_result")),
                "elasticsearch": bool(state.get("es_result"))
            },
            "confidence": {
                "sql": state.get("sql_confidence", 0),
                "elasticsearch": state.get("es_confidence", 0)
            },
            "response_times": {
                "sql_seconds": state.get("sql_time", 0),
                "elasticsearch_seconds": state.get("es_time", 0)
            },
            "routing_decision": state.get("routing_decision", "unknown"),
            "results_combined": state.get("combined_result", False),
            "timestamp": datetime.now().isoformat()
        }
    }
    
    # Include errors if any
    if state.get("sql_error"):
        base_response["metadata"]["errors"] = {"sql": state["sql_error"]}
    if state.get("es_error"):
        if "errors" not in base_response["metadata"]:
            base_response["metadata"]["errors"] = {}
        base_response["metadata"]["errors"]["elasticsearch"] = state["es_error"]
    
    return {"generation": base_response["answer"]}

# Build the workflow
workflow = StateGraph(EnhancedGraphState)

# Add nodes
workflow.add_node("parallel_query", run_parallel_queries)
workflow.add_node("route_question", route_question)
workflow.add_node("analyze_results", analyze_results)
workflow.add_node("format_response", format_final_response)

# Set up edges
workflow.add_edge("parallel_query", "analyze_results")
workflow.add_edge("route_question", "analyze_results")
workflow.add_edge("analyze_results", "format_response")
workflow.add_edge("format_response", END)

# Set entry point
workflow.set_entry_point("parallel_query")

# Add conditional edge for routing
workflow.add_conditional_edges(
    "parallel_query",
    lambda state: "route_question",
    {"route_question": "route_question"}
)

# Compile the graph
app = workflow.compile()

async def run_decision_agent(question: str):
    """Run the decision agent with a given question."""
    try:
        result = await app.ainvoke({"question": question})
        return result
    except Exception as e:
        return {
            "generation": f"An error occurred while processing your request: {str(e)}",
            "error": True
        }

# Example usage
if __name__ == "__main__":
    async def main():
        print("Multi-Agent Decision System")
        print("Enter your query or 'quit' to exit.\n")
        
        while True:
            user_query = input("Query: ").strip()
            if user_query.lower() in ('quit', 'exit'):
                break
                
            print("\nProcessing...\n")
            response = await run_decision_agent(user_query)
            
            if "generation" in response:
                print(response["generation"])
            else:
                print("Unexpected response format:", response)
            
            print("\n" + "="*80 + "\n")
    
    asyncio.run(main())
====================================================

# Old version:
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    sql_result: str
    es_result: str

# New version:
class State(TypedDict):
    question: Annotated[str, "The user's question"]
    generation: Annotated[Optional[str], "The final generated answer"]
    sql_result: Annotated[Optional[str], "Result from SQL agent"]
    es_result: Annotated[Optional[str], "Result from ElasticSearch agent"]
    sql_confidence: Annotated[Optional[float], "Confidence score from SQL agent"]
    es_confidence: Annotated[Optional[float], "Confidence score from ES agent"]
    sql_time: Annotated[Optional[float], "Time taken by SQL agent"]
    es_time: Annotated[Optional[float], "Time taken by ES agent"]
    routing_decision: Annotated[Optional[str], "Routing decision"]
    combined_result: Annotated[Optional[bool], "Whether results were combined"]
    error: Annotated[Optional[str], "Error message if any"]



************
# Old version:
workflow = StateGraph(GraphState)

# New version:
workflow = StateGraph(State)

*********
# Example for one function - all node functions were updated similarly:
def run_parallel_queries(state: State) -> Dict[str, Any]:  # Changed return type
    # ... implementation ...
    return {
        "question": question,  # Only returning fields we modify
        "sql_result": sql_response["result"],
        "es_result": es_response["result"],
        # ... other fields ...
    }********

================================================
    import os
from typing import Literal, TypedDict, List, Optional, Dict, Any, Annotated
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import asyncio
import json
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import Tool
from langgraph.graph import END, StateGraph
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from elasticsearch import Elasticsearch

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Data model for routing
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["sql_db", "elasticsearch"] = Field(
        ...,
        description="Given a user question choose to route it to SQL database or ElasticSearch.",
    )

# Create structured LLM router
structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt for router
system = """You are an expert at routing a user question to either SQL database or ElasticSearch.
The SQL database contains structured relational data with tables, rows, and columns.
ElasticSearch contains document-based data with full-text search capabilities.

Use SQL database for:
- Questions about specific records or relationships between entities
- Questions requiring aggregation, filtering, or joining of structured data
- Questions about database schema or table structures

Use ElasticSearch for:
- Full-text search queries
- Questions about unstructured or semi-structured data
- Questions requiring fuzzy matching or relevance scoring"""
route_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}"),
])

# Initialize SQL Agent
db = SQLDatabase.from_uri("your_database_uri")  # Replace with your actual DB URI
sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_agent = create_sql_agent(
    llm=llm,
    toolkit=sql_toolkit,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=False
)

# ElasticSearch Configuration and Tools
class ElasticConfig:
    def __init__(self):
        self.elastic_index_data_from = 0
        self.elastic_index_data_size = 10
        self.elastic_index_data_max_size = 50
        self.aggs_limit = 5
        self.max_search_retries = 3
        self.token_limit = 3000
        self.langchain_verbose = True

cfg = ElasticConfig()

# Initialize Elasticsearch client
es = Elasticsearch(
    os.getenv("ELASTIC_ENDPOINT"),
    api_key=os.getenv("ELASTIC_API_KEY"),
    verify_certs=False
)

# Tool Input Models
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
        indices = list(es.indices.get(index="*").keys())
        return f"Available indices:{separator}{separator.join(indices)}"
    except Exception as e:
        return f"Error listing indices: {str(e)}"

def get_index_details(index_name: str) -> str:
    """Gets details about a specific index including mappings and settings"""
    try:
        if not es.indices.exists(index=index_name):
            return f"Index '{index_name}' does not exist"
        details = {
            "aliases": es.indices.get_alias(index=index_name).get(index_name, {}).get("aliases", {}),
            "mappings": es.indices.get_mapping(index=index_name).get(index_name, {}).get("mappings", {}),
            "settings": es.indices.get_settings(index=index_name).get(index_name, {}).get("settings", {})
        }
        return json.dumps(details, indent=2)
    except Exception as e:
        return f"Error getting index details: {str(e)}"

def get_index_data(index_name: str, size: int = 5) -> str:
    """Gets sample documents from an index to understand its structure"""
    try:
        if not es.indices.exists(index=index_name):
            return f"Index '{index_name}' does not exist"
        result = es.search(
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
        if not es.indices.exists(index=index_name):
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

        result = es.search(
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
es_tools = [
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
def create_es_agent_executor() -> AgentExecutor:
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful AI ElasticSearch Expert Assistant"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_openai_functions_agent(
        llm=llm,
        tools=es_tools,
        prompt=prompt
    )
    
    return AgentExecutor(
        agent=agent,
        tools=es_tools,
        verbose=cfg.langchain_verbose,
        max_iterations=15,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

es_agent_executor = create_es_agent_executor()

# Define the State with Annotated fields
class State(TypedDict):
    question: Annotated[str, "The user's question"]
    generation: Annotated[Optional[str], "The final generated answer"]
    sql_result: Annotated[Optional[str], "Result from SQL agent"]
    es_result: Annotated[Optional[str], "Result from ElasticSearch agent"]
    sql_confidence: Annotated[Optional[float], "Confidence score from SQL agent"]
    es_confidence: Annotated[Optional[float], "Confidence score from ES agent"]
    sql_time: Annotated[Optional[float], "Time taken by SQL agent"]
    es_time: Annotated[Optional[float], "Time taken by ES agent"]
    routing_decision: Annotated[Optional[str], "Routing decision"]
    combined_result: Annotated[Optional[bool], "Whether results were combined"]
    error: Annotated[Optional[str], "Error message if any"]

def run_parallel_queries(state: State) -> State:
    """Execute both agents in parallel and collect results with metrics."""
    question = state["question"]
    
    def run_agent(agent_func, input_data, agent_type):
        import time
        start_time = time.time()
        try:
            result = agent_func(input_data)
            elapsed = time.time() - start_time
            
            # Calculate confidence based on response length and content
            output = result.get("output", "") if isinstance(result, dict) else str(result)
            confidence = min(0.3 + (len(output) / 1000), 0.95)  # Basic confidence heuristic
            
            return {
                "result": output,
                "time": elapsed,
                "confidence": confidence,
                "error": None
            }
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "result": str(e),
                "time": elapsed,
                "confidence": 0.0,
                "error": str(e)
            }
    
    with ThreadPoolExecutor() as executor:
        sql_future = executor.submit(
            run_agent, 
            sql_agent.invoke, 
            {"input": question},
            "sql"
        )
        es_future = executor.submit(
            run_agent,
            es_agent_executor.invoke,
            {"input": question},
            "es"
        )
        
        sql_response = sql_future.result()
        es_response = es_future.result()
    
    return {
        "question": question,
        "sql_result": sql_response["result"],
        "es_result": es_response["result"],
        "sql_confidence": sql_response["confidence"],
        "es_confidence": es_response["confidence"],
        "sql_time": sql_response["time"],
        "es_time": es_response["time"],
        "sql_error": sql_response["error"],
        "es_error": es_response["error"]
    }

async def route_question(state: State) -> Dict[str, Any]:
    """Determine which data source is most appropriate for the question."""
    question = state["question"]
    try:
        source = await structured_llm_router.ainvoke({"question": question})
        return {
            "routing_decision": source.datasource,
            "question": question
        }
    except Exception as e:
        return {
            "routing_decision": "sql_db",  # Default to SQL on routing failure
            "question": question,
            "error": f"Routing error: {str(e)}"
        }

def analyze_results(state: State) -> State:
    """Analyze and combine results from both agents with sophisticated logic."""
    question = state["question"]
    sql_res = state.get("sql_result", "")
    es_res = state.get("es_result", "")
    sql_conf = state.get("sql_confidence", 0)
    es_conf = state.get("es_confidence", 0)
    routing = state.get("routing_decision", "sql_db")
    
    # Validate results
    def is_valid_result(result):
        if not result or not isinstance(result, str):
            return False
        result_lower = result.lower()
        error_indicators = [
            "error", "not found", "no results", "exception",
            "does not exist", "unable to", "failed"
        ]
        return not any(indicator in result_lower for indicator in error_indicators)
    
    sql_valid = is_valid_result(sql_res)
    es_valid = is_valid_result(es_res)
    
    # Decision logic
    if sql_valid and es_valid:
        # Both valid - combine based on confidence and routing
        if routing == "sql_db" and sql_conf >= es_conf * 0.7:
            selected = sql_res
            source = "SQL Database (primary choice)"
            combined = False
            if es_conf > 0.6:
                selected += f"\n\nElasticSearch corroborates this with {es_conf:.0%} confidence."
                combined = True
        elif routing == "elasticsearch" and es_conf >= sql_conf * 0.7:
            selected = es_res
            source = "ElasticSearch (primary choice)"
            combined = False
            if sql_conf > 0.6:
                selected += f"\n\nSQL Database corroborates this with {sql_conf:.0%} confidence."
                combined = True
        else:
            # Present both with explanation
            selected = (f"## SQL Database Response (confidence: {sql_conf:.0%}):\n{sql_res}\n\n"
                       f"## ElasticSearch Response (confidence: {es_conf:.0%}):\n{es_res}\n\n"
                       "The systems provided different answers. Please evaluate both responses.")
            source = "Both systems"
            combined = True
    elif sql_valid:
        selected = sql_res
        source = "SQL Database"
        combined = False
        if routing == "elasticsearch":
            selected += "\n\nNote: ElasticSearch didn't return valid results, showing SQL results instead."
    elif es_valid:
        selected = es_res
        source = "ElasticSearch"
        combined = False
        if routing == "sql_db":
            selected += "\n\nNote: SQL Database didn't return valid results, showing ElasticSearch results instead."
    else:
        selected = "I couldn't find relevant information in either data source."
        source = "Neither system"
        combined = False
    
    # Format final response
    generation = f"## Response from {source}:\n\n{selected}"
    if combined:
        generation += "\n\nNote: Results were combined from both data sources."
    
    return {
        "generation": generation,
        "combined_result": combined,
        "question": question,
        "routing_decision": routing
    }

def format_final_response(state: State) -> Dict[str, Any]:
    """Prepare the final response with metadata."""
    base_response = {
        "answer": state["generation"],
        "question": state["question"],
        "metadata": {
            "sources_used": {
                "sql": bool(state.get("sql_result")),
                "elasticsearch": bool(state.get("es_result"))
            },
            "confidence": {
                "sql": state.get("sql_confidence", 0),
                "elasticsearch": state.get("es_confidence", 0)
            },
            "response_times": {
                "sql_seconds": state.get("sql_time", 0),
                "elasticsearch_seconds": state.get("es_time", 0)
            },
            "routing_decision": state.get("routing_decision", "unknown"),
            "results_combined": state.get("combined_result", False),
            "timestamp": datetime.now().isoformat()
        }
    }
    
    # Include errors if any
    if state.get("sql_error"):
        base_response["metadata"]["errors"] = {"sql": state["sql_error"]}
    if state.get("es_error"):
        if "errors" not in base_response["metadata"]:
            base_response["metadata"]["errors"] = {}
        base_response["metadata"]["errors"]["elasticsearch"] = state["es_error"]
    
    return {"generation": base_response["answer"]}

# Build the workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("parallel_query", run_parallel_queries)
workflow.add_node("route_question", route_question)
workflow.add_node("analyze_results", analyze_results)
workflow.add_node("format_response", format_final_response)

# Set up edges
workflow.add_edge("parallel_query", "analyze_results")
workflow.add_edge("route_question", "analyze_results")
workflow.add_edge("analyze_results", "format_response")
workflow.add_edge("format_response", END)

# Set entry point
workflow.set_entry_point("parallel_query")

# Add conditional edge for routing
workflow.add_conditional_edges(
    "parallel_query",
    lambda state: "route_question",
    {"route_question": "route_question"}
)

# Compile the graph
app = workflow.compile()

async def run_decision_agent(question: str):
    """Run the decision agent with a given question."""
    try:
        result = await app.ainvoke({"question": question})
        return result
    except Exception as e:
        return {
            "generation": f"An error occurred while processing your request: {str(e)}",
            "error": True
        }

# Example usage
if __name__ == "__main__":
    async def main():
        print("Multi-Agent Decision System")
        print("Enter your query or 'quit' to exit.\n")
        
        while True:
            user_query = input("Query: ").strip()
            if user_query.lower() in ('quit', 'exit'):
                break
                
            print("\nProcessing...\n")
            response = await run_decision_agent(user_query)
            
            if "generation" in response:
                print(response["generation"])
            else:
                print("Unexpected response format:", response)
            
            print("\n" + "="*80 + "\n")
    
    asyncio.run(main())

*********************************
import os
from typing import Literal, TypedDict, List, Optional, Dict, Any, Annotated
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import asyncio
import json
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import Tool
from langgraph.graph import END, StateGraph
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from elasticsearch import Elasticsearch

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Data model for routing
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["sql_db", "elasticsearch"] = Field(
        ...,
        description="Given a user question choose to route it to SQL database or ElasticSearch.",
    )

# Create structured LLM router
structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt for router
system = """You are an expert at routing a user question to either SQL database or ElasticSearch.
The SQL database contains structured relational data with tables, rows, and columns.
ElasticSearch contains document-based data with full-text search capabilities.

Use SQL database for:
- Questions about specific records or relationships between entities
- Questions requiring aggregation, filtering, or joining of structured data
- Questions about database schema or table structures

Use ElasticSearch for:
- Full-text search queries
- Questions about unstructured or semi-structured data
- Questions requiring fuzzy matching or relevance scoring"""
route_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}"),
])

# Initialize SQL Agent (unchanged)
db = SQLDatabase.from_uri("your_database_uri")  # Replace with your actual DB URI
sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_agent = create_sql_agent(
    llm=llm,
    toolkit=sql_toolkit,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=False
)

# ElasticSearch Configuration and Tools (unchanged)
class ElasticConfig:
    def __init__(self):
        self.elastic_index_data_from = 0
        self.elastic_index_data_size = 10
        self.elastic_index_data_max_size = 50
        self.aggs_limit = 5
        self.max_search_retries = 3
        self.token_limit = 3000
        self.langchain_verbose = True

cfg = ElasticConfig()

# Initialize Elasticsearch client (unchanged)
es = Elasticsearch(
    os.getenv("ELASTIC_ENDPOINT"),
    api_key=os.getenv("ELASTIC_API_KEY"),
    verify_certs=False
)

# Tool Input Models (unchanged)
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

# Elasticsearch Tools Implementation (unchanged)
def list_indices(separator: str = ", ") -> str:
    """Lists all available Elasticsearch indices"""
    try:
        indices = list(es.indices.get(index="*").keys())
        return f"Available indices:{separator}{separator.join(indices)}"
    except Exception as e:
        return f"Error listing indices: {str(e)}"

def get_index_details(index_name: str) -> str:
    """Gets details about a specific index including mappings and settings"""
    try:
        if not es.indices.exists(index=index_name):
            return f"Index '{index_name}' does not exist"
        details = {
            "aliases": es.indices.get_alias(index=index_name).get(index_name, {}).get("aliases", {}),
            "mappings": es.indices.get_mapping(index=index_name).get(index_name, {}).get("mappings", {}),
            "settings": es.indices.get_settings(index=index_name).get(index_name, {}).get("settings", {})
        }
        return json.dumps(details, indent=2)
    except Exception as e:
        return f"Error getting index details: {str(e)}"

def get_index_data(index_name: str, size: int = 5) -> str:
    """Gets sample documents from an index to understand its structure"""
    try:
        if not es.indices.exists(index=index_name):
            return f"Index '{index_name}' does not exist"
        result = es.search(
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
        if not es.indices.exists(index=index_name):
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

        result = es.search(
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

# Create Elasticsearch Tools (unchanged)
es_tools = [
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

# Elasticsearch Agent Setup (unchanged)
def create_es_agent_executor() -> AgentExecutor:
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful AI ElasticSearch Expert Assistant"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_openai_functions_agent(
        llm=llm,
        tools=es_tools,
        prompt=prompt
    )
    
    return AgentExecutor(
        agent=agent,
        tools=es_tools,
        verbose=cfg.langchain_verbose,
        max_iterations=15,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

es_agent_executor = create_es_agent_executor()

# Define the State with Annotated fields (updated)
class State(TypedDict):
    question: Annotated[str, "The user's question"]
    generation: Annotated[Optional[str], "The final generated answer"]
    sql_result: Annotated[Optional[str], "Result from SQL agent"]
    es_result: Annotated[Optional[str], "Result from ElasticSearch agent"]
    sql_confidence: Annotated[Optional[float], "Confidence score from SQL agent"]
    es_confidence: Annotated[Optional[float], "Confidence score from ES agent"]
    sql_time: Annotated[Optional[float], "Time taken by SQL agent"]
    es_time: Annotated[Optional[float], "Time taken by ES agent"]
    routing_decision: Annotated[Optional[str], "Routing decision"]
    combined_result: Annotated[Optional[bool], "Whether results were combined"]
    error: Annotated[Optional[str], "Error message if any"]

# Updated parallel query execution
def run_parallel_queries(state: State) -> Dict[str, Any]:
    """Execute both agents in parallel and collect results with metrics."""
    question = state["question"]
    
    def run_agent(agent_func, input_data, agent_type):
        import time
        start_time = time.time()
        try:
            result = agent_func({"input": input_data})
            elapsed = time.time() - start_time
            
            # Get output from agent result
            output = result.get("output", "") if isinstance(result, dict) else str(result)
            confidence = min(0.3 + (len(output) / 1000), 0.95)
            
            return {
                "result": output,
                "time": elapsed,
                "confidence": confidence,
                "error": None
            }
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "result": str(e),
                "time": elapsed,
                "confidence": 0.0,
                "error": str(e)
            }
    
    with ThreadPoolExecutor() as executor:
        sql_future = executor.submit(run_agent, sql_agent.invoke, question, "sql")
        es_future = executor.submit(run_agent, es_agent_executor.invoke, question, "es")
        
        sql_response = sql_future.result()
        es_response = es_future.result()
    
    return {
        "question": question,
        "sql_result": sql_response["result"],
        "es_result": es_response["result"],
        "sql_confidence": sql_response["confidence"],
        "es_confidence": es_response["confidence"],
        "sql_time": sql_response["time"],
        "es_time": es_response["time"],
        "sql_error": sql_response["error"],
        "es_error": es_response["error"]
    }

# Rest of the functions remain the same as in previous implementation
# (route_question, analyze_results, format_final_response)

# Build the workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("parallel_query", run_parallel_queries)
workflow.add_node("route_question", route_question)
workflow.add_node("analyze_results", analyze_results)
workflow.add_node("format_response", format_final_response)

# Set up edges
workflow.add_edge("parallel_query", "analyze_results")
workflow.add_edge("route_question", "analyze_results")
workflow.add_edge("analyze_results", "format_response")
workflow.add_edge("format_response", END)

# Set entry point
workflow.set_entry_point("parallel_query")

# Add conditional edge for routing
workflow.add_conditional_edges(
    "parallel_query",
    lambda state: "route_question",
    {"route_question": "route_question"}
)

# Compile the graph
app = workflow.compile()

async def run_decision_agent(question: str):
    """Run the decision agent with a given question."""
    try:
        result = await app.ainvoke({"question": question})
        return result
    except Exception as e:
        return {
            "generation": f"An error occurred while processing your request: {str(e)}",
            "error": True
        }

# Example usage
if __name__ == "__main__":
    async def main():
        print("Multi-Agent Decision System")
        print("Enter your query or 'quit' to exit.\n")
        
        while True:
            user_query = input("Query: ").strip()
            if user_query.lower() in ('quit', 'exit'):
                break
                
            print("\nProcessing...\n")
            response = await run_decision_agent(user_query)
            
            if "generation" in response:
                print(response["generation"])
            else:
                print("Unexpected response format:", response)
            
            print("\n" + "="*80 + "\n")
    
    asyncio.run(main())
