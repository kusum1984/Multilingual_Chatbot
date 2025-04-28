from typing import Literal, TypedDict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities.elasticsearch import ElasticsearchStore
from langchain_community.tools.elasticsearch import ElasticsearchDocs
from langchain.schema import Document
from langchain_groq import ChatGroq
import os

# Initialize your components (replace with your actual configurations)
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

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

# Initialize your agents (replace with your actual implementations)
# 1. SQL Agent
db = SQLDatabase.from_uri("your_database_uri")
sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_agent = create_sql_agent(
    llm=llm,
    toolkit=sql_toolkit,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# 2. ElasticSearch Agent
es_store = ElasticsearchStore(
    es_url="your_elasticsearch_url",
    index_name="your_index_name",
    strategy=ElasticsearchStore.SparseVectorRetrievalStrategy()
)
es_tool = ElasticsearchDocs(store=es_store)
es_tools = [es_tool]

es_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant specialized in searching ElasticSearch data."),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

es_agent = create_openai_functions_agent(llm, es_tools, es_prompt)
es_agent_executor = AgentExecutor(agent=es_agent, tools=es_tools, verbose=True)

# Define the graph state
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        sql_result: result from SQL agent
        es_result: result from ElasticSearch agent
    """
    question: str
    generation: str
    documents: List[Document]
    sql_result: str
    es_result: str

# Define nodes
def retrieve_from_sql(state: GraphState):
    """Retrieve data from SQL database using the SQL agent."""
    print("---RETRIEVING FROM SQL---")
    question = state["question"]
    result = sql_agent.invoke({"input": question})
    return {"sql_result": result["output"]}

def retrieve_from_es(state: GraphState):
    """Retrieve data from ElasticSearch using the ES agent."""
    print("---RETRIEVING FROM ELASTICSEARCH---")
    question = state["question"]
    result = es_agent_executor.invoke({"input": question})
    return {"es_result": result["output"]}

def route_question(state: GraphState):
    """
    Route question to SQL or ElasticSearch based on the question content.
    """
    print("---ROUTING QUESTION---")
    question = state["question"]
    source = structured_llm_router.invoke({"question": question})
    
    if source.datasource == "sql_db":
        print("---ROUTING TO SQL AGENT---")
        return "sql_agent"
    elif source.datasource == "elasticsearch":
        print("---ROUTING TO ELASTICSEARCH AGENT---")
        return "es_agent"

def decide_result(state: GraphState):
    """
    Decide which result to use if both agents were run,
    or format the single result if only one was run.
    """
    print("---DECIDING BEST RESULT---")
    sql_res = state.get("sql_result", "")
    es_res = state.get("es_result", "")
    
    if sql_res and es_res:
        # Both results available - choose the more comprehensive one
        if len(sql_res) > len(es_res):
            return {"generation": f"SQL Database provided the most comprehensive answer:\n\n{sql_res}"}
        else:
            return {"generation": f"ElasticSearch provided the most comprehensive answer:\n\n{es_res}"}
    elif sql_res:
        return {"generation": f"SQL Database result:\n\n{sql_res}"}
    elif es_res:
        return {"generation": f"ElasticSearch result:\n\n{es_res}"}
    else:
        return {"generation": "No results found from either data source."}

# Build the workflow
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("route_question", route_question)
workflow.add_node("sql_agent", retrieve_from_sql)
workflow.add_node("es_agent", retrieve_from_es)
workflow.add_node("decide_result", decide_result)

# Add edges
workflow.add_edge("sql_agent", "decide_result")
workflow.add_edge("es_agent", "decide_result")
workflow.add_edge("decide_result", END)

# Conditional edges from router
workflow.add_conditional_edges(
    "route_question",
    route_question,
    {
        "sql_agent": "sql_agent",
        "es_agent": "es_agent",
    }
)

# Set entry point
workflow.set_entry_point("route_question")

# Compile the graph
app = workflow.compile()

# Run the agent
def run_decision_agent(question: str):
    result = app.invoke({"question": question})
    return result["generation"]

# Example usage
if __name__ == "__main__":
    while True:
        user_query = input("Enter your query (or 'quit' to exit): ")
        if user_query.lower() == 'quit':
            break
        print("\nProcessing your query...\n")
        response = run_decision_agent(user_query)
        print("\nResponse:")
        print(response)
        print("\n" + "="*50 + "\n")
