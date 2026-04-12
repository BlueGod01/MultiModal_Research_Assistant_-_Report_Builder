from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain_core.runnables import chain
from RAG_Pipeline.retriever import retrieve_multimodal, query_corrected_retrieval
from RAG_Pipeline.reranker import reranked_retriever
from typing import Dict, List
from pydantic import BaseModel, Field, HttpUrl
from Graph_Workflow.Schema import ResearchResponse, RAG_Response, Web_Response
from Graph_Workflow.Schema import QAresearchstate
from langgraph.graph import StateGraph, END
import re
from tavily import TavilyClient

from langchain.tools import tool
from Brain import google_llm
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

# -----------------------------
# Tavily Client
# -----------------------------
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# -----------------------------
# Tools
# -----------------------------
@tool
def tavily_search(query: str) -> str:
    """
    Uses TavilySearchResults if installed and TAVILY_API_KEY is set.
    Returns list of dict with common fields. Note: published date is often missing.
    """
    results = tavily_client.search(query=query, max_results=5)
    normalized: List[dict] = []
    for r in results or []:
        normalized.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            }
        )
    return normalized


@tool
def tavily_extract(url: str) -> str:
    """Extract content from a given URL using Tavily."""
    response = tavily_client.extract(urls=[url])
    return str(response)

@tool
def reranked_retriever_tool(query: str, top_k: int) -> str:
    """Retrieve and rerank relevant documents and images."""
    text_context, images = reranked_retriever(query, top_k)
    return f"Text Context:\n{text_context}\n\nImages:\n{images}"
# -----------------------------
# Prompt
# -----------------------------
react_prompt = PromptTemplate.from_template(
    """
You are a expert research assistant with extensive knowledge in various domains, specifically in research and information retrieval.
Your job is to research from both the web and Pinecone vector database and get the most relevant information according to users demands. You have access to the following tools:
1. `tavily_search`: Use this tool to search the web for relevant information. It returns a list of dictionaries with title, url, snippet, published_at, and source.
2. `tavily_extract`: Use this tool to extract content from a given URL. It returns the extracted content as a string.
3. `reranked_retriever_tool`: Use this tool to retrieve and rerank relevant documents and images from the Pinecone vector database. It returns two sections: Text Context and Images.
Text Context will contain relevant text chunks along with their page numbers, sources, and heading trails. Images will contain list of dictionaries. Each dictionary contains relevant images in base64 format along with their paths.
**Tool Usage INSTRUCTIONS:
- If input contains a URL, use `tavily_extract`, and extract content from the URL. Use this information to answer the query.
- Otherwise, use `tavily_search`, Tavily will return a list of dictionaries of search results with title, url, snippet, published_at, source. Use this information to answer the query.
- If user query asks for information from uploaded documents in Pinecone vector database, use `reranked_retriever_tool` to retrieve relevant documents and images. Use this information to answer the query.
- Always try to use `reranked_retriever_tool` first to check if the relevant information is present in the uploaded documents in Pinecone vector database. If the retrieved information is not sufficient to answer the query, then use `tavily_search` and `tavily_extract` to get more information from the web.
- Choose the top_k value wisely according to the query. For specific queries, you can choose a smaller top_k (eg. 5) value to get more relevant information. For broader queries, you can choose a larger top_k (eg. 10) value to get more information. Always try to balance between relevance and comprehensiveness when choosing the top_k value.
- If `reranked_retriever_tool` returns "No relevant text found." and "No relevant images found.", it means that there are no relevant documents or images in the Pinecone vector database for the given query. In this case, you should rely solely on the web search results from `tavily_search` to answer the user's query. Do not hallucinate or make up information that is not present in the retrieved results. Always base your answers on the retrieved information from the tools.

After researching using the tools, provide a concise and structured answer to the user's query. Make sure to include all relevant information in your answer, and avoid hallucination. 
If the retrieved information is not sufficient to answer the query, you can mention that in your answer.

Make sure to answer like a professional researcher, and honest research procedure.
*FINAL ANSWER INSTRUCTIONS:
- RAG_Research: This section should include the relevant information retrieved from the Pinecone vector database using `reranked_retriever_tool`. Include the retrieved text chunks along with their page numbers, sources, and heading trails in the answer. Also include any relevant images in base64 format along with their paths. iF the `reranked_retriever_tool` returns "No relevant text found." and "No relevant images found.", then leave this section empty and do not include it in the final answer.
- Web_Search_Results: This section should include the relevant information retrieved from the web using `tavily_search` and `tavily_extract`. Include the title, url, snippet, published_at, and source for each relevant search result. If there are no relevant search results, then leave this section empty and do not include it in the final answer.
- Summary: This section should contain a information rich summary that answers the user's query in a concise and structured manner, based on the information retrieved from both the reranked_retriever_tool and the web search results. Make sure to include all relevant information in the summary, and avoid hallucination. If the retrieved information is not sufficient to answer the query, you can mention that in the summary. But share your opinion as a professional researcher based on the retrieved information. Always be honest about the research procedure and findings.

User Query:
{input}

{agent_scratchpad}
"""
)
#-----------------------------
#Calling the main brain for the Research Agent
#-----------------------------

llm = google_llm(model="gemini-1.5-pro", temperature=0.0, streaming=False)
# -----------------------------
# LLM with Structured Output
# -----------------------------
structured_llm = llm.with_structured_output(ResearchResponse)


# -----------------------------
# ReAct-Agent Building
# -----------------------------
tools = [tavily_search, tavily_extract, reranked_retriever_tool]

agent = create_react_agent(
    llm=llm,  # reasoning model (ReAct loop)
    tools=tools,
    prompt=react_prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)


# -----------------------------
# Research Agent Node
# -----------------------------
def research_agent_node(state: QAresearchstate) -> Dict:
    agent_result = agent_executor.invoke({"input": state["user_demand"]})

    structured_response = structured_llm.invoke(
            f"""
            User Query: {state["user_input"]}

            Agent Output:
            {agent_result['output']}

            Convert into structured format.
            """
        )

    return {"research_response": structured_response}

def QA_Chatbot(state: QAresearchstate) -> Dict:
    system_prompt = ('''You are a helpful research helper and a great user-friendly chatbot. 
    You understand user demands very well and convey the user_demands to the research agent,
    so that the research agent can provide the most relevant information according to user demands.
    You have a friend Research_Agent, who is a expert research agent with extensive knowledge in various domains, specifically in research and information retrieval.
    Your friend Research_Agent has access to the following tools: 
                     1. `tavily_search`: Use this tool to search the web for relevant information. It returns a list of dictionaries with title, url, snippet, published_at, and source.
                     2. `tavily_extract`: Use this tool to extract content from a given URL. It returns the extracted content as a string.
                     3. `reranked_retriever_tool`: Use this tool to retrieve and rerank relevant documents and images from the Pinecone vector database. It returns two sections: Text Context and Images. This is based on the user uploaded pdfs, which may contain both text and images.
    If the Research Assistant doesnt return any texts and image relevance from the uploaded pdfs, then either the pdf parsing is still going on in the background
    or there is no relevant information in the uploaded pdfs. In both cases, you should convey this message to the user clearly and crisp way
    If the user asks questions anything outside the scope of research. Then politely ask the user to ask research related questions, .
    Dont tolerate any kind of abusive, offensive, or harmful language from the user. If the user uses such language, then politely ask the user to refrain from using such language and ask them to ask research related questions.
    Final Answer Instructions:
    1)If the user-query needs research, set the needs_research flag to True. After undertanding if user needs to do research turn the needs_research flag as True, for general questions, set it as False otherwise.
    2)Understand user-demands and fill the user_demands field with information rich context after meticulously understanding the user query. This user_demands field will be used by the research agent to get the most relevant information according to user demands. So make sure to fill this field with the most relevant and information rich context according to user query.
    3)Finally fill the field 'qa_response' after synthesizing the Research_Agent's response and user demands. For general questions then you can also use your own knowledge and information to fill the Q&A_response field. You are a Q&A bot whose job is to act as a mediator between the user and the research agent. But make sure to  avoid hallucination. The answers you dont know, be humble and say I dont know.Always be honest about the research procedure and findings. If the user query does not need research, then you can directly answer the user query in the Q&A_response field without using the research agent's response. Always try to provide a concise and informative answer to the user query in the Q&A_response field.
    ''')
    Bot = google_llm(model="gemini-3.5-flash", temperature=0.0, streaming=True)
    QnA_bot = Bot.with_structured_output(QAresearchstate)
    response = QnA_bot.invoke({"messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": state["user_input"]}, {"role": "Research_Agent", "content": f"Research Agent's Response: {state.get('research_response').summary}\n\nRAG Research: {state.get('research_response').RAG_Research}\n\nWeb Search Results: {state.get('research_response').Web_Search_Results}"}]})
    return {"chat_history":response.qa_response, "user_demand": response.user_demand, "needs_research": response.needs_research}

# -----------------------------
# Building the State Graph
subgraph = StateGraph(QAresearchstate)
subgraph.add_node("research_agent", research_agent_node)
subgraph.add_node("QA_Chatbot", QA_Chatbot)

#------------------------------
#Navigation Logic
def navigation_logic(state: QAresearchstate):
    if state.get("needs_research"):
        return "research_agent"
    return END


#Adding the Q&A_Research Subgraph Edges:
subgraph.set_entry_point("QA_Chatbot")
subgraph.add_edge("research_agent", "QA_Chatbot")
subgraph.add_conditional_edges(
    "QA_Chatbot",
    navigation_logic,
    {
        "research_agent": "research_agent",
        END: END
    }
)

QA_Research_Subgraph = subgraph.compile()
