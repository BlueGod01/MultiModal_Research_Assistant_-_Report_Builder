from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain_core.runnables import chain
from RAG_Pipeline.retriever import retrieve_multimodal, query_corrected_retrieval
from RAG_Pipeline.reranker import reranked_retriever
from typing import Dict, List
from pydantic import BaseModel, Field, HttpUrl
from Graph_Workflow.Schema import ResearchResponse, RAG_Response, Web_Response
from Graph_Workflow.Schema import State, QA
from Graph_Workflow.tools import tavily_search, tavily_extract
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
import re
from langchain.tools import tool
from Brain import google_llm
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

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
- If input contains a URL, use `tavily_extract`, and extract content from the URL. Understand the extracted content and use this information to answer the query.
- Otherwise, use `tavily_search`, Tavily will return a list of dictionaries of search results with title, url, snippet, published_at, source. Use this information to answer the query.
- If user query asks for information from uploaded documents in Pinecone vector database, use `reranked_retriever_tool` to retrieve relevant documents and images. Use this information to answer the query.
- reranked_retriever_tool will return two sections: Text Context and Images. Text Context will contain relevant text chunks along with their page numbers, sources, and heading trails. Images will contain list of dictionaries. Each dictionary contains relevant images in base64 format along with their paths. Make sure to hold the structure of images returned.
- Always try to use `reranked_retriever_tool` first to check if the relevant information is present in the uploaded documents in Pinecone vector database. If the retrieved information is not sufficient to answer the query, then use `tavily_search` and `tavily_extract` to get more information from the web.
- Choose the 'top_k' parameter value wisely according to the query. For specific queries, you can choose a smaller top_k (eg. 5) value to get more relevant information. For broader queries, you can choose a larger top_k (eg. 10) value to get more information. Always try to balance between relevance and comprehensiveness when choosing the top_k value.
- If `reranked_retriever_tool` returns "No relevant text found." and "No relevant images found.", it means that there are no relevant documents or images in the Pinecone vector database for the given query. In this case, you should rely solely on the web search results from `tavily_search` to answer the user's query. Do not hallucinate or make up information that is not present in the retrieved results. Always base your answers on the retrieved information from the tools.

After researching using the tools, provide a concise and structured answer to the user's query. Make sure to include all relevant information in your answer, and avoid hallucination. 
If the retrieved information is not sufficient to answer the query, you can mention that in your answer.

Make sure to answer like a professional researcher, and honest research procedure.
*FINAL ANSWER INSTRUCTIONS:
- RAG_Research: This section should include the relevant information retrieved from the Pinecone vector database using `reranked_retriever_tool`. Include the retrieved text chunks along with their page numbers, sources, and heading trails in the answer. Also include any relevant images in base64 format. Understand them thoroughly both textually and visually. If the `reranked_retriever_tool` returns "No relevant text found." and "No relevant images found.", then leave this section empty and do not include it in the final answer.
- Web_Search_Results: This section should include the relevant information retrieved from the web using `tavily_search` and `tavily_extract`. Include the title, url, snippet, published_at, and source for each relevant search result. If there are no relevant search results, then leave this section empty and do not include it in the final answer.
- Summary: This section should contain a information rich summary that answers the user's query in a concise and structured manner, based on the information retrieved from both the reranked_retriever_tool and the web search results. Make sure to include all relevant information in the summary, and avoid hallucination. If the retrieved information is not sufficient to answer the query, you can mention that in the summary. But share your opinion as a professional researcher based on the retrieved information. Always be honest about the research procedure and findings.
The answer should strictly follow the instructions and structure. The results of 'reranked_retriever_tool' should be included in RAG_Research section along with page_numbers, sources, and heading trails, and the results of 'tavily_search' and 'tavily_extract' should be included in the Web_Search_Results section. The summary should synthesize all the retrieved information to answer the user's query.
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
def research_agent_node(state: State) -> Dict:
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

def QA_Chatbot(state: State) -> Dict:
    system_prompt = ('''You are a Q&A chatbot, a user and research agent mediator and expert empathizer to understand user demands.

Role: Q&A Chatbot and smart routing agent
- Understand user queries and decide if Research is needed or not and it needed then whats the point of research.
- If needed, translate the query into a precise, information-rich `user_demands` for the Research_Agent.
- Act as a bridge between user and Research_Agent.

Research_Agent:
- Has access to:
  1. tavily_search → web search (title, url, snippet, etc.)
  2. tavily_extract → extract content from URLs
  3. reranked_retriever_tool → retrieves text + images from user-uploaded PDFs (Pinecone)

Rules:
- If no results from PDFs: inform user clearly (parsing ongoing OR no relevant data OR research not yet executed).
- If query is NOT research-related: provide a polite answer and suggest ideas to user based on your understanding of user demand.
- Reject abusive/offensive language politely. And ask the user to be respectful and ask research and educative queries
Output Instructions:
1. needs_research (bool):
   - True → if query requires external info/research. Or the Research_Agent's response is empty and the query is not a general knowledge question. Or if the user explicitly demands research.
   - False → for general knowledge queries and if the Research_Agent's response is already there and user doesn't need further research. Or if the user explicitly says that they don't need research.

2. user_demands (str):
   - Clear, structured, and information-rich representation of user intent, understand from user query. If user is already very precise then just rephrase it in a more structured way. This will be used as input for the Research_Agent to do the research. Always try to include all the relevant information from the user query in the user_demands, and make it as clear and structured as possible. This will help the Research_Agent to understand the user's needs better and provide more accurate and relevant information in the research response.
   - Optimized for research accuracy

3. qa_response (str):
   - Final user-facing answer
   - If research used → synthesize Research_Agent output
   - If not → answer directly using your knowledge
   - Be concise, factual, and avoid hallucination
   - If unsure → say "I don’t know"
4. need_to_create_report: bool:
  - Analyse the user query properly. If the user want to have report based on the conversations set it True.
  - For all other cases its False by default.
  - For first time conversation it iss definately False                   

General:
- Be clear, concise, and user-friendly
- Always prioritize accuracy and transparency
- In input: summary from Research_Agent's response, including RAG research and web search results.
-Synthesize this information to understand user demands and provide a final answer in the `qa_response` field. Always try to provide a concise and informative answer to the user query in the `qa_response` field, and avoid hallucination. If you dont know the answer, be humble and say I dont know.                                     
    ''')
    Bot = google_llm(model="gemini-3.5-flash", temperature=0.0, streaming=True)
    QnA_bot = Bot.with_structured_output(QA)
    response = QnA_bot.invoke({"messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": state["chat_history"][-1].content if state["chat_history"] else state["user_input"]}, {"role": "Research_Agent", "content": f"Research Agent's Response: {state.get('research_response').summary}\n\nRAG Research: {state.get('research_response').RAG_Research}\n\nWeb Search Results: {state.get('research_response').Web_Search_Results}"}]})
    return {"chat_history":response.qa_response, "user_demand": response.user_demand, "needs_research": response.needs_research, "qa_response": response.qa_response, "need_to_create_report": response.need_to_create_report}

# -----------------------------
# Building the State Graph
subgraph = StateGraph(State)
subgraph.add_node("research_agent", research_agent_node)
subgraph.add_node("QA_Chatbot", QA_Chatbot)

#------------------------------
#Routing Logic
def navigation_logic(state: State):
    if state.get("needs_research"):
        return "research_agent"
    return 'Reply'


#Adding the Q&A_Research Subgraph Edges:
subgraph.set_entry_point("QA_Chatbot")
subgraph.add_edge("research_agent", "QA_Chatbot")
subgraph.add_conditional_edges(
    "QA_Chatbot",
    navigation_logic,
    {
        "research_agent": "research_agent",
        'Reply': END
    }
)

QA_Research_Subgraph = subgraph.compile()
