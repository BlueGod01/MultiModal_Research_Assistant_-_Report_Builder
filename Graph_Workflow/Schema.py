from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, HttpUrl
from typing import Annotated, List, Optional, TypedDict, NotRequired
from langchain_core.messages import BaseMessage
class RAG_Response(BaseModel):
    retrieved_doc: str = Field(description="One chunk of retrieved documents")
    reference: str = Field(description="Reference for the retrieved information, like page number, source, heading trail etc.")
    retrieved_images: List[str] = Field(description="List of retrieved images in base64 format")
class Web_Response(BaseModel):
    title: str = Field(description="Title of the web search result")
    url: str = Field(description="URL of the web search result")
    snippet: str = Field(description="Snippet of the web search result")
    published_at: str = Field(description="Published date of the web search result")
    sources: List[HttpUrl] = Field(description="List of reference URLs")

class ResearchResponse(BaseModel):
    query: str = Field(description="User query")
    summary: str = Field(description="Concise structured summary. Summary that answers the query in a concise and structured manner, based on the RAG_Research and Web_Search_Results. Make sure to include all relevant information in the summary, and avoid hallucination.")
    key_points: List[str] = Field(description="Important content points in bullet form. This bullet points should only include the headings and important points that will help in planning each sections for writing research report or research paper.")
    RAG_Research: List[RAG_Response] = Field(description="List of retrieved documents and images from RAG pipeline, in the from of RAG_Response pydantic schema. Make sure to include all relevant information in the retrieved_doc and reference fields, and include base64 strings of all relevant images in the retrieved_images field. Else leave them empty. If not present avoid hallucination.")
    Web_Search_Results: List[Web_Response] = Field(description="List of web search results, each result is a dict with title, url, snippet, published_at, source")
class QA(BaseModel):
    user_input: str = Field(description="User query or input")
    user_demand: str = Field(description="User demand or requirement for the answer. If user input is a question, then user demand will be the expected answer type or format. If user input is a statement or command, then user demand will be the expected output or action to be performed.")
    needs_research: bool = Field(default=False, description="Flag indicating whether the user query requires research or not. Set True, if the user wants research for general purpose questions, set False.")
    qa_response: str = Field(description="Final answer to the user query after understanding user demands and synthesizing the research agent's response. This field should be filled after understanding user demands and synthesizing the research agent's response. If the user query does not require research, then this field should be filled directly based on user query and user demands without using the research agent's response. Always try to provide a concise and informative answer to the user query in this field, and avoid hallucination. If you dont know the answer, be humble and say I dont know.")

class QAresearchstate(TypedDict):
    user_input: str
    user_demand: str
    needs_research: bool
    research_response: Optional[ResearchResponse]
    chat_history: Annotated[List[BaseMessage], add_messages]