import operator

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, HttpUrl
from typing import Annotated, List, Optional, TypedDict, Literal, Dict
from langchain_core.messages import BaseMessage
#-------------------------------
#Pydantic Schemas for Research Agent:
class RAG_Response(BaseModel):
    retrieved_doc: str = Field(description="One chunk of retrieved documents")
    reference: str = Field(description="Reference for the retrieved information, like page number, source, heading trail etc.")
    retrieved_images: List[Dict] = Field(description="List of Dicts for retrieved images from 'reranked_retriever_tool'. Each dict in the list is in the format: {image: Value, image_path : Value }. If no relevant images found, leave it empty.")
class Web_Response(BaseModel):
    title: str = Field(description="Title of the web search result")
    url: str = Field(description="URL of the web search result")
    snippet: str = Field(description="Snippet of the web search result")
    published_at: str = Field(description="Published date of the web search result")
    images: List[HttpUrl] = Field(default=[],  description="List of image URLs related to the web search results. Leave empty if no relevant images found.")

class ResearchResponse(BaseModel):
    query: str = Field(description="User query")
    summary: str = Field(description="Concise structured summary. Summary that answers the query in a concise and structured manner, based on the RAG_Research and Web_Search_Results. Make sure to include all relevant information in the summary, and avoid hallucination.")
    key_points: List[str] = Field(description="Important content points in bullet form. This bullet points should only include the headings and important points after web search and RAG research that will help in planning each sections for writing research report or research paper.")
    RAG_Research: List[RAG_Response] = Field(description="List of retrieved documents and images from RAG pipeline, in the from of RAG_Response pydantic schema. Make sure to include all relevant information in the retrieved_doc and reference fields, and include base64 strings of all relevant images in the retrieved_images field. Else leave them empty. If not present avoid hallucination.")
    Web_Search_Results: List[Web_Response] = Field(description="List of web search results, each result is a dict with title, url, snippet, published_at, source")
#-------------------------------
#Pydantic Schemas for Q&A Agent:
class QA(BaseModel):
    user_input: str = Field(description="User query or input")
    user_demand: str = Field(description="User demand or requirement for the answer. If user input is a question, then user demand will be the expected answer type or format. If user input is a statement or command, then user demand will be the expected output or action to be performed.")
    needs_research: bool = Field(default=False, description="Flag indicating whether the user query requires research or not. Set True, if the user wants research. If Research Agents Response is empty and its not general purpose question set it True. For general purpose questions set False. If Research response is already there and user doesn't need further research set it False.")
    need_to_create_report: bool = Field(default=False, description = "Analyse the user-demand. If user wants to create report, then set it True. Otherwise in all other cases it should be False by default.")
    qa_response: str = Field(description="Final answer to the user query after understanding user demands and synthesizing the research agent's response. This field should be filled after understanding user demands and synthesizing the research agent's response. If the user query does not require research, then this field should be filled directly based on user query and user demands without using the research agent's response. Always try to provide a concise and informative answer to the user query in this field, and avoid hallucination. If you dont know the answer, be humble and say I dont know.")
#--------------------------------
#Schema for planning and writing research report or research paper:
class Task(BaseModel):
    id: int
    section_heading: str = Field(..., description="Section title (H2). This section titles can be or may start with abstract, methodology, results, and conclusion.")
    supportive_information: Optional[str] = Field(default=None, description="Any supportive information or context for this section that can help in writing this section. This can include important points for this section from the research_response. Include reference URLs in key-value pair for Web_Search_Results. Include reference page numbers, sources, and heading trails for RAG_Research. This field is optional and can be left empty if there is no specific supportive information for this section. Use the keyword "reference: " to metion references. Eg. reference: URL for web_searches, reference: Pageno: 3, Source: document.pdf for RAG_Research results"  )
    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section. This goal will be used to send query to research agent for research for this section."
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=6,
        description="3–6 concrete, non-overlapping subpoints to cover in this section.",
    )
    target_words: int = Field(..., description="Target word count for this section (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    report_heading: str = Field(..., description="Blog title (H1).")
    audience: str
    tone: str
    report_kind: Literal["Technical","Popular Reports","Analytical","Case Study","Survey","Review/Literature report","Experimental","Progress","Policy"]
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]
    
#Schema for Placing Images and writing markdown with placeholders for images:
class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="If the extracted images are there, the filename would be there image_path. Otherwise for AI generated imagesSave under extracted/, e.g. qkv_flow.png")
    alt: str
    caption: str
    AI_Image_needed: bool =Field(description="Set it True if AI generated image is needed for this placeholder. If the chosen image is from extracted images set it False.")
    prompt: str = Field(..., description="Prompt to send to the image model. Leave this empty if relevant images are extracted from provided documents.")
    size: str  = Field(..., description="Image size eg. 1024x1024, 1024x1536, 1536x1024 dimensions. Decide the dimension smartly, such that it can be put in the markdown file or report pdf")
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)

class State(TypedDict):
    # Q&A Research Subgraph:
    user_input: str
    user_demand: str
    needs_research: bool
    need_to_create_report: bool
    research_response: Optional[ResearchResponse]
    qa_response: str
    #Orchestrator-Worker Schema:
    research_topic: str
    plan: Optional[Plan]
     # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)
    # reducer/image
    merged_md: str
    md_with_placeholders: str
    AI_Images_needed: bool
    image_specs: List[dict]

    final: str
    chat_history: Annotated[List[BaseMessage], add_messages]



