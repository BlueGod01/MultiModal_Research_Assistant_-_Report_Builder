from fastapi import FastAPI, UploadFile, File, Cookie, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aiofiles
import os
from Data_Ingestion_Pipeline.Data_Pipeline import run_parallel_pipeline
from Graph_Workflow.Final_Graph import research_graph
from typing import Literal, Optional
import uuid
import logging

# Configure basic logging for the app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Schema for taking user input
class UserInput(BaseModel):
    query: str
    session_id: Optional[str] = None
#Schema for responding AI Output.
class AIResponse(BaseModel):
    response: str
    session_id: str
    is_new_session: bool
    research_report: str

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
#API Endpoint for Uploading PDF/Knowledge Document Source.
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()  # async I/O
        await f.write(content)
    return {"message": "Processing started", "file": file.filename}
#API Endpoint to create Knowledge Base. After uploadng the pdfs.
@app.post("/Knowledge_base")
async def create_knowledge_base(background_tasks: BackgroundTasks, UPLOAD_DIR: str = "uploads", parsing_strategy:Literal["fast", "medium", "deep"] = "medium"):
    if not os.path.exists(UPLOAD_DIR):
        return {"message": "No files uploaded. Please upload files first."}
    if parsing_strategy not in ["fast", "medium", "deep"]:
        return {"message": "Invalid parsing strategy. Choose from 'fast', 'medium', or 'deep'."}
    
    # Run pipeline in background to prevent blocking the event loop
    if os.path.listdir(UPLOAD_DIR):
        background_tasks.add_task(run_parallel_pipeline, directory_path=UPLOAD_DIR, max_workers=4, strategy=parsing_strategy)
        return {"message": "Knowledge base creation started in the background"}
    else:
        return {"message": "No files uploaded. Please upload files first."}

# --------------------------------------------
# SESSION UTILITIES
# --------------------------------------------

def create_session_id() -> str:
    """Create a new unique session id"""
    return str(uuid.uuid4())


def resolve_session_id(
    request_session_id: Optional[str],
    cookie_session_id: Optional[str]
) -> tuple[str, bool]:
    """
    Decide which session_id to use.

    Priority:
    1. Request body session_id (frontend-controlled)
    2. Cookie session_id (browser persistence)
    3. Create new session_id

    Returns:
        session_id, is_new_session
    """
    if request_session_id:
        return request_session_id, False

    if cookie_session_id:
        return cookie_session_id, False

    return create_session_id(), True


#API Endpoint for Interacting with the Agentic AI System.
@app.post("/research_chat", response_model=AIResponse)
def run_agentic_system(request: UserInput, response: Response, session_id_cookie: Optional[str] = Cookie(default=None, alias="session_id")):
    
    #Input must match this schema.
    '''class State(TypedDict):
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
    chat_history: Annotated[List[BaseMessage], add_messages] '''
        # Step 1: Resolve session_id
    session_id, is_new = resolve_session_id(
        request.session_id,
        session_id_cookie
    )

    # Step 2: Set cookie if new session
    if is_new:
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            max_age=60 * 60 * 24 * 7  # 7 days
        )

    # Step 3: Prepare state
    state = {
        "user_input": request.query
        # No need to pass chat_history manually
        # LangGraph will auto-load it via checkpointer
    }

    # Step 4: Invoke LangGraph (auto memory via thread_id)
    result = research_graph.invoke(
        state,
        config={"configurable": {"thread_id": session_id}}
    )

    # Step 5: Return response
    return AIResponse(
        session_id=session_id,
        response=result.get("qa_response", ""),
        is_new_session=is_new,
        research_report=result.get("final", "")
    )
