#Dependencies
from langchain_core.messages import SystemMessage, HumanMessage
from Graph_Workflow.Schema import Plan, Task
from Graph_Workflow.Schema import State
from Brain import google_llm
from langgraph.types import Send
from Graph_Workflow.tools import tavily_search
#-----------------------------
#Constants and LLM Initialization
llm = google_llm(model="gemini-3.5-flash", temperature=0.0)
# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior content writer and research-report advocate.
Your job is to produce a highly actionable plan  for writing a deep research report.
Based on user-demands and the research summary in on going Q&A research conversation, create a detailed plan for writing a research report.
Hard requirements:
- Create 5–9 sections (tasks) suitable for the research topic and audience.
-Make sure to understand Key Points and research summary from the research agent and synthesize them to create the sections.
-Key Points are important content points in bullet form. This bullet points only include the headings and important points after web search and RAG research that will help in planning each sections for writing research report or research paper.")
- Each task must include:
  1)section_heading:  = "Section title (H2). This section titles can be or may start with abstract, methodology, results, and conclusion. Use according the necessity of the topic and audience. For example, if the topic is very technical and the audience is expert researchers, then you can use more technical section headings. If the topic is more general and the audience is non-expert, then you can use more general section headings.")
  2)supportive_information: *"Any supportive information or context for this section that can help in writing this section. This include important points from the research summary to write this section. Include reference URLs in key-value pair for Web_Search_Results. Include reference page numbers, sources, and heading trails for RAG_Research. This field is optional and can be left empty if there is no specific supportive information for this section.
  *Use the keyword "reference: " in supportive information string to metion references. Eg. reference: URL for web_searches, reference: Pageno: 3, Source: document.pdf for RAG_Research results. The references can be used if requires_citations is True for this section. The "reference" portion must be present after the main text information.
  *Use particular RAG_Response and Web_Search_Results for each section according to the relevance of that information for that section. Dont put all the research summary and research response for each section, instead synthesize the research summary and research response and put only the relevant information for each section in supportive_information for that section. This will help the worker node to write better and more focused sections.
  2) goal (1 sentence)
  3) 3–6 bullets that are concrete, specific, and non-overlapping
  4) target word count (120–550)
  5) requires_research: True if the section requires up-to-date information, else False
  6) requires_citations: True if the section requires citations, else False
  7) requires_code: True if the section requires code, else False

Quality bar:
- According to the research topic and audience; use correct terminology.
-Use codes and scientific terms and formulas where appropriate to the topic and audience.
- Bullets must be actionable: build/compare/measure/verify/debug.
- Ensure the overall plan includes at least 2 of these somewhere:
  * minimal code sketch / MWE (set requires_code=True for that section)
  * edge cases / failure modes
  * performance/cost considerations
  * security/privacy considerations (if relevant)
  * debugging/observability tips if the topic demands it.
Output must strictly match the Plan schema.
"""
#-----------------------------
# Orchestrator Node
def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan)

    research_summary = state.get("research_response", []).summary if state.get("research_response") else ""
    key_points = state.get("research_response", []).key_points if state.get("research_response") else []
    RAG_response = state.get("research_response", []).RAG_Research if state.get("research_response") else []
    Web_Search_Results = state.get("research_response", []).Web_Search_Results if state.get("research_response") else []
    key_points_txt = "\n- ".join(key_points)
    RAG_reponse_text = "\n".join(
        f"Text- {r.retrieved_doc} | Reference: {r.reference}\n" for r in RAG_response
    )
    Web_Search_Results_text = "\n".join(
        f"Title: {w.title} | URL: {w.url} | Web Result: {w.snippet} | Published_at: {w.published_at}\n" for w in Web_Search_Results
    )
    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"User_demand: {state['user_demand']}\n"
                    f"Key Points: {key_points_txt}\n\n"
                    f"Research Summary (ONLY if user performed research; may be empty): {research_summary}\n"
                    f"RAG_Response: {RAG_reponse_text}\n\n"
                    f"Web_Search_Results: {Web_Search_Results_text}\n\n"
                )
            ),
        ]
    )

    return {"plan": plan, "research_topic": plan.report_heading}

#-----------------------------
#Fanout logic for worker nodes (if we want to have separate workers for each section):
def fanout(state: State):
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "research_topic": state["research_topic"],
                "plan": state["plan"].model_dump()
            },
        )
        for task in state["plan"].tasks
    ]

# -----------------------------
# Worker (write one section)
# -----------------------------
WORKER_SYSTEM = """You are a senior research writer and report-writing advocate.
Write ONE section of a research report in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to Target words (±15%).
-Take into account the supportive information for this section, if provided, and use it to write this section.
-The references provided in the supportive information are the ONLY sources you can cite for this section. If requires_citations is True, cite reference URLs or page_number and source from the supportive information or research summary using Markdown links. If requires_citations is True but no references are provided in the supportive information, write: "Not found in provided sources."
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a '## <Section Title>' heading/ Markdown H2.
-Make sure to write according to the report kind, audience, and tone provided in the plan.
- If requires_citations is True, cite reference URLs or page_number and source from the supportive information or research summary using Markdown links. If not provided leave it empty."
- If requires_code is True, include at least one minimal, correct code snippet relevant to the section's goal and bullets.

Style:
- Short but context-rich paragraphs, bullets where helpful, code fences for code.
- Avoid fluff/marketing. Be precise and implementation-oriented.
- Use technical terminology and scientific/mathematical formulas where appropriate to the topic and audience.
"""

def worker_node(payload: dict) -> dict:

    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    research_topic = payload["research_topic"]
    supportive_information = task.supportive_information or ""
    bullets_text = "\n- " + "\n- ".join(task.bullets)
    if task.requires_research and not supportive_information:
        extra_web_research = tavily_search(f"{research_topic} {task.section_heading} {task.goal}")
        supportive_information += f"\n\nExtra Web Research:\n{extra_web_research}"
    section_md = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Research topic: {research_topic}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Report kind: {plan.report_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Supportive information:\n{supportive_information}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}