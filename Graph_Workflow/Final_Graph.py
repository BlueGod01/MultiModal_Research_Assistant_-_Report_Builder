from langgraph.graph import StateGraph, END, START
from Graph_Workflow.Schema import State
from Graph_Workflow.QnA_Research_Subgraph import QA_Research_Subgraph
from Graph_Workflow.Orchestrator_Planner_Worker import orchestrator_node, fanout, worker_node
from Graph_Workflow.ReducerWithImages_subgraph import reducer_subgraph
from langgraph.checkpoint.memory import InMemorySaver
#---------------------------------------------------
#Creating short-term memory
memory = InMemorySaver()
#---------------------------------------------------
#Creating the final Subgraph:
#Creating the nodes of the graph
g = StateGraph(State)
g.add_node("Q&Anode", QA_Research_Subgraph)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

#--------------------------------
# Creating router function
def route(state:State):
    if state["need_to_create_report"]:
        return "orchestrator"
    else: 
        return END
#--------------------------------
#Creating edges.    
g.set_entry_point("Q&Anode")

g.add_conditional_edges(
        "Q&Anode",
        route,
        {
            "orchestrator": "orchestrator",
            END: END
        }
    )

g.add_edge("orchestrator", "worker")
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

research_graph = g.compile(checkpointer=memory)

