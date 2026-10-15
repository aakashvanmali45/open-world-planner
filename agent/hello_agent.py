from __future__ import annotations
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

@tool
def get_object_count(location: str) -> int:
    """Return how many objects the robot has catalogued at a given location."""
    fake_db = {"kitchen": 3, "lab": 7, "hallway": 0}
    return fake_db.get(location.lower(), 0)

def build_agent(model: str = "llama3.1:8b"):
    llm = ChatOllama(model=model, temperature=0)
    return create_agent(llm, tools=[get_object_count])

if __name__ == "__main__":
    agent = build_agent()
    q = "How many objects have I catalogued in the lab? Then add 10 to that number"
    result = agent.invoke({"messages":[("user", q)]})
    for m in result["messages"]:
        role = m.__class__.__name__.replace("Message", "")
        print(f"[{role}] {getattr(m, 'content', '')}")
        if getattr(m, "tool_calls", None): 
            print(f" ->tool_calls: {m.tool_calls}")