import os
from langgraph.graph import StateGraph, START, END
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
fastmcp_key = os.getenv("FASTMCP_API_KEY")

llm = ChatGroq(model="llama-3.1-8b-instant",temperature=0) #llama-3.3-70b-versatile

client = MultiServerMCPClient(
    {
        "expenses": {
            "transport":"stdio",
            "command":"python3",
            "args":["/home/ganesh/Analysis/my_project/mcp_campusx/expenses_server/test.py"],
        },
        "arith": {
            "transport":"streamable_http",
            "url": "https://middle-orange-quail.fastmcp.app/mcp",
            "headers": {
                "Authorization": f"Bearer {fastmcp_key}"
            }
        }
    }
)   


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

async def build_graph():

    tools = await client.get_tools()
    print(tools)
    llm_with_tools = llm.bind_tools(tools)
    
    async def chat_node(state:ChatState):
        """ LLM node that may answer or request a tool call."""
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages":[response]}

    tool_node = ToolNode(tools)

    graph = StateGraph(ChatState)
    graph.add_node("chat_node",chat_node)
    graph.add_node("tools",tool_node)

    graph.add_edge(START,"chat_node")

    graph.add_conditional_edges("chat_node",tools_condition)
    graph.add_edge('tools','chat_node')

    chatbot = graph.compile()

    return chatbot


async def main():

    chatbot = await build_graph()

    #ruuing the graph
    result = await chatbot.ainvoke({'messages': [HumanMessage(content="what is total expenses found and give answer like acricket commentator.")]})

    print(result['messages'][-1].content)

if __name__=="__main__":
    asyncio.run(main())