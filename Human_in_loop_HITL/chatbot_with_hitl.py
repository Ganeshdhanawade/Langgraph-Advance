from langgraph.graph import StateGraph, START
from typing import TypedDict,Annotated,List,Literal
from langchain_core.messages import AIMessage,HumanMessage,BaseMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.tools import tool
from langgraph.types import interrupt,Command
from dotenv import load_dotenv
import requests

load_dotenv()

#------------------
# 1. LLM
#------------------
llm = ChatGroq(model="llama-3.1-8b-instant")

#------------------
# 2. tools
#------------------
@tool
def get_stock_price(symbol:str) -> dict:
    """
    fetch latest stock price of the given symbol (e.g. 'AAPL','TSLA')
    using Alpha Vantage with API key in the URL
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=ZGNXYJ4YFBRBG71F"
    r = requests.get(url)
    return r.json()

@tool
def perchase_stock(symbol:str,quantity: int) -> dict:
    """
    Simulate perchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Befor confirming the purchase, this tool will inturrupt
    and wait for a humna decion("yes"/ anything else).
    """
    decision = interrupt(f"approve buying {quantity} share of {symbol}? (yes/no)")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "succsess",
            "messages": f"Purchase order placed for {quantity} share of {symbol}.",
            "symbol":symbol,
            'quantity':quantity
        }
    
    else:
        return {
            "status": "cancelled",
            "messages": f"Purchase order placed for {quantity} share of {symbol} was declined by human.",
            "symbol":symbol,
            'quantity':quantity
        }
    
tools = [get_stock_price,perchase_stock]
llm_with_tool = llm.bind_tools(tools)

#------------------
# 3. State
#------------------
class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]

#------------------
# 4. nodes
#------------------
def chat_node(state:ChatState):
    """ LLM node that may answer the request a tool call."""
    messages = state["messages"]
    response = llm_with_tool.invoke(messages)
    return {"messages":[response]}

tool_node =ToolNode(tools)

#------------------
# 5. checkpointer
#------------------
memory = MemorySaver()

#------------------
# 6. graph
#------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node",chat_node)
graph.add_node("tools",tool_node)

graph.add_edge(START,"chat_node")

graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge("tools","chat_node")

chatbot = graph.compile(checkpointer=memory)

#------------------
# 7. use example
#------------------

if __name__ == "__main__":

    #use fixed thread id for the persisted in memory
    thread_id = "demo-thread"

    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in {"exit","quit"}:
            print("goodbye!")
            break

        #bui.d inital state for turn
        state = {"messages":[HumanMessage(content=user_input)]}

        #run the graph
        result = chatbot.invoke(
            state,
            config={"configurable":{"thread_id":thread_id}},
        )

        interrupts = result.get('__interrupt__',[])
        if interrupts:
            #our inturrupt payload is the string we passto inturupt function
            prompt_to_human = interrupts[0].value
            print(f"HITL: {prompt_to_human}")
            decision = input("your decision:").strip().lower()
            
            #resume graph with the human decistion("yes"/"no"/"whatever")
            result = chatbot.invoke(
                Command(resume=decision),
                config={"configurable":{"thread_id":thread_id}}
            )
        
        #get the message from assistant
        messages = result["messages"]
        last_msg = messages[-1]
        print(f"Bot: {last_msg.content} \n")