from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from langchain_tavily import TavilySearch

load_dotenv()

# llm = ChatOpenAI()
llm = ChatOllama(model="gemma4:e2b")
tools = [TavilySearch()]
agent = create_agent(
    model=llm,
    tools=tools,
)


def main():
    print("Hello from langchain-course!")
    search_prompt = """
Search for 3 data scientist job postings in the past week in banking and finance domain in Hanoi on LinkedIn and list their details.
"""
    result = agent.invoke({"messages": HumanMessage(content=search_prompt)})
    print(result)

if __name__ == "__main__":
    main()
