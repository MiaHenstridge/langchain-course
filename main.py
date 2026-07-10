from typing import List, Optional
from pydantic import BaseModel, Field

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

load_dotenv()

# define structured output
class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="The list of sources used to generate the answer")


# create the agent with the tools
llm = ChatOpenAI(model="gpt-5")
# llm = ChatOllama(model="gemma4:e2b")
tools = [TavilySearch()]
agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse,
)


def main():
    print("Hello from langchain-course!")
    search_prompt = """
Search for 3 current data scientist job postings in banking and finance domain in Hanoi on LinkedIn and list their details.
"""
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content=search_prompt
            )
        }
    )
    print(result)

if __name__ == "__main__":
    main()
