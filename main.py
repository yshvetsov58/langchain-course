from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv(verbose=True)
class Source(BaseModel):
    """Schema for the source by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for the agent response with answer and sources"""

    answer: str = Field(description="The agent answer to the query")
    sources: List[Source] = Field(
        default_factory=list,
        description="The list of sources used to generate the answer.",
    )


llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(llm, tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="search for 3 job posting for ai engineer using langchain in the bay area on linkedin",
            )
        }
    )
    print(result)


if __name__ == "__main__":
    main()
