from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch
load_dotenv(verbose=True)
tavily = TavilyClient()


@tool
def search(search_query: str) -> dict[str, Any]:
    """
    Tool that searches over the internet
    :ars search_query: str The search query
    Returns:
        The search result
    """
    return tavily.search(query=search_query)


llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(llm, tools)


def main():
    result = agent.invoke({"messages": HumanMessage(content="What is the weather in Tokyo?")})
    print(result)

if __name__ == "__main__":
    main()
