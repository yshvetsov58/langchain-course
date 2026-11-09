from dotenv import load_dotenv

load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "gpt-5.5"


# ------- Tools (LangChain @tool decorator) -------


@tool
def get_product_price(product_name: str) -> str:
    """Get the price of a product.
    Arguments:
        product_name: The name of the product to look up.
    Returns:
        The price of the product if found, otherwise a message indicating the product was not found.
    """
    # In a real application, this would query a database or an API.
    prices = {"laptop": "$999", "smartphone": "$699", "headphones": "$199"}
    return prices.get(product_name.lower(), "Product not found.")


@tool
def get_discount_price(original_price: float, discount_category: str) -> float:
    """Get the discounted price of a product based on its original price and discount category.
    Arguments:
        original_price: The original price of the product.
        discount_category: The category of the discount (e.g., "student", "holiday").
    Returns:
        The discounted price if the category is valid, otherwise a message indicating the category was not found.
    """
    discounts = {
        "white": 0.10,  # 10% discount
        "green": 0.20,  # 20% discount
        "gold": 0.30,  # 30% discount
    }
    if discount_category.lower() in discounts:
        discount_rate = discounts[discount_category.lower()]
        discounted_price = original_price * (1 - discount_rate)
        return discounted_price
    else:
        return "Discount category not found."


@traceable(
    name="LangChain Agent with Tool Calling",
    description="An agent that can call tools to get product prices and calculate discounted prices.",
)
def run_agent():
    tools = [get_product_price, get_discount_price]
    llm_agent = ChatOpenAI(model_name=MODEL, temperature=0).bind_tools(tools)
    messages = [
        SystemMessage(
            content="""
    You are a helpful assistant that can provide product prices and calculate discounted prices. Please do not guess the priceof a product. 
    Use the provided tools to get accurate information.For procuct prices, use the get_product_price tool. For discounted prices,
    use the get_discount_price tool. User must provide the product name annd discount category to get the discounted price. If the product or discount category is not found, inform the user accordingly."""
        ),
        HumanMessage(
            content="What is the price of a laptop if I have a green discount?"
        ),
    ]

    tools_by_name = {t.name: t for t in tools}

    for iteration in range(MAX_ITERATIONS):
        response = llm_agent.invoke(messages)
        messages.append(response)

        print(f"\nIteration {iteration + 1}")
        print("Model content:", response.content)
        print("Tool calls:", response.tool_calls)
        if not response.tool_calls:
            print("\nFinal answer:", response.content)
            break
        for tool_call in response.tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args")
            tool_call_id = tool_call.get("id")
            print(f"\nTool call: {tool_name} with args: {tool_args}")
            selected_tool = tools_by_name.get(tool_name)

            if selected_tool is None:
                tool_result = f"Unknown tool: {tool_name}"
            else:
                try:
                    tool_result = selected_tool.invoke(tool_args)
                except Exception as error:
                    tool_result = f"Error invoking tool {tool_name}: {str(error)}"

                    print(f"Calling tool: {tool_name}")
            print(f"Arguments: {tool_args}")
            print(f"Result: {tool_result}")

            messages.append(
                ToolMessage(content=str(tool_result), tool_call_id=tool_call_id)
            )


run_agent()
