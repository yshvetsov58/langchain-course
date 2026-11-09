from dotenv import load_dotenv
from langsmith._openapi_client.types import run_type
from openai import OpenAI

load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
import json

MAX_ITERATIONS = 10
MODEL = "gpt-5.5"


# ---- Tools (LangChain @tool decorator) ---
@traceable(run_type="tool")
def get_product_price(product: str) -> int:
    """Look up the price of a product in the catalog."""
    print(f"   >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 999, "smartphone": 699, "headphones": 199}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: int, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available discount tiers are: bronze, silver, gold."""
    print(
        f"   >> Executing apply_discount(price='{price}', discount_tier={discount_tier})"
    )
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


tools = [
    {
        "type": "function",
        "name": "get_product_price",
        "description": "Look up the price of a product in the catalog.",
        "parameters": {
            "type": "object",
            "properties": {
                "product": {
                    "type": "string",
                    "enum": ["laptop", "smartphone", "headphones"],
                },
            },
            "required": ["product"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "apply_discount",
        "description": "Apply a discount tier and return the final price.",
        "parameters": {
            "type": "object",
            "properties": {
                "price": {
                    "type": "integer",
                    "description": "Price of the product.",
                },
                "discount_tier": {
                    "type": "string",
                    "enum": ["bronze", "silver", "gold"],
                },
            },
            "required": ["price", "discount_tier"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]
tool_dict = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

openAIClient = OpenAI()


@traceable(name="OpenAI chat", run_type="llm")
def openai_traced(messages):
    return openAIClient.responses.create(
        model=MODEL,
        input=messages,
        tools=tools,
    )


messages = [
    {
        "role": "developer",
        "content": """
You are a helpful shopping assistant.

Never guess product prices.
Use get_product_price to retrieve a product price.
Use apply_discount to calculate a discounted price.

The user must provide a product name and a valid discount tier:
bronze, silver, or gold.

If the product or discount tier is unsupported, explain which values are valid.
""",
    },
    {
        "role": "user",
        "content": "What is the price of a laptop if I have a bronze discount?",
    },
]


@traceable(name="Agent loop", run_type="chain")
def main():
    input_items = list(messages)

    for iteration in range(MAX_ITERATIONS):
        print(f"\n>> Executing iteration {iteration + 1}")

        response = openai_traced(messages=input_items)

        input_items += response.output

        tool_calls = [item for item in response.output if item.type == "function_call"]

        if not tool_calls:
            print(f"\nFinal answer: {response.output_text}")
            return response.output_text

        for tool_call in tool_calls:
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)

            print(f">> Tool call: {tool_name}, args: {tool_args}")

            selected_tool = tool_dict.get(tool_name)

            if selected_tool is None:
                raise RuntimeError(f"Unknown tool requested: {tool_name}")

            result = selected_tool(**tool_args)

            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": json.dumps(result),
                }
            )

    raise RuntimeError(f"Agent did not finish after {MAX_ITERATIONS} iterations")


if __name__ == "__main__":
    main()
