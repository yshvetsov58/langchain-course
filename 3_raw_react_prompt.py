import inspect
import re
from typing import Any

from dotenv import load_dotenv
load_dotenv()
from langsmith import traceable
import json

MAX_ITERATIONS = 10
MODEL = "gpt-5.5"

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

tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}



def get_tool_description(tool_dict:dict) -> str:
    descriptions = []
    for tool_name, tool_function in tool_dict.items():
        original_function = getattr(tool_function, "_wrapped_", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)

tool_descriptions = get_tool_description(tools)
tool_names = ", ".join(tools.keys())
pass

def main():
    print("Available tools:")
    print(tool_descriptions)
    print(f"Tool names: {tool_names}")



if __name__ == "__main__":
    main()