import inspect
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
from langsmith import traceable
import json

MAX_ITERATIONS = 10
MODEL = "gpt-5.5"
chatGptClient = OpenAI()

react_prompt = """
Answer the question using the available tools.

Available tools:
{tools_descriptions}

Allowed tool names: {tool_names}

Return exactly one of the following formats.

To call a tool:
Action: <tool name>
Action Input: <JSON object>

To give the final answer:
Final Answer: <answer to the original question>

Rules:
- Return exactly one action OR one final answer per response.
- Action Input MUST be a valid JSON object.
- Its keys MUST match the selected function's parameter names.
- Use double quotes for JSON keys and string values.
- Never return a plain string as Action Input.
- Do not generate Observation. The application supplies tool results.
- Do not include thoughts, explanations, or Markdown code fences.
- Use the observations already provided to choose the next action
  or give the final answer.

Valid action examples:

Action: get_product_price
Action Input: {{"product": "laptop"}}

Action: apply_discount
Action Input: {{"price": 999, "discount_tier": "bronze"}}

Invalid action input:
Action Input: laptop

Question: {question}

{agent_scratchpad}
"""

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
        original_function = inspect.unwrap(tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)

tool_descriptions = get_tool_description(tools)
tool_names = ", ".join(tools.keys())

@traceable(run_type="GPT Agent loop")
def run_agent(question: str):
    scratchpad = ""
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n >>> Iteration {iteration} ---")
        prompt = react_prompt.format(
                    question=question,
                    tools_descriptions=tool_descriptions,
                    tool_names=tool_names,
                    agent_scratchpad=scratchpad,
                )


        full_prompt = prompt + scratchpad
        response = chat_gpt_traced(
            model = MODEL,
            messages=[{"role": "user", "content": full_prompt }],
            options = {}
        )
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError("Answer was aborted due tokens limit")

        text = (choice.message.content or "").strip()
        text = text.split("\nObservation", 1)[0].rstrip()
        print(text)
        if text.startswith("Final Answer:"):
            answer = text.removeprefix("Final Answer:").strip()
            print(f"\nFinal answer is: {answer}")
            return answer
        match = re.fullmatch(
            r"Action:\s*(\w+)\s*\nAction Input:\s*(\{.*\})",
            text,
            flags=re.DOTALL,
        )
        if match is None:
            raise ValueError("Incorrect answer format")
        tool_name = match.group(1)
        tool_args = json.loads(match.group(2))

        if not isinstance(tool_args, dict):
            raise ValueError("Incorrect answer format")
        selected_tool = tools.get(tool_name)
        if selected_tool is None:
            raise ValueError(f"Non known tool, {selected_tool}")
        inspect.signature(selected_tool).bind(**tool_args)
        print(f"\n Tool call: {tool_name} with args: {tool_args}")
        observation = selected_tool(**tool_args)
        print(f"Observation: {observation}")
        scratchpad +=(
            f"{text}\n"
            f"Observation:{json.dumps(observation, ensure_ascii=False)}\n\n"
        )
    raise  RuntimeError(f"Agent do not finish work by {MAX_ITERATIONS} iterations.")


@traceable(name="ChatGpt Chat", run_type="llm")
def chat_gpt_traced(model, messages, options):
    return chatGptClient.chat.completions.create(
        model=model,
        messages=messages,
        **options)

def main():
    print("Available tools:")
    print(tool_descriptions)
    print(f"Tool names: {tool_names}")
    run_agent("What is the price of a laptop if I have a bronze discount?")



if __name__ == "__main__":
    main()