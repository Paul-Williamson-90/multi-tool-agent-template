import os
import asyncio
import logging
from dotenv import load_dotenv
from typing import Union

from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer

from src.skills.base import SkillArgAttr, FunctionCallSkill, SkillMap
from src.agents.router import RouterAgent


# show INFO logs
logging.basicConfig(level=logging.INFO)


class Multiply(FunctionCallSkill):
    def __init__(self):
        name = "multiply"
        description = "Multiply two numbers"
        function_args = [
            SkillArgAttr(
                name="a",
                description="First number",
                dtype="Union[int, float]",
                required=True,
            ),
            SkillArgAttr(
                name="b",
                description="Second number",
                dtype="Union[int, float]",
                required=True,
            ),
        ]
        super().__init__(
            name=name, description=description, function_args=function_args
        )

    def execute(self, a: Union[int, float], b: Union[int, float]) -> str:
        answer = a * b
        return f"The answer is {answer}."


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = OpenAI(api_key=OPENAI_API_KEY, temperature=0.1)
memory = ChatMemoryBuffer(token_limit=40000).from_defaults(llm=llm)
skillmap = SkillMap(skills=[Multiply()])


async def example_test(input: str) -> str:
    workflow = RouterAgent(llm=llm, skill_map=skillmap, memory=memory)
    res = await workflow.run(input=input)
    return res

while True:
    user_input = input("User: ")
    res = asyncio.run(example_test(user_input))
    print("LLM:", res)
