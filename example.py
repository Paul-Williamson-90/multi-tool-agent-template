import asyncio
from typing import Union
from src.skills.base import SkillArgAttr, FunctionCallSkill, SkillMap
from src.agents.router import AgentFlowOpenAI
from llama_index.llms.openai import OpenAI

from dotenv import load_dotenv
import os


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

    def execute(self, args) -> Union[int, float]:
        print("args:", args)
        if isinstance(args, dict) and "input" in args:
            args = args["input"]
        else:
            return 'Invalid input: expected a dictionary with the key "input" that\'s value is a dictionary.'

        answer = args["a"] * args["b"]
        return f"The answer is {answer}."


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = OpenAI(api_key=OPENAI_API_KEY, temperature=0.1)
skillmap = SkillMap(skills=[Multiply()])


async def example_test(input: str):
    workflow = AgentFlowOpenAI(llm=llm, skill_map=skillmap, model="gpt-4o")
    res = await workflow.run(input=input)
    return res


res = asyncio.run(example_test("What is 2 times 3?"))
print(res)
