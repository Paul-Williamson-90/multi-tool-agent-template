import asyncio
import os
import uuid

from dotenv import load_dotenv
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI

from src.routers.base import RouterAgent
from src.routers.skills import SkillMap
from working_example.context import DummyContextModule
from working_example.skills import Multiply

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_llm() -> OpenAI:
    return OpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)


def get_agent() -> RouterAgent:
    dummy_chat_id = uuid.uuid4()
    memory = ChatMemoryBuffer(token_limit=40000)

    llm = get_llm()

    skill_map = SkillMap(skills=[Multiply()])

    context_modules = [DummyContextModule(llm=llm, chat_id=dummy_chat_id)]

    agent = RouterAgent(
        chat_id=dummy_chat_id,
        llm=llm,
        skill_map=skill_map,
        context_modules=context_modules,
        chat_history=memory,
    )
    return agent


def invoke(agent: RouterAgent, user_input: str) -> StreamingAgentChatResponse:
    async def run(input: str) -> StreamingAgentChatResponse:
        response = await agent.run(input=input)
        return response

    return asyncio.run(run(input=user_input))
