import pytest
from typing import Union
from unittest.mock import Mock, MagicMock
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.agents.router import AgentFlowOpenAI, RouterInputEvent, StopEvent, ToolCallEvent, ChatMessage, ToolSelection
from src.skills.base import SkillMap, SkillArgAttr, FunctionCallSkill

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

def test_agent_flow_openai_init():
    skill_map = SkillMap(skills=[Multiply()])
    llm = MagicMock()
    workflow = AgentFlowOpenAI(llm=llm, skill_map=skill_map, model="gpt-4o")
    assert isinstance(workflow, AgentFlowOpenAI)

@pytest.mark.asyncio
async def test_agent_flow_openai_prepare_agent():
    # TODO: Elaborate on test cases
    skill_map = SkillMap(skills=[Multiply()])
    llm = MagicMock()
    workflow = AgentFlowOpenAI(llm=llm, skill_map=skill_map, model="gpt-4o")
    res = await workflow.prepare_agent(Mock(input="cheese"))
    assert isinstance(res, RouterInputEvent)

async def mock_achat_with_tools(*args, **kwargs):
    return Mock(message=Mock(content="cheese"))

def mock_get_tool_calls_from_response_empty(*args, **kwargs):
    return []

def mock_get_tool_calls_from_response_not_empty(*args, **kwargs):
    return [ToolSelection(tool_name="multiply", tool_kwargs={"a": 1, "b": 2}, tool_id="1")]

@pytest.mark.asyncio
async def test_agent_flow_openai_router():
    # No tool calls
    skill_map = SkillMap(skills=[Multiply()])
    llm = MagicMock()
    llm.achat_with_tools = mock_achat_with_tools
    llm.get_tool_calls_from_response = mock_get_tool_calls_from_response_empty
    workflow = AgentFlowOpenAI(llm=llm, skill_map=skill_map, model="gpt-4o")
    res = await workflow.router(Mock(input=[ChatMessage(role="user", content="cheese")]))
    assert isinstance(res, StopEvent)

    # tool calls
    skill_map = SkillMap(skills=[Multiply()])
    llm = MagicMock()
    llm.achat_with_tools = mock_achat_with_tools
    llm.get_tool_calls_from_response = mock_get_tool_calls_from_response_not_empty
    workflow = AgentFlowOpenAI(llm=llm, skill_map=skill_map, model="gpt-4o")
    res = await workflow.router(Mock(input=[ChatMessage(role="user", content="cheese")]))
    assert isinstance(res, ToolCallEvent)