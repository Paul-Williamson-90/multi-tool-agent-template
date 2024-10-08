import pytest
from typing import Union

from pydantic_core._pydantic_core import ValidationError

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.skills.base import (
    SkillArgAttr,
    SkillMap,
    FunctionCallSkill,
    SkillArgException,
)


def test_skill_arg_attr():
    # Success case
    skill_arg = SkillArgAttr(
        name="arg1",
        dtype="Union[str, int]",
        description="This is an argument",
        required=True,
        default=None,
    )
    assert skill_arg.name == "arg1"
    assert skill_arg.dtype == "Union[str, int]"
    assert skill_arg.description == "This is an argument"
    assert skill_arg.required is True
    assert skill_arg.default is None

    # Failure case, required is True but default is not None
    with pytest.raises(SkillArgException):
        skill_arg = SkillArgAttr(
            name="arg1",
            dtype="Union[str, int]",
            description="This is an argument",
            required=True,
            default="test",
        )

    # Failure case, dtype not string
    with pytest.raises(ValidationError):
        skill_arg = SkillArgAttr(
            name="arg1",
            dtype=Union[str, int],
            description="This is an argument",
            required=True,
            default=None,
        )

    # Failure case, dtype not valid type
    with pytest.raises(SkillArgException):
        skill_arg = SkillArgAttr(
            name="arg1",
            dtype='"invalid"',
            description="This is an argument",
            required=True,
            default=None,
        )

    # Failure case, dtype can't eval
    with pytest.raises(SkillArgException):
        skill_arg = SkillArgAttr(
            name="arg1",
            dtype="c * x = Union[str, int], gser [] lkasd",
            description="This is an argument",
            required=True,
            default=None,
        )

    # Failure case, default not correct type
    with pytest.raises(SkillArgException):
        skill_arg = SkillArgAttr(
            name="arg1",
            dtype="str",
            description="This is an argument",
            required=False,
            default=4.56,
        )

    # Success, default correct
    skill_arg = SkillArgAttr(
        name="arg1",
        dtype="Union[str, int]",
        description="This is an argument",
        required=False,
        default="default",
    )
    assert skill_arg.default == "default"


class MockFunctionCallSkill(FunctionCallSkill):
    def execute(self, *args, **kwargs) -> str:
        return "test_successful"


def test_function_call_skill_init():
    skill = MockFunctionCallSkill(
        name="test",
        description="This is a test skill",
        function_args=[
            SkillArgAttr(
                name="arg1",
                dtype="Union[str, int]",
                description="This is an argument",
                required=True,
                default=None,
            )
        ],
    )
    assert isinstance(skill, FunctionCallSkill)


@pytest.fixture
def case_function_call_skill() -> tuple[MockFunctionCallSkill, SkillArgAttr]:
    skill_arg = SkillArgAttr(
        name="arg1",
        dtype="Union[str, int]",
        description="This is an argument",
        required=True,
        default=None,
    )
    skill = MockFunctionCallSkill(
        name="test", description="This is a test skill", function_args=[skill_arg]
    )
    return (skill, skill_arg)


def test_function_call_skill__prepare_function_dict(case_function_call_skill):
    skill: MockFunctionCallSkill = case_function_call_skill[0]
    skill_arg: SkillArgAttr = case_function_call_skill[1]
    expected = {
        "type": "function",
        "function": {
            "name": skill.name,
            "description": skill.description,
            "parameters": {
                "type": "object",
                "properties": {
                    skill_arg.name: {
                        "type": skill_arg.dtype,
                        "description": skill_arg.description,
                    }
                },
                "required": [skill_arg.name],
            },
        },
    }
    assert skill._prepare_function_dict() == expected


def test_function_call_skill_get_function_name(case_function_call_skill):
    skill: MockFunctionCallSkill = case_function_call_skill[0]
    assert skill.get_function_name() == "test"


def test_function_call_skill_get_function_dict(case_function_call_skill):
    skill: MockFunctionCallSkill = case_function_call_skill[0]
    skill_arg: SkillArgAttr = case_function_call_skill[1]
    expected = {
        "type": "function",
        "function": {
            "name": skill.name,
            "description": skill.description,
            "parameters": {
                "type": "object",
                "properties": {
                    skill_arg.name: {
                        "type": skill_arg.dtype,
                        "description": skill_arg.description,
                    }
                },
                "required": [skill_arg.name],
            },
        },
    }
    assert skill.get_function_dict() == expected


def test_function_call_skill_get_function_callable(case_function_call_skill):
    skill: MockFunctionCallSkill = case_function_call_skill[0]
    assert skill.get_function_callable() == skill.handle_router_input


def test_function_call_skill_handle_router_input(case_function_call_skill):
    # no args
    skill = MockFunctionCallSkill(
        name="test", description="This is a test skill", function_args=[]
    )
    assert skill.handle_router_input({}) == "test_successful"

    # no input key
    skill: MockFunctionCallSkill = case_function_call_skill[0]
    skill_arg: SkillArgAttr = case_function_call_skill[1]
    expected = 'Invalid input: expected a dictionary with the key "input" that\'s value is a dictionary.'
    assert skill.handle_router_input({}) == expected

    # fully successful
    expected = "test_successful"
    assert skill.handle_router_input({"input": {skill_arg.name: "test"}}) == expected

    # missing arg
    expected = 'Invalid input: missing required argument "arg1"'
    assert skill.handle_router_input({"input": {}}) == expected

    # wrong arg type
    expected = 'Invalid input: argument "arg1" must be of type Union[str, int]'
    assert skill.handle_router_input({"input": {skill_arg.name: [4.56]}}) == expected

    # default arg
    skill_arg.default = "default"
    skill_arg.required = False
    expected = "test_successful"
    assert skill.handle_router_input({"input": {}}) == expected


@pytest.fixture
def case_skill_map() -> tuple[SkillMap, MockFunctionCallSkill]:
    skill = MockFunctionCallSkill(
        name="test",
        description="This is a test skill",
        function_args=[
            SkillArgAttr(
                name="arg1",
                dtype="Union[str, int]",
                description="This is an argument",
                required=True,
                default=None,
            )
        ],
    )
    skill_map = SkillMap(skills=[skill])
    return (skill_map, skill)


def test_skill_map_init(case_skill_map):
    skill_map: SkillMap = case_skill_map[0]
    assert isinstance(skill_map, SkillMap)


def test_skill_map_get_function_callable_by_name(case_skill_map):
    skill_map: SkillMap = case_skill_map[0]
    skill: MockFunctionCallSkill = case_skill_map[1]
    assert skill_map.get_function_callable_by_name("test") == skill.handle_router_input


def test_skill_map_get_combined_function_description_for_agent(case_skill_map):
    skill_map: SkillMap = case_skill_map[0]
    expected = [
        {
            "type": "function",
            "function": {
                "name": "test",
                "description": "This is a test skill",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "arg1": {
                            "type": "Union[str, int]",
                            "description": "This is an argument",
                        }
                    },
                    "required": ["arg1"],
                },
            },
        }
    ]
    assert skill_map.get_combined_function_description_for_agent() == expected


def test_skill_map_get_function_list(case_skill_map):
    skill_map: SkillMap = case_skill_map[0]
    skill: MockFunctionCallSkill = case_skill_map[1]
    assert skill_map.get_function_list() == [skill.name]


def test_skill_map_get_list_of_function_callables(case_skill_map):
    skill_map: SkillMap = case_skill_map[0]
    skill: MockFunctionCallSkill = case_skill_map[1]
    assert skill_map.get_list_of_function_callables() == [skill.handle_router_input]


def test_skill_map_get_function_dict_by_name(case_skill_map):
    skill_map: SkillMap = case_skill_map[0]
    assert (
        skill_map.get_function_dict_by_name("test")
        == "{'name': 'test', 'description': 'This is a test skill', 'parameters': {'type': 'object', 'properties': {'arg1': {'type': 'Union[str, int]', 'description': 'This is an argument'}}, 'required': ['arg1']}}"
    )
