import inspect
import typing
from typing import Any, Callable, types
from pydantic import BaseModel, validator
from abc import ABC, abstractmethod


class SkillArgAttr(BaseModel):
    """
    Attributes:
    - name: str - name of the argument
    - dtype: str - data type of the argument (typing or python type)
    - description: str - description of the argument
    - required: bool - whether the argument is required or not
    """

    name: str
    dtype: str
    description: str
    required: bool = False

    @validator("dtype")
    def dtype_validation(cls, v: str) -> Any:
        try:
            eval_type = eval(
                v, {"__builtins__": __builtins__}, {"typing": typing, **vars(typing)}
            )
            if not any(
                [
                    isinstance(eval_type, types.GenericAlias),
                    isinstance(eval_type, typing._SpecialGenericAlias),
                    isinstance(eval_type, type),
                ]
            ):
                raise ValueError(f"{v} is not a valid type")
        except Exception as e:
            raise ValueError(f"{v} is not a valid type: {e}")
        return v

    @staticmethod
    def is_typing_type(ex) -> bool:
        return hasattr(ex, "__origin__") and inspect.getmodule(ex) is typing

    @staticmethod
    def is_python_type(ex) -> bool:
        return isinstance(ex, type)


class FunctionCallSkill(ABC):
    def __init__(
        self,
        name: str,
        description: str,
        function_args: list[SkillArgAttr],
    ):
        """
        Instantiates a FunctionCallSkill object.
        This object is used to define a skill that can be called by the LLM router agent.

        Args:
        - name: str - name of the function
        - description: str - description of the function
        - function_args: list[SkillArgAttr] - list of SkillArgAttr objects that define the arguments of the function
        """
        self.name = name
        self.description = description
        self.function_args = function_args
        self.function_callable = self.execute
        self.function_dict = self._prepare_function_dict()

    def _prepare_function_dict(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        arg.name: {"type": arg.dtype, "description": arg.description}
                        for arg in self.function_args
                    },
                    "required": [
                        arg.name for arg in self.function_args if arg.required
                    ],
                },
            },
        }

    def get_function_name(self) -> str:
        return self.name

    def get_function_dict(self) -> dict:
        return self.function_dict

    def get_function_callable(self) -> Callable:
        return self.function_callable

    @abstractmethod
    def execute(self, *args, **kwargs):
        """
        Abstract method that should be implemented by the child class.
        This method should contain the logic of the function that the skill is supposed to execute.
        """
        pass


class SkillMap:
    def __init__(self, skills: list[FunctionCallSkill]):
        """
        Instantiates a SkillMap object.
        This object is used to store a list of FunctionCallSkill objects.

        Args:
        - skills: list[FunctionCallSkill] - list of FunctionCallSkill objects
        """
        skills = [skill() for skill in skills]

        self.skill_map = {}
        for skill in skills:
            self.skill_map[skill.get_function_name()] = {
                "function_dict": skill.get_function_dict(),
                "function_callable": skill.get_function_callable(),
            }

    def get_function_callable_by_name(self, skill_name) -> Callable:
        return self.skill_map[skill_name]["function_callable"]

    def get_combined_function_description_for_agent(self) -> list[dict]:
        combined_dict: list[dict] = []
        for _, function_attr in self.skill_map.items():
            combined_dict.append(function_attr["function_dict"])
        return combined_dict

    def get_function_list(self) -> list[str]:
        return list(self.skill_map.keys())

    def get_list_of_function_callables(self) -> list[Callable]:
        return [skill["function_callable"] for skill in self.skill_map.values()]

    def get_function_description_by_name(self, skill_name) -> str:
        return str(self.skill_map[skill_name]["function_dict"]["function"])
