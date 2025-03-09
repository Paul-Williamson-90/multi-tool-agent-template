import inspect
import typing
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable

from pydantic import BaseModel, TypeAdapter, field_validator, model_validator

from src.routers.skills.errors import SkillArgException


class SkillOutput(BaseModel):
    response_to_llm: str

    def __str__(self) -> str:
        return self.response_to_llm


class SkillArgAttr(BaseModel):
    """Defines an input argument for a skill.

    Parameters
    ----------
    name : str
        The name of the argument.
    dtype : str
        The data type of the argument.
    description : str
        The description of the argument.
    required : bool, optional
        Whether the argument is required or not, by default False.
    default : Any, optional
        The default value of the argument, by default None.
    """

    name: str
    dtype: str
    description: str
    required: bool = False
    default: Any = None

    @field_validator("dtype")
    def dtype_validation(cls, v: str) -> Any:
        """Validates the data type of the argument.

        Parameters
        ----------
        v : str
            The data type of the argument.

        Returns
        -------
        Any
            The data type of the argument.
        """
        try:
            eval_type = eval(
                v,
                {"__builtins__": __builtins__},
                {"typing": typing, "uuid": uuid, **vars(typing)},
            )
            if not any(
                [
                    inspect.getmodule(eval_type) is typing,
                    inspect.getmodule(eval_type) is uuid,
                    isinstance(eval_type, type),
                ]
            ):
                raise SkillArgException(
                    f'dtype {v} is not a valid type (e.g. "Union[str, int]")'
                )
        except Exception as e:
            raise SkillArgException(
                f'dtype {v} is not a valid type (e.g. "Union[str, int]"): {e}'
            )
        return v

    @model_validator(mode="before")
    def required_and_default_validation(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validates the required and default values of the argument.

        Parameters
        ----------
        values : dict[str, Any]
            The values of the argument.

        Returns
        -------
        dict[str, Any]
            The values of the argument.
        """
        required = values.get("required")
        default = values.get("default")
        if required and default is not None:
            raise SkillArgException(
                "If 'required' is set to True, 'default' must be None"
            )
        return values

    @model_validator(mode="after")
    def default_correct_dtype(self) -> "SkillArgAttr":
        """Validates the default value of the argument.

        Returns
        -------
        SkillArgAttr
            The argument object.
        """
        dtype = self.dtype
        default = self.default
        if default is not None:
            eval_type = eval(
                dtype,
                {"__builtins__": __builtins__},
                {"typing": typing, "uuid": uuid, **vars(typing)},
            )
            adapter = TypeAdapter(eval_type)
            try:
                adapter.validate_python(default)
            except Exception:
                raise SkillArgException(
                    f"default value {default} is not of type {dtype}"
                )
        return self

    def validate_input_arg(self, input: Any) -> bool:
        """Validates the input argument.

        Parameters
        ----------
        input : Any
            The input argument.

        Returns
        -------
        bool
            Whether the input argument is valid or not.
        """
        try:
            eval_type = eval(
                self.dtype,
                {"__builtins__": __builtins__},
                {"typing": typing, "uuid": uuid, **vars(typing)},
            )
            adapter = TypeAdapter(eval_type)
            adapter.validate_python(input)
            return True
        except Exception:
            return False


class FunctionCallSkill(ABC):
    """Parent class for defining a skill that can be called by the LLM router agent.

    Example Usage:
    -------------
    ```python
    from src.routers.skills.base import FunctionCallSkill, SkillArgAttr, SkillOutput


    class Multiply(FunctionCallSkill):
        def __init__(
            self,
            name: str = "multiply",
            description: str = "Use this tool to multiply two numbers.",
            function_args=[
                SkillArgAttr(
                    name="a", dtype="float", description="First number", required=True
                ),
                SkillArgAttr(
                    name="b", dtype="float", description="Second number", required=True
                ),
            ],
            visible_to_human: bool = True,
        ):
            super().__init__(
                name=name,
                description=description,
                function_args=function_args,
                visible_to_human=visible_to_human,
            )

        def execute(self, a: float, b: float) -> float:  # type: ignore
            return SkillOutput(response_to_llm=f"{a} times {b} is {a*b}")
    ```
    """
    def __init__(
        self,
        name: str,
        description: str,
        function_args: list[SkillArgAttr] = [],
        visible_to_human: bool = False,
    ):
        """Instantiates a FunctionCallSkill object.

        Parameters
        ----------
        name : str
            The name of the skill.
        description : str
            The description of the skill.
        function_args : list[SkillArgAttr], optional
            The arguments of the skill, by default [].
        visible_to_human : bool, optional
            Whether the RouterAgent should let the user know this skill exists when asked, by default False
        """
        self.name = name
        self.description = description
        self.function_args = function_args
        self.function_callable = self.handle_router_input
        self.function_dict = self._prepare_function_dict()
        self.visible_to_human = visible_to_human

    def _prepare_function_dict(self) -> dict:
        """Prepares the function dictionary / schema.

        Returns
        -------
        dict
            The function dictionary / schema.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        arg.name: {
                            "type": arg.dtype,
                            "description": arg.description,
                        }
                        if arg.required
                        else {
                            "type": arg.dtype,
                            "description": arg.description,
                            "default": arg.default,
                        }
                        for arg in self.function_args
                    },
                    "required": [
                        arg.name for arg in self.function_args if arg.required
                    ],
                },
            },
        }

    def get_function_name(self) -> str:
        """Gets the name of the function.

        Returns
        -------
        str
            The name of the function.
        """
        return self.name

    def get_function_dict(self) -> dict:
        """Gets the function dictionary / schema.

        Returns
        -------
        dict[str, dict[str, Union[str, dict]]]
            The function dictionary / schema.
        """
        return self.function_dict

    def get_function_callable(self) -> Callable:
        """Get the skill's function callable.

        Returns
        -------
        Callable
            The skill's function callable.
        """
        return self.function_callable

    def handle_router_input(self, input: dict[str, Any]) -> SkillOutput:
        """Handles the input from the RouterAgent.

        Parameters
        ----------
        input : dict[str, Any]
            The input from the RouterAgent.

        Returns
        -------
        SkillOutput
            The output from the skill.
        """
        if len(self.function_args) == 0:
            return self.execute()

        input_args = input

        parsed_args: dict[str, Any] = dict()

        for arg in self.function_args:
            if arg.name in input_args:
                if not arg.validate_input_arg(input_args[arg.name]):
                    return SkillOutput(
                        response_to_llm=f'Invalid input: argument "{arg.name}" must be of type {arg.dtype}'
                    )

                parsed_args[arg.name] = input_args[arg.name]
            elif arg.required and not arg.default:
                return SkillOutput(
                    response_to_llm=f'Invalid input: missing required argument "{arg.name}"'
                )

            else:
                parsed_args[arg.name] = arg.default

        return self.execute(**parsed_args)  # type: ignore

    @abstractmethod
    def execute(self) -> SkillOutput:
        """Abstract method to be implemented by the child class.
        This method should contain the logic for the skill execution.

        Returns
        -------
        SkillOutput
            The output from the skill.
        """


class SkillMap:
    """A class to manage a collection of FunctionCallSkill objects that the RouterAgent can use.
    The module is a component of the RouterAgent.

    Example Usage:
    -------------
    ```python
    from src.routers.base import RouterAgent
    from src.routers.skills import SkillMap
    from working_example.skills import Multiply
    ...
    
    skill_map = SkillMap(skills=[Multiply()])

    agent = RouterAgent(
        chat_id=hat_id,
        llm=llm,
        skill_map=skill_map,
        context_modules=context_modules,
        chat_history=memory,
    )
    ```
    """
    def __init__(self, skills: list[FunctionCallSkill]):
        """Instantiates a SkillMap object.

        Parameters
        ----------
        skills : list[FunctionCallSkill]
            A list of FunctionCallSkill objects.
        """
        self.skill_map: dict = dict()
        for skill in skills:
            self.add_skill(skill)
        self._add_available_tools_to_map()

    def add_skill(self, skill: FunctionCallSkill):
        """Adds a skill to the skill map.

        Parameters
        ----------
        skill : FunctionCallSkill
            The skill to be added.
        """
        self.skill_map[skill.get_function_name()] = {
            "function_dict": skill.get_function_dict(),
            "function_callable": skill.get_function_callable(),
            "visible_to_human": skill.visible_to_human,
        }

    def _add_available_tools_to_map(self):
        """Adds the available_tools skill to the skill map.
        This skill is used to get the necessary information to generate a response to the user
        when the user asks what the RouterAgent can do / what tools are available.
        """
        if any(self.skill_map[skill]["visible_to_human"] for skill in self.skill_map):
            self.skill_map["available_tools"] = {
                "function_dict": {
                    "type": "function",
                    "function": {
                        "name": "available_tools",
                        "description": (
                            "When the user asks what you can do / what tools are available, "
                            "use this tool to get the necessary information to generate "
                            "a response to the user."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    },
                },
                "function_callable": self._get_available_tools_description,
                "visible_to_human": False,
            }

    def _get_available_tools_description(self, *args, **kwargs) -> SkillOutput:
        """Gets the description of the available tools. This information is passed to \
        the RouterAgent when the user asks what tools it has and the Agent calls the \
        'available_tools' skill.

        Returns
        -------
        SkillOutput
            The description of the available tools
        """
        content = "**Here are the tools that you have available:**\n\n"
        for skill in self.skill_map:
            if self.skill_map[skill]["visible_to_human"]:
                content += (
                    self.skill_map[skill]["function_dict"]["function"]["name"]
                    + ": "
                    + self.skill_map[skill]["function_dict"]["function"]["description"]
                    + "\n\n"
                )
        content += (
            "**When reporting back to the user, you must provide a more user-friendly description of the tools"
            " for non-technical audience.**\n"
            "**You should also provide helpful information on how best to get you (the agent) to use the tools.**\n"
            "**This should include advising the user to be clear with their instructions and to provide all "
            "the necessary context (e.g. avoid ambiguity, try not to use acronyms, etc.).**"
        )
        return SkillOutput(response_to_llm=content)

    def get_function_callable_by_name(self, skill_name: str) -> Callable:
        """Gets the function callable of a skill by name.

        Parameters
        ----------
        skill_name : str
            The name of the skill.

        Returns
        -------
        Callable
            The function callable of the skill.
        """
        return self.skill_map[skill_name]["function_callable"]

    def get_combined_function_description_for_agent(self) -> list[dict]:
        """Gets the combined function description for the RouterAgent. This information is \
        passed to the RouterAgent during its reasoning and tool selection steps.

        Returns
        -------
        list[dict]
            The combined function description for the RouterAgent.
        """
        combined_dict: list[dict] = []
        for _, function_attr in self.skill_map.items():
            combined_dict.append(function_attr["function_dict"])
        return combined_dict

    def get_function_list(self) -> list[str]:
        """Gets the list of skill names in the skill map.

        Returns
        -------
        list[str]
            The list of skill names in the skill map.
        """
        return list(self.skill_map.keys())

    def get_list_of_function_callables(self) -> list[Callable]:
        """Gets the list of function callables in the skill map.

        Returns
        -------
        list[Callable]
            The list of function callables in the skill map
        """
        return [skill["function_callable"] for skill in self.skill_map.values()]

    def get_function_dict_by_name(self, skill_name: str) -> str:
        """Gets the function dictionary of a skill by name.

        Parameters
        ----------
        skill_name : str
            The name of the skill.

        Returns
        -------
        str
            The function dictionary of the skill.
        """
        return str(self.skill_map[skill_name]["function_dict"]["function"])

    @property
    def info(self) -> str:
        """Gets the information about the tools available in the skill map. This information \
        is used in the RouterAgent reasoning and tool selection steps.

        Returns
        -------
        str
            The information about the tools available in the skill map.
        """
        tools_meta_list: list[str] = []
        for func in self.get_function_list():
            tools_meta_list.append(self.get_function_dict_by_name(func))
        return "\n\n".join(str(tool) for tool in tools_meta_list)
