from typing import Literal

from pydantic import BaseModel
from llama_index.core.tools import ToolSelection


class Step(BaseModel):
    """
    Use this schema to think through the problem step-by-step before generating your response/action.

    Attributes:
        - thought: str - Your thoughts on what you need to do / any considerations you need to make.
        - conclusion: str - The conclusion you've come to after thinking through your thoughts.
    """
    
    thought: str
    conclusion: str

    def __str__(self) -> str:
        result = f"<thought>{self.thought}</thought>\n<conclusion>{self.conclusion}</conclusion>"
        return result


class ResponseType(BaseModel):
    """
    Use this schema to synthesize your thoughts and decide on generating a response OR action tool calls to gather more information \
    to better generate a response to the user. Your decision process should be based on critical evaluation on whether you have enough \
    information to response to the user or whether you need to call a tool to gather more information.
    - If you need to call a tool, you should outline the reasons why and how you would use the tool in your thought process.
    - If you have all the information you need to response to the user, you should plan your response in your thought process.
    
    Attributes:
        - steps: list[Step] - Your step-by-step thought process for planning your response or tool calls.
        - next_action: Literal["tool_call", "response"] - The next action you need to take. If you need to call a tool, set this to \
        "tool_call". If you have all the information you need to response to the user, set this to "response".
    """

    steps: list[Step]
    next_action: Literal["tool_call", "response"]

    def __str__(self) -> str:
        result = "\n".join([str(step) for step in self.steps])
        return result


class ToolCallResponse(BaseModel):
    """
    Use this schema to select the toold to call based on your thought process prior.

    Attributes:
        output: list[ToolSelection] - The tools you would like to call based on your thought process.
    """

    output: list[ToolSelection]

    def __str__(self) -> str:
        result = f"Output: {[str(tool) for tool in self.output]}"
        return result