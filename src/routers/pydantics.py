from enum import Enum

from pydantic import BaseModel
from llama_index.core.tools import ToolSelection
from llama_index.core.llms import ChatMessage
from llama_index.core.base.llms.types import MessageRole


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


class NextAction(Enum):
    TOOL_CALL = "tool_call"
    RESPONSE = "response"


class PlanningStep(BaseModel):
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
    next_action: NextAction

    def __str__(self) -> str:
        result = "\n".join([str(step) for step in self.steps])
        return result

    def as_msg(self) -> ChatMessage:
        return ChatMessage(
            content="\n".join([str(step) for step in self.steps]),
            user=MessageRole.ASSISTANT,
        )


class ToolCallResponse(BaseModel):
    """
    Use this schema to select the toold to call based on your thought process prior.

    Attributes:
        output: list[ToolSelection] - The tools you would like to call based on your thought process.
    """

    output: ToolSelection

    def __str__(self) -> str:
        return str(self.output)

    def as_msg(self) -> ChatMessage:
        return ChatMessage(content=str(self.output), user=MessageRole.ASSISTANT)


class ContextExtraction(BaseModel):
    """
    Use this schema to select a context in memory to extract facts from which will be useful for generating a response to the user.
    The sort of instructions for extraction should be the type of content you would like to extract from the context.

    Attributes:
        memory_object: str - The name of the context memory object that contains the context_id you would like to extract facts from.
        context_id: str - The context_id you would like to extract facts from.
        instructions: str - The instructions you would like to provide to the LLM for extracting facts from the context.
    """

    memory_object: str
    context_id: str
    instructions: str

    def __str__(self) -> str:
        return f"Memory Object: {self.memory_object}\nContext ID: {self.context_id}"


class ContextSelection(BaseModel):
    """
    Use this schema to select contexts that are held in memory with specific instructions as to the sort of information you would like to \
    extract from the context.

    Attributes:
        contexts: list[ContextExtraction] - The contexts you would like to extract facts from, default is an empty list.
    """

    contexts: list[ContextExtraction] = []

    def as_msg(self) -> ChatMessage:
        requests = "- " + "\n- ".join([str(context) for context in self.contexts])
        content = (
            f"I will use the following contexts to extract facts from:\n{requests}"
        )
        return ChatMessage(
            content=content,
            role=MessageRole.ASSISTANT,
        )


class SelectedContext(BaseModel):
    question: str
    facts: str
    memory_object: str
    context_id: str

    def __str__(self) -> str:
        return (
            f"# {self.memory_object}: {self.context_id}\n"
            f"Question: {self.question}\nExtracted Facts:\n{self.facts}"
        )
