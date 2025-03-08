from llama_index.core.llms import ChatMessage
from llama_index.core.tools import ToolSelection
from llama_index.core.workflow import Event
    

class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]


class RouterInputEvent(Event):
    input: list[ChatMessage]


class RouterResponseEvent(Event):
    response: str
    messages: list[ChatMessage]


class RouterToolSelectionEvent(Event):
    response: str
    messages: list[ChatMessage]


class RouterEscapeEvent(Event):
    hint: str