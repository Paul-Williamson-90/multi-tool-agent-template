from llama_index.core.tools import ToolSelection
from llama_index.core.workflow import Event
    

class ToolCallEvent(Event):
    tool_call: ToolSelection


class RouterInputEvent(Event):
    pass


class RouterResponseEvent(Event):
    pass


class RouterToolSelectionEvent(Event):
    pass


class RouterEscapeEvent(Event):
    hint: str