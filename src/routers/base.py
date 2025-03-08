import uuid
from typing import Union, Any, Optional
import inspect
import logging
from datetime import datetime

from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step
from llama_index.core.llms.llm import LLM
from llama_index.core.base.llms.types import MessageRole

from src.routers.events import (
    RouterInputEvent, 
    ToolCallEvent, 
    RouterResponseEvent, 
    RouterToolSelectionEvent,
    RouterEscapeEvent
)
from src.routers.pydantics import PlanningStep, ToolCallResponse, NextAction
from src.routers.prompts import (
    SYSTEM_PROMPT,
    ROUTER_AGENT_PROMPT_TEMPLATE,
    ESCAPE_PROMPT,
    ROUNDS_EXCEEDED_HINT,
    ACTION_DECISION_INSTRUCTIONS,
    ERROR_HINT,
    RESPONSE_INSTRUCTIONS,
    TOOL_DECISION_INSTRUCTIONS,
)
from src.skills.base import SkillMap
from src.routers.constants import DEFAULT_TOKEN_LIMIT
from src.routers.condensers import CondenseModuleType
from src.invocations import structured_invocation, non_structured_invocation


logger = logging.getLogger(__name__)


class RouterAgent(Workflow):
    _generation_kwargs: dict[str, Any] = {"max_tokens": 4000}
    _tool_selection_kwargs: dict[str, Any] = {"max_tokens": 500}
    _rounds_limit: int = 5
    _round: int = 1

    def __init__(
        self,
        llm: LLM,
        skill_map: SkillMap,
        condense_module: CondenseModuleType,
        chat_history: Optional[ChatMemoryBuffer] = None,
        timeout: int = 300,
        system_prompt: str = SYSTEM_PROMPT,
        chat_id: Optional[uuid.UUID] = None,
    ):
        self.chat_id = chat_id or uuid.uuid4()
        logger.info(f"[{self.chat_id}]: Initializing RouterAgent")
        
        super().__init__(timeout=timeout)
        
        self.llm: LLM = llm
        self.skill_map: SkillMap = skill_map
        self.condense_module: CondenseModuleType = condense_module
        self.system_prompt: str = self._prepare_system_prompt(system_prompt)
        self.memory: ChatMemoryBuffer = self._prepare_chat_memory(chat_history)
        self.internal_memory: ChatMemoryBuffer = self._prepare_internal_memory()

    @step
    async def prepare_agent(self, ev: StartEvent) -> RouterInputEvent:
        logger.info(f"[{self.chat_id}]: Preparing RouterAgent")
        
        self._round = 0
        self.condense_module.reset()
        self.internal_memory.reset()

        user_input = ev.input
        user_msg = ChatMessage(role=MessageRole.USER, content=user_input)
        self.memory.put(user_msg)
        return RouterInputEvent()

    @step
    async def router(self, ev: RouterInputEvent) -> Union[RouterResponseEvent, RouterEscapeEvent, RouterToolSelectionEvent]:
        logger.info(f"[{self.chat_id}]: RouterAgent router")
        self._round += 1

        if self._round > self._rounds_limit:
            return RouterEscapeEvent(hint=ROUNDS_EXCEEDED_HINT)

        try:
            condensed = self.condense_module(self.memory)
            thoughts = self._gather_thoughts()
            context = ROUTER_AGENT_PROMPT_TEMPLATE.format(
                chat_history=str(condensed),
                system=self.system_prompt,
                tools=self.skill_map.tool_metadata_str,
                thoughts=thoughts,
                instructions=ACTION_DECISION_INSTRUCTIONS,
            )
            response: PlanningStep = structured_invocation(
                llm=self.llm,
                context=context,
                pydantic_object=PlanningStep,
                llm_kwargs=self._generation_kwargs,
            )

        except Exception as e:
            logger.error(f"[{self.chat_id}]: RouterAgent encountered an error: {e}")
            return RouterEscapeEvent(hint=ERROR_HINT)

        next_action = response.next_action
        self.internal_memory.put(response.as_msg())

        if next_action == NextAction.RESPONSE:
            return RouterResponseEvent()
        else:
            return RouterToolSelectionEvent()
        
    @step
    async def response(self, ev: RouterResponseEvent) -> StopEvent:
        logger.info(f"[{self.chat_id}]: RouterAgent response")
        # TODO: Exchange for streamed response

        thoughts = self._gather_thoughts()
        condensed = self.condense_module(self.memory)

        output = non_structured_invocation(
            llm=self.llm,
            prompt=RESPONSE_INSTRUCTIONS.format(
                chat_history=str(condensed),
                thoughts=thoughts,
                system=self.system_prompt,
            ),
            inference_kwargs=self._generation_kwargs,
        )
        self.memory.put(
            ChatMessage(
                content=str(output), 
                role=MessageRole.ASSISTANT
            )
        )
        return StopEvent(result=output)
        
    @step
    async def tool_selection(self, ev: RouterToolSelectionEvent) -> ToolCallEvent:
        logger.info(f"[{self.chat_id}]: RouterAgent tool selection")

        condensed = self.condense_module(self.memory)
        thoughts = self._gather_thoughts()

        context = ROUTER_AGENT_PROMPT_TEMPLATE.format(
            chat_history=str(condensed),
            system=self.system_prompt,
            tools=self.skill_map.tool_metadata_str,
            thoughts=thoughts,
            instructions=TOOL_DECISION_INSTRUCTIONS,
        )

        response: ToolCallResponse = structured_invocation(
            llm=self.llm,
            context=context,
            pydantic_object=ToolCallResponse,
            llm_kwargs=self._tool_selection_kwargs,
        )

        logger.info(f"[{self.chat_id}]: RouterAgent tool call: {response.output}")

        self.internal_memory.put(response.as_msg())

        return ToolCallEvent(tool_call=response.output)

    @step
    async def tool_call_handler(self, ev: ToolCallEvent) -> RouterInputEvent:
        logger.info(f"[{self.chat_id}]: RouterAgent tool call handler")

        tool_call = ev.tool_call

        function_name = tool_call.tool_name
        arguments = tool_call.tool_kwargs

        logger.info(
            f"[{self.chat_id}]: RouterAgent calling tool {function_name} using arguments {arguments}"
        )

        try:
            function_callable = self.skill_map.get_function_callable_by_name(
                function_name
            )

            if inspect.iscoroutinefunction(function_callable):
                function_result = await function_callable(arguments)
            else:
                function_result = function_callable(arguments)

        except KeyError:
            logger.warning(
                f"[{self.chat_id}]: RouterAgent tool {function_name} not found in skill map."
            )
            function_result = "Error: Unknown tool name."

        except Exception as e:
            logger.error(
                f"[{self.chat_id}]: RouterAgent encountered an error calling tool {function_name}: {e}"
            )
            function_result = (
                f"**{function_name} tool call failed.**\n"
                f"Error: {e}"
            )

        message = ChatMessage(
            role=MessageRole.TOOL,
            content=function_result,
            additional_kwargs={"tool_call_id": tool_call.tool_id},
        )

        self.internal_memory.put(message)

        return RouterInputEvent()

    @step
    async def escape_route(self, ev: RouterEscapeEvent) -> RouterResponseEvent:
        logger.info(f"[{self.chat_id}]: RouterAgent escape route: {ev.hint}")
        self.internal_memory.put(
            ChatMessage(
                content=ESCAPE_PROMPT.format(hint=ev.hint),
                role=MessageRole.SYSTEM,
            )
        )
        return RouterResponseEvent()
    
    def _prepare_system_prompt(self, system_prompt: str) -> str:
        if "{date}" in system_prompt:
            system_prompt = system_prompt.format(
                date=datetime.now().strftime("%Y-%m-%d")
            )
        return system_prompt

    def _prepare_internal_memory(self) -> ChatMemoryBuffer:
        internal_memory: ChatMemoryBuffer = self._prepare_chat_memory()
        internal_memory.put_messages(self.memory.get_all())
        return internal_memory

    def _prepare_chat_memory(self, memory: Optional[ChatMemoryBuffer] = None) -> ChatMemoryBuffer:
        return (
            memory or ChatMemoryBuffer(token_limit=DEFAULT_TOKEN_LIMIT).from_defaults(llm=self.llm)
        )
    
    def _gather_thoughts(self) -> str:
        thoughts = self.internal_memory.get_all()
        return "\n".join([str(msg) for msg in thoughts])