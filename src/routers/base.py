import inspect
import logging
import uuid
from datetime import datetime
from typing import Any, Optional, Union

from llama_index.core.base.llms.types import MessageRole
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step
from tenacity import before_log, retry, stop_after_attempt, wait_fixed

from src.invocations import non_structured_streamed_invocation, structured_invocation
from src.routers.condensers import CondenseModuleBase, StandardCondenser
from src.routers.constants import DEFAULT_TOKEN_LIMIT
from src.routers.context_modules import ContextModuleBase
from src.routers.events import (
    RouterContextSelectionEvent,
    RouterEscapeEvent,
    RouterInputEvent,
    RouterResponseEvent,
    RouterToolSelectionEvent,
    ToolCallEvent,
)
from src.routers.prompts import (
    ACTION_DECISION_INSTRUCTIONS,
    CONTEXT_SELECTION_INSTRUCTIONS,
    ERROR_HINT,
    ESCAPE_PROMPT,
    RESPONSE_INSTRUCTIONS,
    ROUNDS_EXCEEDED_HINT,
    ROUTER_AGENT_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    TOOL_DECISION_INSTRUCTIONS,
)
from src.routers.pydantics import (
    ContextSelection,
    NextAction,
    PlanningStep,
    SelectedContext,
    ToolCallResponse,
)
from src.routers.skills import SkillMap, SkillOutput

logger = logging.getLogger(__name__)


class RouterAgent(Workflow):
    _round: int = 0

    def __init__(
        self,
        llm: LLM,
        skill_map: SkillMap,
        condense_module: Optional[CondenseModuleBase] = None,
        context_modules: list[ContextModuleBase] = [],
        chat_history: Optional[ChatMemoryBuffer] = None,
        system_prompt: str = SYSTEM_PROMPT,
        chat_id: Optional[uuid.UUID] = None,
        generation_kwargs: dict[str, Any] = {"max_tokens": 8000},
        tool_selection_kwargs: dict[str, Any] = {"max_tokens": 500},
        rounds_limit: int = 8,
    ):
        self.chat_id = chat_id or uuid.uuid4()
        logger.info(f"[{self.chat_id}]: Initializing RouterAgent")

        super().__init__(timeout=None)

        self.llm: LLM = llm
        self.skill_map: SkillMap = skill_map
        self.condense_module: CondenseModuleBase = condense_module or StandardCondenser(
            llm=self.llm
        )
        self.system_prompt: str = self._prepare_system_prompt(system_prompt)
        self.memory: ChatMemoryBuffer = self._prepare_chat_memory(chat_history)
        self.internal_memory: ChatMemoryBuffer = self._prepare_internal_memory()
        self.context_modules: dict[
            str, ContextModuleBase
        ] = self._prepare_context_modules(context_modules)
        self._generation_kwargs: dict[str, Any] = generation_kwargs
        self._tool_selection_kwargs: dict[str, Any] = tool_selection_kwargs
        self._rounds_limit: int = rounds_limit

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
    async def router(
        self, ev: RouterInputEvent
    ) -> Union[
        RouterContextSelectionEvent, RouterEscapeEvent, RouterToolSelectionEvent
    ]:
        logger.info(f"[{self.chat_id}]: RouterAgent router")
        self._round += 1

        if self._round > self._rounds_limit:
            return RouterEscapeEvent(hint=ROUNDS_EXCEEDED_HINT)

        try:
            context = self._structured_response_template(
                instructions=ACTION_DECISION_INSTRUCTIONS
            )
            response: PlanningStep = structured_invocation(  # type: ignore
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
            return RouterContextSelectionEvent()
        else:
            return RouterToolSelectionEvent()

    @step
    async def context_selection(
        self, ev: RouterContextSelectionEvent
    ) -> RouterResponseEvent:
        logger.info(f"[{self.chat_id}]: RouterAgent context selection")

        if len(self.context_modules) == 0:
            return RouterResponseEvent()
        if sum([len(module) for module in self.context_modules.values()]) == 0:
            return RouterResponseEvent()

        return self._context_selection()

    @step
    async def response(self, ev: RouterResponseEvent) -> StopEvent:
        logger.info(f"[{self.chat_id}]: RouterAgent response")

        thoughts = self._gather_thoughts()
        condensed = self.condense_module(self.memory)

        output = non_structured_streamed_invocation(
            llm=self.llm,
            prompt=RESPONSE_INSTRUCTIONS.format(
                chat_history=str(condensed),
                thoughts=thoughts,
                system=self.system_prompt,
            ),
            inference_kwargs=self._generation_kwargs,
            memory=self.memory,
        )
        return StopEvent(result=output)

    @step
    async def tool_selection(self, ev: RouterToolSelectionEvent) -> ToolCallEvent:
        logger.info(f"[{self.chat_id}]: RouterAgent tool selection")

        context = self._structured_response_template(
            instructions=TOOL_DECISION_INSTRUCTIONS
        )

        response: ToolCallResponse = structured_invocation(  # type: ignore
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
                function_result: SkillOutput = await function_callable(arguments)
            else:
                function_result = function_callable(arguments)

        except KeyError:
            logger.warning(
                f"[{self.chat_id}]: RouterAgent tool {function_name} not found in skill map."
            )
            function_result = SkillOutput(response_to_llm="Error: Unknown tool name.")

        except Exception as e:
            logger.error(
                f"[{self.chat_id}]: RouterAgent encountered an error calling tool {function_name}: {e}"
            )
            function_result = SkillOutput(
                response_to_llm=f"**{function_name} tool call failed.**\nError: {e}"
            )

        message = ChatMessage(
            role=MessageRole.TOOL,
            content=str(function_result),
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

    def _prepare_chat_memory(
        self, memory: Optional[ChatMemoryBuffer] = None
    ) -> ChatMemoryBuffer:
        return memory or ChatMemoryBuffer(
            token_limit=DEFAULT_TOKEN_LIMIT
        ).from_defaults(llm=self.llm)

    def _gather_thoughts(self) -> str:
        thoughts = self.internal_memory.get_all()
        return "\n".join([str(msg) for msg in thoughts])

    def _prepare_context_modules(
        self, context_modules: list[ContextModuleBase]
    ) -> dict[str, ContextModuleBase]:
        module_dict: dict[str, ContextModuleBase] = {}
        for module in context_modules:
            if module.get_name() in module_dict:
                raise ValueError(
                    f"Duplicate context module name found: {module.get_name()}"
                )
            module_dict[module.get_name()] = module
            self.skill_map.add_skill(module.verify_tool)
        return module_dict

    def _structured_response_template(self, instructions: str) -> str:
        condensed = self.condense_module(self.memory)
        thoughts = self._gather_thoughts()

        context = ROUTER_AGENT_PROMPT_TEMPLATE.format(
            chat_history=str(condensed),
            system=self.system_prompt,
            tools=self.skill_map.info,
            thoughts=thoughts,
            instructions=instructions,
        )
        return context

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(1),
        before=before_log(logger, logging.INFO),
    )
    def _context_selection(self) -> RouterResponseEvent:
        context = self._structured_response_template(
            instructions=CONTEXT_SELECTION_INSTRUCTIONS
        )
        response: ContextSelection = structured_invocation(  # type: ignore
            llm=self.llm,
            context=context,
            pydantic_object=ContextSelection,
            llm_kwargs=self._tool_selection_kwargs,
        )

        if len(response.contexts) == 0:
            return RouterResponseEvent()

        try:
            self.internal_memory.put(response.as_msg())

            extraction_requests = response.contexts

            extracted_facts: list[SelectedContext] = []
            for request in extraction_requests:
                question = request.instructions
                facts = self.context_modules[
                    request.memory_object
                ].extract_from_context(request.context_id, question)
                extracted_facts.append(
                    SelectedContext(
                        question=question,
                        facts=str(facts),
                        memory_object=request.memory_object,
                        context_id=request.context_id,
                    )
                )

            self.internal_memory.put(
                ChatMessage(
                    content="\n\n".join([str(fact) for fact in extracted_facts]),
                    role=MessageRole.TOOL,
                    additional_kwargs={"tool_call_id": "context_retrieval"},
                )
            )

            return RouterResponseEvent()

        except Exception as e:
            self.internal_memory.put(
                ChatMessage(
                    content=f"Error in context_retrieval process: {e}",
                    role=MessageRole.TOOL,
                    additional_kwargs={"tool_call_id": "context_retrieval"},
                )
            )
            raise e
