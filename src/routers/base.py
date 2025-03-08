import uuid
from typing import Union, Any, Optional
import inspect
import logging
import json
from json import JSONDecodeError
from datetime import datetime

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed, before_log
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import ToolSelection
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step
from llama_index.core.llms.llm import LLM
from llama_index.core.base.llms.types import MessageRole, CompletionResponse
from llama_index.core import PromptTemplate

from src.routers.events import (
    RouterInputEvent, 
    ToolCallEvent, 
    RouterResponseEvent, 
    RouterToolSelectionEvent,
    RouterEscapeEvent
)
from src.routers.pydantics import ResponseType, ToolCallResponse
from src.routers.prompts import (
    SYSTEM_PROMPT,
    USER_INTENT_CONDENSE,
    CHAT_HISTORY_CONDENSE,
    CONDENSED_TEMPLATE,
    FIX_JSON_PROMPT,
    ROUTER_AGENT_PROMPT_TEMPLATE,
    ESCAPE_PROMPT,
    ROUNDS_EXCEEDED_HINT,
    ACTION_DECISION_INSTRUCTIONS,
    ERROR_HINT,
    RESPONSE_INSTRUCTIONS,
    TOOL_DECISION_INSTRUCTIONS,
)
from src.skills.base import SkillMap


logger = logging.getLogger(__name__)


class RouterAgent(Workflow):
    _generation_kwargs: dict[str, Any] = {"max_tokens": 4000}
    _n_msgs_condense_trigger: int = 4
    _condence_batch_size: int = 2
    _condense_kwargs: dict[str, Any] = {"max_tokens": 1000}
    _rounds_limit: int = 5
    _n_msgs_user_intent: int = 4
    _condensed: str | None = None
    _user_intent: str | None = None
    _round: int = 1

    def __init__(
        self,
        llm: LLM,
        skill_map: SkillMap,
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
        self.system_prompt: str = self._prepare_system_prompt(system_prompt)
        self.memory: ChatMemoryBuffer = self._prepare_chat_memory(chat_history)
        self.internal_memory: ChatMemoryBuffer = self._prepare_internal_memory()

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
            memory or ChatMemoryBuffer(token_limit=40000).from_defaults(llm=self.llm)
        )

    @step
    async def prepare_agent(self, ev: StartEvent) -> RouterInputEvent:
        logger.info(f"[{self.chat_id}]: Preparing RouterAgent")
        user_input = ev.input
        user_msg = ChatMessage(role=MessageRole.USER, content=user_input)
        self.memory.put(user_msg)
        self.internal_memory.put(user_msg)

        input = self.memory.get_all()
        return RouterInputEvent(input=input)

    @step
    async def router(self, ev: RouterInputEvent) -> Union[ToolCallEvent, StopEvent]:
        logger.info(f"[{self.chat_id}]: RouterAgent router")
        messages: list[ChatMessage] = ev.input

        if self._round > self._rounds_limit:
            return RouterEscapeEvent(hint=ROUNDS_EXCEEDED_HINT)

        try:
            response: ResponseType = self._structured_invocation(
                messages,
                ACTION_DECISION_INSTRUCTIONS,
                ResponseType,
                self._generation_kwargs,
            )

        except Exception as e:
            logger.error(f"[{self.chat_id}]: RouterAgent encountered an error: %s", e)
            return RouterEscapeEvent(hint=ERROR_HINT)

        next_action = response.next_action
        if next_action == "response":
            return RouterResponseEvent(messages=messages, response=response)

        else:
            return RouterToolSelectionEvent(messages=messages, response=response)
        
    @step
    async def response(self, ev: RouterResponseEvent) -> StopEvent:
        logger.info(f"[{self.chat_id}]: RouterAgent response")
        messages = ev.messages
        response = ev.response

        chat_history = self._chat_history_from_messages(messages)
        output = self._non_structured_invocation(
            RESPONSE_INSTRUCTIONS,
            {
                "chat_history": chat_history,
                "thoughts": str(response),
                "system": self.system_prompt,
            },
            self._generation_kwargs,
        )
        self.memory.put(
            ChatMessage(content=str(output), role=MessageRole.ASSISTANT)
        )
        return StopEvent(result=output)
        
    @step
    async def tool_selection(self, ev: RouterToolSelectionEvent) -> ToolCallEvent:
        logger.info(f"[{self.chat_id}]: RouterAgent tool selection")
        messages = ev.messages
        response = ev.response

        tool_response: ToolCallResponse = self._structured_invocation(
            messages,
            TOOL_DECISION_INSTRUCTIONS,
            ToolCallResponse,
            self._condense_kwargs,
            thoughts=str(response),
        )
        output_tool = tool_response.output
        logger.info(f"[{self.chat_id}]: RouterAgent tool calls: %s", output_tool)

        self.internal_memory.put(
            ChatMessage(content=str(response), role=MessageRole.ASSISTANT)
        )

        self.internal_memory.put(
            ChatMessage(content=str(tool_response), role=MessageRole.ASSISTANT)
        )

        self._condensed = None
        return ToolCallEvent(tool_calls=output_tool)

    @step
    async def tool_call_handler(self, ev: ToolCallEvent) -> RouterInputEvent:
        logger.info(f"[{self.chat_id}]: RouterAgent tool call handler")
        self._round += 1

        tool_calls: list[ToolSelection] = ev.tool_calls

        for tool_call in tool_calls:
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

            message = ChatMessage(
                role=MessageRole.TOOL,
                content=function_result,
                additional_kwargs={"tool_call_id": tool_call.tool_id},
            )

            self.internal_memory.put(message)

        return RouterInputEvent(input=self.internal_memory.get_all())

    @step
    async def escape_route(self, ev: RouterEscapeEvent) -> StopEvent:
        logger.info(f"[{self.chat_id}]: RouterAgent has reached the escape route.")

        hint = ev.hint

        internal_msgs = self.internal_memory.get_all()
        user_last_msg = self._get_users_last_message()
        output = self._non_structured_invocation(
            ESCAPE_PROMPT,
            {
                "system": self.system_prompt,
                "user_last_msg": user_last_msg,
                "hint": hint,
                "internal_messages": "\n".join([str(msg) for msg in internal_msgs]),
            },
            self._condense_kwargs,
        )
        self.memory.put(ChatMessage(content=str(output), role=MessageRole.ASSISTANT))
        return StopEvent(result=output)

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), before=before_log(logger, logging.INFO))
    def _structured_invocation(
        self,
        messages: list[ChatMessage],
        instruction: str,
        pydantic_object: type[BaseModel],
        llm_kwargs: dict[str, Any],
        thoughts: str = "",
    ) -> BaseModel:
        chat_history = self._chat_history_from_messages(messages)

        response = self.llm.complete(
            prompt=ROUTER_AGENT_PROMPT_TEMPLATE.format(
                chat_history=chat_history,
                system=self.system_prompt,
                tools=self.skill_map.tool_metadata_str,
                thoughts=thoughts,
                instructions=instruction,
                schema=json.dumps(pydantic_object.model_json_schema()),
            ),
            **llm_kwargs,
        )
        response_str = response.text

        logger.info("RESPONSE:")
        logger.info(response_str)

        try:
            response_json = json.loads(str(response_str).strip())
        except JSONDecodeError:
            logger.warning(
                "JSONDecodeError: Response from LLM is not a valid JSON: %s",
                response_str,
            )
            response_str = self._non_structured_invocation(
                FIX_JSON_PROMPT, {"response": response_str}, self._generation_kwargs
            )
            response_json = json.loads(str(response_str).strip())

        response_object = pydantic_object(**response_json)
        return response_object

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), before=before_log(logger, logging.INFO))
    def _non_structured_invocation(
        self,
        template: PromptTemplate,
        template_kwargs: dict[str, Any],
        inference_kwargs: dict[str, Any],
    ) -> str:
        response: CompletionResponse = self.llm.complete(
            prompt=template.format(**template_kwargs), **inference_kwargs
        )
        response_str = str(response)

        logger.info("RESPONSE:")
        logger.info(response_str)

        return response_str

    def _get_user_intent(self) -> str:
        logger.info("Getting user intent")
        last_n_msgs = self.memory.get_all()[-self._n_msgs_user_intent :]
        last_n_msgs_str = "\n".join([str(msg) for msg in last_n_msgs])
        user_msg = self._get_users_last_message()
        if str(user_msg) not in last_n_msgs_str:
            last_n_msgs_str = f"{user_msg}\n{last_n_msgs_str}"
        user_last_message = self._non_structured_invocation(
            USER_INTENT_CONDENSE,
            {"system": self.system_prompt, "chat_history": last_n_msgs_str},
            self._condense_kwargs,
        )
        return user_last_message

    def _condense_chat_history(self) -> str:
        logger.info("Condensing chat history")
        user_last_message = self._user_intent or self._get_user_intent()
        msgs_to_condense = self.memory.get_all()
        internal_msgs = self.internal_memory.get_all()[len(msgs_to_condense) :]
        condensed = ""
        for batch in range(0, len(msgs_to_condense), self._condence_batch_size):
            batch_str = "\n".join(
                [
                    str(msg)
                    for msg in msgs_to_condense[
                        batch : batch + self._condence_batch_size
                    ]
                ]
            )
            response = self._non_structured_invocation(
                CHAT_HISTORY_CONDENSE,
                {
                    "user_last_message": user_last_message,
                    "condensed": condensed
                    if condensed != ""
                    else "No messages have been condensed yet.",
                    "current_message": batch_str,
                },
                self._condense_kwargs,
            )
            condensed += "\n" + response

        condensed = CONDENSED_TEMPLATE.format(
            condensed=condensed,
            user_last_message=user_last_message,
            thoughts="\n".join([str(msg) for msg in internal_msgs]),
        )
        return condensed

    def _chat_history_from_messages(self, messages: list[ChatMessage]) -> str:
        logger.info(f"RouterAgent has {len(messages)} in the chat history so far.")
        if len(messages) <= self._n_msgs_condense_trigger:
            return "\n".join([str(msg) for msg in messages])
        return self._condensed or self._condense_chat_history()

    def _get_users_last_message(self) -> str:
        return str(self.memory.get_all()[-1])