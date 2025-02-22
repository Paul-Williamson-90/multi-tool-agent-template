from typing import Union, Any, Optional, Literal
import inspect
import logging
import json
from json import JSONDecodeError
from datetime import datetime

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import ToolSelection
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from llama_index.core.llms.llm import LLM
from llama_index.core.base.llms.types import MessageRole, CompletionResponse
from llama_index.core import PromptTemplate

from src.prompt_templates.router_template import (
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


class Step(BaseModel):
    """
    Use this schema to think through the problem step-by-step before generating your response/action.

    Attributes:
        - thought: str - Your thoughts on what you need to do / any considerations you need to make.
        - conclusion: str - The conclusion you've come to after thinking through your thoughts.
    """


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


class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]


class RouterInputEvent(Event):
    input: list[ChatMessage]


class RouterAgent(Workflow):
    _generation_kwargs: dict[str, Any] = {"max_tokens": 4000}
    _n_msgs_condense_trigger: int = 4
    _condence_batch_size: int = 2
    _condense_kwargs: dict[str, Any] = {"max_tokens": 1000}
    _rounds_limit: int = 5
    _n_msgs_user_intent: int = 4

    def __init__(
        self,
        llm: LLM,
        skill_map: SkillMap,
        memory: Optional[ChatMemoryBuffer] = None,
        timeout: int = 300,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        logger.info("Initializing RouterAgent")
        super().__init__(timeout=timeout)
        self.llm: LLM = llm
        self.skill_map: SkillMap = skill_map
        self.system_prompt: str = system_prompt.format(
            date=datetime.now().strftime("%Y-%m-%d")
        )
        self.memory: ChatMemoryBuffer = memory or ChatMemoryBuffer(
            token_limit=40000
        ).from_defaults(llm=llm)
        self._internal_memory: ChatMemoryBuffer = ChatMemoryBuffer(
            token_limit=40000
        ).from_defaults(llm=llm)
        self._internal_memory.put_messages(self.memory.get_all())

        self.tools: list[str] = []
        for func in self.skill_map.get_function_list():
            self.tools.append(self.skill_map.get_function_dict_by_name(func))
        self._condensed: str | None = None
        self._user_intent: str | None = None
        self._round: int = 1

    @step
    async def prepare_agent(self, ev: StartEvent) -> RouterInputEvent:
        logger.info("Preparing RouterAgent")
        user_input = ev.input
        user_msg = ChatMessage(role=MessageRole.USER, content=user_input)
        self.memory.put(user_msg)
        self._internal_memory.put(user_msg)

        input = self.memory.get_all()
        return RouterInputEvent(input=input)

    @step
    async def router(self, ev: RouterInputEvent) -> Union[ToolCallEvent, StopEvent]:
        logger.info("RouterAgent router")
        messages: list[ChatMessage] = ev.input

        if self._round > self._rounds_limit:
            return self._escape_route(ROUNDS_EXCEEDED_HINT)

        try:
            response: ResponseType = self._structured_invocation(
                messages,
                ACTION_DECISION_INSTRUCTIONS,
                ResponseType,
                self._generation_kwargs,
            )

        except Exception as e:
            logger.error("RouterAgent encountered an error: %s", e)
            return self._escape_route(ERROR_HINT)

        next_action = response.next_action
        if next_action == "response":
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

        else:
            tool_response: ToolCallResponse = self._structured_invocation(
                messages,
                TOOL_DECISION_INSTRUCTIONS,
                ToolCallResponse,
                self._condense_kwargs,
                thoughts=str(response),
            )
            output_tool = tool_response.output
            logger.info("RouterAgent tool calls: %s", output_tool)

            self._internal_memory.put(
                ChatMessage(content=str(response), role=MessageRole.ASSISTANT)
            )

            self._internal_memory.put(
                ChatMessage(content=str(tool_response), role=MessageRole.ASSISTANT)
            )

            self._condensed = None
            return ToolCallEvent(tool_calls=output_tool)

    @step
    async def tool_call_handler(self, ev: ToolCallEvent) -> RouterInputEvent:
        logger.info("RouterAgent tool call handler")
        self._round += 1

        tool_calls: list[ToolSelection] = ev.tool_calls

        for tool_call in tool_calls:
            function_name = tool_call.tool_name
            arguments = tool_call.tool_kwargs

            logger.info(
                f"RouterAgent calling tool {function_name} using arguments {arguments}"
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
                    f"RouterAgent tool {function_name} not found in skill map."
                )
                function_result = "Error: Unknown tool name."

            message = ChatMessage(
                role=MessageRole.TOOL,
                content=function_result,
                additional_kwargs={"tool_call_id": tool_call.tool_id},
            )

            self._internal_memory.put(message)

        return RouterInputEvent(input=self._internal_memory.get_all())

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
        internal_msgs = self._internal_memory.get_all()[len(msgs_to_condense) :]
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

    def _tool_metadata_str(self) -> str:
        return "\n\n".join(str(tool) for tool in self.tools)

    def _escape_route(self, hint: str) -> StopEvent:
        logger.info("RouterAgent has reached the escape route.")
        internal_msgs = self._internal_memory.get_all()
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

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
    def _structured_invocation(
        self,
        messages: list[ChatMessage],
        instruction: str,
        pydantic_object: type[BaseModel],
        llm_kwargs: dict[str, Any],
        thoughts: str = "",
    ) -> BaseModel:
        chat_history = self._chat_history_from_messages(messages)
        tool_metadata = self._tool_metadata_str()

        response = self.llm.complete(
            prompt=ROUTER_AGENT_PROMPT_TEMPLATE.format(
                chat_history=chat_history,
                system=self.system_prompt,
                tools=tool_metadata,
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

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
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
