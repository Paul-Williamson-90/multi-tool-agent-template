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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    # handlers=[
    #     logging.FileHandler("temp.log"),
    # ],
)
logger = logging.getLogger(__name__)


class RouterAgent(Workflow):
    """The RouterAgent class is the core multi-tool agent architecture, using Llama-Index \
    Workflow, which is an event-driven process. The flow of the agent is defined by @step decorated \
    methods which define the agent's behavior at each step of the process and determine which step \
    to take next.

    Parameters
    ----------
    _round : int
        Tracks how many rounds the agent has been through as part of a termination condition.

    Example Usage:
    -------------
    ```python
    def get_agent(chat_id: uuid.UUID, memory: ChatMemoryBuffer) -> RouterAgent:
        llm = get_llm()

        skill_map = SkillMap(skills=[Multiply()])

        context_modules = [DummyContextModule(llm=llm, chat_id=chat_id)]

        agent = RouterAgent(
            chat_id=chat_id,
            llm=llm,
            skill_map=skill_map,
            context_modules=context_modules,
            chat_history=memory,
        )
        return agent


    def invoke(agent: RouterAgent, user_input: str) -> StreamingAgentChatResponse:
        async def run(input: str) -> StreamingAgentChatResponse:
            response = await agent.run(input=input)
            return response

        return asyncio.run(run(input=user_input))

    if __name__=="__main__":
        chat_id = ...
        memory = ...
        user_input = ...
        agent = get_agent(chat_id, memory)
        response = invoke(agent, user_input)
        for words in response.chat_stream:
            if words.delta:
                print(words.delta, end="", flush=True)
    ```
    """

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
        """Initialize the RouterAgent with the necessary components.

        Parameters
        ----------
        llm : LLM
            The LLM instance to be used for generating responses.
        skill_map : SkillMap
            The SkillMap instance to be used for managing tool calls.
        condense_module : Optional[CondenseModuleBase], optional
            The module used for condensing chat history for better context window utilisation, by default None
        context_modules : list[ContextModuleBase], optional
            A list of context modules that can be combined with retrieval processes for better context management \
            in the context window, by default []
        chat_history : Optional[ChatMemoryBuffer], optional
            The chat history so far with the user, by default None
        system_prompt : str, optional
            A system message that is used in the prompts throughout the Agent's flow (must contain {date} in a formatted string), \
            by default SYSTEM_PROMPT
        chat_id : Optional[uuid.UUID], optional
            The UUID of the chat, for use with deployments that utilise a database for storing chats, by default None
        generation_kwargs : _type_, optional
            The llm kwargs to pass for generation steps, by default {"max_tokens": 8000}
        tool_selection_kwargs : _type_, optional
            The llm kwargs to pass for tool selection steps, by default {"max_tokens": 500}
        rounds_limit : int, optional
            The maximum number of reasoning rounds the Agent can perform before its forced to give up, by default 8

        Attributes
        ----------
        internal_memory : ChatMemoryBuffer
            The internal memory of the Agent which stores all the messages created by the Agent in its reasoning process.
        """
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
        """Prepares the agent for the conversation.
        - Resets the round counter.
        - Resets the condense module.
        - Resets the internal memory.
        - Adds the user input to the memory.

        Parameters
        ----------
        ev : StartEvent
            An event that triggers the start of the agent process.

        Returns
        -------
        RouterInputEvent
            An event that triggers the Agent's reasoning process.
        """
        logger.info(f"[{self.chat_id}]: Preparing RouterAgent")

        self._round = 0
        self.condense_module.reset()
        self.internal_memory.reset()

        user_input = ev.input
        user_msg = ChatMessage(role=MessageRole.USER, content=user_input)
        self.memory.put(user_msg)
        self.condense_module(self.memory)
        return RouterInputEvent()

    @step
    async def router(
        self, ev: RouterInputEvent
    ) -> Union[
        RouterContextSelectionEvent, RouterEscapeEvent, RouterToolSelectionEvent
    ]:
        """The Agent's reasoning step where it reasons with the context thus far and makes decision \
        on what to do next (respond to user or call a tool).

        Parameters
        ----------
        ev : RouterInputEvent
            An event that triggers the Agent's reasoning process

        Returns
        -------
        Union[ RouterContextSelectionEvent, RouterEscapeEvent, RouterToolSelectionEvent ]
            An event that triggers the next step in the Agent's reasoning process
                - RouterContextSelectionEvent: If the Agent decides to respond to the user
                - RouterEscapeEvent: If the Agent is forced to give up due to an error or too many rounds
                - RouterToolSelectionEvent: If the Agent decides to call a tool
        """
        logger.info(f"[{self.chat_id}]: RouterAgent router")
        self._round += 1

        if self._round > self._rounds_limit:
            return RouterEscapeEvent(hint=ROUNDS_EXCEEDED_HINT)

        try:
            context = self._structured_response_template(
                instructions=ACTION_DECISION_INSTRUCTIONS
            )
            logger.info(
                f"[{self.chat_id}]: RouterAgent planning step\n<prompt>{context}</prompt>"
            )
            response: PlanningStep = structured_invocation(  # type: ignore
                llm=self.llm,
                context=context,
                pydantic_object=PlanningStep,
                llm_kwargs=self._generation_kwargs,
            )
            logger.info(
                f"[{self.chat_id}]: RouterAgent planning step response: {response}"
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
    ) -> Union[RouterResponseEvent, RouterEscapeEvent]:
        """The Agent's context selection step where it decides whether to inject context from a context module \
        into its response prompt.

        Parameters
        ----------
        ev : RouterContextSelectionEvent
            An event that triggers the Agent's context selection process

        Returns
        -------
        Union[RouterResponseEvent, RouterEscapeEvent]
            An event that triggers the next step in the Agent's reasoning process
                - RouterResponseEvent: An event where the Agent responds to the user
                - RouterEscapeEvent: If the Agent is forced to give up due to an error or too many rounds
        """
        logger.info(f"[{self.chat_id}]: RouterAgent context selection")

        if len(self.context_modules) == 0:
            return RouterResponseEvent()
        if sum([len(module) for module in self.context_modules.values()]) == 0:
            return RouterResponseEvent()

        try:
            return self._context_selection()

        except Exception as e:
            logger.error(f"[{self.chat_id}]: RouterAgent context selection error: {e}")
            return RouterEscapeEvent(hint=ERROR_HINT)

    @step
    async def response(self, ev: RouterResponseEvent) -> StopEvent:
        """The Agent's response step where it generates a response to the user.

        Parameters
        ----------
        ev : RouterResponseEvent
            An event that triggers the Agent's response generation

        Returns
        -------
        StopEvent
            An event that signals the end of the Agent's flow
        """
        logger.info(f"[{self.chat_id}]: RouterAgent response")

        thoughts = self._gather_thoughts()
        condensed = self.condense_module(self.memory)
        prompt = RESPONSE_INSTRUCTIONS.format(
            chat_history=str(condensed),
            thoughts=thoughts,
            system=self.system_prompt,
        )

        logger.info(
            f"[{self.chat_id}]: RouterAgent response step\n<prompt>{prompt}</prompt>"
        )

        output = non_structured_streamed_invocation(
            llm=self.llm,
            prompt=prompt,
            inference_kwargs=self._generation_kwargs,
            memory=self.memory,
        )
        return StopEvent(result=output)

    @step
    async def tool_selection(
        self, ev: RouterToolSelectionEvent
    ) -> Union[ToolCallEvent, RouterEscapeEvent]:
        """The Agent's tool selection step where it decides which tool to call.

        Parameters
        ----------
        ev : RouterToolSelectionEvent
            An event that triggers the Agent's tool selection process

        Returns
        -------
        Union[ToolCallEvent, RouterEscapeEvent]
            An event that triggers the next step in the Agent's reasoning process
                - ToolCallEvent: An event where the chosen tool is called
                - RouterEscapeEvent: If the Agent is forced to give up due to an error or too many rounds
        """
        logger.info(f"[{self.chat_id}]: RouterAgent tool selection")

        context = self._structured_response_template(
            instructions=TOOL_DECISION_INSTRUCTIONS
        )

        logger.info(
            f"[{self.chat_id}]: RouterAgent tool selection step\n<prompt>{context}</prompt>"
        )

        try:
            response: ToolCallResponse = structured_invocation(  # type: ignore
                llm=self.llm,
                context=context,
                pydantic_object=ToolCallResponse,
                llm_kwargs=self._tool_selection_kwargs,
            )

            logger.info(f"[{self.chat_id}]: RouterAgent tool call: {response.output}")

            self.internal_memory.put(response.as_msg())

            return ToolCallEvent(tool_call=response.output)

        except Exception as e:
            logger.error(f"[{self.chat_id}]: RouterAgent tool selection error: {e}")
            return RouterEscapeEvent(tool_call=ERROR_HINT)

    @step
    async def tool_call_handler(self, ev: ToolCallEvent) -> RouterInputEvent:
        """Handles the tool calling process by calling the tool and storing the result in the internal memory.

        Parameters
        ----------
        ev : ToolCallEvent
            An event that triggers the Agent's tool calling

        Returns
        -------
        RouterInputEvent
            An event that triggers the Agent's reasoning process
        """
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
        """Handles the escape route of the Agent, where it is forced to give up due to an error or too many rounds.

        Parameters
        ----------
        ev : RouterEscapeEvent
            An event that triggers the Agent's escape route

        Returns
        -------
        RouterResponseEvent
            An event that triggers the Agent to respond to the user with an escape hint
        """
        logger.info(f"[{self.chat_id}]: RouterAgent escape route: {ev.hint}")
        self.internal_memory.put(
            ChatMessage(
                content=ESCAPE_PROMPT.format(hint=ev.hint),
                role=MessageRole.SYSTEM,
            )
        )
        return RouterResponseEvent()

    def _prepare_system_prompt(self, system_prompt: str) -> str:
        """Prepares the system prompt by formatting the date into the prompt.

        Parameters
        ----------
        system_prompt : str
            The system prompt to be prepared

        Returns
        -------
        str
            The prepared system prompt
        """
        if "{date}" in system_prompt:
            system_prompt = system_prompt.format(
                date=datetime.now().strftime("%Y-%m-%d")
            )
        return system_prompt

    def _prepare_internal_memory(self) -> ChatMemoryBuffer:
        """Prepares the internal memory of the Agent.

        Returns
        -------
        ChatMemoryBuffer
            The internal memory of the Agent
        """
        internal_memory: ChatMemoryBuffer = self._prepare_chat_memory()
        internal_memory.put_messages(self.memory.get_all())
        return internal_memory

    def _prepare_chat_memory(
        self, memory: Optional[ChatMemoryBuffer] = None
    ) -> ChatMemoryBuffer:
        """Prepares the chat memory of the Agent. One is created if not provided.

        Parameters
        ----------
        memory : Optional[ChatMemoryBuffer], optional
            The chat history with the user, by default None

        Returns
        -------
        ChatMemoryBuffer
            The chat memory
        """
        return memory or ChatMemoryBuffer(
            token_limit=DEFAULT_TOKEN_LIMIT
        ).from_defaults(llm=self.llm)

    def _gather_thoughts(self) -> str:
        """Gathers the Agent's internal memory (thoughts) and tool call outputs.

        Returns
        -------
        str
            The thoughts of the Agent
        """
        thoughts = self.internal_memory.get_all()
        return "\n".join([str(msg) for msg in thoughts])

    def _prepare_context_modules(
        self, context_modules: list[ContextModuleBase]
    ) -> dict[str, ContextModuleBase]:
        """Prepares the context modules of the Agent.

        Parameters
        ----------
        context_modules : list[ContextModuleBase]
            A list of context modules to be used by the Agent

        Returns
        -------
        dict[str, ContextModuleBase]
            A dictionary of context modules with their names as keys

        Raises
        ------
        ValueError
            If a duplicate context module name
        """
        module_dict: dict[str, ContextModuleBase] = {}
        for module in context_modules:
            if module.get_name() in module_dict:
                raise ValueError(
                    f"Duplicate context module name found: {module.get_name()}"
                )
            module_dict[module.get_name()] = module
            self.skill_map.add_skill(module.verify_tool)
        return module_dict

    def _structured_response_template(
        self, instructions: str, tools_available: bool = True
    ) -> str:
        """Creates a structured response template for the Agent's reasoning steps.

        Parameters
        ----------
        instructions : str
            The instructions for the Agent to follow
        tools_available : bool, optional
            Whether to include tool information in the prompt, by default True

        Returns
        -------
        str
            The structured response template
        """
        condensed = self.condense_module(self.memory)
        thoughts = self._gather_thoughts()

        context = ROUTER_AGENT_PROMPT_TEMPLATE.format(
            chat_history=str(condensed),
            system=self.system_prompt,
            tools=self.skill_map.info
            if tools_available
            else "No tools available at this time.",
            context_modules="\n".join(
                [module.info for module in self.context_modules.values()]
            ),
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
        """The Agent's context selection process where it selects context from a context module \
        and stores the extracted facts in the internal memory.

        Returns
        -------
        RouterResponseEvent
            An event that triggers the Agent's response to the user

        Raises
        ------
        Exception
            If an error occurs during the context selection process
        """
        context = self._structured_response_template(
            instructions=CONTEXT_SELECTION_INSTRUCTIONS, tools_available=False
        )

        logger.info(
            f"[{self.chat_id}]: RouterAgent context selection step\n<prompt>{context}</prompt>"
        )

        response: ContextSelection = structured_invocation(  # type: ignore
            llm=self.llm,
            context=context,
            pydantic_object=ContextSelection,
            llm_kwargs=self._tool_selection_kwargs,
        )

        logger.info(
            f"[{self.chat_id}]: RouterAgent context selection response: {response}"
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
