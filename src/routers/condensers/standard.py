import logging
from textwrap import dedent
from typing import Any, Optional

from llama_index.core import PromptTemplate
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import ChatMemoryBuffer

from src.invocations import non_structured_invocation
from src.routers.condensers.base import CondenseModuleBase

logger = logging.getLogger(__name__)


USER_INTENT_CONDENSE = PromptTemplate(
    dedent(
        """# SYSTEM:
<system>The user has sent you a message and your task is to re-write the user's message \
in a way that includes relevant information from prior messages that the user is referring to. \
For example, when the user refers to information (such as facts, entities, or prior conversations) in a \
prior message but does not directly state it in their last message (presupposition):
```example
# CHAT HISTORY:
<chat history>User: What can you tell me about the new iPhone?
Assistant: The new iPhone has a better camera and a faster processor.</chat history>

# USER'S LAST MESSAGE:
<user>User: How much does it cost?</user>

# RESPONSE:
<response>User: How much does the new iPhone cost?</response>
```
In this example, the user is referring to the new iPhone in their last message, \
which was mentioned in the prior message. The user's message was re-written to include the relevant information \
from the prior message.

**Your response should be in first-person from the perspective of the user.**
**Your re-write of the user's message must not be embelished**
**If there is no prior message that contains relevant information to the user's current query, \
simply repeat the user's last message word for word.**</system>

# CHAT HISTORY:
<chat history>{chat_history}</chat history>

# USER'S LAST MESSAGE:
<user>{user_last_message}</user>

Re-write the user's message now, you do not need to include any other information or preamble.
"""
    )
)

CHAT_HISTORY_CONDENSE = PromptTemplate(
    dedent(
        """# SYSTEM:\n
<system>You are an agentic ChatBot currently in conversation with a user, however your context window is too small \
to fit the entire chat history into the prompt. Your task is to condense the chat history to short bullet-points \
that capture the most important information from the conversation so far. This will allow you to refer back to the \
conversation without having to scroll through the entire chat history.

**Where the current messages are of high relevance to the user's last message, you should ensure more detail is captured, \
otherwise you should only provide a high-level summary.**

# USER'S LAST MESSAGE FOR CHECKING RELEVANCE AGAINST
<user last message>{user_last_message}</user last message>
</system>

# CURRENT MESSAGES TO BE CONDENSED\n
<current message>{current_message}</current message>

# RESPONSE
"""
    )
)


CONDENSED_TEMPLATE = """# CHAT HISTORY:
**This is a condensed chat history to save space:**
{condensed}
"""


class StandardCondenser(CondenseModuleBase):
    """A standard condenser module that condenses the chat history to save space in the context window of the LLM.
    A condense module is a chat history management module that condenses the chat history \
    to save space in the context window of the LLM. This helps reduce the needle-in-the-haystack \
    problem that causes LLM performance to degrade over time.

    Parameters
    ----------
    _user_intent_kwargs : dict[str, Any], optional
        The inference kwargs for the user intent prompt, by default {"max_tokens": 300}
    _condense_kwargs : dict[str, Any], optional
        The inference kwargs for the condense prompt, by default {"max_tokens": 1000}
    """
    _user_intent_kwargs: dict[str, Any] = {"max_tokens": 300}
    _condense_kwargs: dict[str, Any] = {"max_tokens": 1000}

    def __init__(
        self,
        llm: LLM,
        n_msg_trigger: Optional[int] = 5,
        n_tokens_trigger: Optional[int] = None,
        n_msgs_user_intent: int = 6,
        condense_batch_size: int = 2,
    ):
        """Initializes the StandardCondenser class.

        Parameters
        ----------
        llm : LLM
            The LLM module to be invoked.
        n_msg_trigger : Optional[int], optional
            The number of messages that triggers the module, by default 5
        n_tokens_trigger : Optional[int], optional
            The number of tokens that triggers the module, by default None
        n_msgs_user_intent : int, optional
            The number of prior messages to use when inferring the user's intent, by default 6
        condense_batch_size : int, optional
            The number of messages to be condensed at a time (batch processing), by default 2
        """
        super().__init__(n_msg_trigger, n_tokens_trigger)
        self.llm = llm
        self._n_msgs_user_intent = n_msgs_user_intent
        self._condense_batch_size = condense_batch_size

    def get_user_intent(self, chat_history: ChatMemoryBuffer) -> str:
        """Get the user's intent from the chat history.

        Parameters
        ----------
        chat_history : ChatMemoryBuffer
            The chat history to be condensed.

        Returns
        -------
        str
            The user's intent.
        """
        history = chat_history.get_all().copy()
        user_msg = history.pop()
        last_n_msgs = history[-self._n_msgs_user_intent :]
        user_intent = non_structured_invocation(
            llm=self.llm,
            prompt=USER_INTENT_CONDENSE.format(
                chat_history="\n".join([str(msg) for msg in last_n_msgs]),
                user_last_message=user_msg,
            ),
            inference_kwargs=self._user_intent_kwargs,
        )
        return str(user_intent)

    def _extract_relevant(
        self, messages: list[str], condensed: str, user_intent: str
    ) -> str:
        """Extract relevant information from the chat history that relates to the user's intent.

        Parameters
        ----------
        messages : list[str]
            The messages to be condensed.
        condensed : str
            The condensed chat history so far.
        user_intent : str
            The user's intent.

        Returns
        -------
        str
            The extracted relevant information.
        """
        batch_str = "\n".join([str(msg) for msg in messages])

        prompt = CHAT_HISTORY_CONDENSE.format(
            user_last_message=user_intent,
            current_message=batch_str,
        )

        logger.info(f"_extract_relevant: <prompt>{prompt}</prompt>")

        response = non_structured_invocation(
            llm=self.llm,
            prompt=prompt,
            inference_kwargs=self._condense_kwargs,
        )

        logger.info(f"_extract_relevant: <response>{response}</response>")

        return str(response)

    def condense_chat_history(self, chat_history: ChatMemoryBuffer) -> str:
        """Condense the chat history to save space in the context window of the LLM.

        Parameters
        ----------
        chat_history : ChatMemoryBuffer
            The chat history to be condensed.

        Returns
        -------
        str
            The condensed chat history
        """
        user_intent = self.get_user_intent(chat_history)
        messages = chat_history.get_all()[:-1]
        condensed_list: list[str] = []
        for batch in range(0, len(messages), self._condense_batch_size):
            response = self._extract_relevant(
                [str(m) for m in messages[batch : batch + self._condense_batch_size]],
                "\n".join(condensed_list),
                user_intent,
            )
            if "NO RELEVANT INFORMATION" not in response:
                condensed_list.append(response)

        if not condensed_list:
            return ""

        condensed = "\n".join(condensed_list)

        condensed = CONDENSED_TEMPLATE.format(
            condensed=condensed,
        )
        return condensed
