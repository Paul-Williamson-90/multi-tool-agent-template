from textwrap import dedent
from typing import Any, Optional

from llama_index.core import PromptTemplate
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import ChatMemoryBuffer

from src.invocations import non_structured_invocation
from src.routers.condensers.base import CondenseModuleBase

USER_INTENT_CONDENSE = PromptTemplate(
    dedent(
        """# SYSTEM:\n
        <system>The user has sent you a message and your task is to re-write the user's message \
        in a way that includes relevant information from prior messages that the user is referring to. \
        For example, when the user refers to information (such as facts, entities, or prior conversations) in a \
        prior message but does not directly state it in their last message (presupposition).\n\n

        **Your response should be in first-person from the perspective of the user.**\n
        **Your re-write of the user's message must not be embelished**\n
        **If there is no prior message that contains relevant information to the user's current query, \
        simply repeat the user's last message word for word.**</system>\n\n
        
        # CHAT HISTORY:\n
        <chat history>{chat_history}</chat history>\n\n

        # USER'S LAST MESSAGE:\n
        <user>{user_last_message}</user>

        Re-write the user's message now, you do not need to include any other information or preamble.
        """
    )
)

CHAT_HISTORY_CONDENSE = PromptTemplate(
    dedent(
        """# SYSTEM:\n
        <system>You are an agentic ChatBot currently in conversation with a user, however your context window is too small \
        to fit the entire chat history into the prompt. Your task is to condense the chat history so that only the most relevant information \
        is retained. Relevance should be determined on how useful the information is to the user's current query. \
        You must use concise bullet points for each piece of information to ensure the chat history is easy to read and understand. \
        If the current messages to be condensed are irrelevant to the user's last message, \
        simply output these words 'NO RELEVANT INFORMATION'.\n\n
        
        # CONDENSED CHAT HISTORY SO FAR...\n
        <condensed>{condensed}</condensed>\n\n

        # USER'S LAST MESSAGE FOR CHECKING RELEVANCE AGAINST\n
        <user last message>{user_last_message}</user last message>\n\n
        
        # CURRENT MESSAGES TO BE CONDENSED\n
        <current message>{current_message}</current message>
        """
    )
)


CONDENSED_TEMPLATE = """# CHAT HISTORY:
**This is a condensed chat history to save space:**
{condensed}

# USER'S LAST MESSAGE:
{user_last_message}
"""


class StandardCondenser(CondenseModuleBase):
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
        super().__init__(n_msg_trigger, n_tokens_trigger)
        self.llm = llm
        self._n_msgs_user_intent = n_msgs_user_intent
        self._condense_batch_size = condense_batch_size

    def get_user_intent(self, chat_history: ChatMemoryBuffer) -> str:
        if self.user_intent:
            return self.user_intent
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
        self.user_intent = str(user_intent)
        return self.user_intent

    def _extract_relevant(
        self, messages: list[str], condensed: str, user_intent: str
    ) -> str:
        batch_str = "\n".join([str(msg) for msg in messages])
        response = non_structured_invocation(
            llm=self.llm,
            prompt=CHAT_HISTORY_CONDENSE.format(
                user_last_message=user_intent,
                condensed=(
                    condensed
                    if condensed != ""
                    else "No messages have been condensed yet."
                ),
                current_message=batch_str,
            ),
            inference_kwargs=self._condense_kwargs,
        )
        return str(response)

    def condense_chat_history(self, chat_history: ChatMemoryBuffer) -> str:
        user_intent = self.get_user_intent(chat_history)
        messages = chat_history.get_all()[:-1]
        condensed = ""
        for batch in range(0, len(messages), self._condense_batch_size):
            response = self._extract_relevant(
                [str(m) for m in messages[batch : batch + self._condense_batch_size]],
                condensed,
                user_intent,
            )
            if "NO RELEVANT INFORMATION" not in response:
                condensed += "\n" + response

        condensed = CONDENSED_TEMPLATE.format(
            condensed=condensed,
            user_last_message=user_intent,
        )
        return condensed
