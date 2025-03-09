from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from llama_index.core.memory import ChatMemoryBuffer
from pydantic import BaseModel, model_validator


class TriggerMode(Enum):
    MSG = "msg"
    TOKENS = "tokens"


class CondensedChat(BaseModel):
    user_intent: str
    condensed: str

    def __str__(self) -> str:
        return (
            f"# CHAT HISTORY:\n<chat_history>{self.condensed}</chat_history>\n\n"
            f"# USER'S LAST MESSAGE:\n{self.user_intent}"
        )

    @model_validator(mode="after")
    def user_intent_post_format(self):
        if "user:" != self.user_intent.split()[0].lower():
            self.user_intent = f"user: {self.user_intent}"
        return self


class CondenseModuleBase(ABC):
    """A parent class for all condense modules.
    A condense module is a chat history management module that condenses the chat history \
    to save space in the context window of the LLM. This helps reduce the needle-in-the-haystack \
    problem that causes LLM performance to degrade over time.
    """
    def __init__(
        self,
        n_msg_trigger: Optional[int] = None,
        n_tokens_trigger: Optional[int] = None,
    ):
        """Initializes the CondenseModuleBase class.

        A condense module has two modes:
        - Message mode: The module triggers when the number of messages in the chat history \
            exceeds a certain threshold.
        - Tokens mode: The module triggers when the number of tokens in the chat history \
            exceeds a certain threshold.

        Only one of the two modes can be active at a time. Therefore, only pass:
        - n_msg_trigger to activate the message mode.
        - n_tokens_trigger to activate the tokens mode.

        Parameters
        ----------
        n_msg_trigger : Optional[int], optional
            The number of messages that triggers the module, by default None
        n_tokens_trigger : Optional[int], optional
            The number of tokens that triggers the module, by default None

        Raises
        ------
        ValueError
            If both n_msg_trigger and n_tokens_trigger are provided.
        ValueError
            If neither n_msg_trigger nor n_tokens_trigger are provided.
        """
        if not any([n_msg_trigger, n_tokens_trigger]):
            raise ValueError(
                "At least one of n_msg_trigger or n_tokens_trigger must be provided"
            )
        if all([n_msg_trigger, n_tokens_trigger]):
            raise ValueError(
                "Only one of n_msg_trigger or n_tokens_trigger can be provided"
            )
        self.n_msg_trigger = n_msg_trigger
        self.n_tokens_trigger = n_tokens_trigger
        self.trigger_mode: TriggerMode = (
            TriggerMode.MSG if n_msg_trigger is not None else TriggerMode.TOKENS
        )
        self.condensed: str | None = None
        self.user_intent: str | None = None

    @abstractmethod
    def condense_chat_history(self, chat_history: ChatMemoryBuffer) -> str:
        """Abstract method for defining how the chat history should be condensed.

        Parameters
        ----------
        chat_history : ChatMemoryBuffer
            The chat history to be condensed.

        Returns
        -------
        str
            The condensed chat history.
        """
        pass

    @abstractmethod
    def get_user_intent(self, chat_history: ChatMemoryBuffer) -> str:
        """Abstract method for defining how the user's intent should be extracted from the chat history.

        Parameters
        ----------
        chat_history : ChatMemoryBuffer
            The chat history to extract the user's intent from.

        Returns
        -------
        str
            The user's intent.
        """
        pass

    def _trigger(self, chat_history: ChatMemoryBuffer) -> bool:
        if self.trigger_mode == TriggerMode.MSG:
            assert isinstance(self.n_msg_trigger, int)
            return len(chat_history.get_all()) >= self.n_msg_trigger
        assert isinstance(self.n_tokens_trigger, int)
        return (
            chat_history._token_count_for_messages(chat_history.get_all())
            >= self.n_tokens_trigger
        )

    def __call__(self, chat_history: ChatMemoryBuffer) -> CondensedChat:
        """Condenses the chat history.

        Parameters
        ----------
        chat_history : ChatMemoryBuffer
            The chat history to be condensed.

        Returns
        -------
        CondensedChat
            The condensed chat history
        """
        if len(chat_history.get_all()) > 1:
            user_intent = self._user_intent(chat_history)
        else:
            user_intent = str(chat_history.get_all()[-1])

        if self._trigger(chat_history):
            return CondensedChat(
                user_intent=user_intent,
                condensed=self._condense(chat_history),
            )
        return CondensedChat(
            user_intent=user_intent,
            condensed="\n".join([str(msg) for msg in chat_history.get_all()[:-1]]),
        )

    def _non_trigger_condense(self, chat_history: ChatMemoryBuffer) -> str:
        """Procedure for chat history management when the condense module trigger has not been activated.

        Parameters
        ----------
        chat_history : ChatMemoryBuffer
            The chat history to be managed.

        Returns
        -------
        str
            The chat history.
        """
        messages = chat_history.get_all()
        if len(messages) > 0:
            return "\n".join([str(msg) for msg in messages[:-1]])
        return "**There are no previous messages in the chat history.**"

    def _condense(self, chat_history: ChatMemoryBuffer) -> str:
        """Condenses the chat history.

        Parameters
        ----------
        chat_history : ChatMemoryBuffer
            The chat history to be condensed

        Returns
        -------
        str
            The condensed chat history
        """
        if not self.condensed:
            out = self.condense_chat_history(chat_history)
            if out == "":
                out = "**There was no previous chat history that was relevant to the user's current request.**"
            self.condensed = out
        return self.condensed

    def _user_intent(self, chat_history: ChatMemoryBuffer) -> str:
        """Extracts the user's intent from the chat history.

        Parameters
        ----------
        chat_history : ChatMemoryBuffer
            The chat history to extract the user's intent from.

        Returns
        -------
        str
            The user's intent.
        """
        if not self.user_intent:
            out = self.get_user_intent(chat_history)
            self.user_intent = out
        return self.user_intent

    def reset_condensed(self):
        """Resets the condensed chat history."""
        self.condensed = None

    def reset_user_intent(self):
        """Resets the user's intent."""
        self.user_intent = None

    def reset(self):
        """Resets the condensed chat history and the user's intent."""
        self.reset_condensed()
        self.reset_user_intent()
