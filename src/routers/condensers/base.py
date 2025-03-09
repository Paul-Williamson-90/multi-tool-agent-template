from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from llama_index.core.memory import ChatMemoryBuffer
from pydantic import BaseModel


class TriggerMode(Enum):
    MSG = "msg"
    TOKENS = "tokens"


class CondensedChat(BaseModel):
    user_intent: str
    condensed: str

    def __str__(self) -> str:
        return (
            f"# CHAT HISTORY:\n<chat_history>{self.condensed}</chat_history>\n\n"
            f"# USER'S LAST MESSAGE:\n<user_intent>{self.user_intent}</user_intent>"
        )


class CondenseModuleBase(ABC):
    def __init__(
        self,
        n_msg_trigger: Optional[int] = None,
        n_tokens_trigger: Optional[int] = None,
    ):
        if not any([n_msg_trigger, n_tokens_trigger]):
            raise ValueError(
                "At least one of n_msg_trigger or n_tokens_trigger must be provided"
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
        pass

    @abstractmethod
    def get_user_intent(self, chat_history: ChatMemoryBuffer) -> str:
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
        messages = chat_history.get_all()
        if len(messages) > 0:
            return "\n".join([str(msg) for msg in messages[:-1]])
        return "**There are no previous messages in the chat history.**"

    def _condense(self, chat_history: ChatMemoryBuffer) -> str:
        if not self.condensed:
            out = self.condense_chat_history(chat_history)
            if out == "":
                out = "**There was no previous chat history that was relevant to the user's current request.**"
            self.condensed = out
        return self.condensed

    def _user_intent(self, chat_history: ChatMemoryBuffer) -> str:
        if not self.user_intent:
            out = self.get_user_intent(chat_history)
            self.user_intent = out
        return self.user_intent

    def reset_condensed(self):
        self.condensed = None

    def reset_user_intent(self):
        self.user_intent = None

    def reset(self):
        self.reset_condensed()
        self.reset_user_intent()
