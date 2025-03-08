from abc import ABC, abstractmethod
from typing import Optional, Type
from enum import Enum

from pydantic import BaseModel
from llama_index.core.memory import ChatMemoryBuffer


CondenseModuleType = Type["CondenseModuleBase"]


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
            raise ValueError("At least one of n_msg_trigger or n_tokens_trigger must be provided")
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
            return len(chat_history) >= self.n_msg_trigger
        return chat_history._token_count_for_messages() >= self.n_tokens_trigger

    def __call__(self, chat_history: ChatMemoryBuffer) -> CondensedChat:
        if len(chat_history.get_all()) > 1:
            user_intent = self.get_user_intent(chat_history)
        else:
            user_intent = str(chat_history.get_all()[-1])

        if self._trigger(chat_history):
            return CondensedChat(
                user_intent=user_intent,
                condensed=self.condense_chat_history(chat_history)
            )
        return CondensedChat(
            user_intent=user_intent,
            condensed="\n".join([str(msg) for msg in chat_history[:-1]])
        )

    def reset_condensed(self):
        self.condensed = None

    def reset_user_intent(self):
        self.user_intent = None

    def reset(self):
        self.reset_condensed()
        self.reset_user_intent()