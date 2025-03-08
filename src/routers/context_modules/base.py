from abc import ABC, abstractmethod
from textwrap import dedent
from uuid import UUID

import pandas as pd
from llama_index.core.llms.llm import LLM

from src.routers.context_modules.context.base import Context
from src.routers.context_modules.pydantics import ContextInfo, ExtractedFacts
from src.routers.skills import SkillOutput
from src.routers.skills.base import FunctionCallSkill, SkillArgAttr


class VerifyContext(FunctionCallSkill):
    def __init__(
        self,
        context_module: "ContextModuleBase",
    ):
        name = f"{context_module.get_name()}_verify"
        description = (
            f"Verify whether a {context_module.get_name()} stored context contains information "
            "you need for providing a response back to the user."
        )
        function_args: list[SkillArgAttr] = [
            SkillArgAttr(
                name="context_id",
                dtype="str",
                description="The ID of the context to verify.",
                required=True,
            ),
            SkillArgAttr(
                name="query",
                dtype="str",
                description="The query to verify the context with.",
                required=True,
            ),
        ]
        visible_to_human: bool = False
        super().__init__(
            name=name,
            description=description,
            function_args=function_args,
            visible_to_human=visible_to_human,
        )
        self.context_module = context_module

    def execute(self, context_id: str, query: str) -> SkillOutput:  # type: ignore
        response = self.context_module.summarise_from_context(context_id, query)
        return SkillOutput(
            response_to_llm=response,
        )


class ContextModuleBase(ABC):
    _description: str = (
        "{name} is a memory feature that stores context that has been retrieved via tool calls that you have activated previously. "
        "The purpose of this memory is to allow you to access retrieved information on request without it persisting in your "
        "context window and causing performance degradation or confusion. "
        "The {name} can be accessed using the tools available to you."
    )

    def __init__(
        self,
        chat_id: UUID,
        llm: LLM,
        name: str,
    ):
        self.chat_id = chat_id
        self.llm = llm
        self.name = name
        self.context: dict[str, Context] = {}
        self._load_context(chat_id)
        self.verify_tool = VerifyContext(self)

    @abstractmethod
    def _load_context(self, chat_id: UUID):
        pass

    def get_name(self) -> str:
        return self.name

    def add_context(self, context: Context):
        if not isinstance(context, Context):
            raise TypeError(f"Expected a ContextType object, got {type(context)}")
        self.context[context.context_id] = context

    def save_contexts(self):
        for context in self.context.values():
            context.save_context()

    @property
    def info(self) -> str:
        info = f"# {self.name}:\n"
        info += self._description.format(name=self.name) + "\n\n"
        info += (
            f"Below is a list of contexts inside the {self.name}. "
            "To view the content of a context use the context tools available.\n\n"
        )

        if len(self.context) == 0:
            info += f"**The {self.name} is empty, retrieve data first to add it to the {self.name}.**"
            return info

        info_segments: list[ContextInfo] = []
        for _, context in self.context.items():
            info_segments.append(context.info)

        info_frame = pd.DataFrame([s.model_dump() for s in info_segments])
        markdown = info_frame.to_markdown(index=False)
        info += markdown

        info = dedent(info)
        return info

    def check_context_by_id(self, context_id: str) -> bool:
        return context_id in self.context

    def retrieve_context_by_id(self, context_id: str) -> Context:
        self._existance_check(context_id)
        if context_id not in self.context:
            raise ValueError(f"Context with ID {context_id} not found in {self.name}.")
        return self.context[context_id]

    def _existance_check(self, context_id: str):
        if len(self.context) == 0:
            raise ValueError(
                f"The {self.name} is empty, retrieve data first to add it to the {self.name}."
            )
        if not self.check_context_by_id(context_id):
            raise ValueError(f"Context with ID {context_id} not found in {self.name}.")
        return

    def show_context_by_index_and_reference(
        self,
        context_id: str,
        references: list[str],
    ) -> str:
        context = self.retrieve_context_by_id(context_id)
        return context.get_content_by_reference(references)

    def extract_from_context(
        self, context_id: str, instructions: str
    ) -> ExtractedFacts:
        context = self.retrieve_context_by_id(context_id)
        return context.extract(instructions, self.llm)

    def summarise_from_context(self, context_id: str, query: str) -> str:
        context = self.retrieve_context_by_id(context_id)
        return context.summarise(query, self.llm)

    def get_short_references_by_context_id(self, context_id: str) -> list[str]:
        context = self.retrieve_context_by_id(context_id)
        return context.get_short_references()

    def __len__(self) -> int:
        return sum([context.count for context in self.context.values()])
