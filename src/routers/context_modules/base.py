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
    """A default skill that is used with ContextModuleBase. The skill allows the RouterAgent \
    to interact with context modules to verify whether a context contains information needed \
    for providing a response back to the user.

    The information passed to the RouterAgent in response to a query is a summary of the context \
    in relation to the information the RouterAgent is checking exists. This ensures that the \
    context window is not cluttered with information during the reasoning and tool selection \
    steps that would otherwise potentially degrade the LLM's performance due to attention-span \
    limitations.
    """

    def __init__(
        self,
        context_module: "ContextModuleBase",
    ):
        """Initialise the VerifyContext skill.

        Parameters
        ----------
        context_module : ContextModuleBase
            The context module the skill is associated with.
        """
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
        """Execute the VerifyContext skill.

        Parameters
        ----------
        context_id : str
            The ID of the context in the context module to verify.
        query : str
            The query to verify the context with.

        Returns
        -------
        SkillOutput
            The response to the query.
        """
        response = self.context_module.summarise_from_context(context_id, query)
        return SkillOutput(
            response_to_llm=response,
        )


class ContextModuleBase(ABC):
    """A parent class for defining a context module. A context module is used as a memory feature \
    that enables large context retrieval processes without the retrieved context being in the RouterAgent's \
    context window during reasoning and tool selection steps. This is to prevent performance degradation \
    and confusion due to the context window being cluttered with information.

    The RouterAgent can interact with the context modules in various ways at different steps:
    - Reasoning Step:
        - The RouterAgent can verify whether a context contains information needed for providing a response back to the user.
    - Context Selection Step:
        - The RouterAgent can fetch retrieved context from the module and add it to the prompt in the following \
        response generation step.

    Attributes:
    -----------
    name : str
        The name of the context module.
    _description : str
        The description of the context module for the LLM.
    """

    name: str = "abstract_attribute"
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
    ):
        """Initialise the ContextModuleBase.

        Parameters
        ----------
        chat_id : UUID
            The ID of the chat the context module is associated with.
        llm : LLM
            The LLM module used for interacting with the context module.
        """
        self.chat_id = chat_id
        self.llm = llm
        self.context: dict[str, Context] = {}
        self._load_context(chat_id)
        self.verify_tool = VerifyContext(self)

    @abstractmethod
    def _load_context(self, chat_id: UUID):
        """Load the context into the context module by the chat_id.
        This should be defined in the child class (e.g. fetching context from a database).

        Parameters
        ----------
        chat_id : UUID
            The ID of the chat to load the context for.
        """
        pass

    def get_name(self) -> str:
        """Get the name of the context module.

        Returns
        -------
        str
            The name of the context module.
        """
        return self.name

    def add_context(self, context: Context):
        """Add a context to the context module. This should be combined with a retrieval skill \
        so that the output of the retrieval process is added to the context module.

        Parameters
        ----------
        context : Context
            The context to add to the context module.

        Raises
        ------
        TypeError
            If the context is not an instance of Context.
        """
        if not isinstance(context, Context):
            raise TypeError(f"Expected a ContextType object, got {type(context)}")
        self.context[context.context_id] = context

    def save_contexts(self):
        """Save the context inside the context module. This should be defined in the child class \
        (e.g. saving context to a database).
        """
        for context in self.context.values():
            context.save_context()

    @property
    def info(self) -> str:
        """Get the information about the context module. This information is used inside the RouterAgent \
        prompt to provide the LLM with information on the context modules available.

        Returns
        -------
        str
            The information about the context module.
        """
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
        """Check if a context exists in the context module by the context_id.

        Parameters
        ----------
        context_id : str
            The ID of the context to check for.

        Returns
        -------
        bool
            True if the context exists, False otherwise.
        """
        return context_id in self.context

    def retrieve_context_by_id(self, context_id: str) -> Context:
        """Retrieve a context from the context module by the context_id.

        Parameters
        ----------
        context_id : str
            The ID of the context to retrieve.

        Returns
        -------
        Context
            The context retrieved.

        Raises
        ------
        ValueError
            If the context does not exist in the context module.
        """
        self._existance_check(context_id)
        if context_id not in self.context:
            raise ValueError(f"Context with ID {context_id} not found in {self.name}.")
        return self.context[context_id]

    def _existance_check(self, context_id: str):
        """Check if the context exists in the context module and raise an error if it does not.

        Parameters
        ----------
        context_id : str
            The ID of the context to check for.

        Raises
        ------
        ValueError
            If the context does not exist in the context module.
        """
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
        """Show the content of a context by the context_id and references.

        Parameters
        ----------
        context_id : str
            The ID of the context to show.
        references : list[str]
            The references to show the content for.

        Returns
        -------
        str
            The content of the context.
        """
        context = self.retrieve_context_by_id(context_id)
        return context.get_content_by_reference(references)

    def extract_from_context(
        self, context_id: str, instructions: str
    ) -> ExtractedFacts:
        """Extract facts from a context by the context_id and instructions. This is used during \
        the RouterAgent's context selection step where the agent will query the context for information \
        specific to its needs when responding to the user. This helps ensure only the relevant information \
        is added to the prompt for the LLM during its response (needle in a haystack reduction).

        Parameters
        ----------
        context_id : str
            The ID of the context to extract from.
        instructions : str
            The instructions to extract the facts with.

        Returns
        -------
        ExtractedFacts
            The extracted facts from the context.
        """
        context = self.retrieve_context_by_id(context_id)
        return context.extract(instructions, self.llm)

    def summarise_from_context(self, context_id: str, query: str) -> str:
        """Summarise a context by the context_id and query. This is used during the RouterAgent's \
        reasoning step where the agent will verify whether a context contains information needed for \
        providing a response back to the user.

        Parameters
        ----------
        context_id : str
            The ID of the context to summar
        query : str
            The query to summarise the context with.

        Returns
        -------
        str
            The summary of the context.
        """
        context = self.retrieve_context_by_id(context_id)
        return context.summarise(query, self.llm)

    def get_short_references_by_context_id(self, context_id: str) -> list[str]:
        """Get the short references of a context by the context_id.

        Parameters
        ----------
        context_id : str
            The ID of the context to get the short references for.

        Returns
        -------
        list[str]
            The short references of the context.
        """
        context = self.retrieve_context_by_id(context_id)
        return context.get_short_references()

    def __len__(self) -> int:
        """Get the number of documents across all contexts in the context module.

        Returns
        -------
        int
            The number of documents across all contexts in the context module.
        """
        return sum([context.count for context in self.context.values()])
