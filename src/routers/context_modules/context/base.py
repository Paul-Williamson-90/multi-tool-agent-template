import logging
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import numpy as np
from llama_index.core.llms.llm import LLM
from pydantic import BaseModel
from tenacity import before_log, retry, stop_after_attempt, wait_fixed

from src.invocations import non_structured_invocation, structured_invocation
from src.routers.context_modules.prompts import EXTRACT_TEMPLATE, SUMMARISE_TEMPLATE
from src.routers.context_modules.pydantics import ContextInfo, ExtractedFacts

logger = logging.getLogger(__name__)


class Content(BaseModel, ABC):
    """A parent class for defining a content type that can be used with a ContextModuleBase child class.
    You should define the attributes of this Pydantic model in the child class.
    """

    @abstractmethod
    def __str__(self) -> str:
        """The content as a string that is presented to an LLM.

        Returns
        -------
        str
            The content as a string.
        """
        pass

    @property
    @abstractmethod
    def reference(self) -> str:
        """A reference that the LLM can use during response generation to help the user \
        identify where the information came from.

        Returns
        -------
        str
            The reference.
        """
        pass

    @abstractmethod
    def reference_match(self, reference: str) -> bool:
        """A method to check if a reference matches the content's reference.

        Parameters
        ----------
        reference : str
            The reference to check.

        Returns
        -------
        bool
            True if the reference matches, False otherwise.
        """
        pass

    @property
    def short_reference(self) -> str:
        """A short reference that can be used to identify the content. This for example could \
        link to a database primary key for easy reference and retrieval.

        Returns
        -------
        str
            The short reference.
        """
        return self.reference


class Context(ABC):
    """A parent class for defining a context module's context collection. Each context can contain \
    one or more Content type items. The Context module provides an interface for loading, saving, \
    and interacting with the content.

    When the Context is loaded from history, the class allows for 'lazy-loading' of the content so that \
    it is only truly loaded into memory when the RouterAgent interacts with it.

    When creating a new Context, you must provide:
    - chat_id
    - context
    - content

    When loading a Context from history, you must provide:
    - chat_id
    - context_id
    - _count

    Parameters
    ----------
    _source : str, optional
        The source of the context, by default "abstract_attribute", this should be overriden in the child class.
    _max_show_content : int, optional
        The maximum number of content items to show per page, by default 5.
    _llm_extract_kwargs : dict[str, Any], optional
        The kwargs for the LLM extract method, by default {"max_tokens": 1000, "temperature": 0.2}.
    _llm_summary_kwargs : dict[str, Any], optional
        The kwargs for the LLM summarise method, by default {"max_tokens": 500, "temperature": 0.2}.
    _max_workers : int, optional
        The maximum number of workers for the extract method, by default 10.
    _max_content_in_summarise : int, optional
        The maximum number of content items to summarise, by default 30.
    """

    _source: str = "abstract_attribute"
    _max_show_content: int = 5
    _llm_extract_kwargs: dict[str, Any] = {"max_tokens": 1000, "temperature": 0.2}
    _llm_summary_kwargs: dict[str, Any] = {"max_tokens": 500, "temperature": 0.2}
    _max_workers: int = 10
    _max_content_in_summarise: int = 30

    def __init__(
        self,
        chat_id: uuid.UUID,
        context: str,
        context_id: Optional[str] = None,
        content: Optional[list[Content]] = None,
        _count: Optional[int] = None,
    ):
        """Initialise the Context module.

        When the Context is loaded from history, the class allows for 'lazy-loading' of the content so that \
        it is only truly loaded into memory when the RouterAgent interacts with it.

        When creating a new Context, you must provide:
        - chat_id
        - context
        - content

        When loading a Context from history, you must provide:
        - chat_id
        - context_id
        - _count


        Parameters
        ----------
        chat_id : uuid.UUID
            The chat ID for the context.
        context : str
            This should describe at a high-level what content is stored in the context. It is \
            used to help the LLM understand which contexts to use for a given query.
        context_id : Optional[str], optional
            The id of the context, by default None
        content : Optional[list[Content]], optional
            A list of Content objects that represent the Context's contents, by default None
        _count : Optional[int], optional
            The number of Content objects in the Context, by default None.
            If the Context is loaded from a prior step rather than as part of an addition to a \
            ContextModuleBase class, this should be set to the number of Content objects.
        """
        self.chat_id = chat_id
        self.context_id = context_id or str(uuid.uuid4())
        self.context = context
        self.content = content or []
        self._count = _count or len(self.content)
        self._loaded = True if len(self.content) > 0 else False

    @abstractmethod
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(1),
        before=before_log(logger, logging.INFO),
    )
    def load_from_chat_id_context_id(self):
        """Load the context from the chat ID and context ID. This should link to a database \
        that stores the context information.
        """
        pass

    def _load(self):
        """Load the context from the chat ID and context ID. This should link to a database \
        that stores the context information.

        When the context is loaded the _loaded attribute is set to True.
        """
        self.load_from_chat_id_context_id()
        self._loaded = True

    @abstractmethod
    def save_context(self):
        """Save the context to a database. This should store the context information \
        so that it can be retrieved later.
        """
        pass

    @property
    @abstractmethod
    def contents_summary_info(self) -> str:
        """A property that generates a string of information that the LLM can use to \
        understand the context of the content.

        This can for example provide summary statistics about the content.

        Returns
        -------
        str
            The summary information
        """
        pass

    @property
    def info(self) -> ContextInfo:
        """Get the context information as a ContextInfo Pydantic object.

        Returns
        -------
        ContextInfo
            The context information.
        """
        return ContextInfo(
            context_id=self.context_id,
            source=self._source,
            num_records=self.count,
            context=self.context,
        )

    @property
    def count(self) -> int:
        """Get the number of content items in the context.

        Returns
        -------
        int
            The number of content items
        """
        if self._count:
            return self._count
        if not self._loaded:
            self._load()
        assert isinstance(self.content, list)
        return len(self.content)

    @property
    def pages(self) -> int:
        """Get the number of pages of content that can be shown.

        Returns
        -------
        int
            The number of pages of content
        """
        if self._count:
            return self._count // self._max_show_content + 1
        assert isinstance(self.content, list)
        return len(self.content) // self._max_show_content + 1

    def show_content_by_page(self, page: int) -> str:
        """Show the content for a given page.

        Parameters
        ----------
        page : int
            The page number to show.

        Returns
        -------
        str
            The content for the page
        """
        if not self._loaded:
            self._load()

        assert isinstance(self.content, list)

        if len(self.content) == 0:
            return f"No content avaialble for the {self._source}: {self.context}"
        if page < 1 or page > self.pages:
            return f"Invalid page number. Please provide a number between 1 and {self.pages}"

        start = min(max(page - 1, 0) * self._max_show_content, len(self.content))
        end = min(start + self._max_show_content, len(self.content))
        content = "\n\n".join([str(c) for c in self.content[start:end]])
        return content

    def get_short_references(self) -> list[str]:
        """Get the short references for the content.

        Returns
        -------
        list[str]
            A list of short references
        """
        if not self._loaded:
            self._load()
        assert isinstance(self.content, list)
        return [c.short_reference for c in self.content]

    def get_content_by_reference(self, references: list[str]) -> str:
        """Get the content for a list of references.

        Parameters
        ----------
        references : list[str]
            A list of references to get the content for

        Returns
        -------
        str
            The content for the references
        """
        if not self._loaded:
            self._load()
        assert isinstance(self.content, list)

        response = ""
        for reference in references:
            done = False
            for content in self.content:
                if content.reference_match(reference):
                    response += f"{content}\n\n"
                    done = True
                    break
            if not done:
                response += f"No content found for reference: {reference}\n\n"
        return response

    def extract(self, instructions: str, llm: LLM) -> ExtractedFacts:
        """Extract facts from the content using an LLM.

        Parameters
        ----------
        instructions : str
            The facts to extract from the content.
        llm : LLM
            The LLM to use for the extraction.

        Returns
        -------
        ExtractedFacts
            The extracted facts
        """
        if not self._loaded:
            self._load()

        assert isinstance(self.content, list)

        def worker(page) -> tuple[int, ExtractedFacts]:
            output = structured_invocation(
                llm=llm,
                context=EXTRACT_TEMPLATE.format(
                    brief=instructions, contexts=self.show_content_by_page(page)
                ),
                pydantic_object=ExtractedFacts,
                llm_kwargs=self._llm_extract_kwargs,
            )
            assert isinstance(output, ExtractedFacts)
            return page, output

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(worker, page): page for page in range(1, self.pages + 1)
            }
            extracted_facts = [future.result() for future in as_completed(futures)]

        extracted_facts.sort(key=lambda x: x[0])
        extracted_facts_list = [fact for _, fact in extracted_facts]
        combined_facts = ExtractedFacts(
            facts=[fact for facts in extracted_facts_list for fact in facts.facts]
        )
        return combined_facts

    def summarise(self, query: str, llm: LLM) -> str:
        """Summarise the content for a given query.

        Parameters
        ----------
        query : str
            The query to summarise the content for.
        llm : LLM
            The LLM to use for the summarisation.

        Returns
        -------
        str
            The summarised content
        """
        if not self._loaded:
            self._load()

        assert isinstance(self.content, list)

        if len(self.content) == 0:
            return f"No content avaialble for the {self._source}: {self.context}"

        content_list = self.content
        summarised_str = ""

        if len(content_list) > self._max_content_in_summarise:
            indexes = np.random.choice(
                len(content_list), self._max_content_in_summarise, replace=False
            )
            indexes.sort()
            content_list = [content_list[i] for i in indexes]
            summarised_str = (
                "**Too many documents to show, only showing a random selection**"
            )

        content = "\n\n".join([str(c) for c in content_list])

        response = non_structured_invocation(
            llm=llm,
            prompt=SUMMARISE_TEMPLATE.format(
                query=query,
                content=content,
                summarised_str=summarised_str,
            ),
            inference_kwargs=self._llm_summary_kwargs,
        )

        return f"# SUMMARY STATISTICS:\n{self.contents_summary_info}\n\n# SUMMARY:\n{response}"
