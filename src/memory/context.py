from abc import ABC, abstractmethod
import asyncio
from textwrap import dedent
from typing import Any, Optional
import json
from json import JSONDecodeError
import logging
from pydantic import BaseModel

from tenacity import retry, stop_after_attempt, wait_fixed

from llama_index.core.llms.llm import LLM
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core import PromptTemplate

logger = logging.getLogger(__name__)


RESPONSE_PROMPT_TEMPLATE = PromptTemplate(
    """# SYSTEM
<system>Given a brief and some retrieved contexts, you must examine the brief and extract the information from the contexts that relates to the brief.</system>

# BRIEF
<brief>{brief}</brief>

# CONTEXTS
<contexts>{contents}</contexts>

**Below is the schema for which your JSON output must be formatted as (YOU MUST ensure that you are double escaping any control characters that \
are not inside strings):**
{schema}"""
)


FIX_JSON_PROMPT = PromptTemplate(
    dedent(
        """This string is meant to be JSON however is raising a JSONDecodeError. \
        Without changing any of the key-value pairs, you must fix the JSON string so that it is valid JSON. \
        Typically the issues reside in unescaped newlines that aren't inside strings, or unescaped control characters.\n\n
        
        Only output the fixed JSON string, do NOT include any other information.\n\n
        
        # JSON TO FIX\n
        {response}
        """
    )
)


class Point(BaseModel):
    """
    Schema for generating a point based on a piece of context.

    Attributes:
    -----------
    point: str
        Point generated from the context.
    reference: str
        Reference String of the context.
    """

    point: str
    reference: str

    def __str__(self) -> str:
        return f"- {self.point} (Reference: {self.reference})"


class Response(BaseModel):
    """
    Schema for generating a response with regard to a given brief and context.

    Attributes:
    -----------
    points: list[Point]
        Points generated from the contexts
    """

    points: list[Point]

    def __str__(self) -> str:
        return "\n".join([str(point) for point in self.points])


class Content(BaseModel, ABC):
    """
    Content class for storing retrieved content into the memory.
    """

    @abstractmethod
    def __str__(self) -> str:
        """
        Abstract method for converting the content into a string.
        """
        pass

    @property
    @abstractmethod
    def reference(self) -> str:
        """
        Abstract property for getting the reference of the content.
        """
        pass


class Context(ABC):
    """
    Context class for storing retrieved context into the memory.

    Attributes:
    -----------
    _source: str
        Source of the context.
    _max_show_content: int
        Maximum number of content to show when displaying the context.
    _llm_kwargs: dict[str, Any]
        Keyword arguments for the LLM model.
    _semaphore: asyncio.Semaphore
        Semaphore for controlling the number of concurrent requests to the LLM model.
    """

    _source: str = "abstract_attribute"
    _max_show_content: int = 10  # WARNING: Changing this may misalign the page numbering for existing contexts.
    _llm_kwargs: dict[str, Any] = {"max_tokens": 1000, "temperature": 0.2}
    _semaphore: asyncio.Semaphore = asyncio.Semaphore(5)

    def __init__(
        self,
        session_id: str,
        index: int,
        context: str,
        content: Optional[list[Content]] = None,
        _pages: Optional[int] = None,
        _count: Optional[int] = None,
    ):
        """
        Instantiates a Context object.

        When instantiating a NEW context, the content must be provided.
        When instantiating an EXISTING context, the content can be None but _pages must be provided.

        Parameters:
        -----------
        session_id: str
            Session ID the context is related to.
        index: int
            Index of the context.
        context: str
            Context of the context.
        content: Optional[list[Content]]
            List of content objects, default is None.
        _pages: Optional[int]
            Number of pages of context, default is None.
        """
        self.session_id = session_id
        self.index = index
        self.content: Optional[list[Content]] = content
        self.context = context
        self._loaded: bool = True if content else False
        self._pages = _pages
        self._count = _count

    @abstractmethod
    def save_context(self) -> "Context":
        """
        Abstract method for saving the context into a database.
        This method should follow either:
        - Upsert logic, where the context is saved if it does not exist, or updated if it does.
        - Insert if not exists logic, where the context is saved if it does not exist already.
        """
        pass

    @abstractmethod
    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
    def load_from_session_id_index(cls, session_id: str, index: int) -> "Context":
        """
        Abstract method for creating a context from session_id, source and context.

        The Context must be loaded with:
        - session_id
        - index
        - context
        - _pages
        - _count

        NOTE: You must set self._loaded to True after loading the context.

        Parameters:
        -----------
        session_id: str
            Session ID the context is related to.
        index: int
            Index of the context.

        Returns:
        --------
        Context
            Context
        """
        pass

    @property
    def info(self) -> str:
        return f"{self._source} ({self.count} records): {self.context}"

    @property
    def count(self) -> int:
        if self._count:
            return self._count
        return len(self.content)

    @property
    def pages(self) -> int:
        if self._pages:
            return self._pages
        return len(self.content) // self._max_show_content + 1

    def show_content_by_page(self, page: int) -> str:
        """
        Shows content by page without title.

        Parameters:
        -----------
        page: int
            Page number to show.
        """
        if not self._loaded:
            self.load_from_session_id_index(self.session_id, self.index)

        if len(self.content) == 0:
            return f"No content available for the {self._source}: {self.context}."
        if page < 1 or page > self.pages:
            return f"Page out of range. Available pages are from 1 to {self.pages}."

        start = min(max(page - 1, 0) * self._max_show_content, len(self.content))
        end = min(start + self._max_show_content, len(self.content))
        content = "\n\n".join([str(content) for content in self.content[start:end]])
        return content

    def get_content_by_reference(self, references: list[str]) -> str:
        """
        Gets content by reference.

        Parameters:
        -----------
        reference: list[str]
            References of the content to get.

        Returns:
        --------
        str
        """
        if not self._loaded:
            self.load_from_session_id_index(self.session_id, self.index)

        response = ""
        for reference in references:
            done = False
            for content in self.content:
                if content.reference == reference:
                    response += str(content)
                    response += "\n\n"
                    done = True
                    break

            if not done:
                response += f"Content not found ({reference}).\n\n"

        return response

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
    def _non_structured_invocation(
        self,
        llm: LLM,
        template: PromptTemplate,
        template_kwargs: dict[str, Any],
        inference_kwargs: dict[str, Any],
    ) -> str:
        response: CompletionResponse = llm.complete(
            prompt=template.format(**template_kwargs), **inference_kwargs
        )
        response_str = str(response)

        return response_str

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
    async def _structured_invocation(
        self,
        idx: int,
        contents: list[Content],
        brief: str,
        llm: LLM,
    ) -> tuple[int, Response]:
        async with self._semaphore:
            response = llm.complete(
                prompt=RESPONSE_PROMPT_TEMPLATE.format(
                    contents="\n\n".join([str(content) for content in contents]),
                    brief=brief,
                    schema=json.dumps(Response.model_json_schema()),
                ),
                **self._llm_kwargs,
            )
            response_str = response.text

            try:
                response_json = json.loads(str(response_str).strip())
            except JSONDecodeError:
                logger.warning(
                    "JSONDecodeError: Response from LLM is not a valid JSON: %s",
                    response_str,
                )
                response_str = self._non_structured_invocation(
                    FIX_JSON_PROMPT, {"response": response_str}, self._llm_kwargs
                )
                response_json = json.loads(str(response_str).strip())

            response_object = Response(**response_json)
            return (idx, response_object)

    async def aextract(self, brief: str, llm: LLM) -> str:
        """
        Generates a response with regard to a given brief.

        Parameters:
        -----------
        brief: str
            Brief to generate a response for.
        llm: LLM
            LLM model to generate the response.

        Returns:
        --------
        str
            Response generated from the brief.
        """
        if not self._loaded:
            self.load_from_session_id_index(self.session_id, self.index)

        tasks = [
            self._structured_invocation(
                contents=self.show_content_by_page(page),
                brief=brief,
                llm=llm,
            )
            for page in range(1, self.pages + 1)
        ]
        responses: list[tuple[int, Response]] = await asyncio.gather(*tasks)
        responses.sort(key=lambda x: x[0])

        response = "\n".join([str(response[1]) for _, response in responses])
        return response


class ContextInFocusBase(ABC):
    _description: str = (
        "{name} is a memory feature that stores context that has been retrieved via tool calls that you activate. \
    The purpose of this memory is to allow you to access retrieved information on request, without it persisting in your context window and causing performance issues. \
    The {name} can be accessed using the tools available."
    )

    def __init__(
        self,
        session_id: str,
        llm: LLM,
        name: str = "ContextInFocus",
    ):
        self.session_id = session_id
        self.llm = llm
        self.name = name
        self.context: list[Context] = self._load_context(session_id)

    @abstractmethod
    def _load_context(self, session_id: str) -> list[Context]:
        """
        Abstract method for loading the context from the database.

        Details on Implementation:
        --------------------------
        A session_id should be linked to a database with a table that details contexts that have:
        - session_id: str
        - index: int
        - source: str
        - context: str
        - _pages: int
        - _count: int

        With a composite key of (session_id, index) to ensure uniqueness.

        Each context source should have a corresponding Context class that is defined to load the context into memory.

        Each context source should have a corresponding table in the database that details which conents are stored in the context
        for loading into memory.

        Parameters:
        -----------
        session_id: str
            Session ID the context is related to.

        Returns:
        --------
        list[Context]
            List of context objects.
        """
        # If session_id does not exist, return an empty list
        # Else, return the list of context objects loaded from the database
        pass

    def add_context(self, context: Context):
        """
        Adds context to the current context list.

        Parameters:
        -----------
        context: Context
            Context object to add to the current context list.
        """
        self.context.append(context)

    def save_contexts(self):
        """
        Saves the context into the database.
        """
        for context in self.context:
            context.save_context()

    @property
    def info(self):
        info = f"# {self.name}\n"
        info += self._description.format(name=self.name) + "\n\n"
        info += f"Below is a list of contexts inside the {self.name}. \
        Each context is listed in the format: 'index. source (number of records): context on what is contained in the collection'. \
        To view the content of a context use the context tools available.\n"

        if len(self.context) == 0:
            info += f"**The {self.name} is empty, retrieve data first to add it to the {self.name}.**"
            return info

        for i, context in enumerate(self.context):
            info += f"{i + 1}. {context.info}\n"
        return info

    def retrieve_context_by_index(self, index: int) -> Context:
        """
        Retrieves context by index.

        Parameters:
        -----------
        index: int
            Index of the context to retrieve.

        Returns:
        --------
        Context
            Context object.
        """
        if index < 1 or index > len(self.context):
            raise ValueError("Index out of range.")
        return self.context[index - 1]

    def show_context_by_index_and_reference(
        self, index: int, references: list[str]
    ) -> str:
        """
        Shows context by index and reference.

        Parameters:
        -----------
        index: int
            Index of the context to show.
        reference: list[str]
            References of the contents to show.

        Returns:
        --------
        str
            Content of the context.
        """
        if len(self.context) == 0:
            return f"The {self.name} is empty, retrieve data first to add it to the {self.name}."
        if index < 1 or index > len(self.context):
            return f"Index out of range. Available indexes are from 1 to {len(self.context)}."
        context = self.retrieve_context_by_index(index)
        return context.get_content_by_reference(references)

    async def extract_from_context(self, index: int, brief: str) -> str:
        """
        Extracts information from the context.

        The brief provided should be framed as instructions on what sort of facts should be extracted from contexts on an
        individual basis. 

        Parameters:
        -----------
        index: int
            Index of the context to extract information from.
        brief: str
            Brief to extract information for.
        llm: LLM
            LLM model to extract information.

        Returns:
        --------
        str
            Extracted information from the context.
        """
        if len(self.context) == 0:
            return f"The {self.name} is empty, retrieve data first to add it to the {self.name}."
        if index < 1 or index > len(self.context):
            return f"Index out of range. Available indexes are from 1 to {len(self.context)}."
        context = self.retrieve_context_by_index(index)
        return await context.aextract(brief, self.llm)
