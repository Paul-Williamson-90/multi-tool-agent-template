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
from src.routers.context_modules.prompts import (EXTRACT_TEMPLATE,
                                                 SUMMARISE_TEMPLATE)
from src.routers.context_modules.pydantics import ContextInfo, ExtractedFacts

logger = logging.getLogger(__name__)


class Content(BaseModel, ABC):
    @abstractmethod
    def __str__(self) -> str:
        pass

    @property
    @abstractmethod
    def reference(self) -> str:
        pass

    @abstractmethod
    def reference_match(self, reference: str) -> bool:
        pass

    @property
    def short_reference(self) -> str:
        return self.reference


class Context(ABC):
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
        pass

    @abstractmethod
    def save_context(self):
        pass

    @property
    @abstractmethod
    def contents_summary_info(self) -> str:
        pass

    @property
    def info(self) -> ContextInfo:
        return ContextInfo(
            context_id=self.context_id,
            source=self._source,
            num_records=self.count,
            context=self.context,
        )

    @property
    def count(self) -> int:
        if self._count:
            return self._count
        if not self._loaded:
            self.load_from_chat_id_context_id()
        assert isinstance(self.content, list)
        return len(self.content)

    @property
    def pages(self) -> int:
        if self._count:
            return self._count // self._max_show_content + 1
        assert isinstance(self.content, list)
        return len(self.content) // self._max_show_content + 1

    def show_content_by_page(self, page: int) -> str:
        if not self._loaded:
            self.load_from_chat_id_context_id()

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
        if not self._loaded:
            self.load_from_chat_id_context_id()
        assert isinstance(self.content, list)
        return [c.short_reference for c in self.content]

    def get_content_by_reference(self, references: list[str]) -> str:
        if not self._loaded:
            self.load_from_chat_id_context_id()
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
        if not self._loaded:
            self.load_from_chat_id_context_id()

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
        if not self._loaded:
            self.load_from_chat_id_context_id()

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
