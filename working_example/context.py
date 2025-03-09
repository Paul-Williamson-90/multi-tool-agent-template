import logging
from uuid import UUID

from tenacity import before_log, retry, stop_after_attempt, wait_fixed

from src.routers.context_modules.base import ContextModuleBase
from src.routers.context_modules.context.base import Content, Context

logger = logging.getLogger(__name__)


class DummyContent(Content):
    content: str
    title: str
    content_id: str

    def __str__(self) -> str:
        return (
            f"<content>{self.title}: <ref>{self.short_reference}</ref>\n"
            f"{self.content}</content>"
        )

    @property
    def reference(self) -> str:
        return f"{self.title}: <ref>{self.reference}</ref>"

    @property
    def short_reference(self) -> str:
        return self.content_id

    def reference_match(self, reference: str) -> bool:
        return reference == self.short_reference


class DummyContext(Context):
    _source: str = "database"

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(1),
        before=before_log(logger, logging.INFO),
    )
    def load_from_chat_id_context_id(self):
        # DUMMY implementation
        ## This would be a data load procedure
        self.content.extend(
            [
                DummyContent(
                    content="This is a dummy content",
                    title="Dummy Content",
                    reference="48375",
                ),
                DummyContent(
                    content="This is a dummy content",
                    title="Dummy Content",
                    reference="48376",
                ),
            ]
        )

    def save_context(self):
        # DUMMY implementation
        ## This would be a data save procedure
        pass

    @property
    def contents_summary_info(self) -> str:
        if not self.content or len(self.content) == 0:
            return "No content available, therefore no summary info to provide."
        num_records = len(self.content)
        return f"Context contains {num_records} records."


class DummyContextModule(ContextModuleBase):
    name: str = "Dummy Context Module"

    def _load_context(self, chat_id: UUID):
        # DUMMY implementation
        ## This would be a data load procedure (historic chat)
        self.add_context(
            DummyContext(
                chat_id=chat_id,
                context="This is a dummy memory object for developer testing.",
            )
        )
