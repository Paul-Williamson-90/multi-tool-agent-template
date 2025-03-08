
from pydantic import BaseModel


class Fact(BaseModel):
    """
    Schema for a fact extracted from the context.

    Attributes:
    -----------
    fact: str
        The fact extracted from the context.
    reference: str
        The reference string of the context the fact was extracted from.
    """

    fact: str
    reference: str

    def __str__(self) -> str:
        return f"- {self.fact} (Reference: {self.reference})"


class ExtractedFacts(BaseModel):
    """
    Schema for extracting facts from the context.

    Attributes:
    -----------
    facts: List[Fact]
        The list of facts extracted from the context, default is an empty list.
    """

    facts: list[Fact] = []

    def __str__(self) -> str:
        return "\n".join([str(fact) for fact in self.facts])

    def __len__(self) -> int:
        return len(self.facts)


class ContextInfo(BaseModel):
    """
    Schema for storing information on what a collection of context contains.
    This is purely for display purposes to the RouterAgent.

    Attributes:
    -----------
    context_id: str
        The unique identifier of the context.
    source: str
        The source of the context.
    num_records: int
        The number of records in the context.
    context: str
        Context of what the collection contains.
    """

    context_id: str
    source: str
    num_records: int
    context: str
