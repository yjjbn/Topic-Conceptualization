"""Output models for classifications with explanations and an LLM codebook."""
from typing import Literal
from pydantic import BaseModel, Field

class StrictModel(BaseModel, extra="forbid", strict=True):
    pass

## the pieces ##

# label
type Label = Literal["0", "1"]


# label + explanation
class ExplainedClassification(StrictModel):
    label: Label
    explanation: str


# document ID + explanation
class DocumentExample(StrictModel):
    document_id: str = Field(
        description="The exact document_id of an illustrative text from the provided documents."
    )
    explanation: str = Field(
        description="Explain why this example meets or does not meet the definition."
    )

## codebook formats ##

class CodebookLLMCodebook(StrictModel):

    definition: str
    clarification: str
    positive_examples: list[DocumentExample] = Field(min_length=1, max_length=3)
    negative_clarification: str
    negative_examples: list[DocumentExample] = Field(min_length=1, max_length=3)
