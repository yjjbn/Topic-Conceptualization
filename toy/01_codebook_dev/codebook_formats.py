"""Add a Pydantic model and a FORMATS entry to support a new output format."""
from dataclasses import dataclass
from typing import Literal
from pydantic import BaseModel, Field

class StrictModel(BaseModel, extra="forbid", strict=True):
    pass


type Label = Literal["0", "1"]


class Example(StrictModel):
    text: str
    explanation: str


class DecisionRulesCodebook(StrictModel):

    class DecisionRule(StrictModel):
        rule_id: Literal["1", "2", "3", "4", "5"]
        rule: str
        positive_example: Example
        negative_example: Example

    definition: str
    decision_rules: list[DecisionRule] = Field(min_length=1, max_length=5)

class CodebookLLMCodebook(StrictModel):

    definition: str
    clarification: str
    positive_examples: list[Example] = Field(min_length=1, max_length=3)
    negative_clarification: str
    negative_examples: list[Example] = Field(min_length=1, max_length=3)


class ExplainedClassification(StrictModel):
    label: Label
    explanation: str


@dataclass(frozen=True)
class CodebookFormat:
    prompt: str
    codebook_model: type[StrictModel]
    classification_type: object = Label


FORMATS = {
    "decision_rules": CodebookFormat(
        "prompt_classify-create-codebook_grimmer.txt", DecisionRulesCodebook
    ),
    "codebook_LLM": CodebookFormat(
        "prompt_codebookLLM_grimmer.txt", CodebookLLMCodebook
    ),
    "explanation_codebookLLM": CodebookFormat(
        "prompt_explain_codebookLLM_grimmer.txt", CodebookLLMCodebook,
        ExplainedClassification,
    ),
}
