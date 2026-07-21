# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: MIT

from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import Literal, Type


class DocumentLocation(BaseModel, extra="forbid"):
    reference_type: Literal[
        "claim",
        "paragraph",
        "page",
        "figure",
        "table",
        "abstract",
        "component",
        "section",
        "other",
    ] = Field(description="The type of the location within the document")
    numbers: list[int | str] | None = Field(
        description="The numbers of the location, e.g., claim numbers, paragraph numbers.",
        default=None,
    )
    quote: str | None = Field(
        description=(
            "A literal quote from the prior art reference that supports the coverage of the feature"
            ", if applicable. Do not repeat the feature here. Use extra for all other extra information."
        ),
        default=None,
    )
    extra: str | None = Field(
        description="Any extra information about the location, if applicable, e.g. line numbers, examiners reasoning etc.",
        default=None,
    )


class PriorArtReference(BaseModel, extra="forbid"):
    document: str = Field(
        description="The document identifier of the prior art reference, e.g., D1, D2"
    )
    location: DocumentLocation | None = Field(
        description="The location within the document, e.g., page number, figure number, component identifier, if applicable"
    )


class ClaimBreakdownFeature(BaseModel, extra="forbid"):
    feature: str = Field(
        description="The feature of the claim under examination. Remove the prior art references!"
    )
    distinguishing_spans: list[str] = Field(
        description=(
            "The part of the feature that distinguishes it from the prior art, usually indicated by "
            "strike-through formatting. Set to null if the feature is fully covered by prior art."
        ),
        default_factory=list,
    )
    prior_art_references: list[PriorArtReference] = Field(
        description=(
            "The references to the prior art document covering the feature, if applicable. "
            "Should be empty if the feature is not covered by prior art."
        ),
        default_factory=list,
    )


class PriorArtDocument(BaseModel, extra="forbid"):
    label: str = Field(
        description="The label of the prior art document used in the office action, e.g., D1, D2"
    )
    type: Literal["patent", "paper", "other"] = Field(
        description="Type of the prior art document"
    )
    identifier: str | None = Field(
        description=(
            "The identifier of the prior art document, e.g., patent publication number, DOI, or XP number."
            "For papers, favor the DOI over the XP number!"
        ),
        default=None,
    )
    applicant_or_authors: str | None = Field(
        description="Applicant of the patent or authors of the document", default=None
    )
    title: str | None = Field(description="Title of the document", default=None)
    date: str | None = Field(
        description="Publication date of the document in ISO 8601 format (YYYY-MM-DD)",
        default=None,
    )


class InventiveStepReasoning(BaseModel, extra="forbid"):
    difference_to_prior_art: str = Field(
        description="Sentence(s) identifying the distinguishing features over the closest prior art."
    )
    objective_problem: str | None = Field(
        description="Sentence(s) defining the objective technical problem to be solved.",
        default=None,
    )
    technical_effect_reasoning: str | None = Field(
        description="Sentence(s) explaining the technical effects achieved by the distinguishing features.",
        default=None,
    )
    skilled_person_reasoning: str | None = Field(
        description="Sentence(s) explaining why the skilled person would arrive at the claimed solution.",
        default=None,
    )


class ClaimBreakdown(BaseModel, extra="forbid"):
    language: Literal["EN", "DE", "FR"]
    prior_art_documents: list[PriorArtDocument]
    breakdown: list[ClaimBreakdownFeature] | None
    reason_for_rejection: Literal["novelty", "inventive step"] | None = Field(
        description="Reason for rejection of claim 1. Set to None if claim 1 is not objected against."
    )
    inventive_step_reasoning: InventiveStepReasoning | None = Field(
        description=(
            "The examiners reasoning why the distinguishing features of claim 1 do not constitute an inventive step. "
            "Extract the full text verbatim. "
            "Only relevant if claim 1 is rejected for missing inventive step. Set to None/null otherwise."
        )
    )

    @field_validator("inventive_step_reasoning")
    @classmethod
    def reasoning_iff_inventive_step(
        cls: Type[BaseModel], reasoning: str | None, values: ValidationInfo
    ) -> str | None:
        reasons: list[str] = values.data.get("reasons_for_rejection")  # type: ignore
        if reasons and "inventive step" in reasons and not reasoning:
            raise ValueError(
                "'inventive_step_reasoning' cannot be empty if claim 1 was rejected for missing inventive step."
            )
        if reasoning and reasons and "inventive step" not in reasons:
            raise ValueError(
                "'inventive_step_reasoning' is set even though 'inventive step' is not among the reasons for rejection of claim 1."
            )
        return reasoning


##############
# types with prior passages, used to store the references after parsing
# we don't use them for parsing directly to avoid complicating the initial extraction task
##############


class PriorArtPassage(BaseModel, extra="forbid"):
    source_reference: PriorArtReference
    reference: tuple[
        Literal[
            "claim",
            "paragraph",
            "page",
            "figure",
            "table",
            "abstract",
            "component",
            "section",
            "other",
        ],
        int | None,
    ]
    text: str
    confidence: float | None = None


class ClaimBreakdownFeatureWithPriorPassages(ClaimBreakdownFeature):
    prior_art_passages: list[PriorArtPassage] = Field(default_factory=list)


class ClaimBreakdownWithPriorPassages(ClaimBreakdown):
    breakdown: list[ClaimBreakdownFeatureWithPriorPassages] | None  # type: ignore


class PatentDocument(BaseModel):
    publication_number: str
    title: str
    abstract: str | None
    description: list[str | None]
    claims: list[str | None]


class ApplicationData(BaseModel):
    app_num: str
    breakdown: ClaimBreakdownWithPriorPassages
    rejected: PatentDocument
    granted: PatentDocument
    cited: PatentDocument
