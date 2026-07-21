# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: MIT

import re
from typing import Literal

import dspy
from pydantic import BaseModel

from whats_novel.dataset import ApplicationData, PatentDocument


class PassageReference(BaseModel):
    kind: Literal["paragraph", "abstract", "claim"]
    number: int | None
    relevance: Literal[
        "Direct disclosure",
        "Implicit disclosure",
        "Partial disclosure",
        "Contextually relevant",
        "Not relevant",
    ]


class FeatureWithRelevantPassages(BaseModel):
    feature: str
    relevant_passages: list[PassageReference]
    disclosed: Literal["Fully disclosed", "Partially disclosed", "Not disclosed"]


class NoveltyExaminationOutput(BaseModel):
    relevant_passages_per_feature: list[FeatureWithRelevantPassages]
    novelty: Literal["Novel", "Not novel"]
    confidence: Literal["Very unsure", "Unsure", "Somewhat sure", "Sure", "Very sure"]


def copy_signature(signature):
    class SignatureCopy(signature):
        pass

    return SignatureCopy


class BaseNoveltyExamination(dspy.Module):
    predict: dspy.Module

    def __init__(self, offer_money: bool = True, callbacks=None, **kwargs) -> None:
        super().__init__(callbacks=callbacks)
        self.offer_money = offer_money

    def forward(
        self, sample: ApplicationData, granted: bool = False
    ) -> NoveltyExaminationOutput: ...

    async def aforward(
        self, sample: ApplicationData, granted: bool = False
    ) -> NoveltyExaminationOutput: ...

    def format_patent(
        self, patent: PatentDocument, paragraphs: list[int] | None = None
    ) -> str:
        s = f"# Title\n\n{patent.title or 'N/A'}\n\n"
        s += f"# Abstract\n\n{patent.abstract or 'N/A'}\n\n"

        s += "# Description\n\n"
        for i, paragraph in enumerate(patent.description or []):
            if paragraphs is not None and i not in paragraphs:
                continue
            paragraph = paragraph or ""
            if not re.match(r"^\[\d+\]", paragraph):
                paragraph = f"[{str(i + 1).zfill(4)}] {paragraph}"
            s += f"{paragraph}\n"

        s += "# Claims\n\n"
        for claim in patent.claims:
            s += claim or "" + "\n"

        return s.strip()
