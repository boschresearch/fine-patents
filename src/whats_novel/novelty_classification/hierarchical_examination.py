import asyncio
import re
from typing import Literal

import dspy
from pydantic import BaseModel

from whats_novel.dataset import ApplicationData
from whats_novel.novelty_classification.shared import (
    BaseNoveltyExamination,
    FeatureWithRelevantPassages,
    NoveltyExaminationOutput,
    PassageReference,
    copy_signature,
)


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-30B-A3B-Thinking-2507", trust_remote_code=True
)
fp = open("feature_prompts.txt", "w")


class NoveltyExaminationSignature(dspy.Signature):
    """
    You are an expert patent examiner trained in European Patent Office (EPO) standards, specifically evaluating novelty under Article 54 EPC.

    Your task is to determine whether claim 1 of the patent application under examination is novel over the closest prior art reference. This assessment concerns ONLY novelty (anticipation), NOT inventive step (obviousness). An application may lack inventive step but still be considered novel.

    ## Task Requirements

    1. **Focus on Claim 1**: Examine only the first independent claim of the patent application.

    2. **Novelty Standard (Article 54 EPC)**: A claim lacks novelty if all of its technical features are explicitly or implicitly disclosed in a prior art document. The prior art must directly and unambiguously disclose every element of the claim.

    3. **Detailed Analysis**: Perform a thorough element-by-element comparison:
       - Identify each feature in claim 1
       - Determine whether each feature is disclosed in the prior art
       - Consider both explicit and implicit disclosures
       - Account for different terminology describing the same technical concept
       - Apply EPO principles for inherent disclosures and genus-species relationships

    4. **Relevant Passages**: You have previously identified the relevant passages from the prior art for claim 1.

    5. **Confidence Assessment**: To differentiate between cases that are clearly novel or not novel, and those that are borderline, select a confidence score from 'Very unsure' to 'Very sure' based.

    Provide your novelty determination and confidence level based on rigorous examination principles. Remember, your objective is to find a way to reasonably reject the claim for lack of novelty. Broadly interpret the prior art, and grant the claim only if it is unquestionably novel.
    """

    claim_under_examination: str = dspy.InputField(
        desc="The text of claim 1 of the patent application being examined"
    )
    closest_prior_art: str = dspy.InputField(desc="The closest prior art reference")
    novelty: Literal["Novel", "Not novel"] = dspy.OutputField(
        desc="Whether the claim is novel over the prior art"
    )
    confidence: Literal[
        "Very unsure", "Unsure", "Somewhat sure", "Sure", "Very sure"
    ] = dspy.OutputField(desc="The confidence level of the novelty judgment")


class PassageRetrievalSignature(dspy.Signature):
    """
    You are an expert patent examiner trained in European Patent Office (EPO) standards, specifically evaluating novelty under Article 54 EPC.

    Your task is to find the most relevant passages from the closest prior art reference for a single feature of claim 1 of the patent application under examination.

    ## Task Requirements

    1. **Available Information**: You have access to the application under examination and the full closest prior art patent. You will also be provided with a single feature from claim 1 of the application, for which you need to find relevant passages in the prior art.

    2. **Relevant Passages**: For the provided feature of claim 1, identify and list the most relevant passages from the prior art:
       - Identify the 5 most relevant passages from the prior art.
       - Use the relevance field to indicate the nature of the disclosure:
         * "Direct disclosure" - the passage explicitly discloses the feature
         * "Implicit disclosure" - the passage inherently or implicitly discloses the feature
         * "Partial disclosure" - the passage discloses only part of the feature
         * "Contextually relevant" - the passage relates to the feature but doesn't disclose it
         * "Not relevant" - the passage is the most relevant available but still doesn't relate to the feature
       - Sort the passages in order of relevance, with the most relevant first.
       - Even if the feature is not disclosed at all, list the passages that are closest to it (marking them with appropriate relevance labels).

    3. **Level of disclosure**: Additionally, provide an overall disclosure level for the feature based on the passages found:
       - "Fully disclosed" - all aspects of the feature are clearly disclosed in the prior art
       - "Partially disclosed" - some aspects of the feature are disclosed, but not all
       - "Not disclosed" - the feature is not disclosed in the prior art

    Provide your findings in the following format:
    ```py
    Output(
        relevant_passages=FeatureWithRelevantPassages(
            feature="...",
            relevant_passages=[
                PassageReference(
                    kind="...",
                    number=...,
                    relevance="..."
                ),
            ],
            disclosed="..."
        )
    )
    ```
    """

    closest_prior_art: str = dspy.InputField(desc="The closest prior art reference")
    claim_under_examination: str = dspy.InputField(
        desc="The text of claim 1 of the patent application being examined"
    )
    feature_to_evaluate: str = dspy.InputField(
        desc="The feature from claim 1 of the patent application to find relevant passages for"
    )
    relevant_passages: FeatureWithRelevantPassages = dspy.OutputField()


class NoveltyExaminationSignatureWithSummaries(dspy.Signature):
    """
    You are an expert patent examiner trained in European Patent Office (EPO) standards, specifically evaluating novelty under Article 54 EPC.

    Your task is to determine whether claim 1 of the patent application under examination is novel over the closest prior art reference. This assessment concerns ONLY novelty (anticipation), NOT inventive step (obviousness). An application may lack inventive step but still be considered novel.

    ## Task Requirements

    1. **Focus on Claim 1**: Examine only the first independent claim of the patent application.

    2. **Novelty Standard (Article 54 EPC)**: A claim lacks novelty if all of its technical features are explicitly or implicitly disclosed in a prior art document. The prior art must directly and unambiguously disclose every element of the claim.

    3. **Feature Examination Summaries**: You have access to summaries of the examination of each feature of claim 1 against the prior art. Use these summaries to inform your overall novelty determination but critically evaluate their content.

    4. **Detailed Analysis**: Perform a thorough element-by-element comparison:
       - Identify each feature in claim 1
       - Determine whether each feature is disclosed in the prior art
       - Consider both explicit and implicit disclosures
       - Account for different terminology describing the same technical concept
       - Apply EPO principles for inherent disclosures and genus-species relationships

    5. **Relevant Passages**: You have previously identified the relevant passages from the prior art for claim 1.

    6. **Confidence Assessment**: To differentiate between cases that are clearly novel or not novel, and those that are borderline, select a confidence score from 'Very unsure' to 'Very sure' based.

    Provide your novelty determination and confidence level based on rigorous examination principles. Remember, your objective is to find a way to reasonably reject the claim for lack of novelty. Broadly interpret the prior art, and grant the claim only if it is unquestionably novel.
    """

    claim_under_examination: str = dspy.InputField(
        desc="The text of claim 1 of the patent application being examined"
    )
    closest_prior_art: str = dspy.InputField(desc="The closest prior art reference")
    feature_examination_summaries: tuple[str, str] = dspy.InputField(
        desc="Summaries of the examination of each feature of claim 1 against the prior art"
    )
    novelty: Literal["Novel", "Not novel"] = dspy.OutputField(
        desc="Whether the claim is novel over the prior art"
    )
    confidence: Literal[
        "Very unsure", "Unsure", "Somewhat sure", "Sure", "Very sure"
    ] = dspy.OutputField(desc="The confidence level of the novelty judgment")


class FeatureWithRelevantPassagesWithSummary(BaseModel):
    feature: str
    relevant_passages: list[PassageReference]
    disclosed: Literal["Fully disclosed", "Partially disclosed", "Not disclosed"]
    examination_summary: str


class PassageRetrievalSignatureWithSummary(dspy.Signature):
    """
    You are an expert patent examiner trained in European Patent Office (EPO) standards, specifically evaluating novelty under Article 54 EPC.

    Your task is to find the most relevant passages from the closest prior art reference for a single feature of claim 1 of the patent application under examination.

    ## Task Requirements

    1. **Available Information**: You have access to the application under examination and the full closest prior art patent. You will also be provided with a single feature from claim 1 of the application, for which you need to find relevant passages in the prior art.

    2. **Relevant Passages**: For the provided feature of claim 1, identify and list the most relevant passages from the prior art:
       - Identify the 5 most relevant passages from the prior art.
       - Use the relevance field to indicate the nature of the disclosure:
         * "Direct disclosure" - the passage explicitly discloses the feature
         * "Implicit disclosure" - the passage inherently or implicitly discloses the feature
         * "Partial disclosure" - the passage discloses only part of the feature
         * "Contextually relevant" - the passage relates to the feature but doesn't disclose it
         * "Not relevant" - the passage is the most relevant available but still doesn't relate to the feature
       - Sort the passages in order of relevance, with the most relevant first.
       - Even if the feature is not disclosed at all, list the passages that are closest to it (marking them with appropriate relevance labels).

    3. **Level of disclosure**: Additionally, provide an overall disclosure level for the feature based on the passages found:
       - "Fully disclosed" - all aspects of the feature are clearly disclosed in the prior art
       - "Partially disclosed" - some aspects of the feature are disclosed, but not all
       - "Not disclosed" - the feature is not disclosed in the prior art

    Provide your findings in the following format:
    ```py
    Output(
        relevant_passages=FeatureWithRelevantPassagesWithSummary(
            feature="...",
            relevant_passages=[
                PassageReference(
                    kind="...",
                    number=...,
                    relevance="..."
                ),
            ],
            disclosed="...",
            examination_summary="..."
        )
    )
    ```
    """

    closest_prior_art: str = dspy.InputField(desc="The closest prior art reference")
    claim_under_examination: str = dspy.InputField(
        desc="The text of claim 1 of the patent application being examined"
    )
    feature_to_evaluate: str = dspy.InputField(
        desc="The feature from claim 1 of the patent application to find relevant passages for"
    )
    relevant_passages: FeatureWithRelevantPassagesWithSummary = dspy.OutputField()


class ClaimSplittingSignature(dspy.Signature):
    """
    You are an expert patent examiner trained in European Patent Office (EPO) standards, specifically evaluating novelty under Article 54 EPC.

    Your task is to split the text of claim 1 of the patent application under examination into its individual technical features for detailed novelty analysis.

    ## Task Requirements

    1. **Feature Identification**: Identify and extract each distinct technical feature from claim 1.
    2. **Output Format**: Provide the extracted features as a list of strings, where each string represents a single feature.
    3. **Completeness**: Ensure the the concatenation of all extracted features fully represents the original claim text without omission except for whitespace or filler words like 'and'.
    """

    claim_under_examination: str = dspy.InputField(
        desc="The text of claim 1 of the patent application being examined"
    )
    features: list[str] = dspy.OutputField(
        desc="The list of individual technical features extracted from claim 1"
    )


class HierarchicalExamination(BaseNoveltyExamination):
    def __init__(
        self,
        offer_money: bool = True,
        use_label_references: bool = False,
        split_using_llm: bool = False,
        with_examination_summaries: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(offer_money=offer_money, **kwargs)
        self.use_label_references = use_label_references
        self.split_using_llm = split_using_llm
        self.with_examination_summaries = with_examination_summaries

        NoveltyExaminationSignature_ = copy_signature(
            NoveltyExaminationSignature
            if not with_examination_summaries
            else NoveltyExaminationSignatureWithSummaries
        )
        if self.offer_money:
            NoveltyExaminationSignature_.instructions += (
                "You will get paid $10k for every rejected claim, so be thorough!"
            )

        self.predict = dspy.Predict(signature=NoveltyExaminationSignature_)
        self.retrieve = dspy.Predict(
            signature=(
                PassageRetrievalSignature
                if not with_examination_summaries
                else PassageRetrievalSignatureWithSummary
            )
        )
        self.split = dspy.Predict(signature=ClaimSplittingSignature)

    def forward(
        self, sample: ApplicationData, granted: bool = False
    ) -> NoveltyExaminationOutput:
        raise NotImplementedError(
            "Synchronous execution is not supported, use 'aforward' instead."
        )

    async def aforward(
        self, sample: ApplicationData, granted: bool = False
    ) -> NoveltyExaminationOutput:
        pue = sample.rejected if not granted else sample.granted
        cue = pue.claims[0]
        summaries = []
        assert cue

        if self.use_label_references:
            relevant_passages_per_feature: list[FeatureWithRelevantPassages] = [
                FeatureWithRelevantPassages(
                    feature=feature.feature,
                    relevant_passages=[
                        PassageReference(
                            kind=passage.reference[0],  # type: ignore
                            number=passage.reference[1],
                            relevance="Direct disclosure",
                        )
                        for passage in feature.prior_art_passages
                    ],
                    disclosed="Fully disclosed",
                )
                for feature in sample.breakdown.breakdown or []
            ]
            selected_paragraphs = [
                passage.reference[1] - 1
                for feature in sample.breakdown.breakdown or []
                for passage in feature.prior_art_passages
                if passage.reference[0] == "paragraph"
                and passage.reference[1] is not None
            ]
        else:
            if self.split_using_llm:
                split_result = await self.split.acall(
                    claim_under_examination=cue,
                )
                features = split_result.features
            else:
                features = [f.strip() for f in re.split(r"[;\n]", cue) if f.strip()]
            relevant_passages_per_feature_ = await asyncio.gather(
                *[
                    self.retrieve.acall(
                        claim_under_examination=cue,
                        feature_to_evaluate=feature,
                        closest_prior_art=self.format_patent(sample.cited),
                    )
                    for feature in features
                ]
            )
            relevant_passages_per_feature: list[FeatureWithRelevantPassages] = [
                result.relevant_passages for result in relevant_passages_per_feature_
            ]
            selected_paragraphs = [
                passage.number - 1
                for feature_passages in relevant_passages_per_feature
                if feature_passages.relevant_passages
                for passage in feature_passages.relevant_passages
                if passage.kind == "paragraph" and passage.number is not None
            ]
            if self.with_examination_summaries:
                summaries = [
                    (fp.feature, fp.examination_summary)  # type: ignore
                    for fp in relevant_passages_per_feature
                ]
                relevant_passages_per_feature = [
                    FeatureWithRelevantPassages(
                        feature=fp.feature,
                        relevant_passages=fp.relevant_passages,
                        disclosed=fp.disclosed,
                    )
                    for fp in relevant_passages_per_feature
                ]

        prediction = await self.predict.acall(
            claim_under_examination=cue,
            closest_prior_art=self.format_patent(
                sample.cited, paragraphs=selected_paragraphs
            ),
            **(
                {"feature_examination_summaries": summaries}
                if self.with_examination_summaries
                else {}
            ),
        )
        return NoveltyExaminationOutput.model_validate(
            {
                **prediction._store,
                "relevant_passages_per_feature": relevant_passages_per_feature,
            }
        )
