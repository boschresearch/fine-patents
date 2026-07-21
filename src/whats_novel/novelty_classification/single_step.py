import asyncio
from collections import Counter
from typing import Iterable, Literal, TypeVar

import dspy

from whats_novel.dataset import ApplicationData
from whats_novel.novelty_classification.evaluate import edit_similarity
from whats_novel.novelty_classification.shared import (
    BaseNoveltyExamination,
    FeatureWithRelevantPassages,
    NoveltyExaminationOutput,
    PassageReference,
    copy_signature,
)
from whats_novel.utils import get_logger

logger = get_logger(__name__)


class NoveltyExaminationSignature(dspy.Signature):
    """
    You are a rigorous patent examiner trained in European Patent Office (EPO) standards, specifically evaluating novelty under Article 54 EPC. Your role is to be skeptical of novelty and carefully scrutinize whether the prior art anticipates the claimed invention.

    Your task is to determine whether claim 1 of the patent application under examination is novel over the closest prior art reference.

    ## Examination Approach

    **Interpret broadly**: Construe the prior art disclosure broadly and generously. Look beyond exact terminology to understand the underlying technical concepts. The prior art should be read through the eyes of a skilled person who understands the technical field.

    **Emphasize implicit disclosure**: A feature is disclosed not only when explicitly stated, but also when it would be understood by a skilled person as inherently present or necessarily flowing from what is described. If implementing the prior art would inevitably include the claimed feature, that feature is disclosed.

    ## Task Requirements

    1. **Focus on Claim 1**: You are given only the first independent claim of the patent application.

    2. **Novelty Standard (Article 54 EPC)**: A claim lacks novelty if all of its technical features are explicitly or implicitly disclosed in a prior art document. A feature is anticipated if a skilled person reading the prior art would understand it as disclosing that feature, whether explicitly stated or necessarily implied by the disclosure.

    3. **Apply Abstract Reasoning**: This task requires highly abstract analysis to identify equivalent disclosures:
       - Look beyond specific terminology to identify the same technical concepts expressed differently
       - Consider functional equivalents: if a feature describes a function or result, the prior art discloses it if it achieves the same function or result, even by different means
       - Example: for "real-time synchronization module", look for any mechanism that keeps data consistent across systems, such as "continuous update protocol", "live data mirroring", or "synchronized database"
       - Example: for "authentication token", look for any credential mechanism like "session key", "security certificate", or "user identifier"
       - Be creative in identifying conceptual overlaps - different words often describe the same technical reality

    4. **Element-by-Element Comparison**: Perform a thorough analysis:
       - Identify each feature in claim 1. The concatenation of all features should reconstruct the full text of claim 1. Include the preamble.
       - For each feature, actively search for both explicit and implicit disclosures in the prior art
       - When you find potential matches, explain how they correspond even if terminology differs

    5. **Relevant Passages**: For each feature of claim 1, identify and list the most relevant passages from the prior art:
       - Always list all features from claim 1.
       - For each feature, always identify at least 5 of the most relevant passages from the prior art.
       - Use the relevance field to indicate the nature of the disclosure:
         * "Direct disclosure" - the passage explicitly discloses the feature
         * "Implicit disclosure" - the passage inherently or implicitly discloses the feature
         * "Partial disclosure" - the passage discloses only part of the feature
         * "Contextually relevant" - the passage relates to the feature but doesn't disclose it
         * "Not relevant" - the passage is the most relevant available but still doesn't relate to the feature
       - Sort the passages in order of relevance, with the most relevant first.
       - Even if a feature is not disclosed at all, list the passages that are closest to it (marking them as "Contextually relevant" or "Not relevant").

    6. **Confidence Assessment**: To differentiate between cases that are clearly novel or not novel, and those that are borderline, select a confidence score from 'Very unsure' to 'Very sure'.

    Provide your novelty determination and confidence level based on rigorous examination principles.
    """

    claim_under_examination: str = dspy.InputField(
        desc="The text of claim 1 of the patent application being examined"
    )
    closest_prior_art: str = dspy.InputField(desc="The closest prior art reference")
    relevant_passages_per_feature: list[FeatureWithRelevantPassages] = dspy.OutputField(
        desc="The relevant passages from the prior art per feature of claim 1"
    )
    novelty: Literal["Novel", "Not novel"] = dspy.OutputField(
        desc="Whether the claim is novel over the prior art"
    )
    confidence: Literal[
        "Very unsure", "Unsure", "Somewhat sure", "Sure", "Very sure"
    ] = dspy.OutputField(desc="The confidence level of the novelty judgment")


class NoveltyExaminationSignatureConstruction(dspy.Signature):
    """
    You are a rigorous patent examiner trained in European Patent Office (EPO) standards, specifically evaluating novelty under Article 54 EPC. Your role is to be skeptical of novelty and carefully scrutinize whether the prior art anticipates the claimed invention.

    Your task is to determine whether claim 1 of the patent application under examination is novel over exactly one prior art document.

    You are given:
    - Claim 1 under examination.
    - One prior art document (the closest prior art).

    ## Examination Approach

    ### 1) Break claim 1 into individual technical features
    - Split claim 1 into a complete set of individual features.
    - The concatenation of all features must reconstruct the full wording of claim 1, including the preamble.
    - Treat every feature as requiring interpretation: each feature must be construed.

    ### 2) Claim construction

    - The wording of claim 1 is decisive.
    - Interpret the claim from the perspective of the skilled person in the art.

    For each feature:
    - Determine its technical meaning in context.
    - Ask:
    - What technical function or effect does this feature define?
    - What would a skilled person reasonably understand this wording to mean?
    - Is the feature structural, functional, or purpose-defined?

    If a term is unclear or broad:
    - Construe it broadly.
    - Ambiguity generally results in a broad (non-restrictive) interpretation.

    Define the skilled person where useful:
    - Type of technical education (e.g., academic or non-academic background).
    - Practical experience (multi-year or long-standing experience in the relevant field).
    - Any specific technical expertise relevant to the subject-matter.

    Non-technical features cannot establish novelty.

    ### 3) Read the prior art broadly, but apply the correct novelty standard

    Read the prior art document broadly and generously:
    - Look beyond exact terminology to the underlying technical teaching.
    - Identify equivalent technical concepts expressed in different language.

    However, apply the strict novelty standard:

    A claim lacks novelty if all of its technical features (as construed) are directly and unambiguously disclosed in the prior art document, either explicitly or implicitly.

    Explicit disclosure:
    - The feature is clearly described using the same or synonymous terminology.

    Implicit disclosure:
    - The feature is necessarily and inevitably present in what is disclosed.
    - The skilled person would immediately recognize that nothing else could be meant.
    - Do not supplement missing features using general knowledge.

    ### 4) Multiple embodiments in the prior art

    - If the prior art document discloses multiple distinct embodiments or alternative implementations, assess novelty separately for each embodiment.
    - If the prior art explicitly and unambiguously incorporates another document by reference, include that content only to the extent that it is clearly incorporated.

    ### 5) Generic vs. specific disclosures

    Apply the following principles:
    - A specific disclosed embodiment can anticipate a generic claim.
    - A generic disclosure does not anticipate a specific claimed embodiment unless the specific variant is directly and unambiguously disclosed.

    ### 6) Abstract reasoning and functional equivalence

    This task requires highly abstract analysis to identify equivalent disclosures:
    - Look beyond terminology to identify the same technical concept expressed differently.
    - Consider functional equivalents: if a feature defines a function or result, the prior art discloses it if it achieves the same function or result, even by different means — but only if that function or result is directly and unambiguously disclosed in the prior art.

    Examples:
    - “Real-time synchronization module” may correspond to any mechanism that keeps data consistent across systems (continuous update protocol, live data mirroring, synchronized database).
    - “Authentication token” may correspond to any credential mechanism (session key, security certificate, user identifier).

    Be creative in identifying conceptual overlaps, but do not fill gaps with speculation or external knowledge.

    ## Task Requirements

    1. Focus exclusively on the provided claim 1.

    2. Element-by-element comparison:
    - Identify each feature in claim 1, including the preamble.
    - For each feature, search for explicit and implicit disclosures in the prior art.
    - Explain how each identified passage corresponds to the feature, even if terminology differs.

    3. Relevant passages:
    - Always list all features from claim 1.
    - For each feature, identify at least 5 of the most relevant passages from the prior art.
    - For each passage, provide a relevance label:
        * "Direct disclosure" — explicitly discloses the feature
        * "Implicit disclosure" — directly and unambiguously implicit
        * "Partial disclosure" — discloses only part of the feature
        * "Contextually relevant" — related but not a disclosure
        * "Not relevant" — closest available but still unrelated
    - Sort passages by relevance, most relevant first.
    - Even if a feature is not disclosed at all, list the closest passages and label them accordingly.
    - Select references from a single embodiment of the prior art document whenever possible.

    4. Novelty determination:
    - Determine whether claim 1 is novel over the prior art document.
    - Novelty exists if at least one construed technical feature is not directly and unambiguously disclosed (explicitly or implicitly) in a given embodiment.
    - If different embodiments lead to different results, explain this clearly.

    5. Confidence assessment:
    - Select a confidence level: "Very unsure", "Unsure", "Somewhat sure", "Sure", or "Very sure".

    Provide your novelty determination and confidence level based on rigorous examination principles.
    """

    claim_under_examination: str = dspy.InputField(
        desc="The text of claim 1 of the patent application being examined"
    )
    closest_prior_art: str = dspy.InputField(desc="The closest prior art reference")
    relevant_passages_per_feature: list[FeatureWithRelevantPassages] = dspy.OutputField(
        desc="The relevant passages from the prior art per feature of claim 1"
    )
    novelty: Literal["Novel", "Not novel"] = dspy.OutputField(
        desc="Whether the claim is novel over the prior art"
    )
    confidence: Literal[
        "Very unsure", "Unsure", "Somewhat sure", "Sure", "Very sure"
    ] = dspy.OutputField(desc="The confidence level of the novelty judgment")


class SingleStepNoveltyExamination(BaseNoveltyExamination):
    def __init__(self, offer_money: bool = True, **kwargs) -> None:
        super().__init__(offer_money=offer_money, **kwargs)
        NoveltyExaminationSignature_ = copy_signature(NoveltyExaminationSignature)
        if self.offer_money:
            NoveltyExaminationSignature_.instructions += (
                " Remember, your objective is to find a way to reasonably reject the claim for lack of novelty. "
                "Broadly interpret the prior art, and grant the claim only if it is unquestionably novel. "
                "You will get paid $10k for every rejected claim, so be thorough!"
            )
        self.predict = dspy.Predict(signature=NoveltyExaminationSignature_)

    def forward(
        self, sample: ApplicationData, granted: bool = False
    ) -> NoveltyExaminationOutput:
        cue = sample.rejected.claims[0] if not granted else sample.granted.claims[0]
        prediction = self.predict(
            claim_under_examination=cue,
            closest_prior_art=self.format_patent(sample.cited),
        )
        return NoveltyExaminationOutput.model_validate(prediction._store)

    async def aforward(
        self, sample: ApplicationData, granted: bool = False
    ) -> NoveltyExaminationOutput:
        cue = sample.rejected.claims[0] if not granted else sample.granted.claims[0]
        prediction = await self.predict.acall(
            claim_under_examination=cue,
            closest_prior_art=self.format_patent(sample.cited),
        )
        return NoveltyExaminationOutput.model_validate(prediction._store)


class SingleStepNoveltyExaminationSC(SingleStepNoveltyExamination):
    def __init__(self, n_shots: int = 5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_shots = n_shots
        assert self.n_shots % 2 != 0

    def forward(
        self, sample: ApplicationData, granted: bool = False
    ) -> NoveltyExaminationOutput:
        raise NotImplementedError("sync not implemented.")

    async def aforward(
        self, sample: ApplicationData, granted: bool = False
    ) -> NoveltyExaminationOutput:
        outputs = await asyncio.gather(
            *[super().aforward(sample, granted=granted) for _ in range(self.n_shots)]
        )
        return self.merge_outputs(outputs)

    def merge_outputs(
        self, outputs: list[NoveltyExaminationOutput]
    ) -> NoveltyExaminationOutput:
        T = TypeVar("T")

        def majority(values: Iterable[T]) -> T:
            counts = Counter(values)
            return counts.most_common(1)[0][0]

        def merge_passages(
            passages_lists: list[list[PassageReference]],
        ) -> list[PassageReference]:
            passage_occurences = {}
            for passages in passages_lists:
                for passage in passages:
                    key = (passage.kind, passage.number)
                    if key not in passage_occurences:
                        passage_occurences[key] = []
                    passage_occurences[key].append(passage)

            merged_passages = []
            for key, occurences in passage_occurences.items():
                if len(occurences) > self.n_shots // 2:
                    relevance = majority(
                        occurence.relevance for occurence in occurences
                    )
                    merged_passages.append(
                        PassageReference(
                            kind=key[0], number=key[1], relevance=relevance
                        )
                    )
            return merged_passages

        majority_novelty = majority(output.novelty for output in outputs)
        majority_outputs = [
            output for output in outputs if output.novelty == majority_novelty
        ]
        majority_confidence = majority(
            [output.confidence for output in majority_outputs]
        )
        # we use the feature segmentation of the first output,
        # then assign the other outputs' passages to the most similar feature
        feature_mapping = [
            [f] for f in majority_outputs[0].relevant_passages_per_feature
        ]
        for output in majority_outputs[1:]:
            for feature in output.relevant_passages_per_feature:
                best_feature_i, _ = max(
                    [
                        (i, f)
                        for i, f in enumerate(
                            majority_outputs[0].relevant_passages_per_feature
                        )
                    ],
                    key=lambda f: edit_similarity(f[1].feature, feature.feature),
                )
                feature_mapping[best_feature_i].append(feature)

        relevant_passages = [
            FeatureWithRelevantPassages(
                feature=features[0].feature,
                relevant_passages=merge_passages(
                    [f.relevant_passages for f in features]
                ),
                disclosed=majority([f.disclosed for f in features]),
            )
            for features in feature_mapping
        ]
        return NoveltyExaminationOutput(
            novelty=majority_novelty,  # type: ignore
            confidence=majority_confidence,  # type: ignore
            relevant_passages_per_feature=relevant_passages,
        )


class SingleStepNoveltyExaminationConstruction(SingleStepNoveltyExamination):
    def __init__(self, offer_money: bool = True, **kwargs) -> None:
        super().__init__(offer_money=offer_money, **kwargs)
        NoveltyExaminationSignature_ = copy_signature(
            NoveltyExaminationSignatureConstruction
        )
        if self.offer_money:
            NoveltyExaminationSignature_.instructions += (
                " Remember, your objective is to find a way to reasonably reject the claim for lack of novelty. "
                "Broadly interpret the prior art, and grant the claim only if it is unquestionably novel. "
                "You will get paid $10k for every rejected claim, so be thorough!"
            )
        self.predict = dspy.Predict(signature=NoveltyExaminationSignature_)


class NoveltyExaminationSignatureClaimOnly(dspy.Signature):
    """
    You are a rigorous patent examiner trained in European Patent Office (EPO) standards, specifically evaluating novelty under Article 54 EPC. Your role is to be skeptical of novelty and carefully scrutinize whether the prior art anticipates the claimed invention.

    Your task is to determine whether claim 1 of the patent application under examination is novel.

    ## Task Requirements

    1. **Focus on Claim 1**: You are given only the first independent claim of the patent application.

    2. **Novelty Standard (Article 54 EPC)**: A claim lacks novelty if all of its technical features are explicitly or implicitly disclosed in a prior art document. A feature is anticipated if a skilled person reading a prior art would understand it as disclosing that feature, whether explicitly stated or necessarily implied by the disclosure.

    3. **Confidence Assessment**: To differentiate between cases that are clearly novel or not novel, and those that are borderline, select a confidence score from 'Very unsure' to 'Very sure'.

    Provide your novelty determination and confidence level based on rigorous examination principles.
    """

    claim_under_examination: str = dspy.InputField(
        desc="The text of claim 1 of the patent application being examined"
    )
    novelty: Literal["Novel", "Not novel"] = dspy.OutputField(
        desc="Whether the claim is novel over the prior art"
    )
    confidence: Literal[
        "Very unsure", "Unsure", "Somewhat sure", "Sure", "Very sure"
    ] = dspy.OutputField(desc="The confidence level of the novelty judgment")


class SingleStepNoveltyExaminationClaimOnly(BaseNoveltyExamination):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.predict = dspy.Predict(signature=NoveltyExaminationSignatureClaimOnly)

    def forward(
        self, sample: ApplicationData, granted: bool = False
    ) -> NoveltyExaminationOutput:
        cue = sample.rejected.claims[0] if not granted else sample.granted.claims[0]
        prediction = self.predict(
            claim_under_examination=cue,
        )
        return NoveltyExaminationOutput.model_validate(
            {**prediction._store, "relevant_passages_per_feature": []}
        )

    async def aforward(
        self, sample: ApplicationData, granted: bool = False
    ) -> NoveltyExaminationOutput:
        cue = sample.rejected.claims[0] if not granted else sample.granted.claims[0]
        prediction = await self.predict.acall(
            claim_under_examination=cue,
        )
        return NoveltyExaminationOutput.model_validate(
            {**prediction._store, "relevant_passages_per_feature": []}
        )
