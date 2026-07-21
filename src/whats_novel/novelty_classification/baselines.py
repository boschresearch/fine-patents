# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: MIT

import asyncio
from pathlib import Path
import random
import re
import pickle

import dspy
import numpy as np
import openai
import torch
from lxml import etree as etree_lxml
from pyrootutils import setup_root
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset

from whats_novel.dataset import ApplicationData
from whats_novel.novelty_classification.shared import (
    BaseNoveltyExamination,
    FeatureWithRelevantPassages,
    NoveltyExaminationOutput,
    PassageReference,
)
from whats_novel.utils import get_logger

logger = get_logger(__name__)
root = setup_root(__file__)


class SpuriousLLM(BaseNoveltyExamination):
    class Sig(dspy.Signature):
        """Determine whether the patent claim is from a granted patent or from a pre-grant patent application."""

        claim: str = dspy.InputField(description="The claim text to classify")
        is_granted: bool = dspy.OutputField(
            description="Whether the patent claim is from a granted patent"
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predict = dspy.Predict(self.Sig)

    async def aforward(
        self, sample: ApplicationData, granted: bool = False
    ) -> NoveltyExaminationOutput:
        cue = sample.rejected.claims[0] if not granted else sample.granted.claims[0]
        novel = (await self.predict.acall(claim=cue)).is_granted

        return NoveltyExaminationOutput(
            relevant_passages_per_feature=[],
            novelty="Novel" if novel else "Not novel",
            confidence="Very sure",
        )


class RandomNoveltyExamination(BaseNoveltyExamination):
    predict = dspy.Predict("a -> b")  # our main script expects a 'predict' attribute

    async def aforward(
        self, sample: ApplicationData, granted: bool = False
    ) -> NoveltyExaminationOutput:
        valid_references = [
            *[
                ("paragraph", i + 1)
                for i in range(len(sample.cited.description))
                if sample.cited.description[i]
            ],
            *[
                ("claim", i + 1)
                for i in range(len(sample.cited.claims))
                if sample.cited.claims[i]
            ],
            ("abstract", None),
        ]

        return NoveltyExaminationOutput(
            relevant_passages_per_feature=[
                FeatureWithRelevantPassages(
                    feature=feature.feature,
                    relevant_passages=[
                        PassageReference(
                            kind=kind,
                            number=number,
                            relevance=random.choice(
                                [
                                    "Direct disclosure",
                                    "Implicit disclosure",
                                    "Partial disclosure",
                                    "Contextually relevant",
                                    "Not relevant",
                                ]
                            ),
                        )
                        for kind, number in random.sample(valid_references, k=5)
                    ],
                    disclosed=random.choice(
                        ["Fully disclosed", "Partially disclosed", "Not disclosed"]
                    ),
                )
                for feature in sample.breakdown.breakdown or []
            ],
            novelty=random.choice(["Novel", "Not novel"]),
            confidence=random.choice(
                ["Very unsure", "Unsure", "Somewhat sure", "Sure", "Very sure"]
            ),
        )


class LogReg(BaseNoveltyExamination):
    predict = dspy.Predict("a -> b")  # our main script expects a 'predict' attribute

    def get_spurious_features(self, claim: str, app_num: str) -> dict[str, float]: ...

    async def train(self, samples: list[dict], run_dir: Path) -> None:
        x_d = [
            self.get_spurious_features(
                s["sample"].rejected.claims[0]
                if not s["granted"]
                else s["sample"].granted.claims[0],
                s["sample"].app_num,
            )
            for s in samples
        ]
        x = [list(feat.values()) for feat in x_d]
        y = [1 if s["granted"] else 0 for s in samples]

        self.scaler = StandardScaler().fit(x)
        x_scaled = self.scaler.transform(x)

        self.model = LogisticRegression().fit(x_scaled, y)
        model_path = run_dir / "logreg_model.pkl"
        with model_path.open("wb") as f:
            pickle.dump(
                {
                    "coef": self.model.coef_,
                    "intercept": self.model.intercept_,
                    "feature_names": list(x_d[0].keys()),
                },
                f,
            )

    async def aforward(
        self, sample: ApplicationData, granted: bool = False
    ) -> NoveltyExaminationOutput:
        cue = sample.rejected.claims[0] if not granted else sample.granted.claims[0]
        assert cue
        features = self.scaler.transform(
            [list(self.get_spurious_features(cue, sample.app_num).values())]
        )[0]
        novelty = self.model.predict(np.array([features]))[0]
        novelty = "Novel" if novelty else "Not novel"

        return NoveltyExaminationOutput(
            relevant_passages_per_feature=[
                FeatureWithRelevantPassages(
                    feature=feature.feature,
                    relevant_passages=[],
                    disclosed="Not disclosed",
                )
                for feature in sample.breakdown.breakdown or []
            ],
            novelty=novelty,
            confidence="Somewhat sure",
        )


class LogRegLength(LogReg):
    predict = dspy.Predict("a -> b")  # our main script expects a 'predict' attribute

    def get_spurious_features(self, claim: str, app_num: str) -> dict[str, float]:
        return {
            "claim_length": len(claim.split()),
        }


class LogRegSpurious(LogReg):
    predict = dspy.Predict("a -> b")  # our main script expects a 'predict' attribute

    async def train(self, samples: list[dict], run_dir: Path) -> None:
        self.tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 4))
        self.tfidf.fit(
            [
                s["sample"].rejected.claims[0]
                if not s["granted"]
                else s["sample"].granted.claims[0]
                for s in samples
            ]
        )
        await super().train(samples, run_dir)

    def get_ipcr(self, app_num: str) -> list[tuple[str, int | None]]:
        pub_path = root / "data" / "samples" / app_num / "publications"
        try:
            xml_path = next(pub_path.glob("*A1.xml"))
        except StopIteration:
            return []
        xml_content = xml_path.read_text()
        tree = etree_lxml.fromstring(xml_content.encode("utf-8"))

        return list(
            set(
                [
                    cls.split()[0][:4]
                    for cls in tree.xpath(".//classification-ipcr/text/text()")  # type: ignore
                ]
            )
        )

    def get_spurious_features(self, claim: str, app_num: str) -> dict[str, float]:
        features = {
            "num_words_claim_1": len(claim.split()),
            "num_features": len(re.split(r"[;\n]", claim)),
            **{
                f"num_{c}": claim.lower().count(c)
                for c in (",", ";", ":", ".", "(", ")")
            },
            "num_ref_numerals": len(
                re.findall(r"(\([\da-zA-Z]+(?:[,-]\s*[\da-zA-Z]+)*\))", claim)
            ),
            **{
                f"cpc_{c}": int(c in self.get_ipcr(app_num))
                for c in [
                    "G06F",
                    "H04L",
                    "H04N",
                    "H04W",
                    "G06K",
                    "H04M",
                    "G06T",
                    "G06Q",
                    "G10L",
                    "A61B",
                ]
            },
        }
        tf_idf_features = self.tfidf.transform([claim]).toarray()[0].tolist()  # type: ignore
        for token, value in zip(self.tfidf.get_feature_names_out(), tf_idf_features):
            features[f"tfidf_{token}"] = value

        return features  # type: ignore


class EmbeddingSimilarity(BaseNoveltyExamination):
    predict = dspy.Predict("a -> b")  # our main script expects a 'predict' attribute
    client = openai.Client(
        base_url="http://rng-dl01-w015:8000/v1",
        api_key="EMPTY",
    )
    model = "Qwen/Qwen3-Embedding-8B"

    def __init__(self, threshold: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def format_query(self, text: str) -> str:
        """Recommended by Qwen docs"""
        return (
            "Instruct: Given a feature of a patent claim, retrieve relevant "
            "passages from a prior art patent that disclose the feature.\n"
            f"Query: {text}"
        )

    async def aforward(
        self, sample: ApplicationData, granted: bool = False
    ) -> NoveltyExaminationOutput:
        cue = sample.rejected.claims[0] if not granted else sample.granted.claims[0]
        assert cue

        valid_references = [
            *[
                ("paragraph", i + 1, sample.cited.description[i])
                for i in range(len(sample.cited.description))
                if sample.cited.description[i]
            ],
            *[
                ("claim", i + 1, sample.cited.claims[i])
                for i in range(len(sample.cited.claims))
                if sample.cited.claims[i]
            ],
            ("abstract", None, sample.cited.abstract),
        ]

        features = [f.strip() for f in re.split(r"[;\n]", cue) if len(f) > 20]

        try:
            queries = [self.format_query(feature) for feature in features]
            documents = [doc for _, _, doc in valid_references]

            outputs = self.client.embeddings.create(
                input=queries + documents, model=self.model
            )
            query_embeddings = torch.tensor(
                [o.embedding for o in outputs.data[: len(queries)]]
            )
            doc_embeddings = torch.tensor(
                [o.embedding for o in outputs.data[len(queries) :]]
            )
            scores = query_embeddings @ doc_embeddings.T

        except Exception:
            logger.exception(f"[{sample.app_num}] Unexpected error")
            scores = torch.zeros((len(features), len(valid_references)))

        output = NoveltyExaminationOutput(
            relevant_passages_per_feature=[
                FeatureWithRelevantPassages(
                    feature=features[i],
                    relevant_passages=[
                        PassageReference(
                            kind=valid_references[j][0],
                            number=valid_references[j][1],
                            relevance=(
                                "Direct disclosure"
                                if scores[i, j] >= self.threshold
                                else "Not relevant"
                            ),
                        )
                        for j in scores[i].topk(k=5).indices.tolist()
                    ],
                    disclosed=(
                        "Fully disclosed"
                        if (scores[i] >= self.threshold).any()
                        else "Not disclosed"
                    ),
                )
                for i in range(len(features))
            ],
            novelty=(
                "Not novel"
                if (scores[1:].max(dim=1).values >= self.threshold).all()
                else "Novel"
            ),
            confidence="Very sure",
        )
        await asyncio.sleep(0.01)
        return output


class RougeSimilarity(BaseNoveltyExamination):
    predict = dspy.Predict("a -> b")  # our main script expects a 'predict' attribute

    def __init__(self, threshold: float = 0.4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.rouge = rouge_scorer.RougeScorer(["rougeL"])

    async def aforward(
        self, sample: ApplicationData, granted: bool = False
    ) -> NoveltyExaminationOutput:
        cue = sample.rejected.claims[0] if not granted else sample.granted.claims[0]
        assert cue

        valid_references = [
            *[
                ("paragraph", i + 1, sample.cited.description[i])
                for i in range(len(sample.cited.description))
                if sample.cited.description[i]
            ],
            *[
                ("claim", i + 1, sample.cited.claims[i])
                for i in range(len(sample.cited.claims))
                if sample.cited.claims[i]
            ],
            ("abstract", None, sample.cited.abstract),
        ]

        features = [f.strip() for f in re.split(r"[;\n]", cue) if len(f) > 20]
        passages = [doc for _, _, doc in valid_references]
        scores = torch.tensor(
            [
                [
                    self.rouge.score(prediction=passage, target=feature)[
                        "rougeL"
                    ].recall
                    for passage in passages
                ]
                for feature in features
            ]
        )

        output = NoveltyExaminationOutput(
            relevant_passages_per_feature=[
                FeatureWithRelevantPassages(
                    feature=features[i],
                    relevant_passages=[
                        PassageReference(
                            kind=valid_references[j][0],
                            number=valid_references[j][1],
                            relevance=(
                                "Direct disclosure"
                                if scores[i, j] >= self.threshold
                                else "Not relevant"
                            ),
                        )
                        for j in scores[i].topk(k=5).indices.tolist()
                    ],
                    disclosed=(
                        "Fully disclosed"
                        if (scores[i] >= self.threshold).any()
                        else "Not disclosed"
                    ),
                )
                for i in range(len(features))
            ],
            novelty=(
                "Not novel"
                if (scores[1:].max(dim=1).values >= self.threshold).all()
                else "Novel"
            ),
            confidence="Very sure",
        )
        await asyncio.sleep(0.01)
        return output


class NoveltyDataset(Dataset):
    def __init__(self, samples: list[dict], tokenizer, max_length: int = 512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        sample: ApplicationData = item["sample"]
        granted = item["granted"]
        claim = sample.rejected.claims[0] if not granted else sample.granted.claims[0]

        encoding = self.tokenizer(
            claim,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        label = 1 if granted else 0
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class BERT(BaseNoveltyExamination):
    predict = dspy.Predict("a -> b")  # our main script expects a 'predict' attribute

    def __init__(
        self,
        model_name: str = "google-bert/bert-base-uncased",
        max_length: int = 512,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.max_length = max_length
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        self.tokenizer = None

    async def train(self, samples: list[dict], run_dir: Path) -> None:
        """Train BERT classifier on novelty classification dataset."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )
        train_dataset = NoveltyDataset(samples, self.tokenizer, self.max_length)
        training_args = TrainingArguments(
            output_dir=str(run_dir / "bert_trained_model"),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=str(run_dir / "training_logs"),
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="no",
            load_best_model_at_end=False,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )
        trainer.train()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    async def aforward(
        self, sample: ApplicationData, granted: bool = False
    ) -> NoveltyExaminationOutput:
        """Make novelty prediction using trained BERT model."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model must be trained before making predictions")

        cue = sample.rejected.claims[0] if not granted else sample.granted.claims[0]
        assert cue
        encoding = self.tokenizer(
            cue,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
            prediction = torch.argmax(outputs.logits, dim=-1).item()

        novelty = "Novel" if prediction == 1 else "Not novel"

        return NoveltyExaminationOutput(
            relevant_passages_per_feature=[
                FeatureWithRelevantPassages(
                    feature=feature.feature,
                    relevant_passages=[],
                    disclosed="Not disclosed"
                    if novelty == "Novel"
                    else "Fully disclosed",
                )
                for feature in sample.breakdown.breakdown or []
            ],
            novelty=novelty,
            confidence="Very sure",
        )
