import json
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import random
from typing import Any

import hydralette as hl
import numpy as np
import pandas as pd
import ranx
from pyrootutils import setup_root
from tqdm import tqdm
from rapidfuzz import fuzz
from rouge_score import rouge_scorer

from whats_novel.dataset import ApplicationData, load_dataset
from whats_novel.dataset.data_model import PatentDocument
from whats_novel.novelty_classification.shared import NoveltyExaminationOutput
from whats_novel.utils import get_logger

logger = get_logger()
root = setup_root(__file__)
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

cfg = hl.Config(
    dataset_path=root / "data" / "packaged",
    runs_path=hl.Field(
        default_factory=lambda: root / "data" / "logs" / "novelty_classification"
    ),
    run_dirs=hl.Field(
        reference=lambda cfg: list(cfg.runs_path.glob("run*")),
        convert=lambda v: [Path(x) for x in v.split(",")],
    ),
)


def safe_div(num: float, den: float) -> float:
    return num / den if den != 0 else 0.0


def compute_metrics_from_confusion(confusion_matrix: pd.DataFrame) -> dict[str, float]:
    tp_novel = float(confusion_matrix.loc["Novel", "Novel"])  # type: ignore[index]
    tp_not_novel = float(confusion_matrix.loc["Not novel", "Not novel"])  # type: ignore[index]
    col_novel = float(confusion_matrix.loc[:, "Novel"].sum())  # type: ignore[index]
    col_not_novel = float(confusion_matrix.loc[:, "Not novel"].sum())  # type: ignore[index]
    row_novel = float(confusion_matrix.loc["Novel", :].sum())  # type: ignore[index]
    row_not_novel = float(confusion_matrix.loc["Not novel", :].sum())  # type: ignore[index]

    precision_novel = safe_div(tp_novel, col_novel)
    recall_novel = safe_div(tp_novel, row_novel)
    precision_not_novel = safe_div(tp_not_novel, col_not_novel)
    recall_not_novel = safe_div(tp_not_novel, row_not_novel)

    f1_novel = safe_div(
        2 * (precision_novel * recall_novel), (precision_novel + recall_novel)
    )
    f1_not_novel = safe_div(
        2 * (precision_not_novel * recall_not_novel),
        (precision_not_novel + recall_not_novel),
    )
    macro_f1 = (f1_novel + f1_not_novel) / 2
    accuracy = safe_div((tp_novel + tp_not_novel), float(confusion_matrix.values.sum()))

    return {
        "metrics/fraction_novel": safe_div(
            col_novel, float(confusion_matrix.values.sum())
        ),
        "metrics/accuracy": accuracy,
        "metrics/precision (novel)": precision_novel,
        "metrics/recall (novel)": recall_novel,
        "metrics/f1 (novel)": f1_novel,
        "metrics/precision (not novel)": precision_not_novel,
        "metrics/recall (not novel)": recall_not_novel,
        "metrics/f1 (not novel)": f1_not_novel,
        "metrics/macro f1": macro_f1,
    }


def edit_similarity(string: str, substring: str) -> float:
    sub = substring.lower()
    s = string.lower()
    if not sub:
        return 0.0
    sm = SequenceMatcher(None, sub, s, autojunk=False)
    matched = sum(block.size for block in sm.get_matching_blocks())
    return matched / len(sub)


def locate_passage(doc: PatentDocument, kind: str, number: int | None) -> str:
    if kind == "paragraph":
        if number is None:
            return ""
        if 0 <= number - 1 < len(doc.description):
            return doc.description[number - 1] or ""
    elif kind == "abstract":
        return doc.abstract or ""
    elif kind == "claim":
        if number is None:
            return ""
        if 0 <= number - 1 < len(doc.claims):
            return doc.claims[number - 1] or ""
    return ""


def compute_retrieval_metrics(
    sample: ApplicationData, output: dict[str, Any]
) -> dict[str, float | None]:
    # --------- claim precision, recall, f1
    predicted_passages = set(
        (passage.get("kind"), passage.get("number"))
        for feature in output.get("relevant_passages_per_feature", []) or []
        for passage in feature.get("relevant_passages", []) or []
        if passage.get("relevance")
        in ("Direct disclosure", "Implicit disclosure", "Partial disclosure")
    )
    label_passages = set(
        passage.reference
        for feature in sample.breakdown.breakdown or []
        for passage in feature.prior_art_passages
    )
    tp = len(predicted_passages & label_passages)
    fp = len(predicted_passages - label_passages)
    fn = len(label_passages - predicted_passages)

    precision = safe_div(tp, tp + fp) if label_passages else None
    recall = safe_div(tp, tp + fn) if label_passages else None
    f1 = (
        safe_div(2 * (precision * recall), (precision + recall))  # type: ignore
        if label_passages
        else None
    )
    hit = 1.0 if tp > 0 else 0.0 if label_passages else None

    # --------- claim soft precision, recall, f1
    predicted_passages_text = [
        passage
        if (passage := locate_passage(sample.cited, kind, number)) and len(passage) > 20
        else None
        for kind, number in predicted_passages
    ]
    label_passages_text = [
        passage
        if (passage := locate_passage(sample.cited, kind, number)) and len(passage) > 20
        else None
        for kind, number in label_passages
    ]
    rouge_scores = np.array([
        [
            (
                rouge.score(prediction=predicted_passage_text, target=label_passage_text)["rougeL"].recall
                if predicted_passage_text is not None and label_passage_text is not None
                else 0.0
            )
            for predicted_passage_text in predicted_passages_text
        ]
        for label_passage_text in label_passages_text
    ])
    rouge_precision = rouge_scores.max(axis=0).mean().item() if rouge_scores.size > 0 else 0.0
    rouge_recall = rouge_scores.max(axis=1).mean().item() if rouge_scores.size > 0 else 0.0
    rouge_f1 = safe_div(
        2 * (rouge_precision * rouge_recall), (rouge_precision + rouge_recall)
    )
    claim_metrics = {
        "retrieval-metrics/precision (claim-level)": precision,
        "retrieval-metrics/recall (claim-level)": recall,
        "retrieval-metrics/f1 (claim-level)": f1,
        "retrieval-metrics/hit (claim-level)": hit,
        "retrieval-metrics/rougeL precision (claim-level)": rouge_precision,
        "retrieval-metrics/rougeL recall (claim-level)": rouge_recall,
        "retrieval-metrics/rougeL f1 (claim-level)": rouge_f1,
    }
    
    # --------- claim ranx metrics
    relevance_stages = {
        "Direct disclosure": 4.,
        "Implicit disclosure": 3.,
        "Partial disclosure": 2.,
        "Contextually relevant": 1.,
        "Not relevant": 0.,
    }
    ranx_metric_names = [
        "map", 
        "ndcg", 
        "ndcg@5", 
        "ndcg@10", 
        "mrr", 
        "precision@5", 
        "precision@10", 
        "precision", 
        "recall@5", 
        "recall@10", 
        "recall",
        "f1",
        "f1@5",
        "f1@10",
    ]


    if label_passages:

        predicted_passages_ranx = {
            f"{passage.get("kind")}_{passage.get("number")}": rel
            for feature in output.get("relevant_passages_per_feature", []) or []
            for passage in feature.get("relevant_passages", []) or []
            if (rel := relevance_stages[passage.get("relevance")]) >= 2
        }
        label_passages_ranx = {
            f"{passage.reference[0]}_{passage.reference[1]}": 1
            for feature in sample.breakdown.breakdown or []
            for passage in feature.prior_art_passages
        }
        
        if predicted_passages_ranx:
            ranx_metrics = ranx.evaluate(
                ranx.Qrels.from_dict({"q1": label_passages_ranx}), 
                ranx.Run.from_dict({"q1": predicted_passages_ranx}),
                metrics=ranx_metric_names
            )
            assert isinstance(ranx_metrics, dict)

            for key, value in ranx_metrics.items():
                claim_metrics[f"retrieval-metrics/ranx/claim/{key}"] = value.item()  # type: ignore
        else:
            for key in ranx_metric_names:
                claim_metrics[f"retrieval-metrics/ranx/claim/{key}"] = 0.0

    else:
        for key in ranx_metric_names:
            claim_metrics[f"retrieval-metrics/ranx/claim/{key}"] = None

    # --------- mapping feature segmentations
    label_predicted_mapping = [
        (
            feature.feature,
            [],
            set(p.reference for p in feature.prior_art_passages),
            set(),
        )
        for feature in sample.breakdown.breakdown or []
    ]
    for pred_feature in output.get("relevant_passages_per_feature", []):
        similarities = [
            (j, edit_similarity(label_feature.feature, pred_feature["feature"]))
            for j, label_feature in enumerate(sample.breakdown.breakdown or [])
        ]
        best_j, best_sim = max(similarities, key=lambda x: x[1])

        predicted_passages = set(
            (passage.get("kind"), passage.get("number"), relevance_stages[passage.get("relevance")])
            for passage in pred_feature.get("relevant_passages", []) or []
            if passage.get("relevance")
            in ("Direct disclosure", "Implicit disclosure", "Partial disclosure")
        )
        for kind, number, relevance in predicted_passages:
            existing_matches = [r_ for k_, n_, r_ in label_predicted_mapping[best_j][3] if (k_, n_) == (kind, number)]
            highest_previous_rel = max(existing_matches) if existing_matches else -1
            relevance = max(relevance, highest_previous_rel)
            label_predicted_mapping[best_j][3].update({(kind, number, relevance)})
        label_predicted_mapping[best_j][1].append(pred_feature["feature"])

    feature_metrics = defaultdict(list)
    for (
        label_feature,
        pred_features,
        label_passages,
        pred_passages,
    ) in label_predicted_mapping:

        if not label_passages:
            continue

        # --------- feature precision, recall, f1
        pred_passages_without_rel = set((k, n) for k, n, r in pred_passages)

        tp = len(pred_passages_without_rel & label_passages)
        fp = len(pred_passages_without_rel - label_passages)
        fn = len(label_passages - pred_passages_without_rel)

        precision = safe_div(tp, tp + fp) if label_passages else None
        recall = safe_div(tp, tp + fn) if label_passages else None
        f1 = (
            safe_div(2 * (precision * recall), (precision + recall))  # type: ignore
            if label_passages
            else None
        )
        hit = 1.0 if tp > 0 else 0.0 if label_passages else None

        feature_metrics["retrieval-metrics/precision (feature-level)"].append(precision)
        feature_metrics["retrieval-metrics/recall (feature-level)"].append(recall)
        feature_metrics["retrieval-metrics/f1 (feature-level)"].append(f1)
        feature_metrics["retrieval-metrics/hit (feature-level)"].append(hit)

        # --------- feature soft precision, recall, f1
        predicted_passages_text = [
            passage
            if (passage := locate_passage(sample.cited, kind, number)) and len(passage) > 20
            else None
            for kind, number in pred_passages_without_rel
        ]
        label_passages_text = [
            passage
            if (passage := locate_passage(sample.cited, kind, number)) and len(passage) > 20
            else None
            for kind, number in label_passages
        ]
        rouge_scores = np.array([
            [
                (
                    rouge.score(prediction=predicted_passage_text, target=label_passage_text)["rougeL"].recall
                    if predicted_passage_text is not None and label_passage_text is not None
                    else 0.0
                )
                for predicted_passage_text in predicted_passages_text
            ]
            for label_passage_text in label_passages_text
        ])
        rouge_precision = rouge_scores.max(axis=0).mean().item() if rouge_scores.size > 0 else 0.0
        rouge_recall = rouge_scores.max(axis=1).mean().item() if rouge_scores.size > 0 else 0.0
        rouge_f1 = safe_div(
            2 * (rouge_precision * rouge_recall), (rouge_precision + rouge_recall)
        )

        feature_metrics["retrieval-metrics/rougeL precision (feature-level)"].append(rouge_precision)
        feature_metrics["retrieval-metrics/rougeL recall (feature-level)"].append(rouge_recall)
        feature_metrics["retrieval-metrics/rougeL f1 (feature-level)"].append(rouge_f1)

        # --------- feature ranx metrics
        pred_passages_ranx = {
            f"{kind}_{number}": rel
            for kind, number, rel in pred_passages
        }
        label_passages_ranx = {
            f"{kind}_{number}": 1
            for kind, number in label_passages
        }
        if pred_passages_ranx:
            ranx_metrics = ranx.evaluate(
                ranx.Qrels.from_dict({"q1": label_passages_ranx}), 
                ranx.Run.from_dict({"q1": pred_passages_ranx}),
                metrics=ranx_metric_names
            )
            assert isinstance(ranx_metrics, dict)

            for key, value in ranx_metrics.items():
                feature_metrics[f"retrieval-metrics/ranx/feature/{key}"].append(value.item())  # type: ignore

        else:
            for key in ranx_metric_names:
                feature_metrics[f"retrieval-metrics/ranx/feature/{key}"].append(0.0)

    feature_metrics = {
        k: safe_div(
            sum((values_not_none := [v for v in vs if v is not None])),
            len(values_not_none),
        )
        for k, vs in feature_metrics.items()
    }

    return {**claim_metrics, **feature_metrics}


def get_changed_ranges(s1: str, s2: str) -> list[tuple[int, int]]:
    sm = SequenceMatcher(None, s1, s2, autojunk=False)
    changed_ranges = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert"):
            changed_ranges.append((j1, j2))
    return changed_ranges


def locate_feature_in_claim(
    claim: str, feature: str, threshold: int = 80
) -> tuple[int, int]:
    result = fuzz.partial_ratio_alignment(feature, claim)
    assert result is not None

    if result.score >= threshold:
        # Refine by trying all substrings of the matched region
        best_score = result.score
        best_start = result.dest_start
        best_end = result.dest_end

        matched_region = claim[result.dest_start : result.dest_end]
        for i in range(len(matched_region)):
            for j in range(i + 1, len(matched_region) + 1):
                substring = matched_region[i:j]
                score = fuzz.ratio(feature, substring)
                if score > best_score or (
                    score == best_score and j - i < best_end - best_start
                ):
                    best_score = score
                    best_start = result.dest_start + i
                    best_end = result.dest_start + j

        return (best_start, best_end)

    return (0, 0)


def compute_metrics_for_run(
    run_dir: Path, dataset: list[dict[str, Any]]
) -> dict[str, Any]:
    stats: dict[str, float] = defaultdict(float)
    confusion_matrix = pd.DataFrame(
        {"Novel": [0, 0], "Not novel": [0, 0]}, index=["Novel", "Not novel"]
    )  # type: ignore[call-arg]
    confusion_matrix.columns.name = "prediction"
    confusion_matrix.index.name = "label"

    missing = 0
    total_expected = len(dataset)
    retrieval_metrics = defaultdict(list)

    for item in tqdm(dataset):
        sample: ApplicationData = item["sample"]
        granted: bool = item["granted"]
        label = "Novel" if granted else "Not novel"

        sample_dir = (
            run_dir
            / "samples"
            / (sample.app_num + ("_granted" if granted else "_rejected"))
        )
        output_path = sample_dir / "novelty_examination.json"

        if not output_path.exists():
            missing += 1
            stats[f"missing/{label}"] += 1
            continue

        try:
            output = json.loads(output_path.read_text())
        except Exception as e:
            stats[f"failed/{repr(e)}"] += 1
            continue

        novelty = output.get("novelty")
        confidence = output.get("confidence")
        stats[f"confidence/{confidence}"] += 1
        stats[f"label/{label}"] += 1
        stats[f"prediction/{novelty}"] += 1

        for feature in output.get("relevant_passages_per_feature", []) or []:
            disclosed = feature.get("disclosed")
            if disclosed is not None:
                stats[f"features/{disclosed}"] += 1
            for passage in feature.get("relevant_passages", []) or []:
                relevance = passage.get("relevance")
                if relevance is not None:
                    stats[f"passages/{relevance}"] += 1

        if novelty in ("Novel", "Not novel"):
            confusion_matrix.loc[label, novelty] += 1  # type: ignore[index]
        else:
            stats["failed/invalid_prediction"] += 1

        if not granted:
            for metric_name, metric_value in compute_retrieval_metrics(
                sample, output
            ).items():
                if metric_value is not None:
                    retrieval_metrics[metric_name].append(metric_value)

        else:
            label_changed_ranges = get_changed_ranges(
                sample.rejected.claims[0] or "", sample.granted.claims[0] or ""
            )
            label_changed_indices = set(
                i for start, end in label_changed_ranges for i in range(start, end)
            )
            model_non_disclosed_ranges = [
                locate_feature_in_claim(
                    sample.granted.claims[0] or "",
                    feature["feature"],
                )
                for feature in output.get("relevant_passages_per_feature", [])
                if feature["disclosed"] != "Fully disclosed"
            ]
            model_non_disclosed_indices = set(
                i
                for start, end in model_non_disclosed_ranges
                for i in range(start, end)
            )
            novel_features_p = safe_div(
                len(label_changed_indices & model_non_disclosed_indices),
                len(model_non_disclosed_indices),
            )
            retrieval_metrics["novel_features_p"].append(novel_features_p)
            novel_features_r = safe_div(
                len(label_changed_indices & model_non_disclosed_indices),
                len(label_changed_indices),
            )
            retrieval_metrics["novel_features_r"].append(novel_features_r)
            novel_features_f1 = safe_div(
                2 * (novel_features_p * novel_features_r),
                (novel_features_p + novel_features_r),
            )
            retrieval_metrics["novel_features_f1"].append(novel_features_f1)


    mean_retrieval_metrics = {
        f"metrics/{key}": safe_div(sum(values), len(values))
        for key, values in retrieval_metrics.items()
    }
    stats.update(mean_retrieval_metrics)  # type: ignore[arg-type]
    stats.update(compute_metrics_from_confusion(confusion_matrix))  # type: ignore[arg-type]
    stats["confusion matrix"] = confusion_matrix.to_string(  # type: ignore[index]
        index_names=True, header=True
    )
    stats["counts/expected"] = float(total_expected)
    stats["counts/evaluated"] = float(total_expected - missing)
    stats["counts/missing"] = float(missing)

    if (run_dir / "stats.json").exists():
        inf_stats = json.loads(run_dir.joinpath("stats.json").read_text())
        stats["metrics/prompt_tokens_per_sample"] = safe_div(
            inf_stats.get("usage/prompt_tokens", 0.0), float(total_expected - missing)
        )
        stats["metrics/completion_tokens_per_sample"] = safe_div(
            inf_stats.get("usage/completion_tokens", 0.0),
            float(total_expected - missing),
        )

    return {
        "run_dir": str(run_dir),
        "stats": stats,
    }


def add_unbiased_split(dataset: dict[str, list[dict[str, Any]]]) -> None:
    rng = random.Random(42)
    bert_dir = root / "data/logs/novelty_classification/run_BERT_20260112-142350"
    unbiased_ids_path = root / "data/packaged/unbiased_test_ids.json"

    def sample_key(sample: dict[str, Any]) -> str:
        return sample["sample"].app_num + (
            "_granted" if sample["granted"] else "_rejected"
        )

    if not bert_dir.exists():
        ids = set(json.loads(unbiased_ids_path.read_text()))
        dataset["unbiased_test"] = [
            sample for sample in dataset["test"] if sample_key(sample) in ids
        ]
        return

    samples_per_class = {True: [], False: []}
    for sample in dataset["test"]:
        sample_id = sample_key(sample)
        bert_pred_path = bert_dir / "samples" / sample_id / "novelty_examination.json"
        bert_pred = NoveltyExaminationOutput.model_validate_json(
            bert_pred_path.read_text()
        )
        bert_novel = bert_pred.novelty == "Novel"

        if sample["granted"] != bert_novel:
            samples_per_class[sample["granted"]].append(sample)

    n_per_class = min(len(samples_per_class[True]), len(samples_per_class[False]))
    for granted in (True, False):
        samples_per_class[granted] = rng.sample(
            samples_per_class[granted], n_per_class
        )

    samples = samples_per_class[True] + samples_per_class[False]

    unbiased_ids_path.write_text(json.dumps([sample_key(s) for s in samples], indent=2))

    dataset["unbiased_test"] = samples


def main(cfg: hl.Config) -> None:
    logger.info("Loaded dataset")
    dataset = load_dataset(cfg.dataset_path)
    add_unbiased_split(dataset)
    split_counts = {split: len(dataset[split]) for split in dataset}
    logger.info(f"Loaded {split_counts}")

    run_dirs: list[Path] = cfg.run_dirs
    if not run_dirs:
        logger.warning(f"No run_* directories found under {cfg.runs_path}.")
        return

    for run_dir in sorted(run_dirs):
        if not run_dir.is_dir():
            continue

        metrics = {
            split: compute_metrics_for_run(run_dir, eval_items)["stats"]
            for split, eval_items in dataset.items()
            if split != "train"
        }

        (run_dir / "eval_stats.json").write_text(json.dumps(metrics, indent=2))
        logger.info(
            f"Evaluated {run_dir.name}:\n"
            f"\taccuracy={metrics['test']['metrics/accuracy']:.3f}\n"
            f"\tmacro_f1={metrics['test']['metrics/macro f1']:.3f}\n"
            f"\tevaluated={int(metrics['test']['counts/evaluated'])}/"
            f"{int(metrics['test']['counts/expected'])}"
        )


if __name__ == "__main__":
    cfg.apply()
    main(cfg)
