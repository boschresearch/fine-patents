from datetime import datetime
import difflib
import json
from collections import defaultdict
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Any, Optional

import aiofiles
import litellm.types.utils
import dspy
import dspy.clients.lm
import hydralette as hl
import pandas as pd
from pyrootutils import setup_root
from tqdm import tqdm

from whats_novel.dataset.data_model import (
    ClaimBreakdown,
)
from whats_novel.utils.pydantic_adapter import PydanticAdapter
from whats_novel.utils.logs import OutputRedirector, print_dataframe_rich, get_logger
from whats_novel.dataset.s041_parse_rejection import ExtractClaimBreakdown

if TYPE_CHECKING:
    from dspy.teleprompt.gepa.gepa_utils import DSPyTrace, ScoreWithFeedback

logger = get_logger()
root = setup_root(__file__)


def get_log_dir() -> Path:
    dir = root / "data" / "logs" / "gepa"
    i = 0
    while dir.exists():
        dir = dir.parent / f"gepa_{i}"
        i += 1
    dir.mkdir()
    return dir


cfg = hl.Config(
    model="openai/Qwen/Qwen3-VL-235B-A22B-Thinking-FP8",
    samples_path=root / "data" / "samples",
    log_dir=hl.Field(default_factory=get_log_dir),
)


def load_labelled_rejections(
    samples_path: Path, ignore_ids: set[str] = set()
) -> list[dspy.Example]:
    dataset = []
    json_paths = list(
        tqdm(samples_path.glob("*/rejection.label.json"), desc="Loading samples")
    )
    json_paths.sort(key=lambda p: p.name)

    for json_path in json_paths:
        try:
            labelled = ClaimBreakdown.model_validate_json(json_path.read_text())
        except Exception as e:
            logger.error(f"Failed to load {json_path}: {e}")
            continue

        example = dspy.Example(
            pdf_path=json_path.parent / "rejection.pdf",
            output=labelled,
        ).with_inputs("pdf_path")
        dataset.append(example)

    logger.info(f"Loaded {len(dataset)} labelled rejections from {samples_path}")
    return dataset


def lang_correct(pred: ClaimBreakdown, label: ClaimBreakdown) -> float:
    return float(pred.language == label.language)


def prior_art_docs_labels_correct(pred: ClaimBreakdown, label: ClaimBreakdown) -> float:
    pred_docs = [doc.label for doc in pred.prior_art_documents]
    label_docs = [doc.label for doc in label.prior_art_documents]
    return float(pred_docs == label_docs)


def prior_art_docs_identifiers_correct(
    pred: ClaimBreakdown, label: ClaimBreakdown
) -> float:
    pred_identifiers = [
        doc.identifier.replace(" ", "")
        for doc in pred.prior_art_documents
        if doc.identifier
    ]
    label_identifiers = [
        doc.identifier.replace(" ", "")
        for doc in label.prior_art_documents
        if doc.identifier
    ]
    return float(pred_identifiers == label_identifiers)


def prior_art_docs_types_correct(pred: ClaimBreakdown, label: ClaimBreakdown) -> float:
    pred_types = [doc.type for doc in pred.prior_art_documents]
    label_types = [doc.type for doc in label.prior_art_documents]
    return float(pred_types == label_types)


def prior_art_docs_dates_correct(pred: ClaimBreakdown, label: ClaimBreakdown) -> float:
    pred_dates = [doc.date for doc in pred.prior_art_documents]
    label_dates = [doc.date for doc in label.prior_art_documents]
    return float(pred_dates == label_dates)


def reason_for_rejection_correct(pred: ClaimBreakdown, label: ClaimBreakdown) -> float:
    return float(pred.reason_for_rejection == label.reason_for_rejection)


def breakdown_number_of_features_correct(
    pred: ClaimBreakdown, label: ClaimBreakdown
) -> float:
    num_pred = len(pred.breakdown) if pred.breakdown else 0
    num_label = len(label.breakdown) if label.breakdown else 0
    return float(num_pred == num_label)


def breakdown_edit_similarity(pred: ClaimBreakdown, label: ClaimBreakdown) -> float:
    pred_json = json.dumps(pred.model_dump()["breakdown"], indent=4)
    label_json = json.dumps(label.model_dump()["breakdown"], indent=4)
    matcher = difflib.SequenceMatcher(None, pred_json, label_json)
    similarity_ratio = matcher.ratio()
    return similarity_ratio


def inventive_step_reasoning_correct(
    pred: ClaimBreakdown, label: ClaimBreakdown
) -> float:
    if pred.inventive_step_reasoning is None and label.inventive_step_reasoning is None:
        return 1.0
    elif (
        pred.inventive_step_reasoning is None
        and label.inventive_step_reasoning is not None
    ):
        return 0.0
    elif (
        pred.inventive_step_reasoning is not None
        and label.inventive_step_reasoning is None
    ):
        return 0.0

    assert pred.inventive_step_reasoning is not None
    assert label.inventive_step_reasoning is not None

    scores = []
    for pred_feature, label_feature in (
        (
            pred.inventive_step_reasoning.difference_to_prior_art or "",
            label.inventive_step_reasoning.difference_to_prior_art or "",
        ),
        (
            pred.inventive_step_reasoning.objective_problem or "",
            label.inventive_step_reasoning.objective_problem or "",
        ),
        (
            pred.inventive_step_reasoning.technical_effect_reasoning or "",
            label.inventive_step_reasoning.technical_effect_reasoning or "",
        ),
        (
            pred.inventive_step_reasoning.skilled_person_reasoning or "",
            label.inventive_step_reasoning.skilled_person_reasoning or "",
        ),
    ):
        matcher = difflib.SequenceMatcher(None, pred_feature, label_feature)
        similarity_ratio = matcher.ratio()
        scores.append(similarity_ratio)
    return sum(scores) / len(scores)


def get_scores_table(result: dspy.Prediction) -> pd.DataFrame:
    scores = defaultdict(list)
    for example, _, s in result.results:
        for k, v in s.scores.items():
            scores[k].append(v)
        scores["sample"].append(example.pdf_path.parent.name)

    scores_df = pd.DataFrame(scores).set_index("sample")
    scores_df.loc["average"] = scores_df.mean()

    return scores_df


def get_nl_feedback(pred: ClaimBreakdown, label: ClaimBreakdown) -> str:
    return (
        "Compare the predicted claim breakdown to the manually annotated one.\n\n"
        "## Guidelines\n\n"
        "- The most important part is the breakdown. The model has to extract all features and correctly save the references.\n"
        "- For some references, the exact format might not be unabiguous. That is ok, as long as all information is there.\n"
        "- The order of references is irrelevant.\n"
        f"\n## Label\n\n{label}"
    )


print_lock = threading.Lock()


def score_breakdown(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Optional["DSPyTrace"] = None,
    pred_name: str | None = None,
    pred_trace: Optional["DSPyTrace"] = None,
) -> "ScoreWithFeedback":
    # with print_lock:
    logger.info(f"Scoring {gold.pdf_path}")

    predicted: ClaimBreakdown = pred.output
    label: ClaimBreakdown = gold.output

    score = 0.0
    weights = 0.0
    scores = {}

    s = lang_correct(predicted, label)
    logger.info(f"Language correct: {s}")
    scores["lang_correct"] = s
    score += 1.0 * s
    weights += 1.0

    s = prior_art_docs_labels_correct(predicted, label)
    logger.info(f"Prior art document labels correct: {s}")
    scores["prior_art_docs_labels_correct"] = s
    score += 1.0 * s
    weights += 1.0

    s = prior_art_docs_identifiers_correct(predicted, label)
    logger.info(f"Prior art document identifiers correct: {s}")
    scores["prior_art_docs_identifiers_correct"] = s
    score += 1.0 * s
    weights += 1.0

    s = prior_art_docs_types_correct(predicted, label)
    logger.info(f"Prior art document types correct: {s}")
    scores["prior_art_docs_types_correct"] = s
    score += 1.0 * s
    weights += 1.0

    s = prior_art_docs_dates_correct(predicted, label)
    logger.info(f"Prior art document dates correct: {s}")
    scores["prior_art_docs_dates_correct"] = s
    score += 1.0 * s
    weights += 1.0

    s = reason_for_rejection_correct(predicted, label)
    logger.info(f"Reasons for rejection correct: {s}")
    scores["reasons_for_rejection_correct"] = s
    score += 10.0 * s
    weights += 10.0

    s = inventive_step_reasoning_correct(predicted, label)
    logger.info(f"Inventive step reasoning correct: {s}")
    scores["inventive_step_reasoning_correct"] = s
    score += 10.0 * s
    weights += 10.0

    s = breakdown_number_of_features_correct(predicted, label)
    logger.info(f"Number of features in breakdown correct: {s}")
    scores["breakdown_number_of_features_correct"] = s
    score += 5.0 * s
    weights += 5.0

    s = breakdown_edit_similarity(predicted, label)
    logger.info(f"Breakdown edit similarity: {s:.4f}")
    scores["breakdown_edit_similarity"] = s
    score += 20.0 * s
    weights += 20.0

    score = score / weights
    logger.info(f"Total Score: {score:.4f}")

    return dspy.Prediction(
        score=score, scores=scores, feedback=get_nl_feedback(predicted, label)
    )  # type: ignore


def patch_message_logging(log_dir: Path) -> None:
    messages_dir = log_dir / "lm_messages"
    messages_dir.mkdir(exist_ok=True, parents=True)

    def filter_messages(messages):
        for msg in messages:
            content = msg["content"]
            if isinstance(content, list):
                msg["content"] = "\n".join(
                    part.get("text", "")
                    for part in content
                    if part.get("type") == "text"
                )

    orig_completion = dspy.clients.lm.litellm_completion
    orig_acompletion = dspy.clients.lm.alitellm_completion

    def litellm_completion(
        request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None
    ):
        messages = request.get("messages", [])
        response = orig_completion(request, num_retries, cache)
        assert isinstance(response, litellm.types.utils.ModelResponse)
        messages.append(
            {"role": "assistant", "content": response.choices[0].message.content}  # type: ignore
        )
        filter_messages(messages)
        msg_path = messages_dir / f"messages_{datetime.now().isoformat()}.json"
        msg_path.write_text(json.dumps(messages, indent=4))
        return response

    async def litellm_acompletion(
        request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None
    ):
        messages = request.get("messages", [])
        response = await orig_acompletion(request, num_retries, cache)
        assert isinstance(response, litellm.types.utils.ModelResponse)
        messages.append(
            {"role": "assistant", "content": response.choices[0].message.content}  # type: ignore
        )
        filter_messages(messages)
        msg_path = messages_dir / f"messages_{datetime.now().isoformat()}.json"
        async with aiofiles.open(msg_path, "w") as f:
            await f.write(json.dumps(messages, indent=4))
        return response

    dspy.clients.lm.litellm_completion = litellm_completion
    dspy.clients.lm.alitellm_completion = litellm_acompletion


def main(cfg: hl.Config) -> None:
    patch_message_logging(cfg.log_dir)

    lm = dspy.LM(
        model=cfg.model,
        api_base="http://localhost:59537/v1",
        api_key="EMPTY",
        temperature=0.1,
        max_tokens=60_000,
        cache=True,
    )

    dspy.configure(lm=lm, adapter=PydanticAdapter())

    dataset = load_labelled_rejections(cfg.samples_path)

    extract = ExtractClaimBreakdown()

    evaluate = dspy.Evaluate(
        devset=dataset,
        metric=score_breakdown,
        num_threads=8,
        display_progress=True,
    )
    # result = evaluate(extract)
    # print_dataframe_rich(get_scores_table(result))

    gepa = dspy.GEPA(
        metric=score_breakdown,  # type: ignore
        auto="light",
        num_threads=8,
        track_stats=True,
        reflection_lm=lm,
        log_dir=cfg.log_dir,
    )

    extract_optimized = gepa.compile(
        extract,
        trainset=dataset,
        valset=dataset,
    )

    for name, pred in extract_optimized.named_predictors():
        print("================================")
        print(f"Predictor: {name}")
        print("================================")
        print("Prompt:")
        print(pred.signature.instructions)  # type: ignore
        print("*********************************")

    result = evaluate(extract_optimized)  # noqa
    print_dataframe_rich(get_scores_table(result))

    import pickle

    with open("data/optimized_module.pkl", "wb") as f:
        pickle.dump(extract_optimized, f)


if __name__ == "__main__":
    cfg.apply()

    with OutputRedirector(cfg.log_dir / "stdouterr.log"):
        main(cfg)
