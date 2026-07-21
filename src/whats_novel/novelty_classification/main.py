# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: MIT

import asyncio
import json
from collections import defaultdict
from datetime import datetime
import os
from pathlib import Path
from typing import Type

import dspy
import hydralette as hl
import litellm
import pandas as pd
import requests
import rich.syntax
from pyrootutils import setup_root

from whats_novel.dataset import ApplicationData, load_dataset_async
from whats_novel.novelty_classification.evaluate import (
    compute_metrics_from_confusion,
)
from whats_novel.novelty_classification.shared import (
    BaseNoveltyExamination,
    NoveltyExaminationOutput,
)
from whats_novel.novelty_classification.single_step import (
    SingleStepNoveltyExamination,
    SingleStepNoveltyExaminationClaimOnly as SingleStepNoveltyExaminationClaimOnly,
    SingleStepNoveltyExaminationConstruction as SingleStepNoveltyExaminationConstruction,
    SingleStepNoveltyExaminationSC as SingleStepNoveltyExaminationSC,
)
from whats_novel.novelty_classification.hierarchical_examination import (
    HierarchicalExamination as HierarchicalExamination,
)
from whats_novel.novelty_classification.baselines import (
    RandomNoveltyExamination as RandomNoveltyExamination,
    LogRegSpurious as LogRegSpurious,
    EmbeddingSimilarity as EmbeddingSimilarity,
    BERT as BERT,
    RougeSimilarity as RougeSimilarity,
    SpuriousLLM as SpuriousLLM,
)
from whats_novel.utils import (
    OutputRedirector,
    PydanticAdapter,
    RichTableProgress,
    console,
    get_logger,
    run_async,
)

logger = get_logger()
root = setup_root(__file__)
N_RETRIES = 5


def get_model_name(url: str) -> str:
    return requests.get(url + "/models").json()["data"][0]["id"]


cfg = hl.Config(
    dataset_path=root / "data" / "packaged",
    api_base="http://localhost:59535/v1",
    model=hl.Field(reference=lambda cfg: get_model_name(cfg.api_base)),
    run_dir=hl.Field(
        reference=lambda cfg: root
        / "data"
        / "logs"
        / "novelty_classification"
        / f"run_{cfg.model.split('/')[-1]}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    ),
    n_workers=16,
    offer_money=True,
    use_label_references=False,
    split_using_llm=False,
    with_examination_summaries=False,
    module=hl.Field(
        default=SingleStepNoveltyExamination, convert=lambda s: globals()[s]
    ),
    copy_from=hl.Field(default=None, type=Path),
)


async def process_sample(
    NoveltyExamination: Type[BaseNoveltyExamination],
    sample: ApplicationData,
    granted: bool,
    offer_money: bool,
    use_label_references: bool,
    split_using_llm: bool,
    with_examination_summaries: bool,
) -> tuple[NoveltyExaminationOutput, dict[str, list[dict]], litellm.Usage]:
    def add_response_to_messages(history_entry):
        return [
            *history_entry["messages"],
            {
                "role": "assistant",
                "reasoning_content": (
                    message.reasoning_content
                    if hasattr(
                        (message := history_entry["response"].choices[0].message),
                        "reasoning_content",
                    )
                    else None
                ),
                "content": message.content,
            },
        ]

    def get_submodules(module):
        return {
            name: submodule
            for name, submodule in module.__dict__.items()
            if isinstance(submodule, dspy.Module)
        }

    def merge_usages(usages):
        total = litellm.Usage()
        for usage in usages:
            total.prompt_tokens += usage.prompt_tokens
            total.completion_tokens += usage.completion_tokens
            total.total_tokens += usage.total_tokens
        return total

    for attempt in range(N_RETRIES):
        try:
            module = NoveltyExamination(
                offer_money=offer_money,
                use_label_references=use_label_references,
                split_using_llm=split_using_llm,
                with_examination_summaries=with_examination_summaries,
            )
            result = await module.aforward(sample, granted=granted)
            submodules = get_submodules(module)
            messages = {
                f"{module_name}_{i}": add_response_to_messages(submodule.history[i])
                for module_name, submodule in submodules.items()
                for i in range(len(submodule.history))
            }
            usages = [
                submodule.history[i]["response"].usage
                for module_name, submodule in submodules.items()
                for i in range(len(submodule.history))
            ]
            return result, messages, merge_usages(usages)

        except Exception as e:
            logger.warning(
                f"Error processing sample {sample.app_num} (attempt {attempt + 1}/{N_RETRIES}): {e}"
            )
            if attempt == N_RETRIES - 1:
                raise
    raise ValueError


async def main() -> None:
    run_dir: Path = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    console.print(rich.syntax.Syntax(cfg.to_yaml(), "yaml"))
    with open(run_dir / "config.yaml", "w") as f:
        f.write(cfg.to_yaml())

    dataset = await load_dataset_async(cfg.dataset_path)
    train_dataset = dataset["train"]
    logger.info(f"Loaded {len(train_dataset)} train samples for novelty prediction.")

    if hasattr(cfg.module, "train"):
        logger.info(f"Training {cfg.module.__name__}...")
        module_instance = cfg.module()
        await module_instance.train(train_dataset, run_dir)
        logger.info(f"Finished training {cfg.module.__name__}.")
        module = (  # noqa
            lambda *args, **kwargs: module_instance
        )  # dummy "class" to use the trained module in each worker
    else:
        module = cfg.module

    test_dataset = [
        {
            **item,
            "offer_money": cfg.offer_money,
            "use_label_references": cfg.use_label_references,
            "split_using_llm": cfg.split_using_llm,
            "with_examination_summaries": cfg.with_examination_summaries,
            "NoveltyExamination": module,
        }
        for split in ("test", "val")
        for item in dataset[split]
    ]
    logger.info(f"Loaded {len(test_dataset)} test+val samples for novelty prediction.")

    lm = dspy.LM(
        model=f"openai/{cfg.model}",
        api_base=cfg.api_base,
        api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
        temperature=0.3,
        top_p=0.9,
        top_k=20,
        max_tokens=None,  # type: ignore
        cache=False,
        stream=False,
    )
    dspy.configure(lm=lm, adapter=PydanticAdapter())

    if cfg.copy_from is not None:
        copied_ids = set()
        logger.info(f"Copying from {cfg.copy_from}...")
        for sample_dir in cfg.copy_from.glob("samples/*"):
            target_dir = run_dir / "samples" / sample_dir.name
            files = list(sample_dir.iterdir())
            if files:
                for file in files:
                    target_file = target_dir / file.name
                    if not target_file.exists():
                        target_dir.mkdir(parents=True, exist_ok=True)
                        target_file.write_text(file.read_text())

                app_num = sample_dir.name.split("_")[0]
                granted = sample_dir.name.endswith("_granted")
                copied_ids.add((app_num, granted))

        logger.info(f"Copied {len(copied_ids)} samples from {cfg.copy_from}.")

        test_dataset = [
            item
            for item in test_dataset
            if (item["sample"].app_num, item["granted"]) not in copied_ids
        ]
        logger.info(f"Processing {len(test_dataset)} samples after copying.")

    with RichTableProgress(total=len(test_dataset), persistent_every=100) as pbar:
        stats = defaultdict(float)
        confusion_matrix = pd.DataFrame(
            {"Novel": [0, 0], "Not novel": [0, 0]},
            index=pd.Series(["Novel", "Not novel"]),
        )
        confusion_matrix.columns.name = "prediction"
        confusion_matrix.index.name = "label"
        pbar.update(0, data=stats)

        async for inputs, outputs, error in run_async(
            test_dataset, process_sample, n_workers=cfg.n_workers, ordered=False
        ):
            sample: ApplicationData = inputs["sample"]
            granted: bool = inputs["granted"]
            sample_dir: Path = (
                run_dir
                / "samples"
                / (sample.app_num + ("_granted" if granted else "_rejected"))
            )
            sample_dir.mkdir(parents=True)

            if error:
                # raise error
                logger.error(f"Error processing sample {sample.app_num}: {error}")
                stats[f"failed/{str(error.__class__)}"] += 1

            else:
                assert outputs is not None
                output: NoveltyExaminationOutput = outputs[0]
                history: dict[str, list[dict]] = outputs[1]
                usage: litellm.Usage = outputs[2]

                output_path: Path = sample_dir / "novelty_examination.json"
                output_path.write_text(output.model_dump_json(indent=2))

                for history_name, history_messages in history.items():
                    history_path: Path = sample_dir / f"history_{history_name}.json"
                    history_path.write_text(json.dumps(history_messages, indent=2))

                stats[f"confidence/{output.confidence}"] += 1

                label = "Novel" if granted else "Not novel"

                stats[f"label/{label}"] += 1
                stats[f"prediction/{output.novelty}"] += 1

                for feature in output.relevant_passages_per_feature:
                    stats[f"features/{feature.disclosed}"] += 1
                    for passage in feature.relevant_passages:
                        stats[f"passages/{passage.relevance}"] += 1

                confusion_matrix.loc[label, output.novelty] += 1  # type: ignore[index]
                stats.update(compute_metrics_from_confusion(confusion_matrix))  # type: ignore[arg-type]
                stats["confusion matrix"] = confusion_matrix.to_string(  # type: ignore[index]
                    index_names=True, header=True
                )

                stats["usage/prompt_tokens"] += usage.prompt_tokens
                stats["usage/prompt_tokens / s"] = RichTableProgress.AvgPerSec(  # type: ignore
                    "usage/prompt_tokens", "t/s"
                )
                stats["usage/completion_tokens"] += usage.completion_tokens
                stats["usage/completion_tokens / s"] = RichTableProgress.AvgPerSec(  # type: ignore
                    "usage/completion_tokens", "t/s"
                )
                stats["usage/total_tokens"] += usage.total_tokens
                stats["usage/total_tokens / s"] = RichTableProgress.AvgPerSec(  # type: ignore
                    "usage/total_tokens", "t/s"
                )

            pbar.update(1, data=stats)

    run_dir.joinpath("stats.json").write_text(
        json.dumps(
            {
                k: v
                for k, v in stats.items()
                if not isinstance(v, RichTableProgress.AvgPerSec)
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    cfg.apply()
    cfg.run_dir.mkdir()
    with OutputRedirector(cfg.run_dir / "stdout_stderr.log"):
        asyncio.run(main())
