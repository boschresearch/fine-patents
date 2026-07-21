# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: MIT

import asyncio
import json
from collections import defaultdict
from pathlib import Path

import aiofiles
from tqdm import tqdm

from .data_model import (
    ApplicationData as ApplicationData,
    PatentDocument as PatentDocument,
    ClaimBreakdown as ClaimBreakdown,
    ClaimBreakdownWithPriorPassages as ClaimBreakdownWithPriorPassages,
    ClaimBreakdownFeature as ClaimBreakdownFeature,
    ClaimBreakdownFeatureWithPriorPassages as ClaimBreakdownFeatureWithPriorPassages,
    InventiveStepReasoning as InventiveStepReasoning,
    PriorArtDocument as PriorArtDocument,
    PriorArtPassage as PriorArtPassage,
    PriorArtReference as PriorArtReference,
    DocumentLocation as DocumentLocation,
)
from whats_novel.utils import run_async


async def _load_sample(sample_dir: Path) -> list[dict]:
    if not sample_dir.is_dir():
        return []

    app_num = sample_dir.name

    breakdown_path = sample_dir / "breakdown.json"
    async with aiofiles.open(breakdown_path, "r") as f:
        breakdown = ClaimBreakdownWithPriorPassages.model_validate_json(await f.read())

    rejected_path = sample_dir / "rejected_patent.json"
    async with aiofiles.open(rejected_path, "r") as f:
        rejected_data = json.loads(await f.read())
        rejected = PatentDocument(**rejected_data)

    granted_path = sample_dir / "granted_patent.json"
    async with aiofiles.open(granted_path, "r") as f:
        granted_data = json.loads(await f.read())
        granted = PatentDocument(**granted_data)

    cited_path = sample_dir / "cited_patent.json"
    async with aiofiles.open(cited_path, "r") as f:
        cited_data = json.loads(await f.read())
        cited = PatentDocument(**cited_data)

    metadata_path = sample_dir / "metadata.json"
    async with aiofiles.open(metadata_path, "r") as f:
        metadata = json.loads(await f.read())

    samples = []
    for version in metadata["include_versions"]:
        sample = ApplicationData(
            app_num=app_num,
            breakdown=breakdown,
            rejected=rejected,
            granted=granted,
            cited=cited,
        )
        is_granted = version == "granted"
        split = metadata["split"]
        samples.append((split, {"sample": sample, "granted": is_granted}))

    return samples


async def load_dataset_async(dataset_path: Path) -> dict[str, list[dict]]:
    samples = defaultdict(list)

    args = [{"sample_dir": d} for d in dataset_path.glob("*")]
    pbar = tqdm(total=len(args), desc="Loading dataset")
    async for _, samples_, _ in run_async(args, _load_sample, n_workers=32):
        assert samples_ is not None
        pbar.update(1)
        for split, sample in samples_:
            samples[split].append(sample)

    return dict(samples)


def load_dataset(dataset_path: Path) -> dict[str, list[dict]]:
    return asyncio.run(load_dataset_async(dataset_path))