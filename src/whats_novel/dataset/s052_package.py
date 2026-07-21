# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: MIT

import asyncio
import difflib
import json
import pickle
import random
import re
from collections import defaultdict
from itertools import chain
from pathlib import Path

import hydralette as hl
import numpy as np
from pyrootutils import setup_root

from whats_novel.dataset.data_model import (
    ClaimBreakdown,
    ClaimBreakdownFeatureWithPriorPassages,
    PriorArtPassage,
)
from whats_novel.dataset.s051_download_cited import (
    cfg,
    extract_cited_documents,
    list_samples,
    parse_breakdowns,
)
from whats_novel.utils.logs import OutputRedirector, RichTableProgress, get_logger

logger = get_logger()
root = setup_root(__file__)

random.seed(42)

SPLITS = {"train": 0.4, "test": 0.5, "val": 0.1}


class PackagingError(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def validate_breakdown_matches_claim(
    breakdown: ClaimBreakdown, claim: str, threshold: float = 0.95
) -> None:
    def remove_claim_number(text: str) -> str:
        return re.sub(r"^\d+\.\s+", "", text).strip()

    def remove_parantheses(text: str) -> str:
        return re.sub(r"\([^)]+\)", "", text).strip()

    def remove_punctuation(text: str) -> str:
        return re.sub(r"[.,;:!?]", " ", text).strip()

    def unify_spaces(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def normalize_text(text: str) -> str:
        text = text.lower()
        text = remove_claim_number(text)
        text = remove_parantheses(text)
        text = remove_punctuation(text)
        text = unify_spaces(text)
        return text

    if not breakdown.breakdown:
        raise PackagingError("Breakdown empty")

    features = [feature_breakdown.feature for feature_breakdown in breakdown.breakdown]
    claim_from_breakdown = normalize_text(" ".join(features))

    claim_from_application = normalize_text(claim)

    seq = difflib.SequenceMatcher(a=claim_from_application, b=claim_from_breakdown)
    ratio = seq.ratio()
    if ratio < threshold:
        raise PackagingError("Mismatched claim")


def locate_cited_passages(
    breakdown: ClaimBreakdown,
    cited_patent: dict,
) -> ClaimBreakdown:
    cited_description: list[str | None] = cited_patent.get("description", [])
    cited_claims: list[str | None] = cited_patent.get("claims", [])
    cited_abstract: str | None = cited_patent.get("abstract", None)

    def resolve_number(num: str | int) -> list[int]:
        if re.match(r"^\s*\[?\d+\]?\s*(-|to)\s*\[?\d+\]?\s*$", str(num)):
            start, end = [int(n.lstrip("0")) for n in re.findall(r"\d+", str(num))]
            return list(range(start, end + 1))
        elif isinstance(num, str):
            return [int(num.replace("[", "").replace("]", "").lstrip("0"))]
        else:
            return [num]

    assert breakdown.breakdown

    for feature_idx in range(len(breakdown.breakdown)):
        feature = breakdown.breakdown[feature_idx]
        passages = []

        for prior_art_ref in feature.prior_art_references:
            loc = prior_art_ref.location
            if loc is None:
                continue

            if loc.reference_type == "paragraph":
                for num in loc.numbers or []:
                    for n in resolve_number(num):
                        idx = n - 1
                        if not (0 <= idx <= len(cited_description) - 1):
                            raise PackagingError(
                                f"Invalid cited paragraph number {num}"
                            )
                        cited_paragraph = cited_description[idx]
                        if cited_paragraph is None:
                            raise PackagingError(
                                f"Cited paragraph {num} is missing in cited description"
                            )
                        passages.append(
                            PriorArtPassage(
                                reference=("paragraph", n),
                                text=cited_paragraph,
                                source_reference=prior_art_ref,
                            )
                        )

            elif loc.reference_type == "claim":
                for num in loc.numbers or []:
                    for n in resolve_number(num):
                        idx = n - 1
                        if not (0 <= idx <= len(cited_claims) - 1):
                            raise PackagingError(f"Invalid cited claim number {num}")
                        cited_claim = cited_claims[idx]
                        if not isinstance(cited_claim, str):
                            raise PackagingError(
                                f"Cited claim {num} is missing in cited claims"
                            )
                        passages.append(
                            PriorArtPassage(
                                source_reference=prior_art_ref,
                                reference=("claim", n),
                                text=cited_claim,
                            )
                        )

            elif loc.reference_type == "abstract":
                if not cited_abstract:
                    raise PackagingError("Cited abstract is missing")
                passages.append(
                    PriorArtPassage(
                        source_reference=prior_art_ref,
                        reference=("abstract", None),
                        text=cited_abstract,
                    )
                )

            # else:
            #     logger.warning(
            #         f"Unsupported reference type for locating cited passages: {loc.reference_type}"
            #     )

        if passages:
            breakdown.breakdown[feature_idx] = ClaimBreakdownFeatureWithPriorPassages(
                prior_art_passages=passages,
                **feature.model_dump(),
            )

    return breakdown


def load_and_filter_breakdown(
    app_num: str,
    breakdown: ClaimBreakdown,
    sample_path: Path,
    cited_docs_path: Path,
) -> tuple[
    ClaimBreakdown,
    dict[str, str | list[str | None]],
    dict[str, str | list[str | None]],
    dict[str, str | list[str | None]],
]:
    if breakdown.language != "EN":
        raise PackagingError("Non english breakdown")

    if breakdown.reason_for_rejection != "novelty":
        raise PackagingError("Non-novelty rejection")

    try:
        rejected_patent_path = next(sample_path.glob("publications/*A1.json"))
        rejected_patent: dict = json.loads(rejected_patent_path.read_text())
        assert rejected_patent.get("claims", [])[0], "First claim missing"
        assert rejected_patent.get("description"), "Description missing"
        rejected_patent["application_number"] = app_num
        rejected_patent["publication_number"] = rejected_patent_path.stem

    except Exception as e:
        raise PackagingError("Invalid rejected patent") from e

    validate_breakdown_matches_claim(breakdown, rejected_patent["claims"][0])

    try:
        granted_patent_path = next(
            chain(
                sample_path.glob("publications/*B1.json"),
                sample_path.glob("publications/*B2.json"),
            )
        )
        granted_patent: dict = json.loads(granted_patent_path.read_text())
        assert granted_patent.get("claims", [])[0], "First claim missing"
        assert granted_patent.get("description"), "Description missing"
        granted_patent["application_number"] = app_num
        granted_patent["publication_number"] = granted_patent_path.stem

    except Exception as e:
        raise PackagingError("Invalid granted patent") from e

    cited_documents = extract_cited_documents([breakdown])
    if not cited_documents:
        raise PackagingError("No documents cited")

    if len(cited_documents) > 1:
        raise PackagingError("Multiple documents cited")

    cited_id_docdb = list(cited_documents)[0]
    cited_path = cited_docs_path / (str(cited_id_docdb) + ".json")

    if not cited_path.exists():
        raise PackagingError("Cited document not found")

    try:
        cited_patent: dict = json.loads(cited_path.read_text())
        assert cited_patent.get("claims", [])[0], "First claim missing"
        assert cited_patent.get("description"), "Description missing"
        if not cited_path.stem.startswith("US"):
            cited_patent["application_number"] = None
            cited_patent["publication_number"] = cited_path.stem

    except Exception as e:
        raise PackagingError("Invalid cited patent") from e

    try:
        locate_cited_passages(breakdown, cited_patent)
    except PackagingError as e:
        raise PackagingError("Unlocatable citations") from e

    if not any(
        isinstance(feature, ClaimBreakdownFeatureWithPriorPassages)
        and feature.prior_art_passages
        for feature in breakdown.breakdown or []
    ):
        raise PackagingError("No passages located from cited document")

    return breakdown, rejected_patent, granted_patent, cited_patent


def breakdown_stats(breakdowns: dict[str, ClaimBreakdown]) -> None:
    stats = defaultdict(int)
    for breakdown in breakdowns.values():
        stats["breakdowns"] += 1
        if breakdown.breakdown and all(
            isinstance(feature, ClaimBreakdownFeatureWithPriorPassages)
            for feature in breakdown.breakdown[1:]
        ):
            stats["breakdowns/complete"] += 1
        for feature in breakdown.breakdown or []:
            stats["features"] += 1
            if isinstance(feature, ClaimBreakdownFeatureWithPriorPassages):
                stats["features_with_passages"] += 1
                for passage in feature.prior_art_passages:
                    stats["passages"] += 1
                    stats[
                        f"passages/{passage.reference[0] if passage.reference else 'N/A'}"
                    ] += 1

    logger.info("Breakdown Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key:<30}: {value}")


def format_breakdown(breakdown: ClaimBreakdown, topic_id: str) -> str:
    md = [f"# Claim Breakdown for {topic_id}\n"]
    md.append("## Prior Art Documents\n")
    for doc in breakdown.prior_art_documents:
        md.append(f"- **{doc.label}** ({doc.type}): {doc.identifier or 'N/A'}")
        if doc.applicant_or_authors:
            md.append(f"  - Applicant/Authors: {doc.applicant_or_authors}")
        if doc.title:
            md.append(f"  - Title: {doc.title}")
        if doc.date:
            md.append(f"  - Date: {doc.date}")
    md.append("\n## Breakdown Features\n")
    for i, feature in enumerate(breakdown.breakdown or []):
        md.append(f"### Feature {i + 1} \n")
        md.append(f"> {feature.feature} \n")

        if isinstance(feature, ClaimBreakdownFeatureWithPriorPassages):
            for passage in feature.prior_art_passages:
                ref = passage.source_reference
                conf = (
                    f"**Confidence: {passage.confidence:.2f}**"
                    if passage.confidence
                    else ""
                )
                md.append(
                    "- "
                    f"{conf}"
                    f"**Cited in {ref.document}**, "
                    f"type: {ref.location.reference_type if ref.location else 'N/A'}, "
                    f"numbers: {', '.join(str(n) for n in ref.location.numbers or [])}"
                    if ref.location
                    else "N/A"
                )
                md.append(f"  > {passage.text}\n")
        else:
            md.append("*(No prior art passages located)*\n")
            md.append(
                f"References:\n{'\n'.join(ref.model_dump_json() for ref in feature.prior_art_references)}\n"
            )
    return "\n".join(md)


def bin_samples(samples: dict[str, tuple], bins: list[float]):
    binned_samples = {b: ([], []) for b in bins}
    for app_num, sample in samples.items():
        for doc, granted in ((sample[1], False), (sample[2], True)):
            include_versions = (
                sample[4].get("include_versions", [])
                if sample[4] is not None
                else ["rejected", "granted"]
            )
            if (granted and "granted" not in include_versions) or (
                not granted and "rejected" not in include_versions
            ):
                continue
            claim_len = len(doc["claims"][0].split())
            for b in bins:
                if claim_len <= b:
                    binned_samples[b][int(granted)].append((app_num, granted))
                    break

    binned_counts = {b: (len(v[0]), len(v[1])) for b, v in binned_samples.items()}
    for b, (n_rejected, n_granted) in binned_counts.items():
        logger.info(
            f"  Bin <= {str(int(b)):5s} words: {str(n_rejected):5s} rejected, {str(n_granted):5s} granted"
        )
    total_rejected = sum(nr for nr, ng in binned_counts.values())
    total_granted = sum(ng for nr, ng in binned_counts.values())
    logger.info(
        f"  Total samples: {total_granted + total_rejected} ({total_rejected} rejected, {total_granted} granted)"
    )

    return binned_samples, binned_counts


def spurious_correlation_stats(data):
    data = list(data)

    def contains_patent_number(s: str, patent_number: str) -> bool:
        match = re.match(r"([A-Z]{2})(\d{4})(\d+)([A-Z]\d)?$", patent_number)
        if not match:
            print("Invalid patent number format:", patent_number)
            return False
        country, year, number, kind_code = match.groups()
        number_int = number.lstrip("0")
        pattern = rf"{country}\s*{year}\s*/?\s*0*{number_int}"
        return bool(re.search(pattern, s, re.IGNORECASE))

    stats = {True: defaultdict(int), False: defaultdict(int)}
    for doc, granted, cited_pub_num in data:
        stats[granted]["total"] += 1
        stats[granted]["n_claims"] += len(doc.get("claims", []))
        stats[granted]["n_claim_1_words"] += len(doc["claims"][0].split())
        stats[granted]["n_claim_1_reference_numerals"] += len(
            re.findall(
                r"(\([\da-zA-Z]+(?:[,-]\s*[\da-zA-Z]+)*\))", doc["claims"][0] or ""
            )
        )
        stats[granted]["n_description_paras"] += len(
            [p for p in doc.get("description", []) if p]
        )
        stats[granted]["n_description_words"] += sum(
            len(para.split()) for para in doc.get("description", []) if para
        )
        stats[granted]["desc_contains_cited_pubnum"] += contains_patent_number(
            "\n".join(p or "" for p in doc.get("description", [])),
            cited_pub_num,
        )
        stats[granted]["n_description_pubnums"] += len(
            re.findall(
                r"([A-Z]{2})\s*(\d{4})\s*/?\s*(\d+)([A-Z]\d)?",
                "\n".join(p or "" for p in doc["description"]),
            )
        )

    logger.info("Spurious Correlation Statistics:")
    for k, count_g in stats[True].items():
        count_r = stats[False][k]
        if k != "total":
            count_g = count_g / stats[True]["total"]
            count_r = count_r / stats[False]["total"]
        logger.info(
            f"   {k:<30}: {float(count_r):.4f} rejected {float(count_g):.4f} granted"
        )


def remove_reference_numerals(samples: dict[str, tuple]) -> None:
    def unify_spaces(text: str) -> str:
        text = re.sub(r" +", " ", text).strip()
        text = re.sub(r" ([.,:;])", r"\1", text)
        return text

    for sample in samples.values():
        for doc in (sample[1], sample[2]):
            for i in range(len(doc["claims"])):
                claim = doc["claims"][i]
                if claim is not None:
                    claim_no_refs = unify_spaces(
                        re.sub(r"\([\da-zA-Z]+(?:[,-]\s*[\da-zA-Z]+)*\)", "", claim)
                    )
                    doc["claims"][i] = claim_no_refs


async def main(cfg: hl.Config) -> None:
    breakdown_paths = await list_samples(cfg.samples_path)
    breakdowns = dict(await parse_breakdowns(breakdown_paths))

    samples = {}

    stats = defaultdict(int)
    with RichTableProgress(
        total=len(breakdowns), persistent_every=100, print_every=5
    ) as pbar:
        for breakdown_path, breakdown in breakdowns.items():
            sample_dir = breakdown_path.parent
            app_num = sample_dir.name

            try:
                (breakdown_with_passages, rejected, granted, cited) = (
                    load_and_filter_breakdown(
                        app_num, breakdown, sample_dir, cfg.cited_path
                    )
                )

            except PackagingError as e:
                stats[e.reason] += 1

            except Exception as e:
                logger.error(f"Unexpected error packaging sample {app_num}: {e}")
                stats[repr(e)] += 1

            else:
                samples[app_num] = (
                    breakdown_with_passages,
                    rejected,
                    granted,
                    cited,
                    None,
                )
                stats["success"] += 1

            pbar.update(1, data=stats)

    logger.info("Breakdown stats before stratification:")
    breakdown_stats({app_num: sample[0] for app_num, sample in samples.items()})

    logger.info("Spurious correlations before stratification:")
    spurious_correlation_stats(
        (doc, granted, sample[3]["publication_number"])
        for sample in samples.values()
        for doc, granted in ((sample[1], False), (sample[2], True))
    )

    remove_reference_numerals(samples)

    logger.info("Stratifying dataset on topic claim length ...")
    max_len = max(
        len(doc["claims"][0].split())
        for sample in samples.values()
        for doc in (sample[1], sample[2])
    )
    min_len = min(
        len(doc["claims"][0].split())
        for sample in samples.values()
        for doc in (sample[1], sample[2])
    )
    n_bins = 100
    bins = np.linspace(min_len, max_len, n_bins + 1)[1:]

    logger.info("Length stats before stratification:")
    binned_samples_ids, binned_counts = bin_samples(samples, bins.tolist())

    stratified_samples_ids = {b: ([], []) for b in bins}
    for b, (rejected_samples, granted_samples) in binned_samples_ids.items():
        n = min(len(rejected_samples), len(granted_samples))
        random.shuffle(rejected_samples)
        random.shuffle(granted_samples)
        rej_samples_chosen = rejected_samples[:n]
        grant_samples_chosen = granted_samples[:n]
        stratified_samples_ids[b] = (rej_samples_chosen, grant_samples_chosen)

    stratified_samples = {}
    for b, (rejected_samples, granted_samples) in stratified_samples_ids.items():
        for app_num, granted in chain(rejected_samples, granted_samples):
            if app_num in stratified_samples:
                key = "granted" if granted else "rejected"
                stratified_samples[app_num][4]["include_versions"].append(key)
            else:
                metadata = {"include_versions": ["granted" if granted else "rejected"]}
                stratified_samples[app_num] = (*samples[app_num][:4], metadata)

    print("\n" * 5)

    logger.info("Length stats after stratification:")
    strat_binned_samples_ids, strat_binned_counts = bin_samples(
        stratified_samples, bins.tolist()
    )

    logger.info("Breakdown stats before stratification:")
    breakdown_stats(
        {app_num: sample[0] for app_num, sample in stratified_samples.items()}
    )

    logger.info("Spurious correlations after stratification:")
    spurious_correlation_stats(
        (doc, granted, sample[3]["publication_number"])
        for sample in stratified_samples.values()
        for doc, granted in ((sample[1], False), (sample[2], True))
        if {True: "granted", False: "rejected"}[granted]
        in sample[4]["include_versions"]
    )

    lengths_r = [
        l
        for sample in samples.values()
        if (l := len(sample[1]["claims"][0].split())) <= 750  # noqa
    ]
    lengths_g = [
        l
        for sample in samples.values()
        if (l := len(sample[2]["claims"][0].split())) <= 750  # noqa
    ]

    with open(root / "data" / "logs" / "claim_lens_before_strat.pkl", "wb") as f:
        pickle.dump((lengths_r, lengths_g), f)

    logger.info(f"Splitting dataset with ratios: {SPLITS} ...")
    split_counts = {split: 0 for split in SPLITS.keys()}
    for sample in stratified_samples.values():
        metadata = sample[4]
        split = np.random.choice(list(SPLITS.keys()), p=list(SPLITS.values())).item()
        metadata["split"] = split
        split_counts[split] += len(metadata["include_versions"])
    total = sum(split_counts.values())
    for split, count in split_counts.items():
        logger.info(f"  {split:<10}: {count} samples ({count/total:.2%})")

    logger.info(f"Saving dataset to {cfg.packaged_path} ...")
    for app_num, (
        breakdown_with_passages,
        rejected,
        granted,
        cited,
        metadata,
    ) in stratified_samples.items():
        out_path: Path = cfg.packaged_path / app_num
        out_path.mkdir(parents=True, exist_ok=True)

        json_path = out_path / "breakdown.json"
        json_path.write_text(
            breakdown_with_passages.model_dump_json(indent=2, serialize_as_any=True)
        )

        md_path = out_path / "breakdown.md"
        md_path.write_text(format_breakdown(breakdown_with_passages, app_num))

        rejected_path = out_path / "rejected_patent.json"
        rejected_path.write_text(json.dumps(rejected, indent=2))

        granted_path = out_path / "granted_patent.json"
        granted_path.write_text(json.dumps(granted, indent=2))

        cited_path = out_path / "cited_patent.json"
        cited_path.write_text(json.dumps(cited, indent=2))

        metadata_path = out_path / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    cfg.apply()
    with OutputRedirector(root / "data" / "logs" / (Path(__file__).stem + ".log")):
        asyncio.run(main(cfg))
