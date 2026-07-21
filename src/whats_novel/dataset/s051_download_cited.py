# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: MIT

import asyncio
import json
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import aiofiles
import aiohttp
import hydralette as hl
from pyrootutils import setup_root
from tqdm import tqdm

from whats_novel.dataset.data_model import ClaimBreakdown
from whats_novel.dataset.s022_parse_topics import parse_publication
from whats_novel.dataset.uspto import USPTOBulkDataClient
from whats_novel.utils.async_utils import run_async
from whats_novel.utils.logs import RichTableProgress, get_logger

logger = get_logger()
root = setup_root(__file__)

cfg = hl.Config(
    cited_path=root / "data" / "cited",
    samples_path=root / "data" / "samples",
    packaged_path=root / "data" / "packaged",
)


@dataclass
class PatentNumber:
    country_code: str
    number: str
    kind_code: str

    @classmethod
    def parse(cls, identifier: str, default_kc: str = "A1") -> "PatentNumber":
        try:
            return cls.parse_default_format(identifier, default_kc)
        except Exception:
            return cls.parse_other_format(identifier, default_kc)

    @classmethod
    def parse_default_format(
        cls, identifier: str, default_kc: str = "A1"
    ) -> "PatentNumber":
        identifier = re.sub(r"[\s/]", "", identifier)
        cc, num, kc = re.findall(r"([A-Z]{2})(\d+)([A-Z](?:\d+)?)?", identifier)[0]
        if cc == "US" and kc in ("A1", "A2") and len(num) <= 11:
            num = num[:4] + "0" * (11 - len(num)) + num[4:]
        return cls(
            country_code=cc,
            number=num,
            kind_code=kc or default_kc,
        )

    @classmethod
    def parse_other_format(
        cls, identifier: str, default_kc: str = "A1"
    ) -> "PatentNumber":
        identifier = re.sub(r"[\s/]", "", identifier)
        cc, num = re.findall(r"([A-Z]{2})-A-(\d+)", identifier)[0]
        return cls(
            country_code=cc,
            number=num,
            kind_code=default_kc,
        )

    def __str__(self) -> str:
        return f"{self.country_code}{self.number}{self.kind_code}"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash(str(self))


def extract_cited_documents(breakdowns: Iterable[ClaimBreakdown]) -> set[str]:
    identifiers = set()
    for breakdown in breakdowns:
        prior_art = {
            prior_art.label: prior_art for prior_art in breakdown.prior_art_documents
        }
        for feature_breakdown in breakdown.breakdown or []:
            for prior_art_ref in feature_breakdown.prior_art_references:
                label = prior_art_ref.document
                if label in prior_art:
                    doc = prior_art[label]
                    if doc.identifier and doc.type == "patent":
                        try:
                            identifier = PatentNumber.parse(doc.identifier)
                            identifiers.add(identifier)
                        except Exception as e:
                            logger.warning(
                                f"Error parsing identifier {doc.identifier}: {e}"
                            )
    return identifiers


def is_complete(path: Path) -> bool:
    return (
        any(path.glob("publications/*A1.json"))
        and (
            any(path.glob("publications/*B1.json"))
            or any(path.glob("publications/*B2.json"))
        )
        and path.joinpath("rejection.pdf").exists()
        and path.joinpath("rejection.md").exists()
        and path.joinpath("rejection.json").exists()
        and path.joinpath("rejection_meta_rule-based.json").exists()
        and path.joinpath("rejection_claim_1_match.log").exists()
    )


async def list_samples(samples_path: Path) -> list[Path]:
    def check_sample(sample_dir: Path) -> Path | None:
        if is_complete(sample_dir):
            return sample_dir / "rejection.json"
        else:
            return None

    files = []
    with tqdm(desc="Listing sample directories") as pbar:
        async for _, path, _ in run_async(
            ({"sample_dir": sd} for sd in samples_path.glob("*")),
            check_sample,
            n_workers=16,
        ):
            if path is not None:
                files.append(path)
                pbar.set_postfix({"complete": len(files)})
            pbar.update(1)
    files.sort(key=lambda p: p.parent.name)
    return files


async def parse_breakdowns(
    breakdown_paths: list[Path],
) -> list[tuple[Path, ClaimBreakdown]]:
    async def parse(path: Path) -> ClaimBreakdown:
        content = path.read_text()
        return ClaimBreakdown.model_validate_json(content)

    with tqdm(desc="Parsing breakdowns") as pbar:
        breakdowns = []
        async for inputs, breakdown, _ in run_async(
            ({"path": p} for p in breakdown_paths),
            parse,
            n_workers=16,
        ):
            breakdowns.append((inputs["path"], breakdown))
            pbar.update(1)
    return breakdowns


async def get_uspto_document(
    identifier: PatentNumber, client: USPTOBulkDataClient
) -> tuple[str, dict]:
    patent = client.get(str(identifier))
    if patent is None:
        return "failed/pickle_not_found", {}
    elif not patent:
        return "failed/pickle_empty", {}

    patent["zip_file"] = str(patent.get("zip_file", ""))
    return f"success/{identifier.country_code}", patent


async def get_epo_document(
    identifier: PatentNumber, session: aiohttp.ClientSession
) -> tuple[str, str]:
    url = (
        "https://data.epo.org/publication-server/rest/v1.2/patents/"
        f"{identifier.country_code}{identifier.number}NW{identifier.kind_code}/document.xml"
    )
    try:
        await asyncio.sleep(random.uniform(0.1, 0.5))
        async with session.get(url, proxy=os.environ["HTTP_PROXY"]) as response:
            if response.status != 200:
                return f"failed/{response.status}", ""
            content = await response.text()
            return f"success/{identifier.country_code}", content
    except Exception as e:
        return f"failed/{type(e).__name__}", ""


epo_supported_ccs = re.split(
    r",\s*",
    "EP, WO, AT, BE, BG, CA, CH, CY, CZ, DK, EE, ES, FR, GB, GR, HR, "
    "IE, IT, LT, LU, MC, MD, ME, NO, PL, PT, RO, RS, SE, SK",
)


async def get_document(
    identifier: PatentNumber,
    session: aiohttp.ClientSession,
    uspto_client: USPTOBulkDataClient,
    out_path: Path,
) -> str:
    if identifier.country_code == "US":
        status, document = await get_uspto_document(identifier, uspto_client)
        document = json.dumps(document)

        async with aiofiles.open(out_path, "w") as f:
            await f.write(document)

    elif identifier.country_code in epo_supported_ccs:
        status, document_xml = await get_epo_document(identifier, session)
        if document_xml:
            out_path_xml = out_path.with_suffix(".xml")
            async with aiofiles.open(out_path_xml, "w") as f:
                await f.write(document_xml)
            await parse_publication(out_path_xml)

    else:
        # logger.warning(f"Skipping unsupported country code: {identifier.country_code}")
        return "skipped/country_code"

    return status


async def main(cfg: hl.Config) -> None:
    breakdown_paths = await list_samples(cfg.samples_path)
    breakdowns = await parse_breakdowns(breakdown_paths)
    identifiers = extract_cited_documents([b for _, b in breakdowns])
    logger.info(f"Extracted {len(identifiers)} identifiers")

    stats = defaultdict(int)
    async with (
        RichTableProgress(total=len(identifiers)) as pbar,
        aiohttp.ClientSession() as session,
        USPTOBulkDataClient(
            root / "data" / "uspto" / "patents.pickle",
            root / "data" / "uspto" / "patents_index.pickle",
        ) as uspto_client,
    ):
        async for inputs, status, error in run_async(
            [
                {
                    "identifier": identifier,
                    "session": session,
                    "uspto_client": uspto_client,
                    "out_path": cfg.cited_path / (str(identifier) + ".json"),
                }
                for identifier in identifiers
            ],
            get_document,
            n_workers=16,
        ):
            identifier: PatentNumber = inputs["identifier"]
            stats[f"processed/{identifier.country_code}"] += 1

            if error:
                # logger.error(f"Error fetching document for {identifier}: {error}")
                stats[f"failed/{error!r}"] += 1

            else:
                stats[status] += 1

            pbar.update(1, data=stats)


if __name__ == "__main__":
    cfg.apply()
    asyncio.run(main(cfg))
