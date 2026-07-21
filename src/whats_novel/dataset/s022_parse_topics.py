# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: MIT

import asyncio
import json
import re
from collections import defaultdict
from pathlib import Path

import aiofiles
import hydralette as hl
import lxml.etree
from pyrootutils import setup_root

import whats_novel.utils.async_utils as async_utils
from whats_novel.utils.logs import RichTableProgress, get_logger, OutputRedirector

logger = get_logger()
root = setup_root(__file__)

cfg = hl.Config(
    n_workers=16, samples_path=root / "data" / "samples", multi_process=True
)


def pub_get_title(tree: lxml.etree._Element) -> str | None:
    title = None
    b540 = tree.find(".//B540")
    if b540 is not None:
        for i, elem in enumerate(b540):
            if elem.tag == "B541" and elem.text == "en":
                if i + 1 < len(b540) and b540[i + 1].tag == "B542":
                    title = b540[i + 1].text
                    break
    return title


def pub_get_abstract(tree: lxml.etree._Element) -> str | None:
    abstract_nodes = tree.xpath(".//abstract[@lang='en']")
    if abstract_nodes:
        return "".join(abstract_nodes[0].itertext()).strip()  # type: ignore


def pub_get_description(tree: lxml.etree._Element) -> list[str | None] | None:
    description_nodes = tree.xpath(".//description[@lang='en']")
    if not isinstance(description_nodes, list) or not description_nodes:
        return None
    description_node: lxml.etree._Element = description_nodes[0]  # type: ignore
    paragraphs_and_numbers: list[tuple[int, str]] = []
    for p in description_node.findall("./p"):
        p_num = p.attrib.get("num", "")
        if p_num:
            paragraphs_and_numbers.append(
                (int(p_num.lstrip("0")), f"[{p_num}] {''.join(p.itertext()).strip()}")  # type: ignore
            )
    max_number = max(paragraphs_and_numbers, key=lambda x: x[0])[0]
    paragraphs: list[str | None] = [None] * max_number
    for p_num, p_text in paragraphs_and_numbers:
        paragraphs[p_num - 1] = p_text
    return paragraphs


def pub_get_claims(tree: lxml.etree._Element, id: str) -> list[str | None] | None:
    claims_nodes = tree.xpath(".//claims[@lang='en']")
    if not isinstance(claims_nodes, list) or not claims_nodes:
        return None
    claims_node: lxml.etree._Element = claims_nodes[0]  # type: ignore
    claims_and_numbers: list[tuple[int, str]] = []
    for claim in claims_node.findall("./claim"):
        claim_text = "".join(claim.itertext()).strip()  # type: ignore
        p_num = (
            claim.attrib.get("num", "")
            or re.search(r"^\d+", claim_text).group(0)  # type: ignore
            or re.search(r"\d+", claim.attrib.get("id", "")).group(0)  # type: ignore
        )
        p_num = int(p_num.lstrip("0"))
        if p_num:
            if not re.search(r"^\d+", claim_text):
                claim_text = f"{p_num}. {claim_text}"
            claims_and_numbers.append((p_num, claim_text))
    max_number = max(claims_and_numbers, key=lambda x: x[0])[0]
    claims: list[str | None] = [None] * max_number
    for p_num, p_text in claims_and_numbers:
        if claims[p_num - 1] is not None:
            if p_num == 1:
                raise ValueError("Claim 1 duplicated")
            else:
                logger.warning(
                    f"[{id}] Duplicate claim number {p_num}, keeping the first one"
                )
                continue
        claims[p_num - 1] = p_text
    return claims


async def parse_publication(pub_path: Path) -> None:
    async with aiofiles.open(pub_path, "rb") as f:
        content = await f.read()

    try:
        tree = lxml.etree.fromstring(content)
    except Exception:
        raise ValueError("Failed to parse XML")

    try:
        title = pub_get_title(tree)
        assert title is not None, "No title found"
    except Exception:
        raise ValueError("No title")

    # B1 documents never have abstracts, so we make it optional
    try:
        abstract = pub_get_abstract(tree)
    except Exception:
        raise ValueError("No abstract")

    try:
        description = pub_get_description(tree)
        assert description is not None, "No description found"
    except Exception:
        raise ValueError("No description")

    try:
        claims = pub_get_claims(tree, id=pub_path.stem)
        assert claims is not None, "No claims found"
    except Exception:
        raise ValueError("No claims")

    data = {
        "title": title,
        "abstract": abstract,
        "description": description,
        "claims": claims,
    }

    with pub_path.with_suffix(".json").open("w") as f:
        json.dump(data, f)


async def main(cfg: hl.Config) -> None:
    logger.info("Listing publications")
    publications = list(cfg.samples_path.glob("*/publications/*.xml"))
    logger.info(f"Found {len(publications)} publications")

    with RichTableProgress(
        total=len(publications), print_every=100, persistent_every=10_000
    ) as pbar:
        stats = defaultdict(int)

        async for inputs, _, error in async_utils.run_async(
            inputs=[{"pub_path": pub_path} for pub_path in publications],
            func=parse_publication,
            n_workers=cfg.n_workers,
            use_processes=cfg.multi_process,
        ):
            if error:
                stats[f"{inputs['pub_path'].stem[-2:]}/{str(error)}"] += 1
                stats[str(error)] += 1
            else:
                stats[f"{inputs['pub_path'].stem[-2:]}/success"] += 1
                stats["success"] += 1

            pbar.update(1, data=stats)


if __name__ == "__main__":
    cfg.apply()
    with OutputRedirector(root / "data" / "logs" / (Path(__file__).stem + ".log")):
        asyncio.run(main(cfg))
