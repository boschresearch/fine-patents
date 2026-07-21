import asyncio
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

import aiofiles
import hydralette as hl
import regex
import rich.text
from lingua import Language, LanguageDetectorBuilder
from pyrootutils import setup_root

from whats_novel.utils.async_utils import run_async
from whats_novel.utils.logs import (
    OutputRedirector,
    RichTableProgress,
    get_logger,
)

logger = get_logger()
root = setup_root(__file__)

cfg = hl.Config(
    samples_path=root / "data" / "samples",
)


def find_claim_in_rejection(claim: str, rejection: str, **kwargs) -> dict | None:
    def normalize_text(text: str) -> str:
        text = text.lower()
        text = text.strip()
        text = re.sub(r"^\d+\.", "", text)
        text = re.sub(r"[^\S\n]+", " ", text)
        for punct in string.punctuation:
            if punct not in ["(", ")", "[", "]"]:
                text = text.replace(punct, "")
        text = text.strip()
        return text

    if not claim or not rejection:
        return None

    claim = normalize_text(claim)
    rejection = normalize_text(rejection)

    words = [w for w in re.split(r"\s+", claim) if w]
    pieces = []
    for i, word in enumerate(words):
        next_word = words[i + 1] if i + 1 < len(words) else None
        if next_word:
            reference_pat = (
                rf"(?P<reference_{i}>(?:(?!\b{re.escape(next_word)}\b).)*)\s*"
            )
        else:
            reference_pat = rf"(?P<reference_{i}>.*?)\s*"
        pieces.append(rf"(?P<word_{i}>{re.escape(word)})\s+" + reference_pat)

    pattern = regex.compile(
        "".join(pieces) + r"\n",
        flags=regex.IGNORECASE | regex.DOTALL,
    )

    match = None
    match_offset = 0
    search_start = 0
    while search_start < len(rejection):
        match_ = pattern.search(rejection[search_start:])
        if not match_:
            break
        match = match_
        match_offset = search_start
        search_start += match.start() + 1

    if match is None:
        return None

    match_dict = {
        "rejection_normalized": rejection,
        "start": match_offset + match.start(),
        "end": match_offset + match.end(),
        "groups": {
            groupname: {
                "start": match_offset + match.start(i + 1),
                "end": match_offset + match.end(i + 1),
                "text": match.group(groupname),
            }
            for i, groupname in enumerate(match.groupdict())
            if match.group(groupname) is not None
        },
    }
    return match_dict


def highlight_match(rejection: str, match: dict) -> rich.text.Text:
    rejection_styled = rich.text.Text(rejection[match["start"] : match["end"]])
    for groupname, group_info in match["groups"].items():
        color = "red" if groupname.startswith("word_") else "blue"
        start = group_info["start"] - match["start"]
        end = group_info["end"] - match["start"]
        rejection_styled.stylize(f"bold {color}", start, end)
    return rejection_styled


def first_n(it, n):
    for i, x in enumerate(it):
        if i >= n:
            break
        yield x


async def main(cfg: hl.Config) -> None:
    logger.info("Listing OCR'ed rejection files...")
    # md_rejection_paths = list(first_n(cfg.samples_path.glob("*/rejection.md"), 1000))
    md_rejection_paths = list(cfg.samples_path.glob("*/rejection.md"))

    async def read_file(path):
        async with aiofiles.open(path, "r") as f:
            return await f.read()

    logger.info("Loading OCR'ed rejection files...")
    md_rejections = await asyncio.gather(
        *[read_file(path) for path in md_rejection_paths]
    )

    async def read_first_claim(path):
        pub_a1_path = next(path.parent.glob("publications/*A1.json"))
        content = await read_file(pub_a1_path)
        return json.loads(content)["claims"][0]

    logger.info("Loading first claims...")
    first_claims = await asyncio.gather(
        *[read_first_claim(path) for path in md_rejection_paths]
    )

    logger.info("Detecting languages...")
    languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    languages = [detector.detect_language_of(text) for text in md_rejections]
    lang_counts = Counter(lang for lang in languages if lang is not None)
    logger.info(
        "Language statistics: "
        + ", ".join(f"{lang.name}: {count}" for lang, count in lang_counts.items())
    )

    samples = [
        {
            "path": path,
            "rejection": rejection,
            "claim": claim,
            "language": lang.name if lang else None,
        }
        for path, rejection, lang, claim in zip(
            md_rejection_paths, md_rejections, languages, first_claims
        )
    ]

    logger.info("Finding claims in rejections...")
    stats = defaultdict(int)
    with RichTableProgress(total=len(samples), persistent_every=100) as pbar:
        matches = []
        async for sample, match, error in run_async(
            [{"sample_index": i, **sample} for i, sample in enumerate(samples)],
            find_claim_in_rejection,
            n_workers=16,
            use_processes=True,
        ):
            # due to multiprocessing, sample might be a copy, so we need the index in the original list
            samples[sample["sample_index"]]["claim_1_found_in_rejection"] = (
                match is not None
            )
            samples[sample["sample_index"]]["claim_1_match"] = (
                highlight_match(match["rejection_normalized"], match) if match else None
            )
            matches.append(match)
            if error:
                stats[error] += 1
                logger.error(f"Error: {error}")
            elif match:
                stats["matched"] += 1
            else:
                stats["not_matched"] += 1
            pbar.update(1, data=stats)

    logger.info("Saving information")
    successful = 0
    for sample in samples:
        metadata_path = sample["path"].parent / "rejection_meta_rule-based.json"
        claim_match_path = sample["path"].parent / "rejection_claim_1_match.log"
        if sample["language"] == Language.ENGLISH.name and sample.get(
            "claim_1_found_in_rejection", False
        ):
            async with aiofiles.open(metadata_path, "w") as f:
                await f.write(
                    json.dumps(
                        {
                            "language": sample["language"],
                            "claim_1_found_in_rejection": sample[
                                "claim_1_found_in_rejection"
                            ],
                        },
                        indent=2,
                    )
                )
                successful += 1

                with open(claim_match_path, "w") as fp:
                    file_console = rich.console.Console(file=fp)
                    file_console.print(sample["claim_1_match"])

        else:
            if metadata_path.exists():
                metadata_path.unlink()
            if claim_match_path.exists():
                claim_match_path.unlink()

    logger.info(f"Saved {successful} samples with claim 1 found in rejection.")


if __name__ == "__main__":
    cfg.apply()
    with OutputRedirector(root / "data" / "logs" / (Path(__file__).stem + ".log")):
        asyncio.run(main(cfg))
