# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: MIT

import asyncio
import os
from collections import defaultdict
from pathlib import Path

import aiofiles
import fitz
import hydralette as hl
from mineru_vl_utils import MinerUClient
from pyrootutils import setup_root
import tqdm

from whats_novel.utils.logs import OutputRedirector, RichTableProgress, get_logger
from whats_novel.utils import async_utils

logger = get_logger()
root = setup_root(__file__)

cfg = hl.Config(
    samples_path=root / "data" / "samples",
    n_parallel_pdfs=8,
    n_parallel_requests=100,
    mineru_vl_server="http://localhost:59535",
    reversed=False,
)

PREVIOUS_SAMPLES_DIR = root / "data" / "samples_v1"


async def process_pdf(
    path: Path, mineru_client: MinerUClient, semaphore: asyncio.Semaphore
) -> str:
    if path.with_suffix(".md").exists():
        return "skipped/exists"

    previous_path = PREVIOUS_SAMPLES_DIR / path.relative_to(cfg.samples_path)
    if previous_path.with_suffix(".md").exists():
        async with (
            aiofiles.open(
                previous_path.with_suffix(".md"), "r", encoding="utf-8"
            ) as f_src,
            aiofiles.open(path.with_suffix(".md"), "w", encoding="utf-8") as f_dst,
        ):
            async for line in f_src:
                await f_dst.write(line)
            return "skipped/previous"

    async with aiofiles.open(path, "rb") as f:
        pdf_bytes = await f.read()
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [page.get_pixmap().pil_image() for page in pdf_document]  # type: ignore
    extracted_blocks_futures = [
        mineru_client.aio_two_step_extract(page, semaphore=semaphore) for page in pages
    ]
    extracted_blocks_per_page = await asyncio.gather(*extracted_blocks_futures)

    async with aiofiles.open(path.with_suffix(".md"), "w", encoding="utf-8") as f:
        for blocks in extracted_blocks_per_page:
            for block in blocks:
                if block["type"] == "text":
                    await f.write(block["content"] + "\n")

    return "success"


def is_complete(path: Path) -> bool:
    return (
        any(path.glob("publications/*A1.json"))
        and (
            any(path.glob("publications/*B1.json"))
            or any(path.glob("publications/*B2.json"))
        )
        and path.joinpath("rejection.pdf").exists()
    )


async def list_samples(samples_path: Path, reversed: bool) -> list[Path]:
    def check_sample(sample_dir: Path) -> Path | None:
        if is_complete(sample_dir):
            return sample_dir / "rejection.pdf"
        else:
            return None

    files = []
    with tqdm.tqdm(desc="Listing sample directories") as pbar:
        async for _, path, _ in async_utils.run_async(
            ({"sample_dir": sd} for sd in samples_path.glob("*")),
            check_sample,
            n_workers=16,
        ):
            if path is not None:
                files.append(path)
                pbar.set_postfix({"complete": len(files)})
            pbar.update(1)
    files.sort(key=lambda p: p.parent.name, reverse=reversed)
    return files


async def main(cfg: hl.Config) -> None:
    logger.info("Initializing MinerU client ...")
    os.environ["MINERU_VL_SERVER"] = cfg.mineru_vl_server
    mineru_client = MinerUClient("http-client")

    logger.info("Listing rejection documents ...")
    rejections = await list_samples(cfg.samples_path, cfg.reversed)
    logger.info(f"Found {len(rejections)} rejection documents")

    # mineru launches a lot of parallel requests per pdf, which differs unpredictably
    # therefore, in addition to limiting the number of parallel pdfs being processed,
    # this semaphore limits the total number of parallel requests to vllm
    semaphore = asyncio.Semaphore(cfg.n_parallel_requests)

    stats = defaultdict(int)
    with RichTableProgress(total=len(rejections), persistent_every=50) as pbar:
        async for _, status, error in async_utils.run_async(
            [
                {
                    "path": rejection,
                    "mineru_client": mineru_client,
                    "semaphore": semaphore,
                }
                for rejection in rejections
            ],
            process_pdf,
            n_workers=cfg.n_parallel_pdfs,
        ):
            if error:
                stats[repr(error)] += 1
            else:
                stats[status] += 1
            pbar.update(
                1,
                increment_empty=int(status.startswith("skipped"))
                if isinstance(status, str)
                else 0,
                data=stats,
            )


if __name__ == "__main__":
    cfg.apply()
    with OutputRedirector(root / "data" / "logs" / (Path(__file__).stem + ".log")):
        asyncio.run(main(cfg))
