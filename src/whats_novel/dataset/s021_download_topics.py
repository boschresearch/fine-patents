# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: MIT

import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path

import aiofiles
import aiohttp
import hydralette as hl
from pyrootutils import setup_root

from whats_novel.utils import async_utils
from whats_novel.utils.logs import RichTableProgress, get_logger, OutputRedirector

logger = get_logger()
root = setup_root(__file__)

cfg = hl.Config(
    application_numbers=hl.Field(
        default=root / "data" / "applications_publications.json", type=Path
    ),
    save_path=root / "data" / "samples",
    n_workers=10,
)

PREVIOUS_DATA_DIR = root / "data" / "samples_v1"


async def download_publication_data(
    session: aiohttp.ClientSession, pub_num: str, app_num: str
) -> tuple[str, str | None, str | None]:
    out_path = PREVIOUS_DATA_DIR / app_num / "publications" / f"{pub_num}.xml"
    if out_path.exists():
        async with aiofiles.open(out_path, "r") as f:
            content = await f.read()
        return (pub_num, content, "copied")

    url = f"https://data.epo.org/publication-server/rest/v1.2/patents/{pub_num[:-2]}NW{pub_num[-2:]}/document.xml"
    try:
        async with session.get(url, proxy=os.environ["HTTP_PROXY"]) as response:
            if response.status == 200:
                content = await response.text()
                return (pub_num, content, "downloaded")
            else:
                return (pub_num, None, f"ERROR: HTTP {response.status}")
    except Exception as e:
        return (pub_num, None, f"ERROR: {type(e).__name__}")


async def process_application(
    session: aiohttp.ClientSession,
    app_num: str,
    pubs: list[str],
    save_path: Path,
) -> dict:
    stats = defaultdict(int)
    tasks = [download_publication_data(session, pub_num, app_num) for pub_num in pubs]
    results = await asyncio.gather(*tasks)

    for _, _, status in results:
        stats[f"PUBS/{status}"] += 1

    if any(not r[1] for r in results):
        stats["APPS/failed"] += 1
        return stats

    pub_dir = save_path / app_num / "publications"
    pub_dir.mkdir(parents=True, exist_ok=True)

    for pub_num, content, _ in results:
        assert content
        file_path = pub_dir / f"{pub_num}.xml"
        file_path.write_text(content)

    if all(r[2] == "copied" for r in results):
        stats["APPS/copied"] += 1
    else:
        stats["APPS/downloaded"] += 1

    return stats


async def main(cfg: hl.Config) -> None:
    """Main async function to download all publications."""
    applications: dict[str, list[str]] = json.loads(cfg.application_numbers.read_text())
    logger.info(f"Total applications: {len(applications)}")

    applications_granted = {
        app_num: pubs
        for app_num, pubs in applications.items()
        if (
            any(any(pub.endswith(suffix) for suffix in ("B1", "B2")) for pub in pubs)
            and any(pub.endswith("A1") for pub in pubs)
        )
    }
    logger.info(f"Applications with granted publications: {len(applications_granted)}")

    total_pubs = sum(len(pubs) for pubs in applications_granted.values())
    logger.info(f"Total publications to download: {total_pubs}")

    stats = defaultdict(int)
    async with (
        aiohttp.ClientSession() as session,
        RichTableProgress(
            total=len(applications_granted), print_every=50, persistent_every=1000
        ) as pbar,
    ):
        async for _, stats_, error in async_utils.run_async(
            inputs=[
                {
                    "app_num": app_num,
                    "pubs": pubs,
                    "session": session,
                    "save_path": cfg.save_path,
                }
                for app_num, pubs in applications_granted.items()
            ],
            func=process_application,
            n_workers=cfg.n_workers,
        ):
            if error:
                stats[f"APPS/{error}"] += 1
            else:
                assert stats_
                for k, v in stats_.items():
                    stats[k] += v

            pbar.update(1, data=stats, increment_empty=stats.get("APPS/copied", 0))


if __name__ == "__main__":
    cfg.apply()
    with OutputRedirector(root / "data" / "logs" / (Path(__file__).stem + ".log")):
        asyncio.run(main(cfg))
