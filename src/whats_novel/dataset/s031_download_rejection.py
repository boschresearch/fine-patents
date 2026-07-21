import asyncio
import os
import random
from collections import defaultdict
from pathlib import Path
import shutil

import aiohttp
import hydralette as hl
from bs4 import BeautifulSoup
from pyrootutils import setup_root

from whats_novel.utils.logs import OutputRedirector, RichTableProgress, get_logger

logger = get_logger()
root = setup_root(__file__)

cfg = hl.Config(
    samples_path=root / "data" / "samples",
)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

sleep_after_request = 0.3  # seconds
sleep_after_429 = 10  # seconds
sleep_random_before_request = lambda: random.uniform(0, 1)  # noqa

PREVIOUS_SAMPLES_PATH = root / "data" / "samples_v1"


async def get_european_search_opinion_id(
    session: aiohttp.ClientSession, app_number: str
) -> tuple[str | None, str | None]:
    url = f"https://register.epo.org/application?number={app_number}&lng=en&tab=doclist"
    try:
        proxy = os.environ.get("HTTP_PROXY")
        async with session.get(url, proxy=proxy, headers=headers) as response:
            if response.status != 200:
                if response.status == 429:
                    logger.info(
                        f"Cloudflare challenge detected, retrying after {sleep_after_429}s..."
                    )
                    await asyncio.sleep(sleep_after_429)
                    return await get_european_search_opinion_id(session, app_number)
                return (None, f"ERROR: HTTP {response.status}")
            content = await response.text()
            soup = BeautifulSoup(content, "html.parser")
            for link in soup.find_all("a"):
                if link.get_text(strip=True) == "European search opinion":
                    return (link.get("id"), None)  # type: ignore
            return (None, "ERROR: Not found")
    except Exception as e:
        return (None, f"ERROR: {type(e).__name__}")


async def download_search_opinion_pdf(
    session: aiohttp.ClientSession, app_number: str, doc_id: str, path: Path
) -> tuple[bool, str | None]:
    pdf_url = f"https://register.epo.org/application?showPdfPage=all&documentId={doc_id}&appnumber={app_number}&proc="
    try:
        proxy = os.environ.get("HTTP_PROXY")
        async with session.get(pdf_url, proxy=proxy, headers=headers) as resp:
            if not resp.ok:
                if resp.status == 429:
                    logger.info(
                        f"Cloudflare challenge detected, retrying after {sleep_after_429}s..."
                    )
                    await asyncio.sleep(sleep_after_429)
                    return await download_search_opinion_pdf(
                        session, app_number, doc_id, path
                    )
                return (False, f"ERROR: HTTP {resp.status}")
            if not resp.headers.get("Content-Type", "").startswith("application/pdf"):
                return (False, "ERROR: Invalid content type")
            content = await resp.read()
            with open(path, "wb") as f:
                f.write(content)
            return (True, None)
    except Exception as e:
        return (False, f"ERROR: {type(e).__name__}")


def get_non_exist_marker_path(path: Path) -> Path:
    return path.with_suffix(".NOT_EXIST")


def write_non_exist_marker(path: Path) -> None:
    marker_path = get_non_exist_marker_path(path)
    with open(marker_path, "w") as f:
        f.write("")


async def process_application(
    session: aiohttp.ClientSession,
    app_num: str,
    out_path: Path,
) -> dict:
    stats = defaultdict(int)

    previous_out_path = PREVIOUS_SAMPLES_PATH / out_path.relative_to(cfg.samples_path)
    if previous_out_path.exists():
        shutil.copy(previous_out_path, out_path)
        stats["PDF/copied_previous"] += 1
        return stats

    if get_non_exist_marker_path(previous_out_path).exists():
        write_non_exist_marker(out_path)
        stats["PDF/skipped_not_exist_previous"] += 1
        return stats

    if out_path.exists():
        stats["PDF/skipped_exists"] += 1
        return stats

    if get_non_exist_marker_path(out_path).exists():
        stats["PDF/skipped_not_exist"] += 1
        return stats

    await asyncio.sleep(sleep_random_before_request())
    try:
        doc_id, error = await get_european_search_opinion_id(session, app_num)
        if error or not doc_id:
            stats[f"ID/{error or 'ERROR: Unknown'}"] += 1
            write_non_exist_marker(out_path)
            return stats
        await asyncio.sleep(sleep_after_request)

        success, download_error = await download_search_opinion_pdf(
            session, app_num, doc_id, out_path
        )
        if success:
            stats["PDF/downloaded"] += 1
        else:
            stats[f"PDF/{download_error or 'ERROR: Unknown'}"] += 1
            write_non_exist_marker(out_path)

    except Exception as e:
        logger.error(f"[{app_num}] Unexpected error: {e}")
        stats[f"ERROR/{type(e).__name__}"] += 1

    finally:
        await asyncio.sleep(sleep_after_request)

    return stats


async def process_application_worker(
    work_queue: asyncio.Queue[tuple[str, Path, RichTableProgress]],
    session: aiohttp.ClientSession,
    stats: dict,
) -> None:
    while True:
        app_num, out_path, pbar = await work_queue.get()
        stats_ = {}
        try:
            stats_ = await process_application(session, app_num, out_path)
        except Exception as e:
            logger.error(f"[{app_num}] Worker error: {e}")
        finally:
            work_queue.task_done()
            for k, v in stats_.items():
                stats[k] += v
            if pbar is not None:
                pbar.update(
                    1,
                    increment_empty=stats_.get("PDF/skipped_not_exist", 0)
                    + stats_.get("PDF/skipped_exists", 0),
                    data=stats,
                )


def is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    if not any(path.glob("publications/*A1.json")):
        return False
    if not (
        any(path.glob("publications/*B1.json"))
        or any(path.glob("publications/*B2.json"))
    ):
        return False
    return True


async def main(cfg: hl.Config) -> None:
    logger.info("Listing applications")
    sample_dirs = [p for p in cfg.samples_path.glob("*") if is_complete(p)]
    sample_dirs.sort(key=lambda p: p.name)
    logger.info(f"Total applications: {len(sample_dirs)}")

    stats = defaultdict(int)
    # EPO has strict rate limits for the register, even one concurrent request will reach the limit quickly
    num_workers = 1

    async with aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar()) as session:
        pbar = RichTableProgress(total=len(sample_dirs), persistent_every=10)

        work_queue = asyncio.Queue()
        for sample_dir in sample_dirs:
            await work_queue.put((sample_dir.name, sample_dir / "rejection.pdf", pbar))

        workers = [
            asyncio.create_task(process_application_worker(work_queue, session, stats))
            for _ in range(num_workers)
        ]

        await work_queue.join()
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)


if __name__ == "__main__":
    cfg.apply()
    with OutputRedirector(root / "data" / "logs" / (Path(__file__).stem + ".log")):
        asyncio.run(main(cfg))
