import asyncio
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import dspy
import fitz
import hydralette as hl
import rich.console
import rich.panel
import rich.traceback
from pydantic import BaseModel
from pyrootutils import setup_root
from tqdm import tqdm

from whats_novel.dataset.data_model import (
    ClaimBreakdown,
)
from whats_novel.utils import (
    OutputRedirector,
    PydanticAdapter,
    RichTableProgress,
    get_logger,
    run_async,
    console,
)

logger = get_logger()
root = setup_root(__file__)

cfg = hl.Config(
    model="openai/Qwen/Qwen3-VL-235B-A22B-Thinking-FP8",
    base_url="http://localhost:59535/v1",
    samples_path=root / "data" / "samples",
    num_workers=16,
    skip_percent=0.0,
    world_size=1,
    local_rank=0,
)


class ExtractClaimBreakdownSignature(dspy.Signature):
    claim: str = dspy.InputField(desc="The text of claim 1 from the patent application")
    pdf_pages: list[dspy.Image] = dspy.InputField(
        desc="PDF pages of the rejection document"
    )
    output: ClaimBreakdown = dspy.OutputField(
        desc="Structured representation of the rejection of claim 1"
    )


ExtractClaimBreakdownSignature.instructions = (
    Path(__file__).with_suffix(".prompt.md").read_text()
)


class ExtractClaimBreakdown(dspy.Module):
    def __init__(self, n_retries: int = 3) -> None:
        self.extract = dspy.Predict(ExtractClaimBreakdownSignature)
        self.n_retries = n_retries

    @staticmethod
    def load_pdf_pages(path: str) -> list[dspy.Image]:
        doc = fitz.open(path)
        pages = [dspy.Image.from_PIL(page.get_pixmap().pil_image()) for page in doc]  # type: ignore
        return pages

    @staticmethod
    def load_claim_text(path: Path) -> str:
        patent_json_path = next(iter(path.parent.glob("publications/*A1.json")))
        with open(patent_json_path, "r") as f:
            patent_data = json.load(f)
        claim_1_text = patent_data["claims"][0]
        return claim_1_text

    async def aforward(self, pdf_path: Path) -> dspy.Prediction:
        last_exception = None
        for _attempt in range(self.n_retries):
            try:
                breakdown = await self.extract.acall(
                    pdf_pages=self.load_pdf_pages(str(pdf_path)),
                    claim=self.load_claim_text(pdf_path),
                )
                breakdown.n_retries = _attempt
                return breakdown
            except Exception as e:
                logger.warning(
                    f"[{pdf_path.parent.name}] Attempt {_attempt + 1} failed with error: {e}"
                )
                last_exception = e
        raise RuntimeError(
            f"All {self.n_retries} attempts failed for {pdf_path}"
        ) from last_exception

    def forward(self, pdf_path: Path) -> dspy.Prediction:
        last_exception = None
        for _attempt in range(self.n_retries):
            try:
                breakdown = self.extract(
                    pdf_pages=self.load_pdf_pages(str(pdf_path)),
                    claim=self.load_claim_text(pdf_path),
                )
                breakdown.n_retries = _attempt
                return breakdown
            except Exception as e:
                logger.warning(
                    f"[{pdf_path.parent.name}] Attempt {_attempt + 1} failed with error: {e}"
                )
                last_exception = e
        raise RuntimeError(
            f"All {self.n_retries} attempts failed for {pdf_path}"
        ) from last_exception


class ProcessingResult(BaseModel, arbitrary_types_allowed=True):
    pdf_path: Path
    success: bool
    skipped: bool = False
    output: ClaimBreakdown | None = None
    n_retries: int = 0

    def print_panel(self) -> None:
        if self.success and self.output:
            console.print(
                rich.panel.Panel(
                    f"📄 {self.pdf_path.relative_to(root)}\n"
                    f"📄 {self.out_path.relative_to(root)}\n"
                    f"🌍 Language: [cyan]{self.output.language}[/cyan]\n"
                    f"📋 Features: [green]{len(self.output.breakdown) if self.output.breakdown else 0}[/green]\n",
                    title=(
                        f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]"
                        f"[green]✅ {self.pdf_path.stem} Breakdown Extracted[/green]"
                    ),
                    border_style="green",
                    width=100,
                )
            )
        else:
            console.print(
                rich.panel.Panel(
                    rich.console.Group(
                        f"📄 {self.pdf_path.relative_to(root)}\n"
                        f"📄 {self.out_path.relative_to(root)}\n",
                        rich.traceback.Traceback(max_frames=5),
                    ),
                    title=(
                        f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]"
                        f"[red]❌ {self.pdf_path.stem} Processing Failed[/red]"
                    ),
                    border_style="red",
                    width=100,
                )
            )

    def save_result(self) -> None:
        if self.success and self.output:
            with open(self.out_path, "w") as f:
                f.write(self.output.model_dump_json(indent=2))

    def save_messages(self, history_item: dict) -> None:
        messages = history_item["messages"] + [
            {
                "role": "assistant",
                "content": history_item["response"]["choices"][0]["message"]["content"],
            }
        ]
        if self.success and self.output:
            for i in range(len(messages)):
                if isinstance(messages[i]["content"], list):
                    messages[i]["content"] = "\n".join(
                        [
                            piece["text"]
                            if piece.get("type") != "image_url"
                            else "<Image>"
                            for piece in messages[i]["content"]
                        ]
                    )
            with open(self.messages_path, "w") as f:
                f.write(json.dumps(messages, indent=4))

    @property
    def out_path(self) -> Path:
        return self.pdf_path.with_suffix(".json")

    @property
    def label_path(self) -> Path:
        return self.pdf_path.with_suffix(".label.json")

    @property
    def messages_path(self) -> Path:
        return self.pdf_path.with_suffix(".messages.json")


async def process_pdf(pdf: Path) -> ProcessingResult:
    extract_breakdown = ExtractClaimBreakdown()

    if pdf.with_suffix(".json").exists():
        result = ProcessingResult(
            pdf_path=pdf,
            success=True,
            skipped=True,
        )
        return result

    try:
        output = await extract_breakdown.acall(pdf)
        assert output is not None
        result = ProcessingResult(
            pdf_path=pdf,
            success=True,
            output=output.output,
            n_retries=output.n_retries,
        )
        result.save_result()
        result.save_messages(extract_breakdown.history[-1])
        result.print_panel()
        return result

    except Exception as e:
        result = ProcessingResult(pdf_path=pdf, success=False)
        if extract_breakdown.history:
            result.save_messages(extract_breakdown.history[-1])
        result.print_panel()
        raise e


def is_complete(path: Path) -> bool:
    return (
        any(path.glob("publications/*A1.json"))
        and (
            any(path.glob("publications/*B1.json"))
            or any(path.glob("publications/*B2.json"))
        )
        and path.joinpath("rejection.pdf").exists()
        and path.joinpath("rejection.md").exists()
        and path.joinpath("rejection_meta_rule-based.json").exists()
        and path.joinpath("rejection_claim_1_match.log").exists()
    )


async def list_pdf_files(
    samples_path: Path, skip_percent: float, local_rank: int, world_size: int
) -> list[Path]:
    files = []

    with tqdm(desc="Listing PDF files") as pbar:
        async for _, path, _ in run_async(
            ({"sample_dir": sd} for sd in samples_path.glob("*")),
            lambda sample_dir: sample_dir / "rejection.pdf"
            if is_complete(sample_dir)
            else None,
            n_workers=16,
        ):
            if path is not None:
                files.append(path)
                pbar.set_postfix({"complete": len(files)})
            pbar.update(1)

    files.sort(key=lambda p: p.parent.name)

    skip_abs = int(len(files) * skip_percent / 100.0)
    if skip_abs > 0:
        files = files[skip_abs:]

    files = [f for i, f in enumerate(files) if i % world_size == local_rank]

    return files


async def main(cfg: hl.Config) -> None:
    logger.info("Listing PDF files to process")
    pdf_files = await list_pdf_files(
        cfg.samples_path, cfg.skip_percent, cfg.local_rank, cfg.world_size
    )
    logger.info(f"Found {len(pdf_files)} PDF files to process")

    lm = dspy.LM(
        model=cfg.model,
        api_base=cfg.base_url,
        api_key="EMPTY",
        max_tokens=None,  # type: ignore
        temperature=0.8,
        top_p=0.9,
        top_k=20,
        cache=False,
    )
    dspy.configure(lm=lm, adapter=PydanticAdapter())

    stats = defaultdict(int)
    with RichTableProgress(total=len(pdf_files), persistent_every=100) as progress:
        async for _, result, error in run_async(
            ({"pdf": pdf} for pdf in pdf_files),
            process_pdf,
            n_workers=cfg.num_workers,
            ordered=False,
        ):
            icrement_empty = 0

            if result is None or error or not result.success:
                stats[f"error/{error.__class__.__name__}"] += 1

            elif result.success:
                if result.skipped:
                    stats["already_exists"] += 1
                    icrement_empty = 1

                elif result.output and result.output.breakdown:
                    stats["with_breakdown"] += 1

                else:
                    stats["without_breakdown"] += 1

                stats["retries/total"] += result.n_retries
                stats["retries/samples"] += min(1, result.n_retries)

            progress.update(1, increment_empty=icrement_empty, data=stats)


if __name__ == "__main__":
    cfg.apply()
    with OutputRedirector(
        root
        / "data"
        / "logs"
        / (
            Path(__file__).stem
            + f"_{cfg.skip_percent}_{cfg.local_rank+1}-{cfg.world_size}"
            + ".log"
        )
    ):
        asyncio.run(main(cfg))
