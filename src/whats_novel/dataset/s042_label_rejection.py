# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: MIT

import shutil
import subprocess
from pathlib import Path

import hydralette as hl
import rich.traceback
from pyrootutils import setup_root
from tqdm import tqdm

from whats_novel.dataset.data_model import ClaimBreakdown
from whats_novel.dataset.s041_parse_rejection import is_complete

root = setup_root(__file__)
cfg = hl.Config(
    samples_path=root / "data" / "samples", skip_to=hl.Field(type=str, default=None)
)


def main(cfg: hl.Config) -> None:
    samples_path: Path = cfg.samples_path
    samples_meta_path = samples_path / "valid_paths.txt"
    if samples_meta_path.exists():
        with samples_meta_path.open("r") as f:
            pdf_paths = [
                root / line.strip() for line in f.read().split("\n") if line.strip()
            ]
    else:
        pdf_paths = [
            p / "rejection.pdf" for p in tqdm(samples_path.glob("*")) if is_complete(p)
        ]
        pdf_paths.sort(key=lambda p: p.name)
        with samples_meta_path.open("w") as f:
            f.write("\n".join(str(p.relative_to(root)) for p in pdf_paths))

    for pdf_path in pdf_paths:
        if cfg.skip_to and pdf_path.parent.name != cfg.skip_to:
            continue
        else:
            cfg.skip_to = None

        print("_" * 80 + "\n\n")

        breakdown_path = pdf_path.with_suffix(".json")
        breakdown_label_path = pdf_path.with_suffix(".label.json")
        breakdown_label_wip_path = pdf_path.with_suffix(".label.wip.json")

        if not breakdown_path.exists():
            print(f"Skipping {pdf_path}, no breakdown found.")
            continue

        wip_source = breakdown_path

        if breakdown_label_path.exists():
            try:
                ClaimBreakdown.model_validate_json(breakdown_label_path.read_text())
            except Exception:
                print(f"Invalid label file {breakdown_label_path}, please fix.")
                rich.print(rich.traceback.Traceback())
                wip_source = breakdown_label_path
            else:
                print(f"Skipping {pdf_path}, label already exists.")
                continue

        print(f"https://worldwide.espacenet.com/patent/search?q={pdf_path.parent.name}")

        shutil.copy(wip_source, breakdown_label_wip_path)
        subprocess.run(
            f"code {pdf_path} {breakdown_label_wip_path}", shell=True, check=True
        )

        while True:
            while (
                feedback := input("save (s)\texit (e)\tskip(sk)\tvalidate (v)?")
            ).lower() not in {"s", "e", "sk", "v"}:
                pass

            if feedback in {"s", "v"}:
                try:
                    ClaimBreakdown.model_validate_json(
                        breakdown_label_wip_path.read_text()
                    )
                except Exception:
                    print("Invalid label file, please fix.")
                    rich.print(rich.traceback.Traceback())
                    continue
                else:
                    print("Label file is valid.")

                    if feedback == "s":
                        shutil.move(breakdown_label_wip_path, breakdown_label_path)
                        print(f"Saved: {breakdown_label_path}")
                        break

            elif feedback == "sk":
                breakdown_label_wip_path.unlink()
                print("Skipping...")
                break

            elif feedback == "e":
                breakdown_label_wip_path.unlink()
                print("Exiting...")
                return


if __name__ == "__main__":
    cfg.apply()
    main(cfg)
