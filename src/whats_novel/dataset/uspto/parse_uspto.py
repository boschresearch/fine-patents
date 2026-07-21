import pickle
import re
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple
from zipfile import BadZipFile, ZipFile

import hydralette as hl
import numpy as np
from lxml import etree as etree_lxml
from pyrootutils import setup_root
from tqdm import tqdm

from whats_novel.utils.logs import get_logger

logger = get_logger()
root = setup_root(__file__)


cfg = hl.Config(
    uspto_path=root / "data" / "uspto",
    patents_file=root / "data" / "uspto" / "patents.pickle",
    patents_index_file=root / "data" / "uspto" / "patents_index.pickle",
    n_processes=32,
    id_field="publication_number",
)


def parse_zip(path: Path) -> Tuple[List[dict], dict]:
    patent_xmls = []
    stats = defaultdict(int)

    try:
        with ZipFile(path) as zf:
            for file in zf.namelist():
                with zf.open(file) as fp:
                    contents = fp.read().decode("utf-8")
                xmls = [
                    p
                    for p in contents.split('<?xml version="1.0" encoding="UTF-8"?>')
                    if p
                ]
                patent_xmls.extend(xmls)
    except BadZipFile as e:
        logger.error(f"Bad Zip File: {path}. Exception: {e}")
        return [], stats

    patents = []

    for idx_in_zip, patent_xml in enumerate(patent_xmls):
        try:
            tree = etree_lxml.fromstring(patent_xml)  # type: ignore

            publication_number_node = tree.xpath("//publication-reference/document-id")
            if not publication_number_node:
                stats["no_pub_number"] += 1
                continue
            publication_number = (
                publication_number_node[0].xpath("country")[0].text  # type: ignore
                + publication_number_node[0].xpath("doc-number")[0].text  # type: ignore
                + publication_number_node[0].xpath("kind")[0].text  # type: ignore
            )

            application_number_node = tree.xpath("//application-reference/document-id")
            if not application_number_node:
                stats["no_app_number"] += 1
                continue
            application_number = (
                application_number_node[0].xpath("country")[0].text  # type: ignore
                + application_number_node[0].xpath("doc-number")[0].text  # type: ignore
            )

            title_node = tree.xpath("//invention-title")
            if not title_node:
                stats["no_title"] += 1
                continue
            title = title_node[0].text  # type: ignore

            abstract_node = tree.xpath("//abstract/p")
            if not abstract_node:
                stats["no_abstract"] += 1
                continue
            abstract = "".join(abstract_node[0].itertext())  # type: ignore

            def get_num(el: etree_lxml._Element) -> list[int]:
                num: str = el.attrib["num"]  # type: ignore
                if re.match(r"^\[?\d+\]?$", num):
                    return [int(num.strip("[]").lstrip("0"))]
                elif re.match(r"^\d+-\d+$", num):
                    start, end = num.split("-")
                    return list(range(int(start.lstrip("0")), int(end.lstrip("0")) + 1))
                elif re.match(r"heading-\d+$", num) or num == "none":
                    return []
                else:
                    # logger.warning(f"Unexpected paragraph/claim number format: {num}")
                    return []

            description_paragraph_nodes: list[etree_lxml._Element] = tree.xpath(
                "//description/p"
            )  # type: ignore
            try:
                max_id = max(n for p in description_paragraph_nodes for n in get_num(p))  # type: ignore
            except Exception as e:  # noqa
                stats["no_description"] += 1
                continue
            description_paragraphs = [None] * max_id
            description_broken = False
            for p in description_paragraph_nodes:
                for n in get_num(p):
                    p_text = "".join(p.itertext())  # type: ignore
                    if description_paragraphs[n - 1] is not None:
                        description_broken = True
                        break
                    description_paragraphs[n - 1] = p_text.strip()
            if not description_paragraphs or description_broken:
                stats["no_description"] += 1
                continue

            claims_nodes: list[etree_lxml._Element] = tree.xpath("//claims/claim")  # type: ignore
            try:
                max_id = max(n for c in claims_nodes for n in get_num(c))  # type: ignore
            except Exception as e:  # noqa
                stats["no_claims"] += 1
                continue
            claims = [None] * max_id
            claims_broken = False
            for c in claims_nodes:
                # canceled claims are often aggregated into ranges, so we need to consider all numbers in the range
                for n in get_num(c):
                    c_text = "".join(c.itertext())  # type: ignore
                    if (
                        claims[n - 1] is not None
                    ):  # the same number shouldnt appear multiple times
                        claims_broken = True
                        break
                    claims[n - 1] = c_text.strip()

            # each claim should start with its number
            for i, claim in enumerate(claims):
                if match := re.match(r"\d+\.", claim or ""):
                    num = int(match.group(0)[:-1])
                    if num != i + 1:
                        claims_broken = True
                        break
                elif match := re.match(r"(\d+)-(\d+)\.", claim or ""):
                    start_num = int(match.group(1))
                    end_num = int(match.group(2))
                    if not (start_num <= i + 1 <= end_num):
                        claims_broken = True
                        break
                else:
                    claims_broken = True
                    break

            if not claims or claims_broken:
                stats["no_claims"] += 1
                continue

            patents.append(
                dict(
                    publication_number=publication_number,
                    application_number=application_number,
                    zip_file=path,
                    index_in_zip=idx_in_zip,
                    title=title,
                    abstract=abstract,
                    claims=claims,
                    description=description_paragraphs,
                )
            )
            stats["parsed"] += 1
        except Exception as e:
            logger.error(f"Error parsing patent in {path} at index {idx_in_zip}: {e}")
            stats["parse_error"] += 1
            continue

    return patents, stats


def main(cfg: hl.Config):
    patents_file: Path = cfg.patents_file
    patents_index_file: Path = cfg.patents_index_file
    uspto_dir: Path = cfg.uspto_path
    n_processes: int = cfg.n_processes

    if patents_file.exists():
        logger.warning(f"Output file {patents_file} exists! Overwriting ...")
        patents_file.unlink()
    if patents_index_file.exists():
        logger.warning(f"Output file {patents_index_file} exists! Overwriting ...")
        patents_index_file.unlink()

    patent_paths = sorted(
        list(uspto_dir.glob("APPXML/*.zip")) + list(uspto_dir.glob("PTGRXML/*.zip")),
        key=lambda x: x.name,
    )

    # only keep the newest revision per zip file
    mapping = defaultdict(list)
    for patent_path in patent_paths:
        name = patent_path.name.replace(".zip", "").split("_r")[0]
        mapping[name].append(patent_path)
    for name, files in mapping.items():
        if len(files) < 2:
            continue
        nums = [
            file.name.split("_r")[1][0] if "_r" in file.name else 0 for file in files
        ]
        to_keep = np.argmax(nums)
        for i in range(len(files)):
            if i != to_keep:
                patent_paths.remove(files[i])

    stats = defaultdict(int)
    patents_index = {}

    def imap_unordered(func, it):
        for item in it:
            yield func(item)

    try:
        with open(patents_file, "wb") as fp, Pool(n_processes) as p:
            for patents, stats_ in (
                pbar := tqdm(
                    p.imap_unordered(parse_zip, patent_paths),
                    total=len(patent_paths),
                    desc="Parsing Zip Files",
                )
            ):
                for k in stats_:
                    stats[k] += stats_[k]

                pbar.set_postfix(stats)

                for patent in patents:
                    if patent[cfg.id_field] in patents_index:
                        logger.warning(
                            f"Patent ID is not unique: {patent[cfg.id_field]}. Overwriting index {patents_index[patent[cfg.id_field]]}"
                        )
                    patents_index[patent[cfg.id_field]] = fp.tell()
                    pickle.dump(patent, fp, protocol=pickle.HIGHEST_PROTOCOL)

    except KeyboardInterrupt:
        logger.info("Detected Keyboard interrupt. Saving index before terminating ...")

    except Exception as e:
        logger.exception(f"'{repr(e)}' raised. Saving index before terminating ...")

    finally:
        with open(patents_index_file, "wb") as fp:
            pickle.dump(patents_index, fp, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    cfg.apply()
    main(cfg)
