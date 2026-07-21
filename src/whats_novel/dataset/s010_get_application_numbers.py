import json
from pathlib import Path
import re

import hydralette as hl
import sparql_dataframe
from pyrootutils import setup_root

from whats_novel.utils.logs import OutputRedirector, get_logger

logger = get_logger(__file__)
root = setup_root(__file__)

cfg = hl.Config(
    output_path=root / "data" / "applications_publications.json",
    cpc_classes=hl.Field(default=["G06", "G10L", "H04N", "H04L"]),
)


query_get_cpc = """
PREFIX cpc: <http://data.epo.org/linked-data/def/cpc/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT ?child
WHERE {
    VALUES ?parent { <CPC-CLASS> }
    ?parent skos:narrower+ ?child .
}
ORDER BY ?cpc
"""


def get_cpc_codes(cpc_class: str) -> list[str]:
    cpc_query = query_get_cpc.replace("<CPC-CLASS>", f"cpc:{cpc_class}")
    cpc_resp = sparql_dataframe.sparql_dataframe.get_sparql_dataframe(
        "https://data.epo.org/linked-data/query", cpc_query
    )
    cpc = cpc_resp["child"].str.split("/").str[-1].str.replace(" ", "").tolist()
    return cpc


query = """
prefix cpc: <http://data.epo.org/linked-data/def/cpc/>
prefix dcterms: <http://purl.org/dc/terms/>
prefix ipc: <http://data.epo.org/linked-data/def/ipc/>
prefix mads: <http://www.loc.gov/standards/mads/rdf/v1.rdf>
prefix owl: <http://www.w3.org/2002/07/owl#>
prefix patent: <http://data.epo.org/linked-data/def/patent/>
prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>
prefix skos: <http://www.w3.org/2004/02/skos/core#>
prefix st3: <http://data.epo.org/linked-data/def/st3/>
prefix text: <http://jena.apache.org/text#>
prefix vcard: <http://www.w3.org/2006/vcard/ns#>
prefix xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT
    ?application
    (SAMPLE(?filingDateNode) AS ?filingDate)
    (GROUP_CONCAT(?publication; separator="\\n") AS ?publications)
WHERE {
    ?application rdf:type patent:Application ;
        patent:applicationNumber ?applicationNumber ;
        patent:publication ?publication ;
        patent:filingDate ?filingDateNode ;
        patent:classificationCPCInventive ?cpcNode .

    FILTER(?filingDateNode > "2012-01-01"^^xsd:date)
    VALUES ?cpcNode { <CPC-VALUES> }
}
GROUP BY ?application
ORDER BY ?filingDateNode
"""


def get_numbers(cpc: list[str]) -> dict[str, list[str]]:
    appl_query = query.replace("<CPC-VALUES>", " ".join(f"cpc:{code}" for code in cpc))

    applications = sparql_dataframe.sparql_dataframe.get_sparql_dataframe(
        "https://data.epo.org/linked-data/query", appl_query, post=True
    )

    app_base_url = "http://data.epo.org/linked-data/id/application/"
    pub_base_url = "http://data.epo.org/linked-data/data/publication"

    applications_dict = {}
    for _, row in applications.iterrows():
        app_num = (
            row["application"]
            .replace(app_base_url, "")
            .replace("/", "")
            .replace("-", "")
        )
        pub_nums = [
            pub_num
            for pub in set(row["publications"].split("\n"))
            if (
                pub_num := pub.replace(pub_base_url, "")
                .replace("/", "")
                .replace("-", "")
            )
            and re.match(r".*[AB]\d$", pub_num)
        ]
        pub_nums.sort(key=lambda p: p[-2:])
        applications_dict[app_num] = pub_nums
    return applications_dict


def print_stats(applications_dict: dict[str, list[str]]) -> None:
    logger.info(f"Found {len(applications_dict)} applications")
    counts_per_kc = {
        kcs: sum(
            all(any(pub.endswith(kc) for pub in pubs) for kc in kcs)
            for pubs in applications_dict.values()
        )
        for kcs in (
            ("A1",),
            ("A2",),
            ("B1",),
            ("B2",),
            ("A1", "B1"),
            ("A1", "B2"),
            ("A2", "B1"),
            ("A2", "B2"),
        )
    }
    logger.info(f"Counts per kind code: {counts_per_kc}")


def main(cfg: hl.Config) -> None:
    results = []
    for cpc_class in cfg.cpc_classes:
        logger.info(f"Getting CPC codes for {cpc_class}...")
        cpc_codes = get_cpc_codes(cpc_class)
        logger.info(f"Found {len(cpc_codes)} CPC codes.")
        logger.info("Getting application numbers...")
        applications_dict = get_numbers(cpc_codes)
        print_stats(applications_dict)
        results.append(applications_dict)

    applications_dict = {}
    for res in results:
        applications_dict.update(res)

    print_stats(applications_dict)

    with open(cfg.output_path, "w") as f:
        json.dump(applications_dict, f, indent=4)

    logger.info(f"Saved to {cfg.output_path}")


if __name__ == "__main__":
    cfg.apply()
    with OutputRedirector(root / "data" / "logs" / (Path(__file__).stem + ".log")):
        main(cfg)
