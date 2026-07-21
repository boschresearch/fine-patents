# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: MIT

import pickle
from pathlib import Path
from typing import Iterator


class USPTOBulkDataClient:
    """
    Simple client for random access to the patents pickle written by parse_uspto.py.
    The index file maps a patent id (publication_number by default) to a byte offset
    inside the patents_file. Each patent is stored as an individual pickle object
    appended sequentially.
    """

    def __init__(
        self,
        patents_file: Path,
        patents_index_file: Path,
        preload_index: bool = True,
    ) -> None:
        self.patents_file = patents_file
        self.patents_index_file = patents_index_file
        if not self.patents_file.exists():
            raise FileNotFoundError(f"patents_file not found: {self.patents_file}")
        if not self.patents_index_file.exists():
            raise FileNotFoundError(
                f"patents_index_file not found: {self.patents_index_file}"
            )
        self._index: dict[str, int] = {}
        if preload_index:
            self._load_index()
        self._fh = open(self.patents_file, "rb")

    def _load_index(self) -> None:
        with self.patents_index_file.open("rb") as f:
            self._index = pickle.load(f)

    @property
    def size(self) -> int:
        return len(self._index)

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, patent_id: str) -> bool:
        return patent_id in self._index

    def get(self, patent_id: str) -> dict | None:
        if patent_id not in self._index:
            return None
        offset = self._index[patent_id]
        self._fh.seek(offset)
        return pickle.load(self._fh)

    def iter_ids(self) -> Iterator[str]:
        return iter(self._index.keys())

    def iter_patents(self) -> Iterator[dict]:
        with open(self.patents_file, "rb") as fp:
            while True:
                try:
                    yield pickle.load(fp)
                except EOFError:
                    break

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "USPTOBulkDataClient":
        return self

    async def __aenter__(self) -> "USPTOBulkDataClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.close()
