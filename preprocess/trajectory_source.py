"""pluggable trajectory path resolution (H5 batch layout today, can swap to direct westpa if we want"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping


class TrajectorySource(ABC):
    """Maps preprocess PDB ids to trajectory file paths."""

    @abstractmethod
    def h5_path(self, pdbid: str) -> str:
        """Absolute or relative path to the per-structure trajectory (e.g. output_*.h5)."""

    @abstractmethod
    def pdb_ids(self) -> list[str]:
        """Stable ordering for logging and job slicing (sorted ids)."""

    def items(self) -> Iterator[tuple[str, str]]:
        for pid in self.pdb_ids():
            yield pid, self.h5_path(pid)

    def as_dict(self) -> dict[str, str]:
        return {pid: self.h5_path(pid) for pid in self.pdb_ids()}


class H5BatchTrajectorySource(TrajectorySource):
    """Default layout from `gen_input_mapping`: pdbid -> path to HDF5."""

    def __init__(self, pdbid_to_h5: Mapping[str, str], *, order: list[str] | None = None):
        self._m = dict(pdbid_to_h5)
        if order is not None:
            self._ids = [k for k in order if k in self._m]
        else:
            self._ids = sorted(self._m.keys())

    def h5_path(self, pdbid: str) -> str:
        return self._m[pdbid]

    def pdb_ids(self) -> list[str]:
        return list(self._ids)

    def filter_to(self, pdbids: list[str]) -> H5BatchTrajectorySource:
        m = {k: self._m[k] for k in pdbids if k in self._m}
        return H5BatchTrajectorySource(m, order=[k for k in pdbids if k in m])
