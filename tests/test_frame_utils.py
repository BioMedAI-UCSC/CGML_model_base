"""Tests for module.frame_utils (no torch/mdtraj; safe for lightweight CI)."""

from __future__ import annotations

import pytest

from module.frame_utils import split_frame_indices, step3_classical_worker_count


def _flatten_in_order(chunks: list[list[int]]) -> list[int]:
    out: list[int] = []
    for c in chunks:
        out.extend(c)
    return out


@pytest.mark.parametrize(
    "frames,n_workers,expected",
    [
        ([], 4, [[]]),
        ([0], 8, [[0]]),
        ([0, 1, 2, 3], 1, [[0, 1, 2, 3]]),
        ([0, 1, 2, 3], 0, [[0, 1, 2, 3]]),
        (
            [0, 1, 2, 3, 4, 5, 6, 7],
            3,
            [[0, 1, 2], [3, 4, 5], [6, 7]],
        ),
        (
            list(range(5)),
            2,
            [[0, 1, 2], [3, 4]],
        ),
    ],
)
def test_split_frame_indices_shape(
    frames: list[int], n_workers: int, expected: list[list[int]]
) -> None:
    assert split_frame_indices(frames, n_workers) == expected


def test_split_preserves_all_indices_in_order() -> None:
    frames = list(range(23))
    for w in (2, 3, 5, 7, 11):
        chunks = split_frame_indices(frames, w)
        assert _flatten_in_order(chunks) == frames
        if len(frames) > 1 and w > 1:
            w_eff = min(w, len(frames))
            assert len(chunks) == w_eff
            sizes = [len(c) for c in chunks]
            assert max(sizes) - min(sizes) <= 1


@pytest.mark.parametrize(
    "n_frames,cpu_count,num_cores,n_pdbs,expected",
    [
        (0, 8, 32, 1, 1),
        (1, 8, 32, 1, 1),
        (10, 8, 32, 1, 8),
        (100, 8, 32, 1, 8),
        (100, 16, 4, 2, 4),
        (100, 8, 32, 2, 2),
        (3, 16, 4, 3, 3),
    ],
)
def test_step3_classical_worker_count(
    n_frames: int,
    cpu_count: int,
    num_cores: int,
    n_pdbs: int,
    expected: int,
) -> None:
    assert (
        step3_classical_worker_count(
            n_frames, n_pdbs, num_cores, cpu_count=cpu_count
        )
        == expected
    )
