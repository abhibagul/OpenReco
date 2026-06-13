"""Tie-point selection logic for the refine stage (pure function — runs in CI)."""

from __future__ import annotations

from openreco.stages.refine import points_to_delete

# (id, reproj_error_px, track_length)
ITEMS = [
    (1, 0.3, 8),
    (2, 0.5, 4),
    (3, 1.5, 6),    # high error
    (4, 0.2, 1),    # short track
    (5, 5.0, 2),    # high error
    (6, 0.4, 3),
]


def test_filter_by_error_and_track():
    out = set(points_to_delete(ITEMS, max_error=1.0, min_track=2))
    assert out == {3, 4, 5}          # high error (3,5) + short track (4)


def test_min_track_only():
    out = set(points_to_delete(ITEMS, max_error=99.0, min_track=4))
    assert out == {4, 5, 6}          # tracks < 4: id4(1), id5(2), id6(3)


def test_gradual_selection_percentile():
    # worst 20% by error -> only the single largest (id 5, error 5.0)
    out = set(points_to_delete(ITEMS, max_error=99.0, min_track=1, max_error_percentile=20.0))
    assert 5 in out
    assert 1 not in out and 2 not in out


def test_nothing_deleted_when_thresholds_loose():
    assert points_to_delete(ITEMS, max_error=99.0, min_track=1) == []
