import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from betting_model.odds import devig_1x2


def test_devig_sums_to_one():
    fair, overround = devig_1x2(2.0, 3.5, 4.0)
    assert abs(sum(fair) - 1.0) < 1e-9


def test_devig_overround_above_one_for_real_odds():
    # Real bookmaker odds always have a margin baked in.
    _, overround = devig_1x2(2.0, 3.5, 4.0)
    assert overround > 1.0


def test_devig_no_margin_case():
    # If odds implied exactly 100% (a fair book), overround should be ~1.
    fair, overround = devig_1x2(3.0, 3.0, 3.0)
    assert abs(overround - 1.0) < 1e-9
    assert all(abs(p - 1 / 3) < 1e-9 for p in fair)
