import numpy as np
import pytest

from hypermm.fillcheck import ceiling
from hypermm.fillcheck.ceiling import EDGES


def test_an_order_is_at_risk_in_every_bin_up_to_the_one_its_life_ended_in():
    # 0.25 s falls in the third bin (0.2, 0.3]
    x = np.array([[7.0]], dtype=np.float32)
    rows, y, weight = ceiling.person_period(x, np.array([0.25]), np.array([True]))
    assert rows.tolist() == [[7, 0], [7, 1], [7, 2]]
    assert y.tolist() == [0, 0, 1]
    assert weight.tolist() == [1, 1, 1]


def test_a_cancelled_order_counts_half_in_the_bin_it_left_in():
    x = np.array([[7.0]], dtype=np.float32)
    _, y, weight = ceiling.person_period(x, np.array([0.25]), np.array([False]))
    assert y.tolist() == [0, 0, 0]
    assert weight.tolist() == [1, 1, 0.5]


def test_an_order_outliving_the_grid_is_at_risk_in_all_of_it_and_never_positive():
    x = np.array([[7.0]], dtype=np.float32)
    _, y, _ = ceiling.person_period(x, np.array([EDGES[-1] + 1]), np.array([True]))
    assert len(y) == len(EDGES)
    assert y.sum() == 0


def test_kaplan_meier_steps_down_at_events_only():
    # at risk 4, 3, 2, 1; events at 1 and 3, censored at 2 and 4
    km = ceiling.KaplanMeier(np.array([1.0, 2.0, 3.0, 4.0]), np.array([True, False, True, False]))
    assert km(1.0) == pytest.approx(0.75)
    assert km(2.5) == pytest.approx(0.75)
    assert km(3.0) == pytest.approx(0.375)
    assert km(3.0, before=True) == pytest.approx(0.75)
    assert km(0.5) == 1


def test_without_censoring_the_auc_at_a_horizon_is_the_plain_auc():
    durations = np.array([1.0, 2.0, 6.0, 7.0])
    events = np.ones(4, dtype=bool)
    g = ceiling.censoring(durations, events)
    perfect = np.array([4.0, 3.0, 2.0, 1.0])
    assert ceiling.auc_at(durations, events, perfect, 5.0, g) == 1
    assert ceiling.auc_at(durations, events, -perfect, 5.0, g) == 0


def test_orders_cancelled_before_the_horizon_take_no_part_in_the_auc():
    durations = np.array([1.0, 2.0, 6.0, 7.0, 3.0])
    events = np.array([True, True, True, True, False])  # the last one cancelled at 3
    g = ceiling.censoring(durations, events)
    risk = np.array([4.0, 3.0, 2.0, 1.0, -99.0])  # wrong about the cancelled order only
    assert ceiling.auc_at(durations, events, risk, 5.0, g) == 1


def test_the_log_likelihood_reads_a_fill_from_its_bin_and_a_cancellation_from_its_start():
    survival = np.tile(np.linspace(0.9, 0.1, len(EDGES)), (2, 1))
    durations = np.array([0.15, 0.15])  # the second bin
    filled = ceiling.log_likelihood(durations, np.array([True, True]), survival)
    assert filled == pytest.approx(np.log(survival[0, 0] - survival[0, 1]))
    cancelled = ceiling.log_likelihood(durations, np.array([False, False]), survival)
    assert cancelled == pytest.approx(np.log(survival[0, 0]))


def test_the_share_of_attainable_signal_runs_from_floor_to_ceiling():
    assert ceiling.share(0.5, 0.82) == 0
    assert ceiling.share(0.82, 0.82) == 1
    assert ceiling.share(0.58, 0.82) == pytest.approx(0.25)  # the proposal's own example


def test_a_rule_that_says_filled_too_often_ranks_well_and_scores_below_the_base_rate():
    filled = np.array([True] + [False] * 99)
    touch = np.array([True] * 6 + [False] * 94)  # every fill caught, five false ones
    assert ceiling.realised_auc(filled, touch) == pytest.approx((1 + 94 / 99) / 2)
    assert ceiling.brier_skill(filled, touch, 0.01) < 0
    assert ceiling.brier_skill(filled, filled, 0.01) == 1


def test_boosting_the_hazard_recovers_which_orders_fill_sooner():
    rng = np.random.default_rng(0)
    n = 20_000
    x = rng.normal(size=(n, 1)).astype(np.float32)
    fill = rng.exponential(1 / np.exp(x[:, 0]))  # hazard rises with the feature
    cancel = rng.exponential(1.0, size=n)
    durations, events = np.minimum(fill, cancel), fill <= cancel
    model = ceiling.HazardBoosting(max_iter=60).fit(x, durations, events)
    survival = model.survival(x[:2000])
    assert survival.shape == (2000, len(EDGES))
    assert np.all(np.diff(survival, axis=1) <= 0)
    assert ceiling.concordance_td(durations[:2000], events[:2000], survival) > 0.7


def test_time_dependent_concordance_compares_at_the_time_of_the_fill():
    # two orders filled in the first bin, two that outlive it; survival at the end of
    # that bin is all that is compared, whatever the curves do later
    durations, events = np.array([0.05, 0.05, 9.0, 9.0]), np.array([True, True, False, False])
    survival = np.ones((4, len(EDGES)))
    survival[:2, 0] = 0.5  # the filled ones were given the lower survival: right
    survival[:2, 1:] = 1.0
    survival[2:, 1:] = 0.1  # and a higher risk later on, which must not count
    assert ceiling.concordance_td(durations, events, survival) == 1
    assert ceiling.concordance_td(durations, events, 1.5 - survival) == 0
