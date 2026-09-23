import pytest

from campus_safety_ai.core.source_retry import ReconnectPolicy


def test_exponential_delays_saturate_without_overflow():
    policy = ReconnectPolicy(10000, 1, 5)
    assert [policy.delay(i) for i in range(1, 7)] == [1, 2, 4, 5, 5, 5]
    assert policy.delay(10000) == 5
    assert ReconnectPolicy(10000, 0, 0).delay(10000) == 0


@pytest.mark.parametrize('args', [(-1,), (True,), (1.5,), (1, -1), (1, float('inf')),
                                 (1, 2, 1), (1, True), (1, 1, float('nan'))])
def test_reject_invalid_retry_policy(args):
    with pytest.raises(ValueError):
        ReconnectPolicy(*args)


def test_zero_budget_disables_retries():
    with pytest.raises(ValueError):
        ReconnectPolicy(0).delay(1)
