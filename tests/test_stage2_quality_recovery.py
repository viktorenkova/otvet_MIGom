from __future__ import annotations

import pytest

from backend.app.bot.routing_v3 import clear_routing_v3_cache, get_routing_v3


@pytest.fixture(scope="module")
def router():
    clear_routing_v3_cache()
    return get_routing_v3()


@pytest.mark.parametrize(
    ("message", "expected_scenario_id", "role"),
    [
        ("страховщик после победы нам не пишет", "transfer.seller_no_response", "guest"),
        ("из поддержки по электронной почте ничего не ответили", "support.email_no_response", "guest"),
        ("в почтвоой переписке ответа нте", "support.email_no_response", "guest"),
        ("как подготовиться перед участием в аукционе", "buyer.first_bid_checklist", "guest"),
        ("где проверить, стал ли я победителем аукциона", "auction.result", "authorized"),
        ("хочу приехать лично без предварительной записи", "support.office_visit", "guest"),
    ],
)
def test_generic_recovery_profiles_route_unseen_phrasings(
    router, message: str, expected_scenario_id: str, role: str
) -> None:
    decision = router.decide(message, role)

    assert decision.confidence == "high"
    assert decision.scenario is not None
    assert decision.scenario.scenario_id == expected_scenario_id


def test_visit_profile_does_not_override_lot_pickup(router) -> None:
    decision = router.decide("куда приехать за выигранным автомобилем")

    assert not decision.scenario or decision.scenario.scenario_id != "support.office_visit"
