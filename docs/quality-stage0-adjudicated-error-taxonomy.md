# Этап 0: таксономия ошибок closed-control

Источник: `reports\quality-stage0-adjudicated-local.json`.
Размечено: **53 из 53** провалов.

## Первичные причины

| Причина | Количество |
|---|---:|
| `irrelevant_answer` | 1 |
| `rerank_wrong` | 22 |
| `retrieval_miss` | 18 |
| `wrong_clarify` | 11 |
| `wrong_facts` | 1 |

## Потери routing-контура

| Подпричина | Количество |
|---|---:|
| `rerank_wrong` | 32 |
| `retrieval_miss` | 18 |

`routing_subcause` дополнительно ставится ошибочному уточнению, чтобы отличить отсутствие правильного сценария в Top-10 от ошибки выбора/порога.

## Все ошибки

| ID | Группа | Причина | Ожидалось | Получено | Top-10 содержит правильный |
|---|---|---|---|---|---|
| `widget-003` | `onboarding` | `retrieval_miss` | `buyer.get_started` | `pickup.receive_lot` | нет |
| `widget-004` | `onboarding` | `retrieval_miss` | `buyer.get_started` | `bid.place` | нет |
| `widget-005` | `onboarding` | `rerank_wrong` | `buyer.get_started` | `account.registration` | да |
| `widget-006` | `onboarding` | `rerank_wrong` | `lot.catalog_search / tariff.demo` | `account.registration` | да |
| `widget-007` | `onboarding` | `retrieval_miss` | `buyer.get_started` | `bid.place` | нет |
| `widget-008` | `onboarding` | `rerank_wrong` | `buyer.get_started` | `support.contact` | да |
| `widget-009` | `onboarding` | `irrelevant_answer` | `<clarification>` | `<clarification>` | нет |
| `widget-013` | `registration` | `rerank_wrong` | `account.activation_pending / notification.delivery_problem` | `technical.site_error` | да |
| `widget-014` | `registration` | `rerank_wrong` | `account.login_problem` | `account.blocked` | да |
| `widget-015` | `registration` | `rerank_wrong` | `account.login_problem` | `tariff.choose` | да |
| `widget-017` | `registration` | `wrong_clarify` | `seller.get_started` | `<clarification>` | да |
| `widget-020` | `registration` | `rerank_wrong` | `account.credential_responsibility` | `support.callback` | да |
| `widget-023` | `bidding` | `retrieval_miss` | `bid.autobid_extension` | `bid.place` | нет |
| `widget-026` | `bidding` | `retrieval_miss` | `bid.position_service` | `bid.place` | нет |
| `widget-028` | `bidding` | `rerank_wrong` | `lot.status_guide` | `transfer.not_confirmed` | да |
| `widget-031` | `bidding` | `retrieval_miss` | `bid.autobid_extension` | `bid.place` | нет |
| `widget-033` | `bidding` | `retrieval_miss` | `auction.result` | `buyer.beginner_lot_selection` | нет |
| `widget-034` | `bidding` | `retrieval_miss` | `auction.status` | `bid.place` | нет |
| `widget-035` | `bidding` | `retrieval_miss` | `auction.result` | `auction.status` | нет |
| `widget-042` | `search_images` | `rerank_wrong` | `technical.lot_image_missing` | `technical.site_error` | да |
| `widget-043` | `search_images` | `retrieval_miss` | `lot.card_information` | `lot.definition` | нет |
| `widget-047` | `search_images` | `wrong_clarify` | `lot.catalog_search` | `<clarification>` | да |
| `widget-050` | `search_images` | `rerank_wrong` | `technical.catalog_search_filter / lot.card_information` | `auction.formats` | да |
| `widget-052` | `payments` | `rerank_wrong` | `tariff.status` | `tariff.choose` | да |
| `widget-053` | `payments` | `rerank_wrong` | `tariff.status` | `tariff.choose` | да |
| `widget-055` | `payments` | `retrieval_miss` | `commission.explained` | `balance.topup.commission` | нет |
| `widget-058` | `payments` | `retrieval_miss` | `payment.methods / balance.topup.commission` | `account.dashboard_sections` | нет |
| `widget-060` | `payments` | `rerank_wrong` | `payment.not_visible` | `transfer.seller_no_response` | да |
| `widget-061` | `payments` | `retrieval_miss` | `payment.accounting_documents` | `tariff.connect` | нет |
| `widget-063` | `payments` | `retrieval_miss` | `commission.explained` | `auction.other_platform_offer` | нет |
| `widget-064` | `payments` | `wrong_clarify` | `lot.payment.details / payment.methods` | `<clarification>` | да |
| `widget-065` | `payments` | `retrieval_miss` | `payment.accounting_documents` | `tariff.choose` | нет |
| `widget-070` | `transfer_docs` | `retrieval_miss` | `pickup.representative` | `pickup.receive_lot` | нет |
| `widget-075` | `transfer_docs` | `rerank_wrong` | `pickup.access_issuer` | `pickup.delay` | да |
| `widget-078` | `transfer_docs` | `rerank_wrong` | `win.next_steps / transfer.seller_no_response` | `lot.status_guide` | да |
| `widget-080` | `transfer_docs` | `rerank_wrong` | `pickup.receive_lot` | `pickup.access_issuer` | да |
| `widget-083` | `refunds_penalties` | `rerank_wrong` | `refund.timing_status` | `refund.eligibility` | да |
| `widget-091` | `office_contact` | `wrong_clarify` | `support.office_visit` | `<clarification>` | да |
| `widget-092` | `office_contact` | `rerank_wrong` | `support.office_visit` | `tariff.choose` | да |
| `widget-095` | `office_contact` | `rerank_wrong` | `pickup.access_issuer` | `lot.location` | да |
| `widget-102` | `complaint_improvement` | `rerank_wrong` | `feedback.platform_complaint / technical.site_error` | `bid.place` | да |
| `widget-103` | `complaint_improvement` | `rerank_wrong` | `feedback.platform_complaint / technical.catalog_search_filter` | `feedback.improvement_suggestion` | да |
| `widget-104` | `complaint_improvement` | `rerank_wrong` | `feedback.improvement_suggestion` | `auction.other_platform_offer` | да |
| `widget-110` | `scope_safety` | `retrieval_miss` | `<clarification>` | `account.login_problem` | нет |
| `independent-ref-001` | `refund_ambiguous` | `wrong_facts` | `<clarification> / refund.eligibility / refund.application / refund.timing_status` | `<clarification>` | да |
| `independent-bid-002` | `bid_place` | `wrong_clarify` | `bid.place` | `<clarification>` | да |
| `independent-bid-006` | `bid_not_visible` | `wrong_clarify` | `bid.not_visible` | `<clarification>` | да |
| `independent-off-008` | `office_explicit` | `wrong_clarify` | `support.office_visit` | `<clarification>` | да |
| `independent-lot-003` | `catalog_search` | `wrong_clarify` | `lot.catalog_search` | `<clarification>` | да |
| `independent-lot-010` | `lot_location` | `wrong_clarify` | `lot.location` | `<clarification>` | да |
| `independent-pic-003` | `pickup_receive` | `wrong_clarify` | `pickup.receive_lot` | `<clarification>` | да |
| `independent-oos-003` | `out_of_scope` | `retrieval_miss` | `<clarification>` | `<clarification>` | нет |
| `independent-rep-001` | `repeated_input` | `wrong_clarify` | `tariff.choose / tariff.one_time / tariff.premium` | `tariff.choose` | да |
