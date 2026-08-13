from typing import Protocol

from backend.app.models.ticket import Ticket


class TicketDeliveryProvider(Protocol):
    def deliver(self, ticket: Ticket) -> Ticket:
        ...
