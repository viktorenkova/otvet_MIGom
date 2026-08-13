from backend.app.models.ticket import Ticket


class BitrixTicketProvider:
    def deliver(self, ticket: Ticket) -> Ticket:
        return ticket
