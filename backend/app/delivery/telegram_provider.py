from backend.app.models.ticket import Ticket


class TelegramTicketProvider:
    def deliver(self, ticket: Ticket) -> Ticket:
        return ticket
