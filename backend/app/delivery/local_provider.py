from backend.app.bot.dialog_logger import DialogLogger
from backend.app.models.ticket import Ticket


class LocalDatabaseTicketProvider:
    def __init__(self, logger: DialogLogger):
        self.logger = logger

    def deliver(self, ticket: Ticket) -> Ticket:
        return self.logger.save_ticket(ticket)
