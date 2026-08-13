import json

from backend.app.config import Settings
from backend.app.integrations.email_client import EmailClient
from backend.app.models.ticket import Ticket


class EmailTicketProvider:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = EmailClient(settings)

    def deliver(self, ticket: Ticket) -> Ticket:
        if not self.settings.ticket_email_enabled:
            return ticket
        body = "\n".join(
            [
                f"Тема: {ticket.topic}",
                f"Контакт: {ticket.contact or 'не указан'}",
                f"Пользователь: {ticket.role}",
                f"user_id: {ticket.user_id or '-'}",
                f"session_id: {ticket.session_id}",
                f"page_type: {ticket.page_type or '-'}",
                f"lot_id: {ticket.lot_id or '-'}",
                f"payment_id: {ticket.payment_id or '-'}",
                f"category: {ticket.category or '-'}",
                f"priority: {ticket.priority}",
                f"scenario_id: {ticket.scenario_id or '-'}",
                f"source_message_id: {ticket.source_message_id or '-'}",
                "",
                "Описание:",
                ticket.description,
                "",
                "История:",
                json.dumps(ticket.dialog_history, ensure_ascii=False, indent=2),
            ]
        )
        self.client.send(f"MIGTORG: {ticket.topic}", body)
        ticket.status = "sent_email"
        return ticket
