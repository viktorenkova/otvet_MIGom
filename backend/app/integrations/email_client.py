from email.message import EmailMessage
import smtplib

from backend.app.config import Settings


class EmailClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def send(self, subject: str, body: str, message_id: str | None = None) -> None:
        if not self.settings.smtp_host:
            raise RuntimeError("SMTP_HOST is not configured")
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self.settings.ticket_email_from
        message["To"] = self.settings.ticket_email_to
        if message_id:
            message["Message-ID"] = message_id
        message.set_content(body)
        with smtplib.SMTP(self.settings.smtp_host, self.settings.smtp_port, timeout=4.0) as smtp:
            smtp.starttls()
            if self.settings.smtp_user:
                smtp.login(self.settings.smtp_user, self.settings.smtp_password)
            refused = smtp.send_message(message)
            if refused:
                raise RuntimeError("smtp_recipient_refused")
