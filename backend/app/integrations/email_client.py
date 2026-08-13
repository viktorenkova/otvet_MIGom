from email.message import EmailMessage
import smtplib

from backend.app.config import Settings


class EmailClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def send(self, subject: str, body: str) -> None:
        if not self.settings.smtp_host:
            raise RuntimeError("SMTP_HOST is not configured")
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self.settings.ticket_email_from
        message["To"] = self.settings.ticket_email_to
        message.set_content(body)
        with smtplib.SMTP(self.settings.smtp_host, self.settings.smtp_port) as smtp:
            smtp.starttls()
            if self.settings.smtp_user:
                smtp.login(self.settings.smtp_user, self.settings.smtp_password)
            smtp.send_message(message)
