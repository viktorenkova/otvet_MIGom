"""Explicit operator entry point for queued SMTP delivery. Never started by chat requests."""
import argparse
import json
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="Keep polling until stopped")
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("--interval must be at least one second")
    from backend.app.config import get_settings
    from backend.app.bot.dialog_logger import DialogLogger
    from backend.app.delivery.email_provider import EmailTicketProvider
    from backend.app.delivery.outbox import drain_outbox
    settings = get_settings()
    if not settings.ticket_email_enabled or not settings.smtp_host:
        parser.error("Email delivery must be enabled and SMTP_HOST configured")
    logger = DialogLogger(settings.database_path)
    provider = EmailTicketProvider(settings)
    try:
        while True:
            result = drain_outbox(logger, provider)
            if any(result.values()) or not args.watch:
                print(json.dumps(result), flush=True)
            if not args.watch:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
