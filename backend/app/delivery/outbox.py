"""One claimed ticket per delivery attempt. SMTP acceptance is not mailbox delivery."""
import smtplib


def drain_outbox(logger, provider, limit=20):
    result = {"sent": [], "failed": [], "unknown": []}
    for _ in range(min(20, max(0, limit))):
        claim = logger.claim_delivery()
        if not claim:
            break
        ticket, token = claim["ticket"], claim["token"]
        try:
            delivered = provider.deliver(ticket)
            state = "accepted" if delivered.status == "sent_email" else "failed"
            error = "" if state == "accepted" else "provider_did_not_accept"
        except (smtplib.SMTPRecipientsRefused, smtplib.SMTPAuthenticationError, smtplib.SMTPConnectError):
            state, error = "failed", "smtp_rejected_before_acceptance"
        except Exception:
            # Includes disconnect/timeout: acceptance might have preceded the error.
            state, error = "unknown", "smtp_acceptance_unconfirmed"
        if logger.complete_delivery(ticket.id, token, state, error):
            result["sent" if state == "accepted" else state].append(ticket.id)
    return result
