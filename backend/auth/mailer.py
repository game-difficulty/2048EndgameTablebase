from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage


def send_verification_email(email: str, code: str) -> bool:
    host = os.getenv("SMTP_HOST", "").strip()
    username = os.getenv("SMTP_USERNAME", "").strip()
    password = os.getenv("SMTP_PASSWORD", "")
    sender = os.getenv("SMTP_FROM", username).strip()
    if not host or not sender:
        return False

    port = int(os.getenv("SMTP_PORT", "587"))
    use_tls = os.getenv("SMTP_TLS", "1") != "0"
    message = EmailMessage()
    message["From"] = sender
    message["To"] = email
    message["Subject"] = "2048tables verification code"
    message.set_content(
        "\n".join(
            [
                "Your 2048tables verification code is:",
                "",
                code,
                "",
                "This code expires in 10 minutes.",
            ]
        )
    )

    with smtplib.SMTP(host, port, timeout=15) as smtp:
        if use_tls:
            smtp.starttls()
        if username:
            smtp.login(username, password)
        smtp.send_message(message)
    return True
