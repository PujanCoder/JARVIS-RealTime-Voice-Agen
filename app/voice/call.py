"""Twilio outbound-call integration for JARVIS."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.sax.saxutils import escape

logger = logging.getLogger(__name__)


class VoiceConfigurationError(RuntimeError):
    """Raised when required Twilio or public URL configuration is missing."""


class VoiceProviderError(RuntimeError):
    """Raised when Twilio rejects an outbound call request."""


@dataclass(frozen=True, slots=True)
class TwilioSettings:
    """Validated settings required to place a Twilio call."""

    account_sid: str
    auth_token: str
    from_number: str
    to_number: str
    public_base_url: str

    @classmethod
    def from_environment(cls) -> "TwilioSettings":
        values = {
            "account_sid": os.getenv("TWILIO_ACCOUNT_SID", "").strip(),
            "auth_token": os.getenv("TWILIO_AUTH_TOKEN", "").strip(),
            "from_number": os.getenv("TWILIO_PHONE_NUMBER", "").strip(),
            "to_number": os.getenv("MY_PHONE_NUMBER", "").strip(),
            "public_base_url": os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/"),
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise VoiceConfigurationError(
                "Missing voice configuration: " + ", ".join(missing)
            )
        if not values["public_base_url"].startswith(("https://", "http://")):
            raise VoiceConfigurationError("PUBLIC_BASE_URL must use http:// or https://.")
        for name in ("from_number", "to_number"):
            if not re.fullmatch(r"\+[1-9]\d{7,14}", values[name]):
                raise VoiceConfigurationError(
                    f"{name} must be an E.164 phone number such as +15551234567."
                )
        return cls(**values)


class JarvisCaller:
    """Create outbound Twilio calls using environment-based configuration."""

    def __init__(self, settings: TwilioSettings | None = None) -> None:
        self.settings = settings or TwilioSettings.from_environment()

    @property
    def voice_webhook_url(self) -> str:
        """Return the public HTTPS endpoint that returns TwiML."""
        return f"{self.settings.public_base_url}/voice/stream"

    @property
    def websocket_url(self) -> str:
        """Return the public WSS endpoint used by Twilio Media Streams."""
        base = self.settings.public_base_url
        scheme = "wss" if base.startswith("https://") else "ws"
        return f"{scheme}://{base.split('://', 1)[1]}/voice/ws"

    def twiml_response(self) -> str:
        """Build TwiML that starts a bidirectional Media Stream."""
        stream_url = escape(self.websocket_url, {'"': "&quot;"})
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response><Connect><Stream url=\""
            f"{stream_url}"
            '"/></Connect></Response>'
        )

    async def create_outbound_call(self) -> str:
        """Create a call and return its Twilio call SID."""
        return await asyncio.to_thread(self._create_outbound_call_sync)

    def _create_outbound_call_sync(self) -> str:
        endpoint = (
            f"https://api.twilio.com/2010-04-01/Accounts/"
            f"{self.settings.account_sid}/Calls.json"
        )
        body = urlencode(
            {"To": self.settings.to_number, "From": self.settings.from_number, "Url": self.voice_webhook_url}
        ).encode()
        credentials = base64.b64encode(
            f"{self.settings.account_sid}:{self.settings.auth_token}".encode()
        ).decode()
        request = Request(
            endpoint,
            data=body,
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode())
        except HTTPError as exc:
            logger.error("Twilio rejected the outbound call request (HTTP %s).", exc.code)
            raise VoiceProviderError("Twilio rejected the outbound call request.") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            logger.error("Could not reach Twilio while creating an outbound call.")
            raise VoiceProviderError("Could not reach the voice provider.") from exc

        call_sid = payload.get("sid")
        if not isinstance(call_sid, str) or not call_sid:
            raise VoiceProviderError("Twilio returned no call identifier.")
        logger.info("Created outbound JARVIS call with SID %s.", call_sid)
        return call_sid