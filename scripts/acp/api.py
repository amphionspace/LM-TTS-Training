"""Authenticated official ACP API requests; credentials remain outside the training repository."""

import base64
import hashlib
import hmac
import json
import time
from email.utils import formatdate
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent


def request(method, url, body=None, *, home=None):
    credentials = json.loads(
        ((Path(home) if home else ROOT.parents[2] / "acp") / ".credentials.json").read_text()
    )
    parsed = urlsplit(url)
    date = formatdate(time.time(), usegmt=True)
    target = parsed.path + ("?" + parsed.query if parsed.query else "")
    signing = f"date: {date}\nhost: {parsed.netloc}\n@request-target: {method.lower()} {target}"
    signature = base64.b64encode(
        hmac.new(
            credentials["access_key_secret"].encode(),
            signing.encode(),
            hashlib.sha256,
        ).digest()
    ).decode()
    headers = {
        "Date": date,
        "Host": parsed.netloc,
        "Authorization": (
            f'hmac accesskey="{credentials["access_key_id"]}", algorithm="hmac-sha256", '
            f'headers="date host @request-target", signature="{signature}"'
        ),
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    data = None if body is None else json.dumps(body).encode()
    invocation = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(invocation, timeout=60) as response:
            return json.load(response)
    except HTTPError as error:
        message = error.read().decode(errors="replace")
        for secret in credentials.values():
            message = message.replace(secret, "[REDACTED]")
        raise RuntimeError(f"ACP API HTTP {error.code}: {message}") from None


def jobs_url(config):
    return (
        f"{config['api_endpoint']}/compute/acp/data/v2/subscriptions/"
        f"{config['subscription']}/resourceGroups/{config['resource_group']}/"
        f"zones/{config['workspace_zone']}/workspaces/{config['workspace']}/trainingJobs"
    )
