"""Protection du dashboard.

Sans mot de passe, le serveur écoute sur 127.0.0.1. `cm dev --lan` ouvre le réseau local pour le
téléphone et exige alors DASHBOARD_PASSWORD : le dashboard peut dépenser sur la clé API.

Dans tous les cas, une requête d'écriture venue d'un autre site est refusée : sans ça, n'importe
quelle page ouverte dans le navigateur pourrait lancer des actions payantes sur localhost.
"""

import hashlib
import hmac
from urllib.parse import urlsplit

from starlette.requests import Request
from starlette.responses import PlainTextResponse, RedirectResponse, Response

from app.config import get_settings

COOKIE = "cm_session"
LOOPBACK = {"127.0.0.1", "::1"}
PUBLIC_PATHS = ("/ui/login", "/static/", "/health")
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def session_token(password: str) -> str:
    return hashlib.sha256(f"contentmachine:{password}".encode()).hexdigest()


def _same_origin(request: Request) -> bool:
    origin = request.headers.get("origin")
    return origin is None or urlsplit(origin).netloc == request.headers.get("host")


def _is_logged_in(request: Request, password: str) -> bool:
    if request.client is not None and request.client.host in LOOPBACK:
        return True
    cookie = request.cookies.get(COOKIE, "")
    return hmac.compare_digest(cookie, session_token(password))


async def protect(request: Request, call_next) -> Response:
    if request.method not in SAFE_METHODS and not _same_origin(request):
        return PlainTextResponse("Requête venue d'un autre site.", status_code=403)

    secret = get_settings().dashboard_password
    if secret is not None and not request.url.path.startswith(PUBLIC_PATHS):
        if not _is_logged_in(request, secret.get_secret_value()):
            if request.url.path.startswith("/ui"):
                return RedirectResponse("/ui/login", status_code=303)
            return PlainTextResponse("Connexion requise.", status_code=401)
    return await call_next(request)
