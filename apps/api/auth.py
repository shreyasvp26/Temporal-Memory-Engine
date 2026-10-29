"""Authentication and JWT verification per docs/LLD.md Section 11.2."""

import logging
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import jwt
from fastapi import Request
from jwt import PyJWKClient
from memory_engine.config import JwtSettings
from memory_engine.domain.errors import Unauthorized
from memory_engine.domain.ids import UserId
from memory_engine.obs.logging import bind_context

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AuthUser:
    """Authenticated user context containing user_id and optional email."""

    user_id: UserId
    email: str | None = None


class JwtVerifier:
    """Verifies JWT tokens using HS256 shared secret or RS256/JWKS endpoint."""

    def __init__(self, cfg: JwtSettings, env: str = 'dev') -> None:
        self.cfg = cfg
        self.env = env
        self._jwks_client: PyJWKClient | None = None
        if cfg.jwks_url and cfg.mode == 'jwks':
            try:
                self._jwks_client = PyJWKClient(cfg.jwks_url, cache_jwk_set=True, lifespan=600)
            except Exception as exc:
                _logger.warning('Failed to initialize JWKS client for %s: %s', cfg.jwks_url, exc)

    async def verify(self, token: str) -> AuthUser:
        """Verify JWT signature, audience, expiration, and UUID subject."""
        if not token or not token.strip():
            raise Unauthorized('Missing token', code='E-AUTH-01')

        try:
            key: Any
            algorithms: list[str]
            if self._jwks_client is not None:
                signing_key = self._jwks_client.get_signing_key_from_jwt(token)
                key = signing_key.key
                algorithms = ['RS256']
            elif self.cfg.secret:
                key = self.cfg.secret
                algorithms = ['HS256', 'HS384', 'HS512']
            else:
                # If no secret or JWKS configured and in dev/test/local, decode unverified or reject
                if self.env in ('dev', 'test', 'local'):
                    key = ''
                    algorithms = ['HS256', 'none']
                else:
                    raise Unauthorized(
                        'JWT verification secret/JWKS not configured',
                        code='E-AUTH-01',
                    )

            decode_kwargs: dict[str, Any] = {
                'algorithms': algorithms,
                'leeway': 30,
                'options': {
                    'verify_exp': True,
                    'verify_iat': False,
                    'verify_aud': bool(self.cfg.audience),
                    'verify_signature': bool(key or self._jwks_client is not None),
                },
            }
            if self.cfg.audience:
                decode_kwargs['audience'] = self.cfg.audience
            if self.cfg.issuer:
                decode_kwargs['issuer'] = self.cfg.issuer
                decode_kwargs['options']['verify_iss'] = True

            payload = jwt.decode(token, key, **decode_kwargs)
        except jwt.PyJWTError as exc:
            _logger.info('JWT decode error: %s', exc)
            raise Unauthorized(f'Invalid token: {exc}', code='E-AUTH-01') from exc
        except Exception as exc:
            _logger.warning('Unexpected JWT verification failure: %s', exc)
            raise Unauthorized('Token verification failed', code='E-AUTH-01') from exc

        sub = payload.get('sub')
        if not sub:
            raise Unauthorized('Token missing sub claim', code='E-AUTH-01')

        try:
            user_uuid = UUID(str(sub))
        except (ValueError, TypeError) as exc:
            raise Unauthorized('Token sub claim must be a valid UUID', code='E-AUTH-01') from exc

        email = payload.get('email')
        return AuthUser(user_id=UserId(user_uuid), email=str(email) if email else None)


async def get_current_user(request: Request) -> UserId:
    """FastAPI dependency extracting and verifying user_id from headers.

    Supports:
    1. Dev/Test mode header: `X-Debug-User-Id: <uuid>` when env in ('dev', 'test').
    2. Bearer JWT: `Authorization: Bearer <token>` verified with JwtVerifier.
    Binds authenticated `user_id` to `memory_engine.obs.logging` context.
    """
    container = getattr(request.app.state, 'container', None)
    env = (
        getattr(container.settings, 'env', 'dev')
        if container and hasattr(container, 'settings')
        else 'dev'
    )

    # 1. Dev / test / local mode bypass header
    debug_user_id = request.headers.get('x-debug-user-id')
    if debug_user_id and env in ('dev', 'test', 'local'):
        try:
            uid = UserId(UUID(debug_user_id.strip()))
            bind_context(user_id=uid)
            return uid
        except (ValueError, TypeError) as exc:
            raise Unauthorized(
                'Invalid X-Debug-User-Id header format; expected UUID',
                code='E-AUTH-01',
            ) from exc

    # 2. Bearer JWT token verification
    auth_header = request.headers.get('Authorization') or request.headers.get('authorization')
    if not auth_header:
        raise Unauthorized('Missing Authorization header', code='E-AUTH-01')

    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != 'bearer':
        raise Unauthorized(
            "Invalid Authorization header format; expected 'Bearer <token>'",
            code='E-AUTH-01',
        )

    token = parts[1]
    verifier: JwtVerifier
    if container is not None and getattr(container, 'jwt_verifier', None) is not None:
        verifier = container.jwt_verifier
    elif container is not None and hasattr(container, 'settings'):
        verifier = JwtVerifier(container.settings.jwt, env=env)
    else:
        verifier = JwtVerifier(JwtSettings(), env=env)

    auth_user = await verifier.verify(token)
    bind_context(user_id=auth_user.user_id)
    return auth_user.user_id


current_user = get_current_user
