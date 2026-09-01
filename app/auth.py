import logging
import os
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

load_dotenv()
logger = logging.getLogger("roams_auth")


ROAMS_TOKEN_VALIDATION_URL = os.getenv("ROAMS_TOKEN_VALIDATION_URL")

ROAMS_USER_INFO_URL: Optional[str] = os.getenv("ROAMS_USER_INFO_URL")

_HTTP_TIMEOUT = httpx.Timeout(5.0, connect=3.0)

oauth2_scheme = HTTPBearer()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
async def _validate_with_roams(token: str) -> httpx.Response:
    """Calls ROAMS validation endpoint with the token exactly as received."""
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            return await client.get(
                ROAMS_TOKEN_VALIDATION_URL,
                headers={"Authorization": f"Bearer {token}"},
            )
    except httpx.RequestError as exc:
        logger.error("ROAMS validation call failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service unavailable. Please try again.",
        )


async def _fetch_role_from_user_info(token: str, user_id: str) -> str:
    """Second call, only used if the validation response has no role."""
    if not ROAMS_USER_INFO_URL:
        logger.error(
            "ROAMS validation response had no role and ROAMS_USER_INFO_URL "
            "is not configured (user_id=%s)",
            user_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to determine user role.",
        )

    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.get(
                ROAMS_USER_INFO_URL,
                headers={"Authorization": f"Bearer {token}"},
            )
    except httpx.RequestError as exc:
        logger.error("ROAMS user-info call failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service unavailable. Please try again.",
        )

    if resp.status_code < 200 or resp.status_code >= 300:
        logger.error(
            "ROAMS user-info returned non-2xx (%s) for user_id=%s",
            resp.status_code,
            user_id,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        )

    data = resp.json()
    role = data.get("role")
    if not role:
        logger.error("ROAMS user-info missing role for user_id=%s", user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to determine user role.",
        )
    return role


# ---------------------------------------------------------------------------
# Layer 1 — Authentication (token -> identity)
# Same name/shape as before (get_current_user) so existing routes that do
# `Depends(auth.get_current_user)` don't need to change.
# ---------------------------------------------------------------------------
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(oauth2_scheme),
) -> dict:
    token = credentials.credentials

    response = await _validate_with_roams(token)

    # Explicit status check — do not rely on library "throw on non-2xx" behavior.
    if response.status_code < 200 or response.status_code >= 300:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        )

    body = response.json()
    user_id = body.get("userId") or body.get("user_id")
    role = body.get("role")

    if not user_id:
        logger.error("ROAMS validation response missing userId")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to resolve user identity.",
        )

    if not role:
        # ROAMS only confirmed validity — fetch role separately.
        role = await _fetch_role_from_user_info(token, user_id)

    return {"user_id": user_id, "role": role}


# ---------------------------------------------------------------------------
# Layer 2 — Authorization (identity -> permission)
# Same name as before (get_current_admin) so existing admin-only routes
# that do `Depends(auth.get_current_admin)` keep working unchanged.
# ---------------------------------------------------------------------------
def get_current_admin(current_user: dict = Depends(get_current_user)) -> dict:
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin can perform this action",
        )
    return current_user