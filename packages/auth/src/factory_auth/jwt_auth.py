"""
factory_auth/jwt_auth.py
JWT Token Encoding and Verification
"""
from typing import Any, Dict, Optional
import jwt
from factory_contracts.errors import UnauthorizedError


def decode_jwt_token(token: str, secret: str, algorithm: str = "HS256", audience: Optional[str] = None) -> Dict[str, Any]:
    try:
        options = {"verify_aud": audience is not None}
        payload = jwt.decode(token, secret, algorithms=[algorithm], audience=audience, options=options)
        return payload
    except jwt.ExpiredSignatureError:
        raise UnauthorizedError(message="JWT token has expired.")
    except jwt.InvalidTokenError as e:
        raise UnauthorizedError(message=f"Invalid JWT token: {e}")
