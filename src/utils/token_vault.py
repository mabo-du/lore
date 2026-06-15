"""
Secure token storage.

Uses the OS keyring (via the `keyring` library) as the primary store.
Falls back to Fernet symmetric encryption (MAC-address derived key) when
keyring is unavailable, maintaining backward compatibility with existing
encrypted tokens.
"""

import base64
import uuid
import logging

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

_ENCRYPTED_PREFIX = "lore_v1:"
_SALT = b"lore-token-vault-2024"

try:
    import keyring

    _HAVE_KEYRING = True
except ImportError:
    _HAVE_KEYRING = False

_KEYRING_SERVICE = "lore"
_KEYRING_KEY = "hf_token"


def _derive_fernet_key() -> bytes:
    """Derive a machine-bound encryption key."""
    machine_id = str(uuid.getnode()).encode()
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=_SALT,
        iterations=600_000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(machine_id))
    return key


def _get_fernet() -> Fernet:
    return Fernet(_derive_fernet_key())


def encrypt_token(token: str) -> str:
    """Encrypt a token for storage.
    Uses OS keyring when available, falls back to Fernet + QSettings."""
    if _HAVE_KEYRING:
        try:
            keyring.set_password(_KEYRING_SERVICE, _KEYRING_KEY, token)
            # Return a marker so decrypt_token knows to read from keyring
            return "keyring:"
        except Exception as e:
            logger.warning("keyring.set_password failed, falling back to Fernet: %s", e)

    f = _get_fernet()
    encrypted = f.encrypt(token.encode())
    return _ENCRYPTED_PREFIX + encrypted.decode()


def decrypt_token(stored: str) -> str:
    """Decrypt a token previously stored with encrypt_token.
    Returns empty string if no token is stored."""
    if not stored:
        return ""

    # Keyring-stored token
    if stored == "keyring:":
        if _HAVE_KEYRING:
            try:
                token = keyring.get_password(_KEYRING_SERVICE, _KEYRING_KEY)
                return token or ""
            except Exception as e:
                logger.warning("keyring.get_password failed: %s", e)
        else:
            logger.warning(
                "Token marker is 'keyring:' but keyring is not installed. "
                "Re-save your token in Settings to store it locally."
            )
        return ""

    if not stored.startswith(_ENCRYPTED_PREFIX):
        # Legacy plaintext token — return as-is (will be re-encrypted on next save)
        return stored

    f = _get_fernet()
    encrypted = stored[len(_ENCRYPTED_PREFIX):].encode()
    return f.decrypt(encrypted).decode()
