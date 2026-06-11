"""
Secure token storage using Fernet symmetric encryption.

Keys are derived from the machine's MAC address, ensuring the encrypted
token is only decryptable on the same machine. This prevents casual
viewing of credentials if the QSettings file is exposed.
"""

import base64
import uuid

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

_ENCRYPTED_PREFIX = "lore_v1:"
_SALT = b"lore-token-vault-2024"


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
    """Encrypt a token for storage. Returns a prefixed string safe for QSettings."""
    f = _get_fernet()
    encrypted = f.encrypt(token.encode())
    return _ENCRYPTED_PREFIX + encrypted.decode()


def decrypt_token(stored: str) -> str:
    """Decrypt a token previously stored with encrypt_token.
    If the value is not encrypted (no prefix), returns it as-is for
    backward compatibility with plaintext tokens."""
    if not stored:
        return ""
    if not stored.startswith(_ENCRYPTED_PREFIX):
        # Legacy plaintext token — return as-is (will be re-encrypted on next save)
        return stored
    f = _get_fernet()
    encrypted = stored[len(_ENCRYPTED_PREFIX):].encode()
    return f.decrypt(encrypted).decode()
