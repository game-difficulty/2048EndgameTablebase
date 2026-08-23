from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
import re
import tempfile
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

from backend.auth.db import get_auth_db_path


DEFAULT_AVATAR_INPUT_BYTES = 512 * 1024
DEFAULT_AVATAR_EDGE = 128
DEFAULT_AVATAR_MAX_PIXELS = 4_000_000
ALLOWED_AVATAR_FORMATS = {"JPEG", "PNG", "WEBP"}
AVATAR_KEY_RE = re.compile(r"^(?P<user_id>[1-9][0-9]*)/(?P<digest>[a-f0-9]{24})\.webp$")


class AvatarValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ProcessedAvatar:
    data: bytes
    sha256: str


def get_avatar_root() -> Path:
    configured = str(os.getenv("CLOUD_AVATAR_ROOT") or "").strip()
    if configured:
        return Path(configured)
    return get_auth_db_path().parent / "avatars"


def get_avatar_input_limit() -> int:
    try:
        return max(1, int(os.getenv("CLOUD_MAX_AVATAR_BYTES", str(DEFAULT_AVATAR_INPUT_BYTES))))
    except ValueError:
        return DEFAULT_AVATAR_INPUT_BYTES


def _resampling_filter():
    return getattr(Image, "Resampling", Image).LANCZOS


def process_avatar_bytes(data: bytes) -> ProcessedAvatar:
    if not data:
        raise AvatarValidationError("Avatar image is empty.")
    if len(data) > get_avatar_input_limit():
        raise AvatarValidationError("Avatar image exceeds the upload size limit.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                image_format = str(source.format or "").upper()
                if image_format not in ALLOWED_AVATAR_FORMATS:
                    raise AvatarValidationError("Avatar must be a JPEG, PNG, or WebP image.")
                if int(getattr(source, "n_frames", 1) or 1) != 1:
                    raise AvatarValidationError("Animated avatars are not supported.")
                width, height = source.size
                if width < 32 or height < 32:
                    raise AvatarValidationError("Avatar image is too small.")
                if width * height > DEFAULT_AVATAR_MAX_PIXELS:
                    raise AvatarValidationError("Avatar image dimensions are too large.")
                source.load()
                oriented = ImageOps.exif_transpose(source)
                image = oriented.convert("RGBA")
    except AvatarValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning, UnidentifiedImageError, OSError) as exc:
        raise AvatarValidationError("Avatar image could not be decoded safely.") from exc

    edge = DEFAULT_AVATAR_EDGE
    fitted = ImageOps.fit(
        image,
        (edge, edge),
        method=_resampling_filter(),
        centering=(0.5, 0.5),
    )
    output = io.BytesIO()
    try:
        fitted.save(output, format="WEBP", quality=78, method=6, exact=True)
    except OSError as exc:
        raise AvatarValidationError("Avatar image could not be encoded.") from exc
    encoded = output.getvalue()
    if not encoded:
        raise AvatarValidationError("Avatar image could not be encoded.")
    digest = hashlib.sha256(encoded).hexdigest()
    return ProcessedAvatar(data=encoded, sha256=digest)


def avatar_key(user_id: int, sha256: str) -> str:
    return f"{int(user_id)}/{str(sha256)[:24]}.webp"


def resolve_avatar_key(key: str, *, expected_user_id: int | None = None) -> Path:
    normalized = str(key or "").replace("\\", "/").strip("/")
    match = AVATAR_KEY_RE.fullmatch(normalized)
    if match is None:
        raise ValueError("Invalid avatar key.")
    if expected_user_id is not None and int(match.group("user_id")) != int(expected_user_id):
        raise ValueError("Avatar key does not belong to this user.")
    root = get_avatar_root().resolve()
    resolved = (root / normalized).resolve()
    if root not in resolved.parents:
        raise ValueError("Avatar path is outside the configured root.")
    return resolved


def save_processed_avatar(user_id: int, avatar: ProcessedAvatar) -> str:
    key = avatar_key(user_id, avatar.sha256)
    target = resolve_avatar_key(key, expected_user_id=user_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return key

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".avatar-",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as output:
            output.write(avatar.data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return key


def delete_avatar_file(key: str | None) -> None:
    if not key:
        return
    try:
        target = resolve_avatar_key(key)
    except ValueError:
        return
    target.unlink(missing_ok=True)
    try:
        target.parent.rmdir()
    except OSError:
        pass

