"""Chat file store for the API server's bidirectional file exchange.

Generalizes the one-shot artifact transport into a chat file store: a larger byte
cap (100 MB), a longer TTL (24 h), a broad deliverable MIME allowlist, and
repeatable loads (not one-shot). Files are bound to an authenticated principal
scope, addressed only by server-minted ``file_<32hex>`` ids, re-verified by
SHA-256 on every read, and written via temp-file-then-rename. Filesystem paths
never appear in receipts or wire payloads.
"""

import hashlib
import json
import os
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

DEFAULT_FILE_MAX_BYTES = 100 * 1024 * 1024
DEFAULT_FILE_TTL_SECONDS = 24 * 60 * 60
DEFAULT_FILE_RATE_LIMIT_PER_MINUTE = 120
_FILE_ID_RE = re.compile(r"file_[0-9a-f]{32}")

# Extension -> MIME for the deliverable set (documents, data, presentations,
# archives, web, images, audio, video). The upload allowlist derives from this map.
_EXTENSION_MIME = {
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".odt": "application/vnd.oasis.opendocument.text",
    ".ods": "application/vnd.oasis.opendocument.spreadsheet",
    ".odp": "application/vnd.oasis.opendocument.presentation",
    ".rtf": "application/rtf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".json": "application/json",
    ".xml": "application/xml",
    ".html": "text/html",
    ".htm": "text/html",
    ".css": "text/css",
    ".js": "application/javascript",
    ".ts": "application/typescript",
    ".py": "text/x-python",
    ".sh": "application/x-sh",
    ".zip": "application/zip",
    ".tar": "application/x-tar",
    ".gz": "application/gzip",
    ".tgz": "application/gzip",
    ".bz2": "application/x-bzip2",
    ".7z": "application/x-7z-compressed",
    ".rar": "application/vnd.rar",
    ".parquet": "application/octet-stream",
    ".sqlite": "application/octet-stream",
    ".db": "application/octet-stream",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
}
DELIVERABLE_EXTENSIONS = frozenset(_EXTENSION_MIME)
DEFAULT_ALLOWED_MIME_TYPES = frozenset(_EXTENSION_MIME.values())
EXTENSION_BY_MIME: dict[str, str] = {}
for _ext, _mime in _EXTENSION_MIME.items():
    EXTENSION_BY_MIME.setdefault(_mime, _ext)


def _coerce_positive_int(raw: Any, default: int) -> int:
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def file_max_bytes() -> int:
    """``API_SERVER_FILE_MAX_BYTES`` tunable (default 100 MB)."""
    return _coerce_positive_int(os.getenv("API_SERVER_FILE_MAX_BYTES"), DEFAULT_FILE_MAX_BYTES)


def file_ttl_seconds() -> int:
    """``API_SERVER_FILE_TTL_SECONDS`` tunable (default 24 h)."""
    return _coerce_positive_int(os.getenv("API_SERVER_FILE_TTL_SECONDS"), DEFAULT_FILE_TTL_SECONDS)


def file_scope_key(profile: str, api_key: str) -> str:
    """Scope binding for a (profile, API key) principal: the same opaque-digest
    namespace convention as the API server's idempotency scope — never the credential."""
    return hashlib.sha256(f"{profile or 'default'}\0{api_key or ''}".encode()).hexdigest()


def is_valid_file_id(file_id: str) -> bool:
    return bool(file_id) and _FILE_ID_RE.fullmatch(file_id) is not None


def mime_for_filename(filename: str) -> str:
    """The deliverable MIME for a filename's extension, or '' when not a deliverable."""
    return _EXTENSION_MIME.get(os.path.splitext(str(filename))[1].lower(), "")


def safe_filename(name: Any) -> str:
    """A display/materialization filename: basename only, control and quote characters
    replaced, length-bounded. Falls back to ``file`` when nothing usable remains."""
    base = os.path.basename(str(name).replace("\\", "/"))
    base = re.sub(r"[\x00-\x1f\"\\\\]", "_", base).strip()
    return base[:255] or "file"


class FileStoreError(Exception):
    """Base for chat file store failures (the HTTP layer maps these to status codes)."""


class FileNotFound(FileStoreError):
    """Unknown id, or an id bound to a different scope (existence is not leaked)."""


class FileExpired(FileStoreError):
    """The file's TTL has passed."""


class FileTooLarge(FileStoreError):
    """The payload exceeds the store's byte cap."""


class MimeRejected(FileStoreError):
    """The content type is not in the allowlist."""


class FileRateLimited(FileStoreError):
    """The scope's per-minute operation budget is exhausted."""


class FileCorrupted(FileStoreError):
    """The on-disk bytes no longer match the recorded SHA-256."""


@dataclass(frozen=True)
class FileReceipt:
    file_id: str
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    created_at: float
    expires_at: float
    scope: str

    def to_dict(self, download_path: Optional[str] = None) -> dict[str, Any]:
        """The wire shape: the minted id and download URL only — never a filesystem path."""
        return {
            "id": self.file_id,
            "filename": self.filename,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "download_url": download_path or f"/v1/files/{self.file_id}/content",
            "expires_at": int(self.expires_at),
        }


def _receipt_from_meta(meta: dict[str, Any]) -> FileReceipt:
    return FileReceipt(
        file_id=meta["file_id"], filename=meta["filename"], content_type=meta["content_type"],
        size_bytes=meta["size_bytes"], sha256=meta["sha256"], created_at=meta["created_at"],
        expires_at=meta["expires_at"], scope=meta["scope"])


class FileStore:
    """A per-profile chat file store: scoped, TTL-expiring, SHA-256-verified and
    repeatable (not one-shot). Writes go through a temp file + rename so a
    concurrent reader never observes a partial file."""

    def __init__(
        self,
        root: Any,
        *,
        max_bytes: int = DEFAULT_FILE_MAX_BYTES,
        ttl_seconds: int = DEFAULT_FILE_TTL_SECONDS,
        allowed_mimetypes: Optional[Any] = None,
        rate_limit_per_minute: int = DEFAULT_FILE_RATE_LIMIT_PER_MINUTE,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._root = Path(root)
        self._max_bytes = max_bytes
        self._ttl = ttl_seconds
        self._allowed_mimetypes = frozenset(
            str(m).split(";")[0].strip().lower()
            for m in (allowed_mimetypes if allowed_mimetypes is not None else DEFAULT_ALLOWED_MIME_TYPES))
        self._rate_limit = rate_limit_per_minute
        self._clock = clock or time.time
        self._rate_windows: dict[str, deque] = {}
        self._lock = threading.Lock()

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def ttl_seconds(self) -> int:
        return self._ttl

    @property
    def allowed_mimetypes(self) -> frozenset:
        return self._allowed_mimetypes

    def _scope_dir(self, scope: str) -> Path:
        return self._root / scope

    def _rate_allow(self, scope: str) -> bool:
        now = time.monotonic()
        with self._lock:
            window = self._rate_windows.setdefault(scope, deque())
            while window and now - window[0] > 60.0:
                window.popleft()
            if len(window) >= self._rate_limit:
                return False
            window.append(now)
            return True

    @staticmethod
    def _normalize_mime(content_type: Any) -> str:
        return str(content_type or "").split(";")[0].strip().lower()

    def store(self, data: bytes, filename: str, content_type: Any, scope: str) -> FileReceipt:
        """Write one file for ``scope`` -> its receipt. Raises FileRateLimited,
        FileTooLarge or MimeRejected."""
        if not isinstance(data, (bytes, bytearray)):
            raise FileTooLarge("payload must be bytes")
        if not self._rate_allow(scope):
            raise FileRateLimited("rate limit exceeded")
        data = bytes(data)
        if len(data) > self._max_bytes:
            raise FileTooLarge(f"file exceeds the {self._max_bytes} byte cap")
        mime = self._normalize_mime(content_type)
        if mime not in self._allowed_mimetypes:
            raise MimeRejected(f"content type {mime or 'unknown'} is not allowed")
        file_id = "file_" + uuid.uuid4().hex
        scope_dir = self._scope_dir(scope)
        scope_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "file_id": file_id,
            "filename": safe_filename(filename),
            "content_type": mime,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "created_at": self._clock(),
            "expires_at": self._clock() + self._ttl,
            "scope": scope,
        }
        tmp = scope_dir / (file_id + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, scope_dir / file_id)
        tmp_meta = scope_dir / (file_id + ".json.tmp")
        tmp_meta.write_text(json.dumps(meta), encoding="utf-8")
        os.replace(tmp_meta, scope_dir / (file_id + ".json"))
        return _receipt_from_meta(meta)

    def load(self, file_id: str, scope: str) -> tuple[bytes, FileReceipt]:
        """Read one file for ``scope`` (repeatable; expiry is by TTL only). Raises
        FileNotFound, FileExpired, FileCorrupted or FileRateLimited."""
        if not is_valid_file_id(file_id):
            raise FileNotFound(f"unknown file id: {file_id}")
        if not self._rate_allow(scope):
            raise FileRateLimited("rate limit exceeded")
        scope_dir = self._scope_dir(scope)
        meta_path = scope_dir / (file_id + ".json")
        if not meta_path.exists():
            raise FileNotFound(f"unknown file id: {file_id}")
        meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        if meta.get("scope") != scope:
            raise FileNotFound(f"unknown file id: {file_id}")
        if self._clock() >= meta["expires_at"]:
            raise FileExpired(f"file {file_id} has expired")
        data = (scope_dir / file_id).read_bytes()
        if hashlib.sha256(data).hexdigest() != meta["sha256"]:
            raise FileCorrupted(f"file {file_id} failed SHA-256 verification")
        return data, _receipt_from_meta(meta)

    def list(self, scope: str) -> list[FileReceipt]:
        """The scope's active (non-expired) files, newest first."""
        if not self._rate_allow(scope):
            raise FileRateLimited("rate limit exceeded")
        scope_dir = self._scope_dir(scope)
        receipts: list[FileReceipt] = []
        if scope_dir.is_dir():
            for meta_path in sorted(scope_dir.glob("file_*.json")):
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
                    if meta.get("scope") != scope or self._clock() >= meta["expires_at"]:
                        continue
                    receipts.append(_receipt_from_meta(meta))
                except (ValueError, OSError, KeyError):
                    continue
        receipts.sort(key=lambda receipt: receipt.created_at, reverse=True)
        return receipts

    def purge_expired(self) -> int:
        """Delete every file past its TTL (bytes + sidecar); returns the count removed."""
        removed = 0
        if not self._root.is_dir():
            return 0
        for scope_dir in self._root.iterdir():
            if not scope_dir.is_dir():
                continue
            for meta_path in scope_dir.glob("file_*.json"):
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
                    if self._clock() < meta["expires_at"]:
                        continue
                except (ValueError, OSError, KeyError):
                    continue
                for path in (scope_dir / meta["file_id"], meta_path):
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                removed += 1
        return removed


_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_PATH_TOKEN_RE = re.compile(r"(?:~/?|/)[\w.+\-]+(?:/[\w.+\-]+)*")
_TRAILING_PUNCTUATION = ".,;:!?"


def _mask_code(text: str) -> str:
    """Blank out fenced and inline code spans (positions preserved) so code samples
    are never treated as deliverable references."""
    text = _CODE_FENCE_RE.sub(lambda m: " " * len(m.group(0)), text)
    return _INLINE_CODE_RE.sub(lambda m: " " * len(m.group(0)), text)


def find_deliverable_paths(text: Any) -> list[str]:
    """Absolute or ``~``-relative paths in ``text`` ending in a deliverable extension,
    in order of first appearance. Code samples and URLs are excluded."""
    if not isinstance(text, str) or not text:
        return []
    masked = _mask_code(text)
    found: list[str] = []
    seen: set[str] = set()
    for match in _PATH_TOKEN_RE.finditer(masked):
        start = match.start()
        if start > 0 and masked[start - 1] in ":@/":
            continue  # a URL scheme separator, not a path
        token = match.group(0).rstrip(_TRAILING_PUNCTUATION)
        if not token:
            continue
        if os.path.splitext(token)[1].lower() not in DELIVERABLE_EXTENSIONS:
            continue
        if token not in seen:
            seen.add(token)
            found.append(token)
    return found
