"""Chat file exchange for the API server adapter (topical sibling of ``api_server.py``).

``ChatFileExchangeMixin`` carries the /v1/files routes (upload/download/list) and the
chat file input/output plumbing (store access, principal scope, materialization,
deliverable detection). The store itself lives in ``gateway/chat_files.py``; the
one-shot ``/v1/artifacts/*`` browser-control transport is a different contract and
is not touched here.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

try:
    from aiohttp import web
except ImportError:  # pragma: no cover - mirrors api_server's optional import
    web = None  # type: ignore[assignment]

# Logger parity with the origin module (moved log records keep their name).
logger = logging.getLogger("gateway.platforms.api_server")


def _require_auth(handler):
    """Late-bound facade auth guard: a test patching ``api_server._require_auth`` still
    intercepts the file routes (the facade import happens at class-definition time)."""
    from gateway.platforms.api_server import _require_auth as _facade_require_auth
    return _facade_require_auth(handler)


class ChatFileExchangeMixin:
    """The API server's bidirectional file exchange (store: ``gateway/chat_files.py``)."""

    def _file_store(self):
        """The routed profile's chat FileStore (per-home cache, like the response store)."""
        from hermes_constants import get_hermes_home
        home = str(get_hermes_home())
        with self._file_store_lock:
            store = self._file_stores.get(home)
            if store is None:
                from gateway.chat_files import FileStore, file_max_bytes, file_ttl_seconds
                store = FileStore(
                    get_hermes_home() / "files",
                    max_bytes=file_max_bytes(), ttl_seconds=file_ttl_seconds())
                self._file_stores[home] = store
            return store

    def _file_scope_key(self) -> str:
        """The authenticated principal's file scope: a (profile, API key) digest — the same
        namespace convention as the idempotency scope (opaque, never the credential)."""
        from gateway.platforms.api_server import _api_request_profile
        from gateway.chat_files import file_scope_key
        return file_scope_key(_api_request_profile.get() or "default", self._expected_api_key())

    def _resolve_chat_file(self, file_id: str) -> dict[str, Any]:
        """Load a stored file and materialize it on the agent host -> the reference dict
        ``_normalize_file_part`` renders. Raises ``ValueError("file_not_found:...")`` for an
        unknown, expired or out-of-scope id (``file_rate_limited`` when the scope is throttled)."""
        from hermes_constants import get_hermes_home
        from gateway.chat_files import FileRateLimited, FileStoreError, safe_filename
        file_id = (file_id or "").strip()
        if not file_id:
            raise ValueError("file_not_found:File parts must include a file_id.")
        try:
            data, receipt = self._file_store().load(file_id, self._file_scope_key())
        except FileRateLimited as exc:
            raise ValueError(f"file_rate_limited:{exc}") from None
        except FileStoreError as exc:
            raise ValueError(f"file_not_found:{exc}") from None
        dest_dir = get_hermes_home() / "cache" / "files" / file_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / safe_filename(receipt.filename)
        dest.write_bytes(data)
        return {"path": str(dest), "filename": receipt.filename,
                "content_type": receipt.content_type, "size_bytes": receipt.size_bytes}

    def _file_output_items(self, text: Any) -> list[dict[str, Any]]:
        """Deliverable files the final text references -> wire ``file`` items (minted id +
        download_url, never a filesystem path). Unreadable or oversized files are skipped with
        a log line; the visible text itself is left untouched."""
        from gateway.chat_files import FileStoreError, find_deliverable_paths, mime_for_filename
        items: list[dict[str, Any]] = []
        seen_expanded: set[str] = set()
        for path_str in find_deliverable_paths(text):
            expanded = os.path.expanduser(path_str)
            if expanded in seen_expanded:
                continue
            seen_expanded.add(expanded)
            path = Path(expanded)
            try:
                if not path.is_file():
                    continue
                data = path.read_bytes()
            except OSError:
                logger.info("[%s] skipping unreadable deliverable path %s", self.name, path_str)
                continue
            store = self._file_store()
            if len(data) > store.max_bytes:
                logger.info("[%s] skipping oversized deliverable %s (%d bytes)", self.name, path_str, len(data))
                continue
            content_type = mime_for_filename(path.name) or "application/octet-stream"
            try:
                receipt = store.store(data, path.name, content_type, self._file_scope_key())
            except FileStoreError:
                continue
            items.append({
                "type": "file", "filename": receipt.filename, "content_type": receipt.content_type,
                "size_bytes": receipt.size_bytes,
                "download_url": f"/v1/files/{receipt.file_id}/content",
                "file_id": receipt.file_id})
        return items

    @_require_auth
    async def _handle_file_upload(self, request: web.Request) -> web.Response:
        """POST /v1/files — store an uploaded file for the authenticated principal -> 201
        receipt (id, filename, content_type, size_bytes, download_url, expires_at)."""
        from gateway.platforms.api_server import _error_response
        from gateway.chat_files import (
            EXTENSION_BY_MIME, FileRateLimited, FileTooLarge, MimeRejected,
            mime_for_filename, safe_filename)
        store = self._file_store()
        content_type = (request.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        raw_name = (request.headers.get("X-File-Filename") or "").strip()
        filename = safe_filename(raw_name) if raw_name else "upload" + EXTENSION_BY_MIME.get(content_type, "")
        if content_type in ("", "application/octet-stream"):
            content_type = mime_for_filename(filename) or content_type
        if content_type not in store.allowed_mimetypes:
            return _error_response("MIME type not allowed", 415, code="mime_not_allowed")
        content_length = request.headers.get("Content-Length")
        if content_length is not None:
            try:
                if int(content_length) > store.max_bytes:
                    return _error_response("File too large", 413, code="file_too_large")
            except ValueError:
                return _error_response("Invalid Content-Length header", 400, code="invalid_content_length")
        data = await request.read()
        if len(data) > store.max_bytes:
            return _error_response("File too large", 413, code="file_too_large")
        try:
            receipt = await asyncio.to_thread(store.store, data, filename, content_type, self._file_scope_key())
        except FileRateLimited:
            return _error_response("Too many file operations", 429, code="rate_limited")
        except FileTooLarge:
            return _error_response("File too large", 413, code="file_too_large")
        except MimeRejected:
            return _error_response("MIME type not allowed", 415, code="mime_not_allowed")
        return web.json_response(receipt.to_dict(), status=201)

    @_require_auth
    async def _handle_file_download(self, request: web.Request) -> web.Response:
        """GET /v1/files/{file_id}/content — the stored bytes (repeatable until TTL expiry)."""
        from gateway.platforms.api_server import _error_response
        from gateway.chat_files import FileExpired, FileRateLimited, FileStoreError, is_valid_file_id
        file_id = request.match_info.get("file_id", "")
        if not is_valid_file_id(file_id):
            return _error_response("Invalid file id", 400, code="invalid_file_id")
        try:
            data, receipt = await asyncio.to_thread(
                self._file_store().load, file_id, self._file_scope_key())
        except FileRateLimited:
            return _error_response("Too many file operations", 429, code="rate_limited")
        except FileExpired:
            return _error_response("File expired", 410, code="file_expired")
        except FileStoreError:
            return _error_response("File not found", 404, code="file_not_found")
        return web.Response(
            body=data, status=200,
            headers={
                "Content-Type": receipt.content_type,
                "Content-Disposition": f'attachment; filename="{receipt.filename}"',
                "X-File-Id": receipt.file_id,
                "X-File-Sha256": receipt.sha256})

    @_require_auth
    async def _handle_file_list(self, request: web.Request) -> web.Response:
        """GET /v1/files — the caller's active files (id, filename, size, expires_at)."""
        from gateway.platforms.api_server import _error_response
        from gateway.chat_files import FileRateLimited
        try:
            receipts = await asyncio.to_thread(self._file_store().list, self._file_scope_key())
        except FileRateLimited:
            return _error_response("Too many file operations", 429, code="rate_limited")
        return web.json_response({"object": "list", "data": [r.to_dict() for r in receipts]})
