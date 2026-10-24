"""Behaviour contracts for the API server chat file store (``gateway/chat_files.py``).

Covers the store's round-trip, SHA-256 re-verification, TTL expiry, MIME/size
policies, scope binding, non-one-shot loads, rate limiting and the deliverable
path scanner used for file output.
"""

import hashlib
import json

import pytest

from gateway.chat_files import (
    FileExpired,
    FileNotFound,
    FileRateLimited,
    FileStore,
    FileStoreError,
    FileTooLarge,
    MimeRejected,
    find_deliverable_paths,
    file_max_bytes,
    file_scope_key,
    file_ttl_seconds,
    is_valid_file_id,
    mime_for_filename,
    safe_filename,
)


class _Clock:
    def __init__(self, start: float = 1_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _store(tmp_path, **kwargs) -> FileStore:
    kwargs.setdefault("clock", _Clock())
    return FileStore(tmp_path / "files", **kwargs)


class TestFileStoreRoundTrip:
    def test_store_load_round_trip(self, tmp_path):
        store = _store(tmp_path)
        receipt = store.store(b"%PDF-1.4", "report.pdf", "application/pdf", "scope-a")
        assert receipt.file_id.startswith("file_") and len(receipt.file_id) == 37
        assert receipt.filename == "report.pdf"
        assert receipt.size_bytes == 8
        data, loaded = store.load(receipt.file_id, "scope-a")
        assert data == b"%PDF-1.4"
        assert loaded.file_id == receipt.file_id
        assert loaded.sha256 == hashlib.sha256(b"%PDF-1.4").hexdigest()

    def test_load_is_repeatable_not_one_shot(self, tmp_path):
        store = _store(tmp_path)
        receipt = store.store(b"hello", "hello.txt", "text/plain", "scope-a")
        for _ in range(3):
            data, _ = store.load(receipt.file_id, "scope-a")
            assert data == b"hello"

    def test_sha256_reverified_on_read(self, tmp_path):
        store = _store(tmp_path)
        receipt = store.store(b"original", "a.txt", "text/plain", "scope-a")
        (tmp_path / "files" / "scope-a" / receipt.file_id).write_bytes(b"tampered")
        with pytest.raises(FileStoreError):
            store.load(receipt.file_id, "scope-a")

    def test_scope_mismatch_is_not_found(self, tmp_path):
        store = _store(tmp_path)
        receipt = store.store(b"x", "a.txt", "text/plain", "scope-a")
        with pytest.raises(FileNotFound):
            store.load(receipt.file_id, "scope-b")

    def test_unknown_id_is_not_found(self, tmp_path):
        store = _store(tmp_path)
        with pytest.raises(FileNotFound):
            store.load("file_" + "0" * 32, "scope-a")

    def test_malformed_id_is_not_found(self, tmp_path):
        store = _store(tmp_path)
        with pytest.raises(FileNotFound):
            store.load("../etc/passwd", "scope-a")


class TestFileStorePolicies:
    def test_ttl_expiry(self, tmp_path):
        clock = _Clock()
        store = FileStore(tmp_path / "files", ttl_seconds=60, clock=clock)
        receipt = store.store(b"x", "a.txt", "text/plain", "scope-a")
        clock.advance(59)
        store.load(receipt.file_id, "scope-a")
        clock.advance(2)
        with pytest.raises(FileExpired):
            store.load(receipt.file_id, "scope-a")
        assert store.list("scope-a") == []

    def test_purge_expired(self, tmp_path):
        clock = _Clock()
        store = FileStore(tmp_path / "files", ttl_seconds=60, clock=clock)
        receipt = store.store(b"x", "a.txt", "text/plain", "scope-a")
        assert store.purge_expired() == 0
        clock.advance(61)
        assert store.purge_expired() == 1
        with pytest.raises(FileNotFound):
            store.load(receipt.file_id, "scope-a")

    def test_size_cap(self, tmp_path):
        store = _store(tmp_path, max_bytes=16)
        with pytest.raises(FileTooLarge):
            store.store(b"x" * 17, "a.bin", "application/octet-stream", "scope-a")
        store.store(b"x" * 16, "a.bin", "application/octet-stream", "scope-a")

    def test_mime_rejection(self, tmp_path):
        store = _store(tmp_path)
        with pytest.raises(MimeRejected):
            store.store(b"x", "a.exe", "application/x-msdownload", "scope-a")

    def test_mime_parameters_ignored(self, tmp_path):
        store = _store(tmp_path)
        store.store(b"x", "a.txt", "text/plain; charset=utf-8", "scope-a")

    def test_rate_limit(self, tmp_path):
        store = _store(tmp_path, rate_limit_per_minute=2)
        store.store(b"1", "a.txt", "text/plain", "scope-a")
        store.store(b"2", "b.txt", "text/plain", "scope-a")
        with pytest.raises(FileRateLimited):
            store.store(b"3", "c.txt", "text/plain", "scope-a")

    def test_list_newest_first_and_scoped(self, tmp_path):
        clock = _Clock()
        store = FileStore(tmp_path / "files", clock=clock)
        r1 = store.store(b"1", "a.txt", "text/plain", "scope-a")
        clock.advance(1)
        r2 = store.store(b"2", "b.txt", "text/plain", "scope-a")
        store.store(b"3", "c.txt", "text/plain", "scope-b")
        listed = store.list("scope-a")
        assert [r.file_id for r in listed] == [r2.file_id, r1.file_id]


class TestFileReceipt:
    def test_to_dict_never_carries_filesystem_paths(self, tmp_path):
        store = _store(tmp_path)
        receipt = store.store(b"x", "a.txt", "text/plain", "scope-a")
        wire = receipt.to_dict()
        assert set(wire) == {"id", "filename", "content_type", "size_bytes", "download_url", "expires_at"}
        assert json.dumps(wire).find(str(tmp_path)) == -1
        assert wire["download_url"] == f"/v1/files/{receipt.file_id}/content"


class TestHelpers:
    def test_file_id_validation(self):
        assert is_valid_file_id("file_" + "a" * 32)
        assert not is_valid_file_id("file_" + "a" * 31)
        assert not is_valid_file_id("artifact_" + "a" * 32)
        assert not is_valid_file_id("")

    def test_file_scope_key_is_opaque_and_stable(self):
        key = file_scope_key("default", "secret-key")
        assert key == file_scope_key("default", "secret-key")
        assert key != file_scope_key("other", "secret-key")
        assert "secret-key" not in key

    def test_mime_for_filename(self):
        assert mime_for_filename("report.PDF") == "application/pdf"
        assert mime_for_filename("noext") == ""

    def test_safe_filename(self):
        assert safe_filename("/etc/passwd") == "passwd"
        assert safe_filename("a\\b.txt") == "b.txt"
        assert safe_filename("") == "file"
        assert safe_filename("a" * 300) == "a" * 255


class TestFindDeliverablePaths:
    def test_absolute_path(self):
        assert find_deliverable_paths("Saved to /home/u/report.pdf") == ["/home/u/report.pdf"]

    def test_tilde_path(self):
        assert find_deliverable_paths("See ~/reports/q3.xlsx") == ["~/reports/q3.xlsx"]

    def test_multiple_unique_in_order(self):
        text = "/a/b.pdf then /a/b.pdf and /c/d.mp4"
        assert find_deliverable_paths(text) == ["/a/b.pdf", "/c/d.mp4"]

    def test_code_block_paths_ignored(self):
        text = "copy this:\n```bash\ncat /home/u/secret.pdf\n```\ndone"
        assert find_deliverable_paths(text) == []

    def test_inline_code_path_ignored(self):
        assert find_deliverable_paths("run `cat /home/u/secret.pdf` now") == []

    def test_urls_ignored(self):
        assert find_deliverable_paths("https://example.com/report.pdf") == []
        assert find_deliverable_paths("file:///home/u/report.pdf") == []

    def test_non_deliverable_extension_ignored(self):
        assert find_deliverable_paths("/home/u/script.sh.bak") == []
        assert find_deliverable_paths("/home/u/passwd") == []

    def test_trailing_punctuation_stripped(self):
        assert find_deliverable_paths("done. /home/u/report.pdf.") == ["/home/u/report.pdf"]

    def test_empty_and_non_string(self):
        assert find_deliverable_paths("") == []
        assert find_deliverable_paths(None) == []
        assert find_deliverable_paths(42) == []


class TestEnvTunables:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("API_SERVER_FILE_MAX_BYTES", raising=False)
        monkeypatch.delenv("API_SERVER_FILE_TTL_SECONDS", raising=False)
        assert file_max_bytes() == 104857600
        assert file_ttl_seconds() == 86400

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_FILE_MAX_BYTES", "1234")
        monkeypatch.setenv("API_SERVER_FILE_TTL_SECONDS", "60")
        assert file_max_bytes() == 1234
        assert file_ttl_seconds() == 60

    def test_invalid_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_FILE_MAX_BYTES", "not-a-number")
        monkeypatch.setenv("API_SERVER_FILE_TTL_SECONDS", "-5")
        assert file_max_bytes() == 104857600
        assert file_ttl_seconds() == 86400
