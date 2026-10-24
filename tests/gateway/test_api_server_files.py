"""Behaviour contracts for the API server's bidirectional file exchange.

Covers the ``/v1/files`` routes (upload/download/list, auth ladder, MIME/size
policy, TTL expiry), file input in chat/responses (materialization + the
``file_not_found`` error), file output items (responses + chat completions,
streaming + non-streaming, code-block exclusion) and the ``file_exchange``
capability block.
"""

import hashlib
import json
import re
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)
from gateway.chat_files import FileStore


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/files", adapter._handle_file_upload)
    app.router.add_get("/v1/files", adapter._handle_file_list)
    app.router.add_get("/v1/files/{file_id}/content", adapter._handle_file_download)
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    return app


@pytest.fixture
def adapter():
    return _make_adapter()


async def _upload_file(cli, data: bytes, *, content_type: str = "application/pdf",
                       filename: str = "report.pdf") -> dict:
    resp = await cli.post(
        "/v1/files", data=data,
        headers={"Content-Type": content_type, "X-File-Filename": filename})
    assert resp.status == 201, await resp.text()
    return await resp.json()


def _sse_frames(body: str) -> list[dict]:
    frames = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        if data == "[DONE]":
            continue
        try:
            frames.append(json.loads(data))
        except json.JSONDecodeError:
            continue
    return frames


class _Clock:
    def __init__(self, start: float = 1_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestFileRoutes:
    @pytest.mark.asyncio
    async def test_upload_201_receipt_shape(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            receipt = await _upload_file(cli, b"%PDF-1.4 test")
        assert re.fullmatch(r"file_[0-9a-f]{32}", receipt["id"])
        assert receipt["filename"] == "report.pdf"
        assert receipt["content_type"] == "application/pdf"
        assert receipt["size_bytes"] == len(b"%PDF-1.4 test")
        assert receipt["download_url"] == f"/v1/files/{receipt['id']}/content"
        assert isinstance(receipt["expires_at"], int) and receipt["expires_at"] > 0

    @pytest.mark.asyncio
    async def test_upload_default_filename_from_mime(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/files", data=b"x", headers={"Content-Type": "application/pdf"})
            payload = await resp.json()
        assert resp.status == 201
        assert payload["filename"] == "upload.pdf"

    @pytest.mark.asyncio
    async def test_download_200_headers_and_repeatable(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            receipt = await _upload_file(cli, b"hello bytes")
            for _ in range(2):
                resp = await cli.get(receipt["download_url"])
                assert resp.status == 200
                assert await resp.read() == b"hello bytes"
                assert resp.headers["Content-Type"] == "application/pdf"
                assert 'filename="report.pdf"' in resp.headers["Content-Disposition"]
                assert resp.headers["X-File-Id"] == receipt["id"]
                assert resp.headers["X-File-Sha256"] == hashlib.sha256(b"hello bytes").hexdigest()

    @pytest.mark.asyncio
    async def test_download_unknown_id_404(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/files/file_" + "0" * 32 + "/content")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_download_invalid_id_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/files/not-a-file-id/content")
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_download_expired_410(self, adapter):
        from hermes_constants import get_hermes_home
        clock = _Clock()
        store = FileStore(get_hermes_home() / "files", ttl_seconds=60, clock=clock)
        adapter._file_stores[str(get_hermes_home())] = store
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            receipt = await _upload_file(cli, b"hello")
            clock.advance(61)
            resp = await cli.get(receipt["download_url"])
        assert resp.status == 410

    @pytest.mark.asyncio
    async def test_upload_401_without_bearer(self, adapter):
        adapter._api_key = "test-api-key-0123456789"
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/files", data=b"x",
                                 headers={"Content-Type": "text/plain", "X-File-Filename": "a.txt"})
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_upload_415_mime_rejected(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/files", data=b"MZ",
                                 headers={"Content-Type": "application/x-msdownload",
                                         "X-File-Filename": "a.exe"})
        assert resp.status == 415

    @pytest.mark.asyncio
    async def test_upload_413_too_large(self, adapter, monkeypatch):
        monkeypatch.setenv("API_SERVER_FILE_MAX_BYTES", "16")
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/files", data=b"x" * 17,
                                 headers={"Content-Type": "text/plain", "X-File-Filename": "a.txt"})
        assert resp.status == 413

    @pytest.mark.asyncio
    async def test_list_returns_caller_files(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            r1 = await _upload_file(cli, b"one")
            r2 = await _upload_file(cli, b"two", filename="b.txt", content_type="text/plain")
            resp = await cli.get("/v1/files")
            data = (await resp.json())["data"]
        assert resp.status == 200
        assert {r["id"] for r in data} == {r1["id"], r2["id"]}


class TestFileInput:
    @pytest.mark.asyncio
    async def test_chat_completions_file_part_materializes(self, adapter):
        from hermes_constants import get_hermes_home
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            receipt = await _upload_file(cli, b"pdf bytes")
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return ({"final_response": "ok", "messages": [], "api_calls": 1},
                            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                mock_run.side_effect = _stub
                resp = await cli.post("/v1/chat/completions", json={
                    "model": "hermes-agent",
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": "Analyze this"},
                        {"type": "file", "file": {"file_id": receipt["id"]}}]}],
                })
                body = await resp.text()
        assert resp.status == 200, body
        user_message = mock_run.captured["user_message"]
        # Text-only content collapses to a plain string (the native shape); a file part
        # alongside an image would stay a list.
        text = (user_message if isinstance(user_message, str) else "\n".join(
            p["text"] for p in user_message if p.get("type") == "text"))
        reference = next(p for p in text.splitlines() if p.startswith("Attached file: "))
        cache_dir = get_hermes_home() / "cache" / "files" / receipt["id"]
        materialized = next(iter(cache_dir.iterdir()))
        assert materialized.name == "report.pdf"
        assert materialized.read_bytes() == b"pdf bytes"
        assert str(materialized) in reference
        assert "application/pdf" in reference

    @pytest.mark.asyncio
    async def test_chat_completions_bad_file_id_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/chat/completions", json={
                "model": "hermes-agent",
                "messages": [{"role": "user", "content": [
                    {"type": "file", "file": {"file_id": "file_" + "0" * 32}}]}],
            })
            payload = await resp.json()
        assert resp.status == 400
        assert payload["error"]["code"] == "file_not_found"

    @pytest.mark.asyncio
    async def test_responses_input_file_part_materializes(self, adapter):
        from hermes_constants import get_hermes_home
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            receipt = await _upload_file(cli, b"pdf bytes")
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return ({"final_response": "ok", "messages": [], "api_calls": 1},
                            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                mock_run.side_effect = _stub
                resp = await cli.post("/v1/responses", json={
                    "model": "hermes-agent",
                    "input": [{"role": "user", "content": [
                        {"type": "input_text", "text": "Analyze this"},
                        {"type": "input_file", "file_id": receipt["id"]}]}],
                })
                body = await resp.text()
        assert resp.status == 200, body
        user_message = mock_run.captured["user_message"]
        text = (user_message if isinstance(user_message, str) else "\n".join(
            p["text"] for p in user_message if p.get("type") == "text"))
        assert any(p.startswith("Attached file: ") for p in text.splitlines())
        cache_dir = get_hermes_home() / "cache" / "files" / receipt["id"]
        materialized = next(iter(cache_dir.iterdir()))
        assert materialized.read_bytes() == b"pdf bytes"


class TestFileOutput:
    def _deliverable(self, tmp_path) -> str:
        path = tmp_path / "q3-revenue.pdf"
        path.write_bytes(b"%PDF-1.4 deliverable")
        return str(path)

    @pytest.mark.asyncio
    async def test_responses_nonstreaming_emits_file_item(self, adapter, tmp_path):
        target = self._deliverable(tmp_path)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    return ({"final_response": f"Done: {target}", "messages": [], "api_calls": 1},
                            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                mock_run.side_effect = _stub
                resp = await cli.post("/v1/responses", json={
                    "model": "hermes-agent", "input": "make a report"})
                payload = await resp.json()
        assert resp.status == 200
        output = payload["output"]
        file_items = [i for i in output if i.get("type") == "file"]
        assert len(file_items) == 1
        item = file_items[0]
        assert item["filename"] == "q3-revenue.pdf"
        assert item["content_type"] == "application/pdf"
        assert item["size_bytes"] == len(b"%PDF-1.4 deliverable")
        assert item["download_url"] == f"/v1/files/{item['file_id']}/content"
        assert re.fullmatch(r"file_[0-9a-f]{32}", item["file_id"])

    @pytest.mark.asyncio
    async def test_chat_completions_nonstreaming_files_side_channel(self, adapter, tmp_path):
        target = self._deliverable(tmp_path)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    return ({"final_response": f"Done: {target}", "messages": [], "api_calls": 1},
                            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                mock_run.side_effect = _stub
                resp = await cli.post("/v1/chat/completions", json={
                    "model": "hermes-agent",
                    "messages": [{"role": "user", "content": "make a report"}]})
                payload = await resp.json()
        assert resp.status == 200
        files = payload["hermes"]["files"]
        assert len(files) == 1
        assert files[0]["filename"] == "q3-revenue.pdf"
        assert files[0]["type"] == "file"

    @pytest.mark.asyncio
    async def test_responses_streaming_emits_file_item_events(self, adapter, tmp_path):
        target = self._deliverable(tmp_path)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    callback = kwargs.get("stream_delta_callback")
                    if callback:
                        callback("working... ")
                    return ({"final_response": f"Done: {target}", "messages": [], "api_calls": 1,
                            "completed": True},
                            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                mock_run.side_effect = _stub
                resp = await cli.post("/v1/responses", json={
                    "model": "hermes-agent", "input": "make a report", "stream": True})
                body = await resp.text()
        assert resp.status == 200
        frames = _sse_frames(body)
        added = [f for f in frames if f.get("type") == "response.output_item.added"
                 and f.get("item", {}).get("type") == "file"]
        done = [f for f in frames if f.get("type") == "response.output_item.done"
                and f.get("item", {}).get("type") == "file"]
        assert len(added) == 1 and len(done) == 1
        assert added[0]["item"]["filename"] == "q3-revenue.pdf"
        completed = [f for f in frames if f.get("type") == "response.completed"]
        assert len(completed) == 1
        assert any(i.get("type") == "file" for i in completed[0]["response"]["output"])

    @pytest.mark.asyncio
    async def test_chat_completions_streaming_hermes_file_event(self, adapter, tmp_path):
        target = self._deliverable(tmp_path)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    return ({"final_response": f"Done: {target}", "messages": [], "api_calls": 1,
                            "completed": True},
                            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                mock_run.side_effect = _stub
                resp = await cli.post("/v1/chat/completions", json={
                    "model": "hermes-agent",
                    "messages": [{"role": "user", "content": "make a report"}],
                    "stream": True})
                body = await resp.text()
        assert resp.status == 200
        assert "event: hermes.file" in body
        file_frames = [f for f in _sse_frames(body) if f.get("type") == "file"]
        assert len(file_frames) == 1
        assert file_frames[0]["filename"] == "q3-revenue.pdf"

    @pytest.mark.asyncio
    async def test_code_block_path_not_emitted(self, adapter, tmp_path):
        target = self._deliverable(tmp_path)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    return ({"final_response": f"```\ncat {target}\n```", "messages": [],
                            "api_calls": 1},
                            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                mock_run.side_effect = _stub
                resp = await cli.post("/v1/responses", json={
                    "model": "hermes-agent", "input": "make a report"})
                payload = await resp.json()
        assert resp.status == 200
        output = payload["output"]
        assert [i for i in output if i.get("type") == "file"] == []


class TestCapabilities:
    @pytest.mark.asyncio
    async def test_file_exchange_block(self, adapter, monkeypatch):
        monkeypatch.delenv("API_SERVER_FILE_MAX_BYTES", raising=False)
        monkeypatch.delenv("API_SERVER_FILE_TTL_SECONDS", raising=False)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/capabilities")
            payload = await resp.json()
        assert resp.status == 200
        fx = payload["file_exchange"]
        assert fx["upload"] == {"method": "POST", "path": "/v1/files"}
        assert fx["download"] == {"method": "GET", "path": "/v1/files/{id}/content"}
        assert fx["list"] == {"method": "GET", "path": "/v1/files"}
        assert fx["max_bytes"] == 104857600
        assert fx["ttl_seconds"] == 86400
        assert payload["endpoints"]["file_upload"] == {"method": "POST", "path": "/v1/files"}
