"""Unit tests for observability middleware."""
import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.observability import RequestIDMiddleware, get_request_id


async def _echo_request_id(request):
  return PlainTextResponse(get_request_id())


@pytest.fixture
def client():
    app = Starlette(routes=[Route("/", _echo_request_id)])
    app.add_middleware(RequestIDMiddleware)
    return TestClient(app)


def test_middleware_generates_request_id(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers.get("X-Request-ID")
    assert len(resp.headers["X-Request-ID"]) == 12
    assert resp.text == resp.headers["X-Request-ID"]


def test_middleware_propagates_incoming_request_id(client):
    resp = client.get("/", headers={"X-Request-ID": "custom-id-99"})
    assert resp.status_code == 200
    assert resp.headers["X-Request-ID"] == "custom-id-99"
    assert resp.text == "custom-id-99"
