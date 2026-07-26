from typing import Any

import pytest
import requests

from ariadne.core.catalog.tmdb import (
    BACKOFF_BASE_SECONDS,
    MAX_RETRY_AFTER_SECONDS,
    RateLimiter,
    TmdbAuthError,
    TmdbClient,
    TmdbError,
    TmdbNotFound,
)


class FakeResponse:
    def __init__(self, status_code: int, body: Any = None, headers: dict[str, str] | None = None):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.headers = headers or {}

    def json(self) -> Any:
        return self._body


class FakeSession:
    """Returns queued responses in order, recording the calls it received."""

    def __init__(self, *responses: FakeResponse | Exception):
        self._queue = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, Any], timeout: int) -> FakeResponse:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        if not self._queue:
            raise AssertionError("FakeSession received more requests than it was given")
        nxt = self._queue.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def make_client(
    *responses: FakeResponse | Exception, **kwargs: Any
) -> tuple[TmdbClient, list[float]]:
    slept: list[float] = []
    client = TmdbClient(
        api_key="test-key",
        requests_per_second=1000.0,
        session=FakeSession(*responses),  # type: ignore[arg-type]
        sleep=slept.append,
        **kwargs,
    )
    return client, slept


# --- auth ------------------------------------------------------------------------------


def test_missing_api_key_fails_immediately(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("ariadne.core.catalog.tmdb.get_settings", lambda: _SettingsStub(""))
    with pytest.raises(TmdbAuthError):
        TmdbClient()


class _SettingsStub:
    def __init__(self, key: str):
        self.tmdb_api_key = key
        self.tmdb_rate_limit = 10.0


def test_api_key_is_sent_as_a_query_parameter():
    client, _ = make_client(FakeResponse(200, {"results": []}))
    client.search_movies("The Godfather", 1972)
    assert client._session.calls[0]["params"]["api_key"] == "test-key"  # type: ignore[attr-defined]


@pytest.mark.parametrize("status", [401, 403])
def test_rejected_key_raises_without_retrying(status: int):
    client, slept = make_client(FakeResponse(status))
    with pytest.raises(TmdbAuthError):
        client.get_movie(238)
    assert client.request_count == 1
    assert slept == []


# --- requests --------------------------------------------------------------------------


def test_search_passes_year_when_given():
    client, _ = make_client(FakeResponse(200, {"results": [{"id": 1}]}))
    client.search_movies("Whiplash", 2014)
    assert client._session.calls[0]["params"]["year"] == 2014  # type: ignore[attr-defined]


def test_search_omits_year_when_absent():
    client, _ = make_client(FakeResponse(200, {"results": []}))
    client.search_movies("Whiplash")
    assert "year" not in client._session.calls[0]["params"]  # type: ignore[attr-defined]


def test_search_returns_empty_list_when_results_missing():
    client, _ = make_client(FakeResponse(200, {}))
    assert client.search_movies("Nothing") == []


def test_missing_film_raises_not_found_without_retrying():
    client, slept = make_client(FakeResponse(404))
    with pytest.raises(TmdbNotFound):
        client.get_movie(999999999)
    assert client.request_count == 1
    assert slept == []


def test_non_object_body_is_rejected():
    client, _ = make_client(FakeResponse(200, ["not", "an", "object"]))
    with pytest.raises(TmdbError):
        client.get_movie(238)


# --- retries ---------------------------------------------------------------------------


def test_transient_server_error_is_retried_then_succeeds():
    client, slept = make_client(FakeResponse(503), FakeResponse(200, {"id": 238}))
    assert client.get_movie(238)["id"] == 238
    assert client.request_count == 2
    assert client.retry_count == 1
    assert slept == [BACKOFF_BASE_SECONDS]


def test_backoff_doubles_between_attempts():
    client, slept = make_client(FakeResponse(500), FakeResponse(500), FakeResponse(200, {"id": 1}))
    client.get_movie(1)
    assert slept == [BACKOFF_BASE_SECONDS, BACKOFF_BASE_SECONDS * 2]


def test_retry_after_header_is_honoured():
    client, slept = make_client(
        FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse(200, {"id": 1})
    )
    client.get_movie(1)
    assert slept == [7.0]


def test_absurd_retry_after_is_capped():
    client, slept = make_client(
        FakeResponse(429, headers={"Retry-After": "9999"}), FakeResponse(200, {"id": 1})
    )
    client.get_movie(1)
    assert slept == [MAX_RETRY_AFTER_SECONDS]


def test_unparseable_retry_after_falls_back_to_backoff():
    client, slept = make_client(
        FakeResponse(429, headers={"Retry-After": "soon"}), FakeResponse(200, {"id": 1})
    )
    client.get_movie(1)
    assert slept == [BACKOFF_BASE_SECONDS]


def test_network_errors_are_retried():
    client, _ = make_client(requests.ConnectionError("boom"), FakeResponse(200, {"id": 1}))
    assert client.get_movie(1)["id"] == 1
    assert client.retry_count == 1


def test_gives_up_after_max_attempts_without_sleeping_on_the_last():
    client, slept = make_client(*[FakeResponse(503) for _ in range(3)], max_attempts=3)
    with pytest.raises(TmdbError):
        client.get_movie(1)
    assert client.request_count == 3
    # Sleeping after the final failed attempt would be wasted time.
    assert len(slept) == 2


def test_unretryable_client_error_raises():
    client, slept = make_client(FakeResponse(422))
    with pytest.raises(TmdbError):
        client.get_movie(1)
    assert client.request_count == 1
    assert slept == []


# --- rate limiter ----------------------------------------------------------------------


def test_rate_limiter_does_not_sleep_before_the_first_request():
    slept: list[float] = []
    RateLimiter(10.0, sleep=slept.append).wait()
    assert slept == []


def test_rate_limiter_spaces_consecutive_requests():
    slept: list[float] = []
    limiter = RateLimiter(2.0, sleep=slept.append)
    limiter.wait()
    limiter.wait()
    assert len(slept) == 1
    assert 0 < slept[0] <= 0.5


def test_rate_limiter_rejects_a_nonpositive_rate():
    with pytest.raises(ValueError):
        RateLimiter(0)
