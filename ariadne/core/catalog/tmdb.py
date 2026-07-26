"""TMDB HTTP client.

Deliberately polite: one shared rate limiter, bounded retries, and a request counter so the
Level 1 metrics can report how many calls a run actually cost. Nothing here knows about the
database — callers decide what to persist.
"""

import time
from collections.abc import Callable
from typing import Any

import requests

from ariadne.settings import get_settings

BASE_URL = "https://api.themoviedb.org/3"

# Retried: TMDB throttling and transient server faults. Everything else fails immediately,
# because retrying a 401 or a 404 only wastes the rate budget.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
DEFAULT_MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 0.5

# Cap on how long a Retry-After header can park us. TMDB should never send more than a few
# seconds; a larger value means something is wrong and failing is better than hanging.
MAX_RETRY_AFTER_SECONDS = 30.0


class TmdbError(Exception):
    pass


class TmdbAuthError(TmdbError):
    pass


class TmdbNotFound(TmdbError):
    pass


class RateLimiter:
    """Spaces requests by wall-clock interval.

    A sleeping limiter rather than a token bucket: this project makes long sequential runs
    over a few thousand films, so smoothing matters more than allowing bursts.
    """

    def __init__(self, requests_per_second: float, sleep: Callable[[float], None] = time.sleep):
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self._min_interval = 1.0 / requests_per_second
        self._sleep = sleep
        self._last_request: float | None = None

    def wait(self) -> None:
        if self._last_request is not None:
            remaining = self._min_interval - (time.monotonic() - self._last_request)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request = time.monotonic()


class TmdbClient:
    def __init__(
        self,
        api_key: str | None = None,
        requests_per_second: float | None = None,
        session: requests.Session | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        settings = get_settings()
        self._api_key = api_key if api_key is not None else settings.tmdb_api_key
        if not self._api_key:
            raise TmdbAuthError("TMDB_API_KEY is not set")

        rate = requests_per_second if requests_per_second is not None else settings.tmdb_rate_limit

        # The limiter keeps its own clock deliberately. Sharing `sleep` with retry backoff
        # would conflate two unrelated concerns and make either one untestable alone.
        self._limiter = RateLimiter(rate)

        self._session = session if session is not None else requests.Session()
        self._max_attempts = max_attempts
        self._sleep = sleep

        # Reported by the Level 1 metrics.
        self.request_count = 0
        self.retry_count = 0

    def search_movies(self, title: str, year: int | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"query": title, "include_adult": "true"}
        if year is not None:
            params["year"] = year
        payload = self._get("/search/movie", params)
        results = payload.get("results", [])
        return results if isinstance(results, list) else []

    def search_tv(self, title: str, year: int | None = None) -> list[dict[str, Any]]:
        """Television search, used only to explain a movie-search failure.

        Letterboxd exports contain some television. Confirming that is what an unresolved
        entry actually is turns an opaque failure into a reportable exclusion.
        """
        params: dict[str, Any] = {"query": title}
        if year is not None:
            params["first_air_date_year"] = year
        payload = self._get("/search/tv", params)
        results = payload.get("results", [])
        return results if isinstance(results, list) else []

    def get_movie(self, tmdb_id: int) -> dict[str, Any]:
        return self._get(f"/movie/{tmdb_id}", {})

    def get_credits(self, tmdb_id: int) -> dict[str, Any]:
        """Full cast and crew, every department.

        All departments are kept even though the model uses a handful of roles: one call
        returns the whole list, so filtering here would save nothing and would mean
        refetching every film after any change to role scope.
        """
        return self._get(f"/movie/{tmdb_id}/credits", {})

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{BASE_URL}{path}"
        query = {**params, "api_key": self._api_key}
        last_error: Exception | None = None

        for attempt in range(self._max_attempts):
            self._limiter.wait()
            self.request_count += 1

            try:
                response = self._session.get(url, params=query, timeout=15)
            except requests.RequestException as exc:
                last_error = exc
                self._backoff(attempt, response=None)
                continue

            if response.status_code == 200:
                body = response.json()
                if not isinstance(body, dict):
                    raise TmdbError(f"{path} returned {type(body).__name__}, expected an object")
                return body

            if response.status_code in (401, 403):
                raise TmdbAuthError(f"TMDB rejected the API key ({response.status_code})")
            if response.status_code == 404:
                raise TmdbNotFound(path)
            if response.status_code not in RETRY_STATUSES:
                raise TmdbError(f"{path} returned {response.status_code}")

            last_error = TmdbError(f"{path} returned {response.status_code}")
            self._backoff(attempt, response=response)

        raise TmdbError(f"{path} failed after {self._max_attempts} attempts") from last_error

    def _backoff(self, attempt: int, response: requests.Response | None) -> None:
        if attempt + 1 >= self._max_attempts:
            return
        self.retry_count += 1
        self._sleep(_backoff_seconds(attempt, response))


def _backoff_seconds(attempt: int, response: requests.Response | None) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                requested = float(retry_after)
            except ValueError:
                requested = None
            if requested is not None:
                return requested if requested < MAX_RETRY_AFTER_SECONDS else MAX_RETRY_AFTER_SECONDS

    doubling: int = 2**attempt
    return BACKOFF_BASE_SECONDS * doubling
