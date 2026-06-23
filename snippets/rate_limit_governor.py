"""
Adaptive rate-limit governor for the scraper fleet.

A plain token bucket caps request rate. The twist here is the *backoff factor*:
when a worker reports a ban/soft-block signal (`on_ban`), the governor halves its
effective refill rate; clean responses (`on_success`) nudge it back up by 5%. So
the fleet self-tunes toward the fastest rate a defended site will tolerate,
instead of a hand-guessed constant — and it backs off automatically the moment a
site starts pushing back, which is what keeps long scrapes from getting blocked.

Excerpt from the market-intelligence pipeline. See docs/dropship-intel.md.
"""
import asyncio
import time
import logging

logger = logging.getLogger(__name__)


class RateLimitGovernor:
    def __init__(self, requests_per_hour: int = 60):
        self.capacity = requests_per_hour
        self.tokens = float(requests_per_hour)
        self.refill_rate = requests_per_hour / 3600.0   # tokens per second
        self.last_refill = time.monotonic()
        self._backoff_factor = 1.0                       # 1.0 = full speed

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_refill
        # Backoff scales the *refill* rate, so a ban slows recovery, not just spend
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate * self._backoff_factor,
        )
        self.last_refill = now

    async def acquire(self):
        """Block until a request token is available, then consume one."""
        self._refill()
        if self.tokens >= 1:
            self.tokens -= 1
            return
        wait = (1 - self.tokens) / (self.refill_rate * self._backoff_factor)
        wait = min(wait, 60)   # cap a single sleep so workers stay responsive
        logger.debug("Rate limit: sleeping %.1fs", wait)
        await asyncio.sleep(wait)
        self.tokens -= 1

    def on_ban(self):
        # Multiplicative decrease — react hard to a block signal
        self._backoff_factor = max(0.1, self._backoff_factor / 2)
        logger.warning(
            "Ban detected — throttling to %.0f%% of normal rate",
            self._backoff_factor * 100,
        )

    def on_success(self):
        # Additive increase — recover gently toward full speed (AIMD)
        self._backoff_factor = min(1.0, self._backoff_factor * 1.05)
