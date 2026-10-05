"""A Toponymy LLM wrapper that names regions with Claude through the Anthropic SDK.

Toponymy's own AnthropicNamer goes through LiteLLM and always sends `temperature` with
a small `max_tokens` (128 for a topic name). Claude Opus 5.5 rejects sampling parameters
and always thinks, which needs headroom beyond the visible answer, so this wrapper drops
the temperature it is handed and sets its own ceiling.

Toponymy drives async wrappers with a fresh `asyncio.run()` per stage, and an async
client or semaphore binds to the loop it was first used on. So the semaphore is rebuilt
whenever the running loop changes, and each request opens and closes its own client: a
client left open when its loop ends raises "Event loop is closed" on cleanup.
Successful responses are cached on disk by prompt, so re-running a fit never pays twice
for the same prompt.

With `dry_run=True` no request is sent: each prompt is recorded and answered with a
placeholder name, which is enough to drive a fit through and measure what the real one
would send.
"""

import asyncio
import hashlib
import sqlite3
from pathlib import Path

import anthropic
from config import DATA_DIR
from toponymy.llm_wrappers import AsyncLLMWrapper

MODEL = "claude-opus-5-5"
EFFORT = "medium"
CACHE_PATH = DATA_DIR / "llm_cache.sqlite"
# Dollars per million tokens.
PRICE_INPUT, PRICE_OUTPUT = 4.00, 20.00

PLACEHOLDER = '{"topic_name": "Placeholder region name", "topic_specificity": 0.5}'


class ClaudeNamer(AsyncLLMWrapper):
    # Errors an identical retry can't fix; Toponymy's base class fails fast on these.
    FAIL_FAST_EXCEPTIONS = (
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
        anthropic.BadRequestError,
        anthropic.NotFoundError,
    )

    def __init__(
        self,
        model: str = MODEL,
        effort: str = EFFORT,
        max_concurrent_requests: int = 8,
        max_tokens: int = 16000,
        cache_path: Path = CACHE_PATH,
        dry_run: bool = False,
    ):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.max_concurrent_requests = max_concurrent_requests
        self.dry_run = dry_run
        self.extra_prompting = ""
        self.callback = None
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0, "cached": 0}
        self.recorded: list[tuple[str | None, str]] = []
        self._loop = None
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache_path)
        self.db.execute("CREATE TABLE IF NOT EXISTS llm (key TEXT PRIMARY KEY, text TEXT)")

    def _semaphore_for_loop(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if loop is not self._loop:
            self._loop = loop
            self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        return self._semaphore

    def _key(self, system: str | None, user: str) -> str:
        raw = "\x00".join([self.model, self.effort, system or "", user])
        return hashlib.sha256(raw.encode()).hexdigest()

    async def create(self, system: str | None, user: str) -> str:
        if self.dry_run:
            self.recorded.append((system, user))
            return PLACEHOLDER

        key = self._key(system, user)
        row = self.db.execute("SELECT text FROM llm WHERE key = ?", (key,)).fetchone()
        if row is not None:
            self.usage["cached"] += 1
            return row[0]

        kwargs = {"system": system} if system else {}
        async with self._semaphore_for_loop(), anthropic.AsyncAnthropic(max_retries=4) as client:
            message = await client.beta.messages.create(
                model=self.model,
                # Toponymy's per-call max_tokens sizes only the visible answer; it would
                # cut thinking short, so the wrapper sets its own ceiling.
                max_tokens=self.max_tokens,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                # If a safety classifier declines a prompt, re-run it on a fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[{"role": "user", "content": user}],
                **kwargs,
            )
        self.usage["calls"] += 1
        self.usage["input_tokens"] += message.usage.input_tokens
        self.usage["output_tokens"] += message.usage.output_tokens
        if message.stop_reason == "refusal":
            raise RuntimeError(f"refusal: {message.stop_details}")
        text = next((block.text for block in message.content if block.type == "text"), None)
        if text is None:
            raise RuntimeError(f"no text block (stop_reason={message.stop_reason})")
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO llm VALUES (?, ?)", (key, text))
        return text

    def cost(self) -> float:
        usage = self.usage
        return (usage["input_tokens"] * PRICE_INPUT + usage["output_tokens"] * PRICE_OUTPUT) / 1e6

    async def _call_single_llm(self, prompt, temperature, max_tokens) -> str:
        return await self.create(None, prompt["combined"])

    async def _call_single_llm_with_system(self, prompt, temperature, max_tokens) -> str:
        return await self.create(prompt["system"], prompt["user"])
