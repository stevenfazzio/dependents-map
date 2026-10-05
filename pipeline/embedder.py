"""Qwen3-Embedding-8B, served by vLLM on a RunPod serverless endpoint, behind a disk cache.

`Embedder.encode(texts, show_progress_bar)` satisfies Toponymy's TextEmbedderProtocol, so
the object that embeds the documents also embeds Toponymy's keyphrases and topic names
later. Every vector is cached on disk by a hash of its text: re-running a stage never
re-embeds a string, and the endpoint only wakes for text it has not seen.

Texts are cut to MAX_TEXT_TOKENS here, with the model's own tokenizer, rather than by
the server. That makes the cache key the exact text that was embedded, and keeps the
start of a long README instead of leaving the choice of which end to drop to the server.

The endpoint is a RunPod load-balancer endpoint, which scales to zero. To recreate it:
    image  vllm/vllm-openai:v0.28.0
    args   --model Qwen/Qwen3-Embedding-8B --runner pooling --convert embed
           --max-model-len 8192 --port 8000
    env    PORT=8000, PORT_HEALTH=8000        ports  8000/http        disk  60 GB
    gpu    pool AMPERE_48 (A40 / A6000), minCudaVersion 13.0
    workers 0-1, idle timeout 600 s, scaling REQUEST_COUNT 8
then set ENDPOINT_ID below. A load balancer has no queue, so the first request after
idle fails until a worker is up; `wait_until_ready` absorbs that.

Both endpoints that built the umap-learn map were deleted once its region names were
final: the first (o9hikcc2pzjwcn) on 2026-10-04, and the one named below, made for the
rebuild after stage 06 began filtering the corpus, on 2026-10-05. Text already embedded
is served from the disk cache; embedding anything new needs a new endpoint.
"""

import hashlib
import os
import random
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from config import DATA_DIR
from tokenizers import Tokenizer

ENDPOINT_ID = "6jxxlypeo4pzj0"
BASE_URL = f"https://{ENDPOINT_ID}.api.runpod.ai"
HEALTH_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/health"
MODEL = "Qwen/Qwen3-Embedding-8B"
CACHE_PATH = DATA_DIR / "embedding_cache.sqlite"

# The server accepts 8192 tokens including the special tokens it appends.
MAX_TEXT_TOKENS = 8000
# One request carries at most this many texts and roughly this many tokens.
BATCH_MAX_TEXTS = 128
BATCH_MAX_TOKENS = 48_000
CONCURRENT_REQUESTS = 4
MAX_RETRIES = 6
# 400 is how the load balancer reports "timed out waiting for a worker".
RETRY_STATUS = {400, 429, 500, 502, 503, 504}
PROGRESS_EVERY_S = 30

_thread = threading.local()


class Embedder:
    def __init__(self, cache_path: Path = CACHE_PATH):
        self.api_key = os.environ["RUNPOD_API_KEY"]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache_path)
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")
        self._tokenizer = None
        self._ready = False
        self._ready_lock = threading.Lock()

    # ── text preparation ────────────────────────────────────────────────────────────

    def truncate(self, texts: list[str]) -> tuple[list[str], list[int]]:
        """Cut each text to MAX_TEXT_TOKENS. Returns the texts and their token counts."""
        if self._tokenizer is None:
            self._tokenizer = Tokenizer.from_pretrained(MODEL)
        out, counts = [], []
        for start in range(0, len(texts), 1000):
            chunk = texts[start : start + 1000]
            encodings = self._tokenizer.encode_batch(chunk, add_special_tokens=False)
            for text, encoding in zip(chunk, encodings, strict=True):
                n = len(encoding.ids)
                if n > MAX_TEXT_TOKENS:
                    text = text[: encoding.offsets[MAX_TEXT_TOKENS - 1][1]]
                    n = MAX_TEXT_TOKENS
                out.append(text)
                counts.append(n)
        return out, counts

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(f"{MODEL}\x00{text}".encode()).hexdigest()

    # ── endpoint ────────────────────────────────────────────────────────────────────

    def _session(self) -> requests.Session:
        if not hasattr(_thread, "session"):
            _thread.session = requests.Session()
            _thread.session.headers["Authorization"] = f"Bearer {self.api_key}"
        return _thread.session

    def wait_until_ready(self, max_wait_s: int = 1800) -> None:
        with self._ready_lock:
            if self._ready:
                return
            start = time.time()
            while time.time() - start < max_wait_s:
                try:
                    # A request to the load balancer is what triggers a scale-up from zero.
                    status = self._session().get(f"{BASE_URL}/ping", timeout=30).status_code
                except requests.RequestException as e:
                    status = type(e).__name__
                if status == 200:
                    print(f"  endpoint ready after {time.time() - start:.0f}s", flush=True)
                    self._ready = True
                    return
                try:
                    workers = self._session().get(HEALTH_URL, timeout=30).json().get("workers")
                except (requests.RequestException, ValueError):
                    workers = "?"
                waited = time.time() - start
                print(
                    f"  waiting for a worker ({waited:.0f}s): ping={status} workers={workers}",
                    flush=True,
                )
                time.sleep(15)
            raise TimeoutError(f"endpoint not ready after {max_wait_s}s")

    def _embed_remote(self, texts: list[str]) -> np.ndarray:
        reason = "unknown"
        for attempt in range(MAX_RETRIES):
            self.wait_until_ready()
            try:
                resp = self._session().post(
                    f"{BASE_URL}/v1/embeddings",
                    json={"model": MODEL, "input": texts, "encoding_format": "float"},
                    timeout=600,
                )
            except (requests.Timeout, requests.ConnectionError) as e:
                reason = type(e).__name__
            else:
                if resp.status_code == 200:
                    data = sorted(resp.json()["data"], key=lambda d: d["index"])
                    return np.asarray([d["embedding"] for d in data], dtype=np.float32)
                reason = f"HTTP {resp.status_code}: {resp.text[:200]!r}"
                if resp.status_code not in RETRY_STATUS or "context length" in resp.text:
                    raise RuntimeError(f"embedding request rejected: {reason}")
                self._ready = False
            if attempt < MAX_RETRIES - 1:
                wait = min(2**attempt * 10, 120) + random.uniform(0, 5)
                print(f"  {reason}; retrying in {wait:.0f}s", flush=True)
                time.sleep(wait)
        raise RuntimeError(f"embedding request failed after {MAX_RETRIES} attempts: {reason}")

    # ── cache ───────────────────────────────────────────────────────────────────────

    def _cached(self, keys: list[str]) -> dict[str, np.ndarray]:
        found = {}
        for i in range(0, len(keys), 900):  # stay under SQLite's variable limit
            chunk = keys[i : i + 900]
            marks = ",".join("?" * len(chunk))
            rows = self.db.execute(f"SELECT key, vec FROM emb WHERE key IN ({marks})", chunk)
            found.update({key: np.frombuffer(vec, dtype=np.float32) for key, vec in rows})
        return found

    # ── public ──────────────────────────────────────────────────────────────────────

    def encode(self, texts, show_progress_bar=None, verbose=None, **kwargs) -> np.ndarray:
        """Embed texts, in order. Vectors are L2-normalised by the model's pooling config."""
        texts, counts = self.truncate([str(t) for t in texts])
        keys = [self._key(t) for t in texts]
        vectors = self._cached(list(set(keys)))

        # Each distinct uncached text once, longest first: memory trouble shows up at
        # the start of a run, and the time estimate only ever improves.
        todo = {}
        for text, key, n_tokens in zip(texts, keys, counts, strict=True):
            if key not in vectors:
                todo[key] = (text, n_tokens)
        ordered = sorted(todo.items(), key=lambda item: -item[1][1])

        batches, batch, batch_tokens = [], [], 0
        for key, (text, n_tokens) in ordered:
            full = len(batch) >= BATCH_MAX_TEXTS or batch_tokens + n_tokens > BATCH_MAX_TOKENS
            if batch and full:
                batches.append(batch)
                batch, batch_tokens = [], 0
            batch.append((key, text, n_tokens))
            batch_tokens += n_tokens
        if batch:
            batches.append(batch)

        loud = bool(show_progress_bar or verbose)
        total_tokens = sum(n for _, n in todo.values())
        if loud:
            print(
                f"embedding: {len(texts)} texts, {len(texts) - len(todo)} cached or repeated, "
                f"{len(todo)} to embed ({total_tokens:,} tokens in {len(batches)} requests)",
                flush=True,
            )
        if not batches:
            return np.vstack([vectors[k] for k in keys])

        self.wait_until_ready()
        start = last_report = time.time()
        done_texts = done_tokens = 0
        with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as pool:
            futures = {
                pool.submit(self._embed_remote, [text for _, text, _ in batch]): batch
                for batch in batches
            }
            for future in as_completed(futures):
                batch = futures[future]
                batch_vectors = future.result()
                rows = [
                    (key, vec.tobytes())
                    for (key, _, _), vec in zip(batch, batch_vectors, strict=True)
                ]
                with self.db:  # committed per request, so a crash loses at most one
                    self.db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)", rows)
                vectors.update(
                    {key: vec for (key, _, _), vec in zip(batch, batch_vectors, strict=True)}
                )
                done_texts += len(batch)
                done_tokens += sum(n for _, _, n in batch)

                now = time.time()
                if loud and (now - last_report >= PROGRESS_EVERY_S or done_texts == len(todo)):
                    last_report = now
                    rate = done_tokens / (now - start)
                    left_min = (total_tokens - done_tokens) / rate / 60
                    print(
                        f"  {done_texts:,}/{len(todo):,} texts, "
                        f"{done_tokens / 1e6:.2f}M/{total_tokens / 1e6:.2f}M tokens, "
                        f"{rate:,.0f} tokens/s, about {left_min:.0f} min left",
                        flush=True,
                    )
        return np.vstack([vectors[k] for k in keys])
