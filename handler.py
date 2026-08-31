from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import threading
import time
from pathlib import Path
from typing import Any


MODEL_NAME = "nanochat-d24-sft-000466"
MODEL_TAG = os.getenv("NANOCHAT_MODEL_TAG", "d24")
MODEL_STEP = int(os.getenv("NANOCHAT_MODEL_STEP", "466"))
EXPECTED_ARCHIVE_SHA256 = os.getenv(
    "NANOCHAT_ARCHIVE_SHA256",
    "c022d37dab53e03cecc1dcd172af8cccbf02c6d20586f4dc03c47fef8f5637df",
)

VOLUME_ROOT = Path(os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume"))
ARCHIVE_PATH = VOLUME_ROOT / os.getenv(
    "NANOCHAT_ARCHIVE_NAME", "nanochat-d24-sft-inference.tar"
)
MODEL_ROOT = VOLUME_ROOT / os.getenv("NANOCHAT_MODEL_DIR", "nanochat_artifacts")
READY_MARKER = MODEL_ROOT / f".ready-{EXPECTED_ARCHIVE_SHA256}"

DEFAULT_MAX_NEW_TOKENS = 160
MAX_NEW_TOKENS = 256
MAX_INPUT_CHARS = 8_000


class RequestError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_tar_members(archive: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if target != destination and destination not in target.parents:
            raise RuntimeError(f"Unsafe path in model archive: {member.name}")
        if member.issym() or member.islnk() or member.isdev():
            raise RuntimeError(f"Unsupported archive member: {member.name}")


def _required_model_files(root: Path) -> list[Path]:
    return [
        root / "tokenizer" / "tokenizer.pkl",
        root / "tokenizer" / "token_bytes.pt",
        root / "chatsft_checkpoints" / MODEL_TAG / f"model_{MODEL_STEP:06d}.pt",
        root / "chatsft_checkpoints" / MODEL_TAG / f"meta_{MODEL_STEP:06d}.json",
    ]


def _layout_is_complete(root: Path) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in _required_model_files(root))


def ensure_model_extracted() -> None:
    """Verify and extract the inference archive exactly once on the Network Volume."""
    if READY_MARKER.is_file() and _layout_is_complete(MODEL_ROOT):
        return
    if not ARCHIVE_PATH.is_file():
        raise FileNotFoundError(
            f"Model archive not found at {ARCHIVE_PATH}. Attach Network Volume 99hnz378fr."
        )

    # Multiple cold-starting workers can share a Network Volume. The file lock makes
    # sure only one of them performs the one-time extraction.
    import fcntl

    lock_path = VOLUME_ROOT / ".nanochat-d24-extract.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if READY_MARKER.is_file() and _layout_is_complete(MODEL_ROOT):
            return

        actual_sha256 = _sha256(ARCHIVE_PATH)
        if actual_sha256 != EXPECTED_ARCHIVE_SHA256:
            raise RuntimeError(
                "Model archive checksum mismatch: "
                f"expected {EXPECTED_ARCHIVE_SHA256}, got {actual_sha256}"
            )

        temporary_root = VOLUME_ROOT / f".nanochat_artifacts.extracting-{os.getpid()}"
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        temporary_root.mkdir(parents=True)

        try:
            with tarfile.open(ARCHIVE_PATH, mode="r:") as archive:
                _validate_tar_members(archive, temporary_root)
                archive.extractall(temporary_root)

            if not _layout_is_complete(temporary_root):
                raise RuntimeError("Extracted model archive is incomplete")

            if MODEL_ROOT.exists():
                shutil.rmtree(MODEL_ROOT)
            temporary_root.replace(MODEL_ROOT)
            READY_MARKER.write_text(EXPECTED_ARCHIVE_SHA256 + "\n", encoding="ascii")
        except Exception:
            if temporary_root.exists():
                shutil.rmtree(temporary_root)
            raise


def _as_int(value: Any, name: str, default: int, minimum: int, maximum: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise RequestError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RequestError(f"{name} must be an integer") from exc
    if not minimum <= parsed <= maximum:
        raise RequestError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _as_float(value: Any, name: str, default: float, minimum: float, maximum: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        raise RequestError(f"{name} must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RequestError(f"{name} must be a number") from exc
    if not minimum <= parsed <= maximum:
        raise RequestError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _normalize_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    if isinstance(payload.get("prompt"), str):
        messages: Any = [{"role": "user", "content": payload["prompt"]}]
    else:
        messages = payload.get("messages")

    if not isinstance(messages, list) or not messages:
        raise RequestError("Provide a non-empty prompt or messages list")

    normalized: list[dict[str, str]] = []
    expected_role = "user"
    total_chars = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise RequestError(f"messages[{index}] must be an object")
        role = message.get("role")
        content = message.get("content")
        if role != expected_role:
            raise RequestError(
                "Messages must alternate user/assistant, begin with user, and end with user"
            )
        if not isinstance(content, str) or not content.strip():
            raise RequestError(f"messages[{index}].content must be non-empty text")
        content = content.strip()
        total_chars += len(content)
        normalized.append({"role": role, "content": content})
        expected_role = "assistant" if expected_role == "user" else "user"

    if normalized[-1]["role"] != "user":
        raise RequestError("The final message must have role=user")
    if total_chars > MAX_INPUT_CHARS:
        raise RequestError(f"Conversation is too long; maximum is {MAX_INPUT_CHARS} characters")
    return normalized


class NanochatRuntime:
    def __init__(self) -> None:
        ensure_model_extracted()
        os.environ["NANOCHAT_BASE_DIR"] = str(MODEL_ROOT)
        os.environ.setdefault("NANOCHAT_DTYPE", "bfloat16")

        import torch
        from nanochat.checkpoint_manager import load_model
        from nanochat.engine import Engine

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for this endpoint")

        self.torch = torch
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        self.model, self.tokenizer, self.metadata = load_model(
            "sft",
            self.device,
            phase="eval",
            model_tag=MODEL_TAG,
            step=MODEL_STEP,
        )
        self.engine = Engine(self.model, self.tokenizer)
        self.context_length = int(self.model.config.sequence_len)

        self.bos = self.tokenizer.get_bos_token_id()
        self.user_start = self.tokenizer.encode_special("<|user_start|>")
        self.user_end = self.tokenizer.encode_special("<|user_end|>")
        self.assistant_start = self.tokenizer.encode_special("<|assistant_start|>")
        self.assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        self.generate_lock = threading.Lock()

    def _tokenize_messages(self, messages: list[dict[str, str]]) -> list[int]:
        tokens = [self.bos]
        for message in messages:
            if message["role"] == "user":
                tokens.append(self.user_start)
                tokens.extend(self.tokenizer.encode(message["content"]))
                tokens.append(self.user_end)
            else:
                tokens.append(self.assistant_start)
                tokens.extend(self.tokenizer.encode(message["content"]))
                tokens.append(self.assistant_end)
        tokens.append(self.assistant_start)
        return tokens

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = _normalize_messages(payload)
        max_new_tokens = _as_int(
            payload.get("max_new_tokens"),
            "max_new_tokens",
            DEFAULT_MAX_NEW_TOKENS,
            1,
            MAX_NEW_TOKENS,
        )
        temperature = _as_float(payload.get("temperature"), "temperature", 0.6, 0.0, 2.0)
        top_k = _as_int(payload.get("top_k"), "top_k", 50, 1, 200)
        seed = _as_int(payload.get("seed"), "seed", 42, 0, 2_147_483_647)

        prompt_tokens = self._tokenize_messages(messages)
        if len(prompt_tokens) + max_new_tokens > self.context_length:
            available = max(0, self.context_length - len(prompt_tokens))
            raise RequestError(
                f"Conversation uses {len(prompt_tokens)} tokens; only {available} output tokens remain"
            )

        generated_tokens: list[int] = []
        started = time.perf_counter()
        with self.generate_lock:
            for token_column, _token_masks in self.engine.generate(
                prompt_tokens,
                num_samples=1,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=seed,
            ):
                token = token_column[0]
                if token in (self.assistant_end, self.bos):
                    break
                generated_tokens.append(token)

        text = self.tokenizer.decode(generated_tokens).strip()
        return {
            "text": text,
            "model": MODEL_NAME,
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(generated_tokens),
            },
            "generation_seconds": round(time.perf_counter() - started, 3),
        }


_runtime: NanochatRuntime | None = None
_runtime_init_lock = threading.Lock()


def get_runtime() -> NanochatRuntime:
    global _runtime
    if _runtime is None:
        with _runtime_init_lock:
            if _runtime is None:
                _runtime = NanochatRuntime()
    return _runtime


def handler(event: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = event.get("input", {})
        if not isinstance(payload, dict):
            raise RequestError("input must be a JSON object")
        return get_runtime().generate(payload)
    except RequestError as exc:
        return {"error": str(exc), "type": "validation_error"}
    except Exception as exc:
        print(f"Worker error: {type(exc).__name__}: {exc}", flush=True)
        return {"error": "The model worker failed to process this request", "type": "worker_error"}


if __name__ == "__main__":
    # Load and validate the model before the worker begins accepting jobs.
    get_runtime()
    import runpod

    runpod.serverless.start({"handler": handler})
