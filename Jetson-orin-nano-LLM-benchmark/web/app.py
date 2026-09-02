#!/usr/bin/env python3
"""Lightweight web chat server for local GGUF models on Jetson."""

from __future__ import annotations

import gc
import json
import os
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

try:
    from llama_cpp import Llama
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "llama-cpp-python is required. Install dependencies from requirements.txt first."
    ) from exc

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODELS_DIR = Path(os.getenv("LLM_MODELS_DIR", PROJECT_ROOT / "models")).resolve()

DEFAULT_N_CTX = int(os.getenv("LLM_N_CTX", "1024"))
DEFAULT_N_GPU_LAYERS = int(os.getenv("LLM_N_GPU_LAYERS", "8"))
DEFAULT_N_THREADS = int(os.getenv("LLM_N_THREADS", str(min(6, os.cpu_count() or 1))))
DEFAULT_N_BATCH = int(os.getenv("LLM_N_BATCH", "256"))
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "256"))


class SelectModelRequest(BaseModel):
    model_id: str = Field(..., description="Relative model path from models directory")
    n_ctx: int = Field(DEFAULT_N_CTX, ge=128, le=4096)
    n_gpu_layers: int = Field(DEFAULT_N_GPU_LAYERS, ge=0, le=120)


class ChatStreamRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: List[Dict[str, str]] = Field(default_factory=list)
    max_tokens: int = Field(DEFAULT_MAX_TOKENS, ge=1, le=1024)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)


def as_sse(event_type: str, payload: Dict[str, object]) -> str:
    data = json.dumps({"type": event_type, **payload}, ensure_ascii=False)
    return f"data: {data}\n\n"


class ModelManager:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.lock = threading.RLock()
        self.llm: Optional[Llama] = None
        self.loaded_model_path: Optional[Path] = None
        self.loaded_n_ctx: Optional[int] = None
        self.loaded_n_gpu_layers: Optional[int] = None
        self.loaded_n_batch: Optional[int] = None
        self.requested_n_ctx: Optional[int] = None
        self.requested_n_gpu_layers: Optional[int] = None

    def list_models(self) -> List[Dict[str, object]]:
        if not self.models_dir.exists():
            return []

        models = []
        for path in sorted(self.models_dir.rglob("*.gguf")):
            try:
                rel_path = path.resolve().relative_to(self.models_dir)
            except ValueError:
                continue

            models.append(
                {
                    "id": rel_path.as_posix(),
                    "name": path.name,
                    "path": str(path.resolve()),
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                    "is_loaded": self.loaded_model_path is not None
                    and path.resolve() == self.loaded_model_path,
                }
            )
        return models

    def _resolve_model_path(self, model_id: str) -> Path:
        candidate = (self.models_dir / model_id).resolve()
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(f"Model file not found: {model_id}")
        if candidate.suffix.lower() != ".gguf":
            raise ValueError("Selected file is not a .gguf model")

        try:
            candidate.relative_to(self.models_dir)
        except ValueError as exc:
            raise ValueError("Model path is outside models directory") from exc

        return candidate

    def _release_gpu_caches(self) -> None:
        gc.collect()
        if torch is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _unload_locked(self) -> None:
        if self.llm is not None:
            old_llm = self.llm
            self.llm = None
            del old_llm

        self.loaded_model_path = None
        self.loaded_n_ctx = None
        self.loaded_n_gpu_layers = None
        self.loaded_n_batch = None
        self.requested_n_ctx = None
        self.requested_n_gpu_layers = None
        self._release_gpu_caches()

    def unload(self) -> None:
        with self.lock:
            self._unload_locked()

    def _build_gpu_layers_schedule(self, requested_layers: int) -> List[int]:
        schedule = [max(0, requested_layers)]
        current = schedule[0]
        while current > 0:
            current = max(0, current - 4)
            if current not in schedule:
                schedule.append(current)
        return schedule

    def _build_n_ctx_schedule(self, requested_ctx: int) -> List[int]:
        candidates = [requested_ctx, 768, 512, 384, 256, 192, 128]
        schedule: List[int] = []
        for value in candidates:
            if value <= requested_ctx and value >= 128 and value not in schedule:
                schedule.append(value)
        if not schedule:
            schedule = [max(128, requested_ctx)]
        return schedule

    def _build_n_batch_schedule(self, n_ctx_value: int) -> List[int]:
        start = max(32, min(DEFAULT_N_BATCH, n_ctx_value))
        candidates = [start, 128, 96, 64, 48, 32]
        schedule: List[int] = []
        for value in candidates:
            if value <= n_ctx_value and value >= 32 and value not in schedule:
                schedule.append(value)
        if not schedule:
            schedule = [max(32, min(64, n_ctx_value))]
        return schedule

    def load(self, model_id: str, n_ctx: int, n_gpu_layers: int) -> Dict[str, object]:
        with self.lock:
            model_path = self._resolve_model_path(model_id)

            same_model = (
                self.llm is not None
                and self.loaded_model_path == model_path
                and self.requested_n_ctx == n_ctx
                and self.requested_n_gpu_layers == n_gpu_layers
            )
            if same_model:
                return self.current_model_info()

            self._unload_locked()

            last_error: Optional[Exception] = None
            selected_n_gpu_layers: Optional[int] = None
            selected_n_ctx: Optional[int] = None
            selected_n_batch: Optional[int] = None
            n_ctx_schedule = self._build_n_ctx_schedule(n_ctx)
            gpu_schedule = self._build_gpu_layers_schedule(n_gpu_layers)

            for candidate_n_ctx in n_ctx_schedule:
                batch_schedule = self._build_n_batch_schedule(candidate_n_ctx)
                for candidate_layers in gpu_schedule:
                    for candidate_n_batch in batch_schedule:
                        try:
                            self.llm = Llama(
                                model_path=str(model_path),
                                n_ctx=candidate_n_ctx,
                                n_gpu_layers=candidate_layers,
                                n_batch=candidate_n_batch,
                                n_threads=DEFAULT_N_THREADS,
                                verbose=False,
                            )
                            selected_n_ctx = candidate_n_ctx
                            selected_n_gpu_layers = candidate_layers
                            selected_n_batch = candidate_n_batch
                            break
                        except Exception as exc:
                            last_error = exc
                            self._unload_locked()
                    if self.llm is not None:
                        break
                if self.llm is not None:
                    break

            if (
                self.llm is None
                or selected_n_gpu_layers is None
                or selected_n_ctx is None
                or selected_n_batch is None
            ):
                detail = f"Failed to load model after retries: {last_error}"
                raise ValueError(detail)

            self.loaded_model_path = model_path
            self.loaded_n_ctx = selected_n_ctx
            self.loaded_n_gpu_layers = selected_n_gpu_layers
            self.loaded_n_batch = selected_n_batch
            self.requested_n_ctx = n_ctx
            self.requested_n_gpu_layers = n_gpu_layers

            info = self.current_model_info()
            info["requested_n_gpu_layers"] = n_gpu_layers
            info["auto_adjusted_n_gpu_layers"] = selected_n_gpu_layers != n_gpu_layers
            info["requested_n_ctx"] = n_ctx
            info["auto_adjusted_n_ctx"] = selected_n_ctx != n_ctx
            info["n_batch"] = selected_n_batch
            return info

    def current_model_info(self) -> Dict[str, object]:
        if self.loaded_model_path is None:
            return {
                "loaded": False,
                "model_id": None,
                "model_path": None,
                "n_ctx": None,
                "n_gpu_layers": None,
                "n_batch": None,
            }

        return {
            "loaded": True,
            "model_id": self.loaded_model_path.relative_to(self.models_dir).as_posix(),
            "model_path": str(self.loaded_model_path),
            "n_ctx": self.loaded_n_ctx,
            "n_gpu_layers": self.loaded_n_gpu_layers,
            "n_batch": self.loaded_n_batch,
            "requested_n_ctx": self.requested_n_ctx,
            "requested_n_gpu_layers": self.requested_n_gpu_layers,
        }

    def _sanitize_history(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        cleaned: List[Dict[str, str]] = []
        for item in history:
            role = item.get("role", "")
            content = item.get("content", "")
            if role not in {"user", "assistant", "system"}:
                continue
            if not isinstance(content, str) or not content.strip():
                continue
            cleaned.append({"role": role, "content": content.strip()})
        return cleaned

    def _fallback_prompt(self, messages: List[Dict[str, str]]) -> str:
        chunks = []
        for msg in messages:
            role = msg["role"].upper()
            chunks.append(f"{role}: {msg['content']}")
        chunks.append("ASSISTANT:")
        return "\n".join(chunks)

    def _extract_token(self, chunk: Dict[str, object]) -> str:
        choices = chunk.get("choices", [{}])
        if not choices:
            return ""
        choice = choices[0]
        if not isinstance(choice, dict):
            return ""

        delta = choice.get("delta")
        if isinstance(delta, dict):
            token = delta.get("content") or delta.get("text") or ""
            return token if isinstance(token, str) else ""

        token = choice.get("text", "")
        return token if isinstance(token, str) else ""

    def stream_chat(
        self,
        message: str,
        history: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Iterable[str]:
        with self.lock:
            if self.llm is None:
                raise RuntimeError("No model loaded. Select a model first.")

            messages = self._sanitize_history(history)
            messages.append({"role": "user", "content": message.strip()})

            try:
                stream = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True,
                )
                for chunk in stream:
                    token = self._extract_token(chunk)
                    if token:
                        yield token
                return
            except Exception:
                # Fallback for models without usable chat templates.
                pass

            prompt = self._fallback_prompt(messages)
            completion_stream = self.llm.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
            )
            for chunk in completion_stream:
                token = self._extract_token(chunk)
                if token:
                    yield token


app = FastAPI(title="Jetson Local LLM Chat", version="1.0.0")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
manager = ModelManager(MODELS_DIR)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "default_n_ctx": DEFAULT_N_CTX,
            "default_n_gpu_layers": DEFAULT_N_GPU_LAYERS,
        },
    )


@app.get("/api/models")
async def get_models() -> Dict[str, object]:
    return {
        "models": manager.list_models(),
        "current": manager.current_model_info(),
        "models_dir": str(MODELS_DIR),
    }


@app.post("/api/select-model")
async def select_model(payload: SelectModelRequest) -> Dict[str, object]:
    try:
        info = manager.load(
            model_id=payload.model_id,
            n_ctx=payload.n_ctx,
            n_gpu_layers=payload.n_gpu_layers,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}") from exc

    return {"ok": True, "current": info}


@app.post("/api/unload-model")
async def unload_model() -> Dict[str, object]:
    manager.unload()
    return {"ok": True, "current": manager.current_model_info()}


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatStreamRequest) -> StreamingResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is empty")

    def event_generator() -> Iterable[str]:
        try:
            for token in manager.stream_chat(
                message=message,
                history=payload.history,
                max_tokens=payload.max_tokens,
                temperature=payload.temperature,
                top_p=payload.top_p,
            ):
                yield as_sse("token", {"text": token})

            yield as_sse("done", {})
        except Exception as exc:
            yield as_sse("error", {"message": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health() -> Dict[str, object]:
    return {"ok": True, "current": manager.current_model_info()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "web.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
