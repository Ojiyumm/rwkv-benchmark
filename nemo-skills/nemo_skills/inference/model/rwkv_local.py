import asyncio
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np

from nemo_skills.inference.model.base import EndpointType
from nemo_skills.inference.model.utils import trim_after_stop_phrases
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))

_ALLOWED_SAMPLING_KEYS = {
    "temperature",
    "max_generate_tokens",
    "top_k",
    "top_p",
    "pad_zero",
    "alpha_presence",
    "alpha_frequency",
    "alpha_decay",
    "stop_tokens",
}


@dataclass(slots=True)
class _PendingGeneration:
    prompt_text: str
    sampling_overrides: Dict[str, Any]
    stop_phrases: List[str]
    random_seed: Optional[int]
    top_logprobs: Optional[int]
    remove_stop_phrases: bool
    future: asyncio.Future


def _ensure_repo_root_on_path() -> None:
    """Ensure the rwkv-benchmark repo root is on sys.path so local_batch_benchmark can be imported."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "local_batch_benchmark").exists():
            repo_root = parent
            break
    else:
        LOG.warning("Could not locate local_batch_benchmark directory relative to %s", current)
        return

    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.append(repo_root_str)


try:
    from local_batch_benchmark.batch_engine import RWKVInferenceEngine, SamplingConfig
except ModuleNotFoundError:
    _ensure_repo_root_on_path()
    from local_batch_benchmark.batch_engine import RWKVInferenceEngine, SamplingConfig


class RWKVLocalModel:
    """Thin async wrapper around the local RWKV batch inference engine used by rwkv-benchmark."""

    MODEL_PROVIDER = "rwkv_local"

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        server_type: Optional[str] = None,
        vocab_path: Optional[str] = None,
        sampling: Optional[Dict[str, Any]] = None,
        sampling_config: Optional[Dict[str, Any]] = None,
        max_workers: int = 1,
        max_batch_size: int = 40,
        batch_collect_timeout_s: float = 0.005,
        vocab_size: int = 65536,
        head_size: int = 64,
        seed: int = 42,
        **_: Any,
    ) -> None:
        if server_type and server_type != self.MODEL_PROVIDER:
            LOG.debug("RWKVLocalModel instantiated with server_type=%s (expected %s).", server_type, self.MODEL_PROVIDER)

        if not model:
            raise ValueError("Parameter `model` must point to the RWKV checkpoint prefix (without .pth extension).")

        sampling_dict = sampling_config or sampling or {}
        sampling_kwargs = self._extract_sampling_kwargs(sampling_dict)

        base_sampling = SamplingConfig(**sampling_kwargs)
        self._base_sampling_kwargs = sampling_kwargs
        self._base_max_tokens = base_sampling.max_generate_tokens

        vocab_loc = vocab_path or tokenizer

        LOG.info("Loading RWKV model from %s (vocab_path=%s)...", model, vocab_loc)
        self.engine = RWKVInferenceEngine(
            model_path=model,
            vocab_path=vocab_loc,
            vocab_size=vocab_size,
            head_size=head_size,
            seed=seed,
            sampling_config=base_sampling,
        )
        LOG.info("RWKV model ready.")

        worker_count = max(1, max_workers)
        self._max_batch_size = max(1, int(max_batch_size))
        self._batch_collect_timeout = max(0.0, float(batch_collect_timeout_s))
        self._executor = ThreadPoolExecutor(max_workers=worker_count)
        self.cfg = SimpleNamespace(max_concurrent_requests=self._max_batch_size)
        self._pending_queue: asyncio.Queue[_PendingGeneration] | None = None
        self._dispatcher_task: asyncio.Task[Any] | None = None
        self._dispatcher_lock: asyncio.Lock | None = None

    def __del__(self) -> None:
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
        try:
            if self._dispatcher_task is not None:
                self._dispatcher_task.cancel()
        except Exception:
            pass

    async def generate_async(
        self,
        prompt: str | List[Dict[str, str]],
        endpoint_type: EndpointType | None = None,
        tokens_to_generate: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = -1,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: Optional[int] = None,
        stop_phrases: Optional[List[str]] = None,
        top_logprobs: Optional[int] = None,
        timeout: float | int | None = 14400,
        remove_stop_phrases: bool = True,
        stream: bool = False,
        reasoning_effort: str | None = None,
        tools: Optional[List[dict]] = None,
        include_response: bool = False,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if stream:
            raise NotImplementedError("Streaming is not supported for RWKVLocalModel.")
        if tools:
            raise NotImplementedError("Tool-augmented generation is not supported for RWKVLocalModel.")
        if reasoning_effort is not None:
            LOG.warning("Ignoring reasoning_effort=%s for RWKVLocalModel.", reasoning_effort)
        if min_p not in (0.0, None):
            LOG.warning("min_p is not supported by RWKVLocalModel; ignoring value %s.", min_p)
        if repetition_penalty != 1.0:
            LOG.warning(
                "repetition_penalty is not supported directly by RWKVLocalModel; ignoring value %s.",
                repetition_penalty,
            )

        prompt_text = self._normalize_prompt(prompt, endpoint_type)
        sampling_overrides = self._collect_sampling_overrides(
            tokens_to_generate=tokens_to_generate,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            extra_body=extra_body,
        )

        loop = asyncio.get_running_loop()
        await self._ensure_async_primitives()

        normalized_seed: Optional[int] = random_seed
        if temperature <= 0 and (random_seed == 0 or random_seed is None):
            # Greedy decoding ignores RNG, so we drop the seed to keep requests batchable.
            normalized_seed = None

        future: asyncio.Future[Dict[str, Any]] = loop.create_future()
        request = _PendingGeneration(
            prompt_text=prompt_text,
            sampling_overrides=dict(sampling_overrides),
            stop_phrases=list(stop_phrases or []),
            random_seed=normalized_seed,
            top_logprobs=top_logprobs,
            remove_stop_phrases=remove_stop_phrases,
            future=future,
        )

        await self._pending_queue.put(request)
        await self._ensure_dispatcher(loop)

        result = await future

        if include_response:
            result["response"] = {
                "sampling_overrides": sampling_overrides,
                "stop_phrases": stop_phrases,
            }

        return result

    async def _ensure_async_primitives(self) -> None:
        if self._pending_queue is None:
            self._pending_queue = asyncio.Queue()
        if self._dispatcher_lock is None:
            self._dispatcher_lock = asyncio.Lock()

    async def _ensure_dispatcher(self, loop: asyncio.AbstractEventLoop) -> None:
        assert self._dispatcher_lock is not None
        async with self._dispatcher_lock:
            if self._dispatcher_task is None or self._dispatcher_task.done():
                self._dispatcher_task = loop.create_task(self._dispatch_loop())

    async def _dispatch_loop(self) -> None:
        assert self._pending_queue is not None
        loop = asyncio.get_running_loop()
        while True:
            try:
                request = await self._pending_queue.get()
            except asyncio.CancelledError:
                return

            batch = [request]
            if self._batch_collect_timeout > 0:
                deadline = loop.time() + self._batch_collect_timeout
                while len(batch) < self._max_batch_size:
                    timeout = max(0.0, deadline - loop.time())
                    if timeout <= 0:
                        break
                    try:
                        batch.append(await asyncio.wait_for(self._pending_queue.get(), timeout=timeout))
                    except asyncio.TimeoutError:
                        break
                    except asyncio.CancelledError:
                        for _ in batch:
                            self._pending_queue.task_done()
                        raise
            else:
                while len(batch) < self._max_batch_size:
                    try:
                        batch.append(self._pending_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

            active_batch = [req for req in batch if not req.future.cancelled()]

            try:
                if active_batch:
                    results = await loop.run_in_executor(self._executor, self._execute_batch_sync, active_batch)
                    for req, res in zip(active_batch, results):
                        if not req.future.done():
                            req.future.set_result(res)
            except Exception as exc:
                for req in active_batch:
                    if not req.future.done():
                        req.future.set_exception(exc)
            finally:
                for _ in batch:
                    self._pending_queue.task_done()

    def _execute_batch_sync(self, batch_requests: List[_PendingGeneration]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = [None] * len(batch_requests)
        signature_groups: Dict[Any, List[int]] = {}

        for idx, req in enumerate(batch_requests):
            if req.random_seed is not None:
                signature_groups[(idx, "seeded")] = [idx]
                continue

            signature = (self._sampling_signature(req.sampling_overrides), req.top_logprobs)
            signature_groups.setdefault(signature, []).append(idx)

        for indices in signature_groups.values():
            if any(batch_requests[i].random_seed is not None for i in indices):
                for i in indices:
                    results[i] = self._generate_single_sync(batch_requests[i])
                continue

            if len(indices) == 1:
                results[indices[0]] = self._generate_single_sync(batch_requests[indices[0]])
                continue

            group_requests = [batch_requests[i] for i in indices]
            group_results = self._generate_group_sync(group_requests)
            for offset, i in enumerate(indices):
                results[i] = group_results[offset]

        for idx, res in enumerate(results):
            if res is None:
                results[idx] = self._generate_single_sync(batch_requests[idx])

        return results

    def _generate_group_sync(self, requests: List[_PendingGeneration]) -> List[Dict[str, Any]]:
        if not requests:
            return []

        sampling_overrides = requests[0].sampling_overrides
        override_config = SamplingConfig(**sampling_overrides)
        prompts = [req.prompt_text for req in requests]

        if requests[0].top_logprobs:
            start_time = time.perf_counter()
            generated = self.engine.generate_with_logprobs(
                prompts,
                max_length=override_config.max_generate_tokens,
                echo=False,
                top_logprobs=requests[0].top_logprobs,
                override_config=override_config,
            )
            generation_time = time.perf_counter() - start_time

            outputs: List[Dict[str, Any]] = []
            for req, res in zip(requests, generated):
                generation = res["text"]
                if req.remove_stop_phrases:
                    generation = trim_after_stop_phrases(generation, list(req.stop_phrases))
                outputs.append(
                    {
                        "generation": generation,
                        "num_generated_tokens": len(res["tokens"]),
                        "logprobs": res["logprobs"],
                        "tokens": res["tokens"],
                        "top_logprobs": res["top_logprobs"],
                        "generation_time": generation_time,
                    }
                )
            return outputs

        tokens, inference_time = self.engine.generate_batch(
            prompts,
            max_length=override_config.max_generate_tokens,
            override_config=override_config,
        )

        outputs = []
        for req, token_seq in zip(requests, tokens):
            trimmed_tokens = self._trim_tokens(np.array(token_seq), override_config.stop_tokens)
            generation = self.engine.tokenizer.decode(trimmed_tokens, utf8_errors="ignore")
            if req.remove_stop_phrases:
                generation = trim_after_stop_phrases(generation, list(req.stop_phrases))
            outputs.append(
                {
                    "generation": generation,
                    "num_generated_tokens": len(trimmed_tokens),
                    "generation_time": inference_time,
                }
            )
        return outputs

    def _generate_single_sync(self, request: _PendingGeneration) -> Dict[str, Any]:
        if request.random_seed is not None:
            self._reset_global_seed(request.random_seed)

        override_config = SamplingConfig(**request.sampling_overrides)

        if request.top_logprobs:
            result = self.engine.generate_with_logprobs(
                [request.prompt_text],
                max_length=override_config.max_generate_tokens,
                echo=False,
                top_logprobs=request.top_logprobs,
                override_config=override_config,
            )[0]
            generation = result["text"]
            num_generated_tokens = len(result["tokens"])
            output = {
                "generation": trim_after_stop_phrases(generation, list(request.stop_phrases))
                if request.remove_stop_phrases
                else generation,
                "num_generated_tokens": num_generated_tokens,
                "logprobs": result["logprobs"],
                "tokens": result["tokens"],
                "top_logprobs": result["top_logprobs"],
            }
            return output

        tokens, inference_time = self.engine.generate_batch(
            [request.prompt_text],
            max_length=override_config.max_generate_tokens,
            override_config=override_config,
        )
        token_seq = self._trim_tokens(tokens[0], override_config.stop_tokens)
        generation = self.engine.tokenizer.decode(token_seq, utf8_errors="ignore")
        if request.remove_stop_phrases:
            generation = trim_after_stop_phrases(generation, list(request.stop_phrases))
        return {
            "generation": generation,
            "num_generated_tokens": len(token_seq),
            "generation_time": inference_time,
        }

    def _sampling_signature(self, sampling_overrides: Dict[str, Any]) -> tuple:
        signature_parts = []
        for key in sorted(sampling_overrides):
            value = sampling_overrides[key]
            if isinstance(value, list):
                norm_value = tuple(value)
            elif isinstance(value, dict):
                norm_value = tuple(sorted(value.items()))
            else:
                norm_value = value
            signature_parts.append((key, norm_value))
        return tuple(signature_parts)

    def _normalize_prompt(self, prompt: str | List[Dict[str, str]], endpoint_type: EndpointType | None) -> str:
        if isinstance(prompt, str):
            return prompt

        if endpoint_type not in (None, EndpointType.chat):
            raise TypeError(f"RWKVLocalModel received unsupported endpoint_type={endpoint_type} for prompt={prompt}.")

        parts = []
        for message in prompt:
            role = message.get("role", "user")
            content = message.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def _collect_sampling_overrides(
        self,
        tokens_to_generate: Optional[int],
        temperature: float,
        top_p: float,
        top_k: int,
        extra_body: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        overrides = dict(self._base_sampling_kwargs)

        if tokens_to_generate is not None:
            overrides["max_generate_tokens"] = tokens_to_generate
        elif "max_generate_tokens" not in overrides:
            overrides["max_generate_tokens"] = self._base_max_tokens

        overrides["temperature"] = temperature
        overrides["top_p"] = top_p
        overrides["top_k"] = max(0, top_k or 0)

        if extra_body:
            for key in _ALLOWED_SAMPLING_KEYS:
                if key in extra_body:
                    overrides[key] = extra_body[key]

        return overrides

    def _trim_tokens(self, token_seq: np.ndarray, stop_tokens: Optional[List[int]]) -> List[int]:
        trimmed: List[int] = []
        stop_tokens_set = set(stop_tokens or [])
        for token in token_seq.tolist():
            if token in stop_tokens_set or token == 0:
                break
            trimmed.append(int(token))
        return trimmed

    def _extract_sampling_kwargs(self, sampling_dict: Dict[str, Any]) -> Dict[str, Any]:
        kwargs = {}
        for key in _ALLOWED_SAMPLING_KEYS:
            if key in sampling_dict and sampling_dict[key] is not None:
                kwargs[key] = sampling_dict[key]
        return kwargs

    def _reset_global_seed(self, seed: int) -> None:
        import random

        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
