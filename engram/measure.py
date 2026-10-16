"""Measure what the experiment plan assumes (issue #12): serving throughput and sleep cost.

    engram-measure serve --model Qwen/Qwen3-8B --base-url http://localhost:8000/v1 --concurrency 8
    engram-measure sleep --base-model Qwen/Qwen3-8B --messages 150

``serve`` fires concurrent chat requests at an OpenAI-compatible endpoint and reports requests/s
and output tokens/s. ``sleep`` times one B4 consolidation (LoRA on N user messages) on the local
device. Results print as JSON so they can be pasted into research/experiment-plan.md.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor

PROMPT = "Summarise the trade-offs of monorepos versus polyrepos in about 150 words."


def measure_serve(model: str, base_url: str, concurrency: int, requests: int) -> dict:
    from engram.models.chat import OpenAIChat

    client = OpenAIChat(model=model, base_url=base_url)
    msgs = [{"role": "user", "content": PROMPT}]

    def one(_: int) -> int:
        return len(client.complete(msgs, max_tokens=256).split())

    one(0)  # warm-up
    start = time.time()
    with ThreadPoolExecutor(concurrency) as pool:
        words = sum(pool.map(one, range(requests)))
    elapsed = time.time() - start
    return {
        "mode": "serve",
        "model": model,
        "concurrency": concurrency,
        "requests": requests,
        "seconds": round(elapsed, 2),
        "requests_per_s": round(requests / elapsed, 3),
        "approx_output_tokens_per_s": round(words * 1.3 / elapsed, 1),  # ~1.3 tokens per word
    }


def measure_sleep(base_model: str, messages: int, epochs: int) -> dict:
    import torch

    from engram.bench import dataset
    from engram.systems.lora import LoRAConfig, LoRAMemory

    bench = dataset.generate(dataset.BenchConfig(users=1, sessions=30, seed=0))
    texts = [t.text for s in bench.users[0].sessions for t in s.turns]
    texts = (texts * (messages // max(1, len(texts)) + 1))[:messages]
    system = LoRAMemory(base_model, LoRAConfig(epochs=epochs))
    system.load_user("u")
    system.start_session()
    for t in texts:
        system.observe(t)
    system.end_session()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    system.sleep()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start
    out = {
        "mode": "sleep",
        "base_model": base_model,
        "messages": messages,
        "epochs": epochs,
        "train_steps": system.train_steps,
        "seconds": round(elapsed, 2),
        "device": system.device,
    }
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
        out["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="engram-measure", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--model", required=True)
    s.add_argument("--base-url", default="http://localhost:8000/v1")
    s.add_argument("--concurrency", type=int, default=8)
    s.add_argument("--requests", type=int, default=64)
    z = sub.add_parser("sleep")
    z.add_argument("--base-model", required=True)
    z.add_argument("--messages", type=int, default=150)
    z.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args(argv)
    if args.cmd == "serve":
        result = measure_serve(args.model, args.base_url, args.concurrency, args.requests)
    else:
        result = measure_sleep(args.base_model, args.messages, args.epochs)
    result["host"] = platform.node()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
